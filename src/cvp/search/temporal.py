"""TRAKE — ordered multi-event search inside one video.

Given k event descriptions, find per-video keyframe sequences
``n_1 < n_2 < ... < n_k`` (strictly increasing in time) that maximize the sum
of event-to-frame similarities, subject to per-step time-gap bounds.

Two-stage, following the CVPRw'25 moment-retrieval recipe:

1. **Pool videos** — each event query hits FAISS for ``per_event_topk``
   candidates; videos are ranked by their best summed per-event evidence.
2. **Exact DP per video** — load that video's full embedding matrix, compute
   the (frames × events) similarity matrix, then run a DP over the exact
   scores (not just the FAISS subset), so weakly-scoring middle events can
   still anchor a sequence.

Two sequence solvers share the gap window ``[min_gap_s, max_gap_s]``, the
``sim_floor`` mask and the soft gap penalty λ = ``gap_penalty_per_s``:

* :func:`dante_best_sequences` — SOICT'25 DANTE exact DP, O(N·T) via a
  sliding-window running max over ``DP[j-1, τ] + λ·time[τ]``
  (default, ``temporal.algo: dante``).
* :func:`dp_best_sequences` — diverse beam DP (``temporal.algo: beam``).

:func:`trake_search` can additionally score events with an **ensemble**: pass
per-member event vectors + index stores and the per-video similarity matrix
becomes the weighted sum of per-member cosine sims, min-max normalized per
event over the pooled videos.

Optional upgrades (session 2026-08-27, every one config-gated, defaults = old
behaviour — see ``TemporalCfg`` in :mod:`cvp.config`):

* **Query variants per event** — event vectors may carry SEVERAL encoded texts
  per event (``event_variant_map`` maps vector rows → event index); per-event
  similarity is the max over that event's variants, mirroring the KIS
  multi-query max-fusion.
* **Pool-context vectors** — ``pool_event_vecs`` (one per event) replace the
  event vectors for the video-POOLING stage only, so the organiser header can
  guide video selection without diluting the DP alignment.
* **Caption step signal** — ``caption_scorer`` (built by
  :func:`caption_scorer_from_signals`) supplies per-event caption-BM25 scores
  per keyframe; ``temporal.caption_signal_weight`` blends them into the DP
  similarity matrix after per-event min-max over the pooled videos.
* **Jitter submit strategy** — ``temporal.submit_strategy: jitter`` keeps the
  head of the ranking and densifies the remaining row budget with frame
  variants around the best per-video chains (:func:`jitter_frame_variants`).
"""

from __future__ import annotations

import heapq
import logging
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol, Sequence

import numpy as np

from cvp.config import Settings, TemporalCfg
from cvp.data.catalog import KeyframeCatalog

log = logging.getLogger(__name__)


class SupportsSearchStore(Protocol):
    """Minimal store surface TRAKE needs (satisfied by ``cvp.index.store.IndexStore``)."""

    def search(self, query_vecs: np.ndarray, topk: int) -> tuple[np.ndarray, np.ndarray]: ...

    def embedding_path(self, video_id: str) -> Path: ...


@dataclass
class TrakeCandidate:
    video_id: str
    ns: list[int]                 # keyframe ordinals, one per event (increasing)
    frame_idxs: list[int]         # what the submission needs
    pts_times: list[float]
    score: float                  # mean of per-event similarities
    per_event: list[float] = field(default_factory=list)


def pool_videos(
    event_hits: list[tuple[np.ndarray, np.ndarray]],
    catalog: KeyframeCatalog,
    max_videos: int,
) -> list[str]:
    """Rank videos by summed best per-event FAISS score; keep top ``max_videos``.

    event_hits: per event, (scores, global_ids) arrays from the dense index.
    """
    df = catalog.load()
    gid_to_vid = df["video_id"].to_numpy()
    per_video: dict[str, np.ndarray] = {}
    k = len(event_hits)
    for e, (scores, gids) in enumerate(event_hits):
        for s, g in zip(scores.ravel(), gids.ravel()):
            if g < 0:
                continue
            vid = gid_to_vid[int(g)]
            arr = per_video.setdefault(vid, np.zeros(k, dtype=np.float32))
            if s > arr[e]:
                arr[e] = s
    # Score = sum of best per-event evidence; videos missing an event get 0 for it
    ranked = sorted(per_video.items(), key=lambda kv: float(kv[1].sum()), reverse=True)
    return [vid for vid, _ in ranked[:max_videos]]


def dp_best_sequences(
    sim: np.ndarray,            # (frames, events) similarity matrix for ONE video
    pts: np.ndarray,            # (frames,) seconds per keyframe
    settings: Settings,
    beam_size: int | None = None,
) -> list[tuple[list[int], float, list[float]]]:
    """Beam DP → up to ``beam_size`` sequences of frame ROW indices (0-based).

    Returns [(rows, mean_score, per_event_scores)], best first.
    """
    cfg = settings.temporal
    beam = beam_size or cfg.beam_size
    n_frames, n_events = sim.shape
    if n_frames == 0 or n_events == 0:
        return []

    # beam of partial paths per event step: (neg_total, rows)
    # Seed: top frames for event 0 above the similarity floor.
    order0 = np.argsort(-sim[:, 0])[: max(beam * 4, 16)]
    paths: list[tuple[float, list[int]]] = [
        (float(sim[r, 0]), [int(r)]) for r in order0 if sim[r, 0] >= cfg.sim_floor
    ]
    if not paths:  # relax: keep best few anyway
        paths = [(float(sim[r, 0]), [int(r)]) for r in order0[:beam]]

    for e in range(1, n_events):
        nxt: list[tuple[float, list[int]]] = []
        col = sim[:, e]
        for total, rows in paths:
            last = rows[-1]
            t_last = pts[last]
            # frames strictly after `last`, inside the time-gap window
            lo_t, hi_t = t_last + cfg.min_gap_s, t_last + cfg.max_gap_s
            candidates = np.where((pts > t_last) & (pts >= lo_t) & (pts <= hi_t))[0]
            candidates = candidates[candidates > last]
            if candidates.size == 0:
                continue
            # soft gap penalty (DANTE): prefer tighter chains at equal similarity
            gains = col[candidates] - cfg.gap_penalty_per_s * (pts[candidates] - t_last)
            top = candidates[np.argsort(-gains)[: beam]]
            for r in top:
                s = float(col[r])
                gain = float(s - cfg.gap_penalty_per_s * (pts[r] - t_last))
                if s < cfg.sim_floor and len(nxt) >= beam:
                    continue
                nxt.append((total + gain, rows + [int(r)]))
        if not nxt:
            return []
        # Keep the best `beam` per distinct last-frame to preserve diversity
        nxt.sort(key=lambda t: -t[0])
        seen_last: set[int] = set()
        pruned: list[tuple[float, list[int]]] = []
        for total, rows in nxt:
            if rows[-1] in seen_last and len(pruned) >= beam:
                continue
            seen_last.add(rows[-1])
            pruned.append((total, rows))
            if len(pruned) >= beam * 4:
                break
        paths = pruned

    paths.sort(key=lambda t: -t[0])
    out = []
    for total, rows in paths[:beam]:
        per_event = [float(sim[r, e]) for e, r in enumerate(rows)]
        out.append((rows, total / n_events, per_event))
    return out


def dante_best_sequences(
    sim: np.ndarray,
    times: np.ndarray,
    cfg: Settings | TemporalCfg,
    top_n: int | None = None,
) -> list[tuple[tuple[int, ...], float, list[float]]]:
    """DANTE exact DP (SOICT'25): O(N·T) chain search with a soft gap penalty.

    Recurrence for event ``j`` at frame ``t`` (λ = ``gap_penalty_per_s``)::

        DP[j, t] = sim[t, j] + max_τ (DP[j-1, τ] − λ·(time[t] − time[τ]))
                 = sim[t, j] − λ·time[t] + max_τ (DP[j-1, τ] + λ·time[τ])

    where ``τ`` ranges over frames strictly before ``t`` whose time gap to
    ``t`` lies in ``[min_gap_s, max_gap_s]``. The inner max is a running max
    over ``G[τ] = DP[j-1, τ] + λ·time[τ]`` maintained with a monotonic deque
    (sliding-window maximum), so each event costs O(N) instead of O(N²).
    Frames with similarity below ``sim_floor`` cannot host an event; the mask
    is relaxed per event when it would eliminate every frame (mirrors the
    beam solver's seed relaxation).

    Args:
        sim: (n_frames, n_events) similarity matrix for ONE video.
        times: (n_frames,) seconds per keyframe, non-decreasing.
        cfg: a ``TemporalCfg`` — or a full ``Settings``, whose ``.temporal``
            section is used.
        top_n: sequences to return (default ``cfg.beam_size``). Returned
            sequences have distinct last frames by construction, matching the
            beam solver's diversity guarantee.

    Returns:
        [(frame_rows_tuple, mean_score, per_event_scores)], best first, where
        ``mean_score`` is the gap-penalized chain total divided by n_events
        (same scoring as :func:`dp_best_sequences`).
    """
    tcfg: TemporalCfg = getattr(cfg, "temporal", cfg)
    sim64 = np.asarray(sim, dtype=np.float64)
    t64 = np.asarray(times, dtype=np.float64).ravel()
    if sim64.ndim != 2:
        return []
    n_frames, n_events = sim64.shape
    if n_frames == 0 or n_events == 0 or t64.shape[0] != n_frames:
        return []
    if n_frames > 1 and bool(np.any(np.diff(t64) < 0)):
        log.warning("DANTE: frame times are not non-decreasing — skipping this video")
        return []

    lam = float(tcfg.gap_penalty_per_s)
    floor = float(tcfg.sim_floor)
    keep = int(top_n or tcfg.beam_size)

    # sim_floor masking, relaxed per event when it would kill the whole column.
    eligible = sim64 >= floor
    for j in range(n_events):
        if not eligible[:, j].any():
            eligible[:, j] = True

    dp = np.full((n_events, n_frames), -np.inf, dtype=np.float64)
    parent = np.full((n_events, n_frames), -1, dtype=np.int64)
    dp[0, eligible[:, 0]] = sim64[eligible[:, 0], 0]

    for j in range(1, n_events):
        g = dp[j - 1] + lam * t64          # running-max payload G[τ]
        dq: deque[int] = deque()           # τ indices, G non-increasing front→back
        r = 0                              # next τ to consider for admission
        col = sim64[:, j]
        ok = eligible[:, j]
        for t in range(n_frames):
            # Admit every τ < t that now satisfies the minimum gap (times are
            # non-decreasing, so once admissible a τ stays admissible on the
            # min side for all later t).
            while r < t and t64[r] < t64[t] and (t64[t] - t64[r]) >= tcfg.min_gap_s:
                if np.isfinite(g[r]):
                    while dq and g[dq[-1]] <= g[r]:
                        dq.pop()
                    dq.append(r)
                r += 1
            # Evict τ that fell out of the maximum-gap window (oldest = front).
            while dq and (t64[t] - t64[dq[0]]) > tcfg.max_gap_s:
                dq.popleft()
            if dq and ok[t]:
                tau = dq[0]
                dp[j, t] = col[t] + g[tau] - lam * t64[t]
                parent[j, t] = tau

    finals = dp[n_events - 1]
    order = np.argsort(-finals)
    out: list[tuple[tuple[int, ...], float, list[float]]] = []
    for t in order[:keep]:
        total = float(finals[int(t)])
        if not np.isfinite(total):
            break  # argsort(desc) puts -inf last — nothing valid beyond here
        rows = [int(t)]
        cur, j = int(t), n_events - 1
        while j > 0:
            cur = int(parent[j, cur])
            if cur < 0:  # defensive: broken back-pointer chain
                rows = []
                break
            rows.append(cur)
            j -= 1
        if not rows:
            continue
        rows.reverse()
        per_event = [float(sim64[r, e]) for e, r in enumerate(rows)]
        out.append((tuple(rows), total / n_events, per_event))
    return out


# ── ensemble helpers ─────────────────────────────────────────────────────────


def _cosine_sim(vecs: np.ndarray, event_vecs: np.ndarray) -> np.ndarray:
    """(n, d) frame vectors × (k, d) L2-normalized event vectors → (n, k) sims."""
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (vecs / norms) @ np.asarray(event_vecs, dtype=np.float32).T


def _identity_map(n: int) -> list[int]:
    return list(range(n))


def _n_events_of(variant_map: Sequence[int]) -> int:
    return (int(max(variant_map)) + 1) if len(variant_map) else 0


def _reduce_variants(sim: np.ndarray, variant_map: Sequence[int], n_events: int) -> np.ndarray:
    """(n_frames, n_variant_rows) sims → (n_frames, n_events) via per-event max.

    Mirrors the KIS multi-query max-fusion (``fusion.aggregate_queries`` with
    ``how="max"``): an event scores with its best-matching text variant. The
    identity map returns ``sim`` unchanged (the classic single-text path).
    """
    vmap = list(variant_map)
    if vmap == _identity_map(n_events) and sim.shape[1] == n_events:
        return sim
    out = np.full((sim.shape[0], n_events), -np.inf, dtype=sim.dtype)
    for row, event in enumerate(vmap):
        np.maximum(out[:, event], sim[:, row], out=out[:, event])
    out[~np.isfinite(out)] = 0.0  # events with no variant rows (defensive)
    return out


def monotonize_frame_idxs(frame_idxs: list[int]) -> list[int] | None:
    """Make a DP chain's frame_idx sequence strictly increasing, or reject it.

    The official map csvs contain adjacent keyframes with EQUAL frame_idx
    (2026 Batch-1: 614 tied pairs across 192/873 fully-mapped videos), so a
    valid chain may carry ties — bump each tied frame to predecessor+1 (GT
    windows span ~10 frames, so +1 stays inside the answer window). A genuine
    DECREASE only arises from fallback estimates mixed with real map values;
    that chain is garbage → return None.
    """
    if any(b < a for a, b in zip(frame_idxs, frame_idxs[1:])):
        return None
    out = list(frame_idxs)
    for i in range(1, len(out)):
        if out[i] <= out[i - 1]:
            out[i] = out[i - 1] + 1
    return out


def _minmax_1d(a: np.ndarray) -> np.ndarray:
    """Min-max normalize a 1-D array; degenerate ranges map to a neutral 0.5."""
    if a.size == 0:
        return a.astype(np.float64)
    lo, hi = float(a.min()), float(a.max())
    if hi - lo < 1e-9:
        return np.full(a.shape, 0.5, dtype=np.float64)
    return (a.astype(np.float64) - lo) / (hi - lo)


def _resolve_algo(settings: Settings, algo: str | None) -> str:
    """Explicit ``algo`` arg > ``settings.temporal.algo`` > 'dante'."""
    name = str(algo or getattr(settings.temporal, "algo", "dante") or "dante").strip().lower()
    if name not in ("dante", "beam"):
        log.warning("TRAKE: unknown temporal algo %r — falling back to 'dante'", name)
        return "dante"
    return name


# One resolved ensemble member: (name, weight, event_vecs, store, variant_map,
# pool_event_vecs) — variant_map maps event-vec ROWS → event index (identity
# when the member has one text per event); pool_event_vecs (n_events, d) or
# None replace the vectors for the video-POOLING stage only.
Member = tuple[str, float, np.ndarray, SupportsSearchStore, list[int], "np.ndarray | None"]


def _resolve_members(
    event_vecs_by_member: dict[str, np.ndarray] | None,
    member_weights: dict[str, float] | None,
    stores_by_member: dict[str, SupportsSearchStore] | None,
    variant_maps_by_member: dict[str, Sequence[int]] | None = None,
    pool_vecs_by_member: dict[str, np.ndarray] | None = None,
) -> list[Member]:
    """Validate ensemble kwargs → resolved members (may be empty)."""
    if not event_vecs_by_member:
        return []
    if not stores_by_member:
        log.warning(
            "TRAKE ensemble: event_vecs_by_member given without stores_by_member — "
            "falling back to the primary model"
        )
        return []
    out: list[Member] = []
    n_events: int | None = None
    for name, vecs in event_vecs_by_member.items():
        store = stores_by_member.get(name)
        if store is None:
            log.warning("TRAKE ensemble: no store for member %r — member skipped", name)
            continue
        arr = np.asarray(vecs, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] == 0:
            log.warning("TRAKE ensemble: bad event vectors for member %r — member skipped", name)
            continue
        vmap = list((variant_maps_by_member or {}).get(name) or _identity_map(arr.shape[0]))
        if len(vmap) != arr.shape[0]:
            log.warning(
                "TRAKE ensemble: member %r variant map has %d entries for %d vectors "
                "— member skipped", name, len(vmap), arr.shape[0],
            )
            continue
        member_events = _n_events_of(vmap)
        if n_events is None:
            n_events = member_events
        elif member_events != n_events:
            log.warning(
                "TRAKE ensemble: member %r covers %d events, expected %d — member skipped",
                name, member_events, n_events,
            )
            continue
        pool = (pool_vecs_by_member or {}).get(name)
        if pool is not None:
            pool = np.asarray(pool, dtype=np.float32)
            if pool.ndim != 2 or pool.shape[0] != member_events:
                log.warning(
                    "TRAKE ensemble: member %r pool vectors have shape %s, expected "
                    "(%d, d) — pooling falls back to the event vectors",
                    name, getattr(pool, "shape", None), member_events,
                )
                pool = None
        weight = float((member_weights or {}).get(name, 1.0))
        if weight <= 0:
            continue
        out.append((name, weight, arr, store, vmap, pool))
    return out


def _pooled_event_hits(
    members: list[Member],
    topk: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Per-event FAISS hits merged across members.

    Member scores are min-max normalized per event (they live on different
    scales) and weighted before concatenation — pooling only needs candidate
    recall; the exact DP re-scores everything downstream. A member's variant
    rows all feed their event's hit list; ``pool_event_vecs`` (when present)
    replace the event vectors for this stage only.
    """
    n_events = _n_events_of(members[0][4])
    scores_acc: list[list[np.ndarray]] = [[] for _ in range(n_events)]
    gids_acc: list[list[np.ndarray]] = [[] for _ in range(n_events)]
    for name, weight, vecs, store, vmap, pool_vecs in members:
        search_vecs, search_map = (
            (pool_vecs, _identity_map(n_events)) if pool_vecs is not None else (vecs, vmap)
        )
        try:
            scores, gids = store.search(search_vecs, topk)
        except Exception as e:  # noqa: BLE001 — one bad member must not sink TRAKE
            log.warning("TRAKE ensemble: index search failed for member %r: %s", name, e)
            continue
        for row, e_i in enumerate(search_map):
            valid = gids[row] >= 0
            if not valid.any():
                continue
            scores_acc[e_i].append(_minmax_1d(scores[row][valid]) * weight)
            gids_acc[e_i].append(gids[row][valid])
    out: list[tuple[np.ndarray, np.ndarray]] = []
    for e_i in range(n_events):
        if scores_acc[e_i]:
            out.append((np.concatenate(scores_acc[e_i]), np.concatenate(gids_acc[e_i])))
        else:
            out.append((np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.int64)))
    return out


def _ensemble_video_sims(
    videos: list[str],
    n_rows_by_vid: dict[str, int],
    members: list[Member],
) -> dict[str, np.ndarray]:
    """Weighted per-event fusion of per-member cosine sims for the pooled videos.

    Each member's (frames × events) matrix is min-max normalized **per event
    over ALL pooled videos** (so cross-video ranking stays meaningful);
    degenerate ranges collapse to a neutral 0.5. Videos missing a member's
    embeddings use the remaining members with renormalized weights. Members
    carrying several text variants per event reduce to (frames × events) via
    per-event max BEFORE normalization.
    """
    raw: dict[str, dict[str, np.ndarray]] = {}
    for name, _weight, evecs, store, vmap, _pool in members:
        n_events = _n_events_of(vmap)
        mats: dict[str, np.ndarray] = {}
        for vid in videos:
            try:
                vecs = np.asarray(np.load(store.embedding_path(vid)), dtype=np.float32)
            except (EOFError, OSError, ValueError) as e:
                log.warning("TRAKE ensemble: cannot load %r embeddings for %s: %s", name, vid, e)
                continue
            if len(vecs) != n_rows_by_vid.get(vid, -1):
                log.warning(
                    "TRAKE ensemble: %r embedding/manifest mismatch for %s — member skipped there",
                    name, vid,
                )
                continue
            mats[vid] = _reduce_variants(
                _cosine_sim(vecs, evecs), vmap, n_events
            ).astype(np.float64)
        if mats:
            raw[name] = mats

    norm: dict[str, dict[str, np.ndarray]] = {}
    for name, mats in raw.items():
        stacked = np.concatenate(list(mats.values()), axis=0)
        lo = stacked.min(axis=0)
        span = stacked.max(axis=0) - lo
        degenerate = span < 1e-9
        span = np.where(degenerate, 1.0, span)
        normed: dict[str, np.ndarray] = {}
        for vid, m in mats.items():
            mm = (m - lo) / span
            if degenerate.any():
                mm[:, degenerate] = 0.5
            normed[vid] = mm
        norm[name] = normed

    weights = {name: w for name, w, _v, _s, _m, _p in members}
    combined: dict[str, np.ndarray] = {}
    for vid in videos:
        acc: np.ndarray | None = None
        w_sum = 0.0
        for name, mats in norm.items():
            m = mats.get(vid)
            if m is None:
                continue
            w = weights[name]
            acc = m * w if acc is None else acc + m * w
            w_sum += w
        if acc is not None and w_sum > 0:
            combined[vid] = (acc / w_sum).astype(np.float32)
    return combined


# ── caption step signal ──────────────────────────────────────────────────────


def caption_scorer_from_signals(
    text_signals, catalog: KeyframeCatalog, event_texts: list[str]
) -> "Callable[[str], np.ndarray | None]":
    """Build a ``video_id → (n_rows, K) raw caption-BM25 matrix`` closure.

    ``text_signals`` needs only ``score_field`` (``cvp.search.text_signals.
    TextSignals``); events are scored with their ORIGINAL Vietnamese text —
    captions are Vietnamese (Vintern), same convention as the KIS BM25 path.
    Returns None for a video with no caption match on any event, so absent
    artifacts degrade to a no-op upstream.
    """

    def _score(video_id: str) -> np.ndarray | None:
        rows = catalog.video_rows(video_id).sort_values("n")
        refs = catalog.refs([int(g) for g in rows["global_id"]])
        cols: list[list[float]] = []
        any_signal = False
        for text in event_texts:
            try:
                m = text_signals.score_field("caption", text, refs) or {}
            except Exception as e:  # noqa: BLE001 — aux signal must never sink TRAKE
                log.warning("TRAKE caption signal failed for %s: %s", video_id, e)
                m = {}
            if m:
                any_signal = True
            cols.append([float(m.get(r.global_id, 0.0)) for r in refs])
        if not any_signal:
            return None
        return np.asarray(cols, dtype=np.float64).T  # (n_rows, K)

    return _score


def _normalized_caption_by_vid(
    videos: list[str],
    n_rows_by_vid: dict[str, int],
    caption_scorer: "Callable[[str], np.ndarray | None]",
    n_events: int,
) -> dict[str, np.ndarray]:
    """Per-video caption matrices, min-max normalized per event over the pool.

    BM25 absent = no signal: a degenerate per-event range (all scores equal,
    typically all zero) maps to 0 — NOT the dense path's neutral 0.5 — so a
    corpus without caption artifacts leaves the DP matrix untouched.
    """
    raw: dict[str, np.ndarray] = {}
    for vid in videos:
        try:
            m = caption_scorer(vid)
        except Exception as e:  # noqa: BLE001 — aux signal must never sink TRAKE
            log.warning("TRAKE caption scorer failed for %s: %s", vid, e)
            continue
        if m is None:
            continue
        m = np.asarray(m, dtype=np.float64)
        if m.ndim != 2 or m.shape[0] != n_rows_by_vid.get(vid, -1) or m.shape[1] != n_events:
            log.warning(
                "TRAKE caption signal: matrix for %s has shape %s, expected (%d, %d) — skipped",
                vid, m.shape, n_rows_by_vid.get(vid, -1), n_events,
            )
            continue
        raw[vid] = m
    if not raw:
        return {}
    stacked = np.concatenate(list(raw.values()), axis=0)
    lo = stacked.min(axis=0)
    span = stacked.max(axis=0) - lo
    dead = span < 1e-9
    span = np.where(dead, 1.0, span)
    out: dict[str, np.ndarray] = {}
    for vid, m in raw.items():
        mm = (m - lo) / span
        if dead.any():
            mm[:, dead] = 0.0
        out[vid] = mm.astype(np.float32)
    return out


# ── jitter submit strategy ───────────────────────────────────────────────────

# Rows of the legacy ranking kept verbatim before the jitter blocks start:
# R@1/R@5 stay EXACTLY what the legacy strategy would score, jitter only
# densifies the deeper cutoffs (R@20/50/100).
JITTER_HEAD = 8


def jitter_frame_variants(rows: Sequence[int], fidx: np.ndarray) -> list[list[int]]:
    """Ordered frame-tuple alternates around one chain (base tuple excluded).

    The submitted frame need NOT be a keyframe — the keyframe grid is sparse
    (~5-7s at Batch-1 density) while GT windows are tight, so midpoints
    BETWEEN keyframes can hit windows no grid frame can. Early variants come
    first: TRAKE queries ask for the FIRST moment of an action, which usually
    starts between the previous keyframe and the one the DP picked.

    Args:
        rows: per-event row indices into the video's keyframe grid (increasing).
        fidx: the video's full frame_idx array (catalog rows sorted by ``n``).

    Returns:
        Frame tuples, best-guess first, each strictly increasing, deduped,
        never containing the base tuple.
    """
    base = [int(fidx[r]) for r in rows]

    def _early(j: int) -> int | None:  # midpoint toward the previous keyframe
        r = int(rows[j])
        return (int(fidx[r - 1]) + int(fidx[r])) // 2 if r >= 1 else None

    def _late(j: int) -> int | None:   # midpoint toward the next keyframe
        r = int(rows[j])
        return (int(fidx[r]) + int(fidx[r + 1])) // 2 if r + 1 < len(fidx) else None

    def _prev(j: int) -> int | None:
        r = int(rows[j])
        return int(fidx[r - 1]) if r >= 1 else None

    def _next(j: int) -> int | None:
        r = int(rows[j])
        return int(fidx[r + 1]) if r + 1 < len(fidx) else None

    k = len(rows)

    def _subst(fn, idxs) -> list[int]:
        out = list(base)
        for j in idxs:
            v = fn(j)
            if v is not None:
                out[j] = int(v)
        return out

    cands: list[list[int]] = [_subst(_early, range(k))]
    cands += [_subst(_early, [j]) for j in range(k)]
    cands.append(_subst(_late, range(k)))
    cands += [_subst(_late, [j]) for j in range(k)]
    cands += [_subst(_prev, [j]) for j in range(k)]
    cands += [_subst(_next, [j]) for j in range(k)]

    seen: set[tuple[int, ...]] = {tuple(base)}
    out: list[list[int]] = []
    for c in cands:
        t = tuple(c)
        if t in seen or c[0] < 0:
            continue
        if any(b <= a for a, b in zip(c, c[1:])):
            continue
        seen.add(t)
        out.append(c)
    return out


def expand_candidates_jitter(
    ranked: "list[TrakeCandidate]",
    rows_by_vid: dict,
    cfg: TemporalCfg,
    max_results: int,
) -> "list[TrakeCandidate]":
    """Densify the row budget around the best per-video chains.

    Layout: the first :data:`JITTER_HEAD` legacy rows verbatim → jitter blocks
    of the best chain of the top ``cfg.jitter_videos`` distinct videos (rank
    order) → the remaining legacy rows. Duplicates collapse, ``max_results``
    caps the total. Variant candidates reuse the base chain's ``ns``/pts (the
    grid rows they were derived from) with an epsilon-decayed score so any
    downstream score sort preserves this order.
    """
    if not ranked:
        return ranked
    picked: dict[str, TrakeCandidate] = {}
    for c in ranked:
        if c.video_id not in picked:
            picked[c.video_id] = c
            if len(picked) >= int(getattr(cfg, "jitter_videos", 4)):
                break
    blocks: list[TrakeCandidate] = []
    for vid, cand in picked.items():
        rows_df = rows_by_vid.get(vid)
        if rows_df is None:
            continue
        ns = rows_df["n"].to_numpy(dtype=np.int64)
        fidx = rows_df["frame_idx"].to_numpy(dtype=np.int64)
        # Round-72 (Cursor-lab audit): video map DỞ DANG trộn frame thật với
        # ước lượng n-1 — neighbor/midpoint tính từ mảng thô sẽ sinh frame rác
        # ở các dòng sâu. Video chưa map đủ thì giữ nguyên chuỗi gốc, bỏ jitter.
        if "has_map" in rows_df.columns and not bool(rows_df["has_map"].all()):
            continue
        pos = {int(n): i for i, n in enumerate(ns)}
        try:
            rows = [pos[int(n)] for n in cand.ns]
        except KeyError:  # defensive: chain ns not on this grid
            continue
        for i, frames in enumerate(jitter_frame_variants(rows, fidx)):
            blocks.append(
                TrakeCandidate(
                    video_id=vid,
                    ns=list(cand.ns),
                    frame_idxs=frames,
                    pts_times=list(cand.pts_times),
                    score=cand.score - 1e-6 * (i + 1),
                    per_event=list(cand.per_event),
                )
            )
    seen: set[tuple] = set()
    out: list[TrakeCandidate] = []
    for c in [*ranked[:JITTER_HEAD], *blocks, *ranked[JITTER_HEAD:]]:
        key = (c.video_id, tuple(int(f) for f in c.frame_idxs))
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
        if len(out) >= max_results:
            break
    return out


# ── full pipeline ────────────────────────────────────────────────────────────


def trake_search(
    event_vecs: np.ndarray,     # (k, dim) L2-normalized event query vectors
    index_search,               # callable (vecs, topk) -> (scores, gids)
    embedding_path,             # callable (video_id) -> Path to (n, dim) .npy
    catalog: KeyframeCatalog,
    settings: Settings,
    max_results: int = 100,
    *,
    algo: str | None = None,
    event_variant_map: Sequence[int] | None = None,
    pool_event_vecs: np.ndarray | None = None,
    caption_scorer: "Callable[[str], np.ndarray | None] | None" = None,
    event_vecs_by_member: dict[str, np.ndarray] | None = None,
    member_weights: dict[str, float] | None = None,
    stores_by_member: dict[str, SupportsSearchStore] | None = None,
    variant_maps_by_member: dict[str, Sequence[int]] | None = None,
    pool_vecs_by_member: dict[str, np.ndarray] | None = None,
) -> list[TrakeCandidate]:
    """Full TRAKE pipeline: pool videos → exact per-video DP → global rank.

    Args:
        event_vecs: (k, dim) L2-normalized event queries for the primary model
            — or (R, dim) variant rows when ``event_variant_map`` is given.
        index_search: primary index callable ``(vecs, topk) -> (scores, gids)``.
        embedding_path: primary callable ``video_id -> Path`` of ``(n, dim)`` .npy.
        catalog: keyframe catalog (global_id ↔ video/frame bridge).
        settings: ``settings.temporal`` drives pooling and the DP.
        max_results: global cap on returned candidates.
        algo: ``"dante"`` (exact O(N·T) DP) or ``"beam"``; ``None`` →
            ``settings.temporal.algo``.
        event_variant_map: optional row→event index map for ``event_vecs``
            (``temporal.event_query_variants: all``); per-event similarity is
            the max over that event's variant rows. None = one row per event.
        pool_event_vecs: optional (n_events, dim) vectors used for the video
            POOLING stage only (``temporal.pool_context: prepend``); the DP
            keeps scoring with ``event_vecs``.
        caption_scorer: optional ``video_id → (n_rows, K) raw caption-BM25``
            (see :func:`caption_scorer_from_signals`); blended into the DP
            matrix when ``settings.temporal.caption_signal_weight > 0``.
        event_vecs_by_member: optional ENSEMBLE scoring — per-member event
            vectors, e.g. ``{"siglip2": (k, d1), "openclip": (k, d2)}``. When
            given (with ``stores_by_member``), per-video similarity matrices
            are the weighted sum of per-member cosine sims after min-max
            normalization per event over the pooled videos (degenerate ranges
            → neutral 0.5); ``event_vecs``/``index_search``/``embedding_path``
            are then ignored. With exactly one member this reduces to the
            classic single-model path on that member's store.
        member_weights: per-member fusion weights (default 1.0 each,
            renormalized over the members available per video).
        stores_by_member: per-member stores exposing ``search`` and
            ``embedding_path`` (e.g. ``cvp.index.store.IndexStore``).
        variant_maps_by_member: optional per-member row→event maps (same
            semantics as ``event_variant_map``).
        pool_vecs_by_member: optional per-member pooling-only vectors (same
            semantics as ``pool_event_vecs``).
    """
    cfg = settings.temporal
    chosen = _resolve_algo(settings, algo)

    members = _resolve_members(event_vecs_by_member, member_weights, stores_by_member,
                               variant_maps_by_member, pool_vecs_by_member)
    if len(members) == 1:  # single member ≡ classic path on that member's lanes
        _name, _w, m_vecs, m_store, m_vmap, m_pool = members[0]
        event_vecs, index_search, embedding_path = m_vecs, m_store.search, m_store.embedding_path
        event_variant_map, pool_event_vecs = m_vmap, m_pool
        members = []

    event_vecs = np.asarray(event_vecs, dtype=np.float32)
    vmap = list(event_variant_map) if event_variant_map is not None else _identity_map(len(event_vecs))
    if len(vmap) != len(event_vecs):
        log.warning("TRAKE: variant map has %d entries for %d event vectors — ignoring it",
                    len(vmap), len(event_vecs))
        vmap = _identity_map(len(event_vecs))
    n_events = _n_events_of(members[0][4]) if members else _n_events_of(vmap)

    if pool_event_vecs is not None and len(pool_event_vecs) != n_events:
        log.warning("TRAKE: pool vectors have %d rows, expected %d — pooling on event vectors",
                    len(pool_event_vecs), n_events)
        pool_event_vecs = None

    if members:
        event_hits = _pooled_event_hits(members, cfg.per_event_topk)
    else:
        if pool_event_vecs is not None:
            scores, gids = index_search(pool_event_vecs, cfg.per_event_topk)
            event_hits = [(scores[i : i + 1], gids[i : i + 1]) for i in range(n_events)]
        else:
            scores, gids = index_search(event_vecs, cfg.per_event_topk)
            if vmap == _identity_map(n_events):  # classic: one row per event
                event_hits = [(scores[i : i + 1], gids[i : i + 1]) for i in range(n_events)]
            else:  # variant rows all feed their event's hit pool
                event_hits = []
                for e_i in range(n_events):
                    rows_e = [r for r, e in enumerate(vmap) if e == e_i]
                    if rows_e:
                        event_hits.append((
                            np.concatenate([np.ravel(scores[r]) for r in rows_e]),
                            np.concatenate([np.ravel(gids[r]) for r in rows_e]),
                        ))
                    else:  # gap in the variant map — that event has no evidence
                        event_hits.append((np.zeros(0, dtype=np.float64),
                                           np.zeros(0, dtype=np.int64)))
    videos = pool_videos(event_hits, catalog, cfg.max_videos)
    if not videos:
        return []

    rows_by_vid = {vid: catalog.video_rows(vid).sort_values("n") for vid in videos}
    sims_by_vid: dict[str, np.ndarray] = {}
    if members:
        sims_by_vid = _ensemble_video_sims(
            videos, {vid: len(df) for vid, df in rows_by_vid.items()}, members
        )

    cap_w = float(getattr(cfg, "caption_signal_weight", 0.0) or 0.0)
    cap_by_vid: dict[str, np.ndarray] = {}
    if cap_w > 0 and caption_scorer is not None:
        cap_by_vid = _normalized_caption_by_vid(
            videos, {vid: len(df) for vid, df in rows_by_vid.items()},
            caption_scorer, n_events,
        )
        if not cap_by_vid:
            # Round-72 (Cursor-lab audit): knob bật mà không thu được tín hiệu
            # nào = chạy Y HỆT baseline — phải LA LỚN, kẻo lượt bench A/B đo ra
            # số trùng baseline và kênh caption bị kết án oan là vô dụng.
            log.warning(
                "TRAKE caption_signal_weight=%.2f nhưng KHÔNG thu được tín hiệu "
                "caption nào từ pool %d video — kết quả sẽ Y HỆT baseline "
                "(kho captions/text_index thiếu trên máy này?)", cap_w, len(videos))

    results: list[TrakeCandidate] = []
    for vid in videos:
        rows_df = rows_by_vid[vid]
        if members:
            sim = sims_by_vid.get(vid)
            if sim is None:
                continue
        else:
            try:
                vecs = np.asarray(np.load(embedding_path(vid)), dtype=np.float32)
            except (EOFError, OSError, ValueError) as e:
                log.warning("TRAKE: cannot load embeddings for %s: %s", vid, e)
                continue
            if len(vecs) != len(rows_df):
                log.warning("TRAKE: embedding/manifest mismatch for %s — skipped", vid)
                continue
            sim = _reduce_variants(_cosine_sim(vecs, event_vecs), vmap, n_events)
        cap = cap_by_vid.get(vid)
        if cap is not None and cap.shape == sim.shape:
            sim = sim + cap_w * cap
        pts = rows_df["pts_time"].to_numpy(dtype=np.float64)
        fidx = rows_df["frame_idx"].to_numpy(dtype=np.int64)
        ns = rows_df["n"].to_numpy(dtype=np.int64)

        if chosen == "beam":
            seqs = dp_best_sequences(sim, pts, settings)
        else:
            seqs = dante_best_sequences(sim, pts, settings)
            if not seqs and len(pts) > 1 and bool(np.any(np.diff(pts) < 0)):
                # Partially-mapped videos can have non-monotonic pts (real map
                # rows mixed with fallback estimates). DANTE requires sorted
                # times, but the beam solver checks gaps per step — degrade to
                # it rather than dropping the whole video from TRAKE.
                log.warning(
                    "TRAKE: %s has non-monotonic times — falling back to beam DP", vid
                )
                seqs = dp_best_sequences(sim, pts, settings)
        for rows, mean_score, per_event in seqs:
            frame_idxs = monotonize_frame_idxs([int(fidx[r]) for r in rows])
            # A genuine DECREASE means garbage frame_idx (fallback n-1 mixed
            # with real map values on partially-mapped videos) — drop; but
            # EQUAL neighbours are real: the official Batch-1 map csvs contain
            # 614 tied adjacent frame_idx pairs across 192/873 videos, and a
            # legal DP chain may land on one. monotonize bumps ties by +1
            # (GT windows span ~10 frames) instead of losing the candidate.
            if frame_idxs is None:
                log.warning("TRAKE: dropped decreasing frame_idx sequence in %s", vid)
                continue
            results.append(
                TrakeCandidate(
                    video_id=vid,
                    ns=[int(ns[r]) for r in rows],
                    frame_idxs=frame_idxs,
                    pts_times=[float(pts[r]) for r in rows],
                    score=float(mean_score),
                    per_event=list(per_event),
                )
            )

    results.sort(key=lambda c: -c.score)
    ranked = heapq.nlargest(max_results, results, key=lambda c: c.score)
    if str(getattr(cfg, "submit_strategy", "legacy")) == "jitter" and ranked:
        ranked = expand_candidates_jitter(ranked, rows_by_vid, cfg, max_results)
    return ranked
