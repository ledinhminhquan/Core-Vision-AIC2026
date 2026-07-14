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
"""

from __future__ import annotations

import heapq
import logging
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

from cvf.config import Settings, TemporalCfg
from cvf.data.catalog import KeyframeCatalog

log = logging.getLogger(__name__)


class SupportsSearchStore(Protocol):
    """Minimal store surface TRAKE needs (satisfied by ``cvf.index.store.IndexStore``)."""

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


def _resolve_members(
    event_vecs_by_member: dict[str, np.ndarray] | None,
    member_weights: dict[str, float] | None,
    stores_by_member: dict[str, SupportsSearchStore] | None,
) -> list[tuple[str, float, np.ndarray, SupportsSearchStore]]:
    """Validate ensemble kwargs → [(name, weight, event_vecs, store)] (may be empty)."""
    if not event_vecs_by_member:
        return []
    if not stores_by_member:
        log.warning(
            "TRAKE ensemble: event_vecs_by_member given without stores_by_member — "
            "falling back to the primary model"
        )
        return []
    out: list[tuple[str, float, np.ndarray, SupportsSearchStore]] = []
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
        if n_events is None:
            n_events = arr.shape[0]
        elif arr.shape[0] != n_events:
            log.warning(
                "TRAKE ensemble: member %r has %d event vectors, expected %d — member skipped",
                name, arr.shape[0], n_events,
            )
            continue
        weight = float((member_weights or {}).get(name, 1.0))
        if weight <= 0:
            continue
        out.append((name, weight, arr, store))
    return out


def _pooled_event_hits(
    members: list[tuple[str, float, np.ndarray, SupportsSearchStore]],
    topk: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Per-event FAISS hits merged across members.

    Member scores are min-max normalized per event (they live on different
    scales) and weighted before concatenation — pooling only needs candidate
    recall; the exact DP re-scores everything downstream.
    """
    n_events = members[0][2].shape[0]
    scores_acc: list[list[np.ndarray]] = [[] for _ in range(n_events)]
    gids_acc: list[list[np.ndarray]] = [[] for _ in range(n_events)]
    for name, weight, vecs, store in members:
        try:
            scores, gids = store.search(vecs, topk)
        except Exception as e:  # noqa: BLE001 — one bad member must not sink TRAKE
            log.warning("TRAKE ensemble: index search failed for member %r: %s", name, e)
            continue
        for e_i in range(n_events):
            valid = gids[e_i] >= 0
            if not valid.any():
                continue
            scores_acc[e_i].append(_minmax_1d(scores[e_i][valid]) * weight)
            gids_acc[e_i].append(gids[e_i][valid])
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
    members: list[tuple[str, float, np.ndarray, SupportsSearchStore]],
) -> dict[str, np.ndarray]:
    """Weighted per-event fusion of per-member cosine sims for the pooled videos.

    Each member's (frames × events) matrix is min-max normalized **per event
    over ALL pooled videos** (so cross-video ranking stays meaningful);
    degenerate ranges collapse to a neutral 0.5. Videos missing a member's
    embeddings use the remaining members with renormalized weights.
    """
    raw: dict[str, dict[str, np.ndarray]] = {}
    for name, _weight, evecs, store in members:
        mats: dict[str, np.ndarray] = {}
        for vid in videos:
            try:
                vecs = np.asarray(np.load(store.embedding_path(vid)), dtype=np.float32)
            except (OSError, ValueError) as e:
                log.warning("TRAKE ensemble: cannot load %r embeddings for %s: %s", name, vid, e)
                continue
            if len(vecs) != n_rows_by_vid.get(vid, -1):
                log.warning(
                    "TRAKE ensemble: %r embedding/manifest mismatch for %s — member skipped there",
                    name, vid,
                )
                continue
            mats[vid] = _cosine_sim(vecs, evecs).astype(np.float64)
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

    weights = {name: w for name, w, _v, _s in members}
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
    event_vecs_by_member: dict[str, np.ndarray] | None = None,
    member_weights: dict[str, float] | None = None,
    stores_by_member: dict[str, SupportsSearchStore] | None = None,
) -> list[TrakeCandidate]:
    """Full TRAKE pipeline: pool videos → exact per-video DP → global rank.

    Args:
        event_vecs: (k, dim) L2-normalized event queries for the primary model.
        index_search: primary index callable ``(vecs, topk) -> (scores, gids)``.
        embedding_path: primary callable ``video_id -> Path`` of ``(n, dim)`` .npy.
        catalog: keyframe catalog (global_id ↔ video/frame bridge).
        settings: ``settings.temporal`` drives pooling and the DP.
        max_results: global cap on returned candidates.
        algo: ``"dante"`` (exact O(N·T) DP) or ``"beam"``; ``None`` →
            ``settings.temporal.algo``.
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
            ``embedding_path`` (e.g. ``cvf.index.store.IndexStore``).
    """
    cfg = settings.temporal
    chosen = _resolve_algo(settings, algo)

    members = _resolve_members(event_vecs_by_member, member_weights, stores_by_member)
    if len(members) == 1:  # single member ≡ classic path on that member's lanes
        _name, _w, m_vecs, m_store = members[0]
        event_vecs, index_search, embedding_path = m_vecs, m_store.search, m_store.embedding_path
        members = []

    if members:
        event_hits = _pooled_event_hits(members, cfg.per_event_topk)
    else:
        scores, gids = index_search(event_vecs, cfg.per_event_topk)
        event_hits = [(scores[i : i + 1], gids[i : i + 1]) for i in range(len(event_vecs))]
    videos = pool_videos(event_hits, catalog, cfg.max_videos)
    if not videos:
        return []

    rows_by_vid = {vid: catalog.video_rows(vid).sort_values("n") for vid in videos}
    sims_by_vid: dict[str, np.ndarray] = {}
    if members:
        sims_by_vid = _ensemble_video_sims(
            videos, {vid: len(df) for vid, df in rows_by_vid.items()}, members
        )

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
            except (OSError, ValueError) as e:
                log.warning("TRAKE: cannot load embeddings for %s: %s", vid, e)
                continue
            if len(vecs) != len(rows_df):
                log.warning("TRAKE: embedding/manifest mismatch for %s — skipped", vid)
                continue
            sim = _cosine_sim(vecs, event_vecs)
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
            frame_idxs = [int(fidx[r]) for r in rows]
            # DP guarantees increasing keyframe rows/times, but frame_idx can
            # regress on partially-mapped videos (fallback frame_idx = n-1
            # mixed with real values) — such a row would be rejected by DRES.
            if any(b <= a for a, b in zip(frame_idxs, frame_idxs[1:])):
                log.warning("TRAKE: dropped non-increasing frame_idx sequence in %s: %s", vid, frame_idxs)
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
    return heapq.nlargest(max_results, results, key=lambda c: c.score)
