"""Score fusion: multi-query aggregation, multi-model ensembling, signal blending.

Conventions
-----------
* A "score map" is ``{global_id: score}``.
* Dense scores are cosine similarities; per-field BM25 scores are unbounded —
  every map is min-max normalized over the candidate pool before blending, so
  configured weights mean the same thing across corpora and queries.
* Model ensembling supports both weighted-sum (default: preserves score
  magnitudes for the UI) and reciprocal-rank fusion (robust when models are
  badly calibrated against each other).
"""

from __future__ import annotations

import numpy as np


def minmax(scores: dict[int, float]) -> dict[int, float]:
    if not scores:
        return {}
    vals = np.fromiter(scores.values(), dtype=np.float64, count=len(scores))
    lo, hi = float(vals.min()), float(vals.max())
    if hi - lo < 1e-9:
        return {k: 1.0 for k in scores}
    # Round-74 perf: one vectorized (v-lo)/(hi-lo) instead of a per-item dict
    # comprehension — same IEEE-754 double ops in the same (v−lo)/(hi−lo)
    # shape, so every value is BIT-identical to the loop it replaced
    # (tests/test_perf_identity.py pins this against a frozen reference).
    normed = (vals - lo) / (hi - lo)
    return dict(zip(scores.keys(), normed.tolist()))


def aggregate_queries(per_query: list[dict[int, float]], how: str = "max") -> dict[int, float]:
    """Combine score maps from query variants (original/translation/expansions)."""
    if not per_query:
        return {}
    if len(per_query) == 1:
        return per_query[0]
    if how == "mean":
        acc: dict[int, list[float]] = {}
        for m in per_query:
            for k, v in m.items():
                acc.setdefault(k, []).append(v)
        return {k: float(np.mean(v)) for k, v in acc.items()}
    # Round-74 perf ("max", the default): a running max in one pass instead of
    # accumulating per-key lists and calling np.max on each (~10× on the
    # 3×2000-gid shape). Key order (first encounter) and values are identical
    # to the old float(np.max(list)) — pinned by tests/test_perf_identity.py.
    out: dict[int, float] = {}
    for m in per_query:
        for k, v in m.items():
            cur = out.get(k)
            if cur is None or v > cur:
                out[k] = v
    return {k: float(v) for k, v in out.items()}


def weighted_sum(maps: list[dict[int, float]], weights: list[float], normalize: bool = True) -> dict[int, float]:
    """Weighted sum over normalized score maps; ids missing from a map get 0."""
    out: dict[int, float] = {}
    for m, w in zip(maps, weights):
        if not m or w == 0:
            continue
        mm = minmax(m) if normalize else m
        for k, v in mm.items():
            out[k] = out.get(k, 0.0) + w * v
    return out


def rrf(maps: list[dict[int, float]], weights: list[float] | None = None, k: int = 60) -> dict[int, float]:
    """Reciprocal-rank fusion: score = Σ w / (k + rank). Rank starts at 1."""
    weights = weights or [1.0] * len(maps)
    out: dict[int, float] = {}
    for m, w in zip(maps, weights):
        for rank, gid in enumerate(sorted(m, key=m.get, reverse=True), start=1):
            out[gid] = out.get(gid, 0.0) + w / (k + rank)
    return out


def neighbor_consistency_boost(
    scores: dict[int, float],
    gid_to_video_span: dict[int, tuple[int, int]],
    weight: float = 0.15,
    window: int = 2,
    decay: float = 0.5,
) -> dict[int, float]:
    """Second-pass consistency boost (round-73, Nhiệm vụ C): plateaus beat spikes.

    Rewards a candidate whose NEIGHBOUR keyframes also carry evidence: the true
    moment in news footage spans several keyframes (a plateau in the fused
    map), while cross-video noise is usually a lone spike whose neighbours
    scored nothing. The complement of :func:`neighbor_boost`, which SPREADS a
    candidate's own score outward (a lone spike still irrigates its
    neighbourhood there); here a candidate only GAINS from what its neighbours
    themselves earned — a spike with silent neighbours gains exactly nothing.

    Formula (researched choices, round-73):

    * normalisation ``n̂(g) = s(g)/max(s)`` — NOT min-max: fused scores are
      already ≥ 0 with a near-zero floor, and min-max would zero out a
      plateau's shoulders whenever they happen to be the candidate-map minimum
      (short candidate lists), erasing exactly the signal we aggregate;
    * support(g) = Σ_{d=1..window} decay^d · (n̂(g−d) + n̂(g+d)) / 2 — both
      sides averaged so a one-sided run counts half; a neighbour outside the
      candidate map (or outside the video span) contributes 0 — that IS the
      spike penalty;
    * out(g) = s(g) + weight · max(s) · support(g)/Σ decay^d — support is
      normalised to [0, 1], so ``weight`` caps the lift at a fixed fraction of
      the top score regardless of window/decay choices.

    ``weight <= 0`` (the knob default), an empty/degenerate map, or a
    non-positive decay returns ``scores`` UNCHANGED (the identical object —
    bit-identical ranking downstream). Never crosses video boundaries.
    """
    if weight <= 0 or window <= 0 or not scores:
        return scores
    top = max(scores.values())
    if top <= 0:
        return scores  # no positive evidence anywhere — nothing to aggregate
    denom = sum(decay ** d for d in range(1, window + 1))
    if denom <= 0:
        return scores
    out = dict(scores)
    for gid, s in scores.items():
        span = gid_to_video_span.get(gid)
        if span is None:
            continue
        start, count = span
        support = 0.0
        for d in range(1, window + 1):
            lo_in = start <= gid - d
            hi_in = gid + d < start + count
            if not lo_in and not hi_in:
                break  # both directions left the video span — no larger d can hit
            left = scores.get(gid - d, 0.0) if lo_in else 0.0
            right = scores.get(gid + d, 0.0) if hi_in else 0.0
            if left > 0 or right > 0:
                support += (decay ** d) * ((left + right) / 2.0) / top
        if support > 0:
            out[gid] = s + weight * top * (support / denom)
    return out


def neighbor_boost(
    scores: dict[int, float],
    gid_to_video_span: dict[int, tuple[int, int]],
    boost: float = 0.10,
    window: int = 2,
) -> dict[int, float]:
    """Temporal-context smoothing: frames adjacent to strong frames get a lift.

    In news footage the target moment usually spans several keyframes; if two
    neighbours both matched, the true frame is very likely between them.
    ``gid_to_video_span`` maps a candidate gid to its video's (start_gid, count)
    so the boost never crosses a video boundary.
    """
    if boost <= 0 or window <= 0 or not scores:
        return scores
    out = dict(scores)
    for gid, s in scores.items():
        span = gid_to_video_span.get(gid)
        if span is None:
            continue
        start, count = span
        for d in range(1, window + 1):
            lo_in = start <= gid - d
            hi_in = gid + d < start + count
            if not lo_in and not hi_in:
                # Both directions left the video span — no larger d can hit
                # (round-10: an oversized neighbor_window burned
                # candidates×window iterations for nothing).
                break
            for other in (gid - d, gid + d):
                if start <= other < start + count and other in scores:
                    contribution = boost * s / d
                    if contribution > 0:
                        out[other] = out.get(other, 0.0) + contribution
    return out
