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
    vals = np.fromiter(scores.values(), dtype=np.float64)
    lo, hi = float(vals.min()), float(vals.max())
    if hi - lo < 1e-9:
        return {k: 1.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


def aggregate_queries(per_query: list[dict[int, float]], how: str = "max") -> dict[int, float]:
    """Combine score maps from query variants (original/translation/expansions)."""
    if not per_query:
        return {}
    if len(per_query) == 1:
        return per_query[0]
    out: dict[int, list[float]] = {}
    for m in per_query:
        for k, v in m.items():
            out.setdefault(k, []).append(v)
    if how == "mean":
        return {k: float(np.mean(v)) for k, v in out.items()}
    return {k: float(np.max(v)) for k, v in out.items()}


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
            for other in (gid - d, gid + d):
                if start <= other < start + count and other in scores:
                    contribution = boost * s / d
                    if contribution > 0:
                        out[other] = out.get(other, 0.0) + contribution
    return out
