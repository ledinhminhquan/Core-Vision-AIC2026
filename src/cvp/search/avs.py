"""AVS (Ad-hoc Video Search) ranking: many *distinct* relevant moments.

AVS is scored against a hidden relevance list — reward comes from covering as
many distinct correct segments as possible, so flooding the 100 rows with
near-identical frames of one video wastes slots. This module re-orders a
ranked result list for coverage:

* at most ``per_video_cap`` rows per video,
* within a video, picked frames must be ≥ ``min_gap_s`` apart (distinct segments),
* optionally MMR over candidate embeddings: near-duplicate scenes across
  *different* videos (news reuse the same b-roll) also waste slots — with
  ``cand_vecs`` given, each pick trades relevance against novelty:
  ``mmr_lambda·score − (1−mmr_lambda)·max cos(candidate, already picked)``.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # avoid circular import — engine imports this module
    from cvp.search.engine import SearchResult


def avs_diversify(
    results: list["SearchResult"],
    per_video_cap: int = 3,
    min_gap_s: float = 10.0,
    limit: int = 100,
    cand_vecs: np.ndarray | None = None,
    mmr_lambda: float = 1.0,
) -> list["SearchResult"]:
    """Coverage-first selection from a score-ranked result list.

    ``cand_vecs`` (len(results), dim), L2-normalized, row-aligned with
    ``results``; with ``mmr_lambda < 1`` selection becomes greedy MMR instead
    of pure score order. Zero rows (missing embeddings) fall back to score-only.
    """
    use_mmr = (
        cand_vecs is not None
        and len(cand_vecs) == len(results)
        and 0.0 <= mmr_lambda < 1.0
        and len(results) > 0
    )
    picked: list[SearchResult] = []
    picked_rows: list[int] = []
    per_video_times: dict[str, list[float]] = defaultdict(list)
    passed_over: list[SearchResult] = []

    if not use_mmr:
        for r in results:
            times = per_video_times[r.video_id]
            if len(times) >= per_video_cap:
                passed_over.append(r)
                continue
            if any(abs(r.ref.pts_time - t) < min_gap_s for t in times):
                passed_over.append(r)
                continue
            picked.append(r)
            times.append(r.ref.pts_time)
            if len(picked) >= limit:
                return picked
    else:
        V = np.asarray(cand_vecs, dtype=np.float32)
        # Normalize relevance to [0,1] so mmr_lambda means the same across queries.
        scores = np.array([r.score for r in results], dtype=np.float64)
        lo, hi = scores.min(), scores.max()
        rel = (scores - lo) / (hi - lo) if hi - lo > 1e-9 else np.ones_like(scores)
        valid = np.linalg.norm(V, axis=1) > 1e-6
        remaining = list(range(len(results)))
        max_sim = np.zeros(len(results), dtype=np.float64)  # cos to closest pick
        while remaining and len(picked) < limit:
            best_i, best_val = -1, -np.inf
            for i in remaining:
                r = results[i]
                times = per_video_times[r.video_id]
                if len(times) >= per_video_cap:
                    continue
                if any(abs(r.ref.pts_time - t) < min_gap_s for t in times):
                    continue
                novelty_penalty = max_sim[i] if valid[i] else 0.0
                val = mmr_lambda * rel[i] - (1.0 - mmr_lambda) * novelty_penalty
                if val > best_val:
                    best_i, best_val = i, val
            if best_i < 0:
                break
            r = results[best_i]
            picked.append(r)
            picked_rows.append(best_i)
            per_video_times[r.video_id].append(r.ref.pts_time)
            remaining.remove(best_i)
            if valid[best_i]:
                sims = V @ V[best_i]
                np.maximum(max_sim, np.where(valid, sims, 0.0), out=max_sim)
        passed_over = [results[i] for i in remaining if results[i] not in picked]

    # Fill remaining slots with the best of what was skipped (score order).
    for r in passed_over:
        if len(picked) >= limit:
            break
        picked.append(r)
    return picked[:limit]
