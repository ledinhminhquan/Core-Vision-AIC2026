"""Evaluation metrics.

* ``qualifier_score`` — Mean of Top-k R-Scores over k ∈ {1, 5, 20, 50, 100}
  for BINARY-hit tasks (KIS/QA): a hit at rank 1 → 1.0, at rank 80 → 0.2.
* ``qualifier_score_fractional`` — same cutoffs over FRACTIONAL per-row
  R-Scores (TRAKE partial credit: 3/4 events in-window = 0.75).
* retrieval metrics (R@K / MRR / median rank) for encoder training eval.

The canonical, full-fidelity scorer is ``cvp.eval.official`` — these helpers
are the lightweight notebook-friendly subset and agree with it on both the
binary and the TRAKE partial-credit formulas.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

QUALIFIER_CUTOFFS = (1, 5, 20, 50, 100)


def qualifier_score(is_correct: Sequence[bool], cutoffs: Sequence[int] = QUALIFIER_CUTOFFS) -> float:
    """Score one query's ranked submission given per-row correctness flags."""
    flags = list(is_correct)
    first_hit = next((i + 1 for i, ok in enumerate(flags) if ok), None)
    if first_hit is None:
        return 0.0
    return sum(1.0 for k in cutoffs if first_hit <= k) / len(cutoffs)


def qualifier_score_fractional(r_scores: Sequence[float],
                               cutoffs: Sequence[int] = QUALIFIER_CUTOFFS) -> float:
    """Mean over cutoffs of max R-Score in the top-k, for fractional row scores."""
    scores = list(r_scores)
    if not scores:
        return 0.0
    prefix_max: list[float] = []
    run = 0.0
    for s in scores:
        run = max(run, float(s))
        prefix_max.append(run)
    return sum(prefix_max[min(k, len(prefix_max)) - 1] for k in cutoffs) / len(cutoffs)


@dataclass
class KisGroundTruth:
    """A KIS/QA answer: a contiguous frame segment inside one video."""

    video_id: str
    frame_start: int
    frame_end: int  # inclusive

    def contains(self, video_id: str, frame_idx: int) -> bool:
        return video_id == self.video_id and self.frame_start <= frame_idx <= self.frame_end


def score_kis_submission(rows: list[tuple[str, int]], gt: KisGroundTruth) -> float:
    return qualifier_score([gt.contains(v, f) for v, f in rows])


def score_trake_submission(rows: list[tuple[str, list[int]]],
                           gt_video: str, gt_segments: list[tuple[int, int]]) -> float:
    """Official TRAKE partial credit: wrong video → 0; right video → the
    fraction of events whose frame lands inside its window (missing or extra
    frames count as misses — the denominator is always the GT event count)."""

    def row_score(video_id: str, frames: list[int]) -> float:
        if video_id != gt_video or not gt_segments:
            return 0.0
        hits = sum(1 for f, (a, b) in zip(frames, gt_segments) if a <= f <= b)
        return hits / len(gt_segments)

    return qualifier_score_fractional([row_score(v, fs) for v, fs in rows])


# ── encoder-training retrieval eval ──────────────────────────────────────────


def retrieval_metrics(
    text_vecs: np.ndarray,   # (N, d) L2-normalized caption embeddings
    image_vecs: np.ndarray,  # (N, d) L2-normalized image embeddings (aligned rows)
    ks: Sequence[int] = (1, 5, 10),
    batch: int = 1024,
) -> dict[str, float]:
    """Text→image retrieval over aligned pairs: R@K, MRR, median rank."""
    n = len(text_vecs)
    ranks = np.zeros(n, dtype=np.int64)
    for i in range(0, n, batch):
        sims = text_vecs[i : i + batch] @ image_vecs.T          # (b, N)
        target = sims[np.arange(sims.shape[0]), np.arange(i, min(i + batch, n))]
        ranks[i : i + batch] = (sims > target[:, None]).sum(axis=1) + 1
    out = {f"R@{k}": float((ranks <= k).mean()) for k in ks}
    out["MRR"] = float((1.0 / ranks).mean())
    out["MedR"] = float(np.median(ranks))
    return out


def compare_models(
    eval_fn_a: Callable[[], dict[str, float]],
    eval_fn_b: Callable[[], dict[str, float]],
) -> dict[str, tuple[float, float, float]]:
    """{metric: (a, b, delta)} — for baseline-vs-finetuned reports."""
    a, b = eval_fn_a(), eval_fn_b()
    return {k: (a[k], b[k], b[k] - a[k]) for k in a if k in b}
