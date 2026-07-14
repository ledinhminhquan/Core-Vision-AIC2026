"""Rocchio relevance feedback over the embedding space.

During a live round the operator marks tiles right/wrong; the query vector is
nudged toward the positives and away from the negatives, then re-searched:

    q' = α·q + β·mean(positives) − γ·mean(negatives)

Classic constants (α=1, β=0.75, γ=0.15) work well when positives are scarce
(1–3 marks). Used by Vortex at AIC'25; costs one extra FAISS call.
"""

from __future__ import annotations

import numpy as np


def rocchio(
    query_vec: np.ndarray,
    positive_vecs: np.ndarray | None,
    negative_vecs: np.ndarray | None = None,
    alpha: float = 1.0,
    beta: float = 0.75,
    gamma: float = 0.15,
) -> np.ndarray:
    """Return the L2-normalized updated query vector."""
    q = np.asarray(query_vec, dtype=np.float32).ravel().copy() * alpha
    if positive_vecs is not None and len(positive_vecs):
        q += beta * np.asarray(positive_vecs, dtype=np.float32).mean(axis=0)
    if negative_vecs is not None and len(negative_vecs):
        q -= gamma * np.asarray(negative_vecs, dtype=np.float32).mean(axis=0)
    norm = float(np.linalg.norm(q))
    return q / (norm or 1.0)
