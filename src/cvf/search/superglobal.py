"""SuperGlobal re-ranking (Shao et al., ICCV'23), as adapted for keyframe
retrieval by GRAB (CVPRw'25).

Refines the top-K visual scores with two nearly-free global operations over
the candidate pool only:

  S1: DB-side refinement — each candidate's vector is GeM-averaged (p=1) with
      its nearest neighbours *within the pool*, then re-scored against q.
  S2: query expansion    — q is averaged with the current top-r vectors and
      all candidates are re-scored against the expanded query.

  final = (S1 + S2) / 2

No training, no extra index, ~1 ms for K=500 — and it reliably pulls the true
scene above near-duplicates of a wrong scene.

Accepts one query vector ``(dim,)`` or a stack of query variants ``(Q, dim)``
(original / enhanced / translation / expansions). With multiple variants the
refined scores are max-aggregated per candidate — matching how the dense
stage fuses variants, so reranking can't demote a hit that only an expansion
found (that mismatch was a known failure mode of the single-variant version).

Zero rows in ``cand_vecs`` (frames whose embedding was unreadable at build
time) are excluded from neighbour pools and query expansion so they cannot
dilute the refinement means; their own score stays 0.
"""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)


def _rerank_single(
    q: np.ndarray,             # (dim,) L2-normalized
    V: np.ndarray,             # (K, dim) L2-normalized candidate vectors
    valid: np.ndarray,         # (K,) bool — non-zero rows
    nn_idx: np.ndarray | None,  # (K, neighbors) precomputed in-pool neighbours
    qe_top: int,
) -> np.ndarray:
    base = V @ q                                    # (K,) — zero rows score 0

    # S2 — query expansion with the current top-r valid candidates
    order = np.argsort(-np.where(valid, base, -np.inf))
    top_r = order[: min(qe_top, int(valid.sum()) or 1)]
    q_exp = q + V[top_r].mean(axis=0)
    q_exp /= (np.linalg.norm(q_exp) or 1.0)
    s2 = V @ q_exp

    # S1 — DB-side descriptor refinement over in-pool neighbours
    if nn_idx is not None:
        refined = V + V[nn_idx].mean(axis=1)
        refined /= np.maximum(np.linalg.norm(refined, axis=1, keepdims=True), 1e-9)
        s1 = refined @ q
    else:
        s1 = base

    return ((s1 + s2) / 2.0).astype(np.float32)


def superglobal_rerank(
    query_vec: np.ndarray,          # (dim,) or (Q, dim) L2-normalized
    cand_vecs: np.ndarray,          # (K, dim) L2-normalized candidate vectors
    neighbors: int = 10,
    qe_top: int = 10,
) -> np.ndarray:
    """Return refined scores (K,) — max over query variants when several given."""
    Q = np.asarray(query_vec, dtype=np.float32)
    if Q.ndim == 1:
        Q = Q[None, :]
    V = np.asarray(cand_vecs, dtype=np.float32)
    if V.ndim != 2 or len(V) == 0:
        return np.zeros((0,), dtype=np.float32)
    k = len(V)
    valid = np.linalg.norm(V, axis=1) > 1e-6

    # In-pool neighbour graph is query-independent — compute once for all variants.
    nn_idx = None
    n_valid = int(valid.sum())
    n_neigh = min(neighbors, max(n_valid - 1, 0))
    if n_neigh >= 1:
        sim = V @ V.T                               # (K, K)
        np.fill_diagonal(sim, -np.inf)
        sim[:, ~valid] = -np.inf                    # zero rows are never neighbours
        nn_idx = np.argpartition(-sim, kth=n_neigh - 1, axis=1)[:, :n_neigh]

    scores = np.full((k,), -np.inf, dtype=np.float32)
    for q in Q:
        s = _rerank_single(q, V, valid, nn_idx, qe_top)
        scores = np.maximum(scores, s)
    scores[~valid] = 0.0
    return scores
