import numpy as np

from cvp.search import fusion
from cvp.search.superglobal import superglobal_rerank


def test_minmax():
    m = fusion.minmax({1: 0.0, 2: 5.0, 3: 10.0})
    assert m[1] == 0.0 and m[3] == 1.0 and abs(m[2] - 0.5) < 1e-9


def test_minmax_constant_map():
    assert fusion.minmax({1: 3.0, 2: 3.0}) == {1: 1.0, 2: 1.0}


def test_aggregate_queries_max_and_mean():
    maps = [{1: 0.2, 2: 0.8}, {1: 0.6}]
    assert fusion.aggregate_queries(maps, "max") == {1: 0.6, 2: 0.8}
    out = fusion.aggregate_queries(maps, "mean")
    assert abs(out[1] - 0.4) < 1e-9


def test_weighted_sum_missing_ids_are_zero():
    out = fusion.weighted_sum([{1: 1.0, 2: 0.0}, {2: 1.0}], [1.0, 1.0])
    assert out[1] == 1.0 and out[2] == 1.0


def test_rrf_ranks():
    out = fusion.rrf([{1: 0.9, 2: 0.5}, {2: 0.9, 1: 0.5}])
    assert abs(out[1] - out[2]) < 1e-9  # symmetric ranks → equal score


def test_neighbor_boost_respects_video_bounds():
    scores = {0: 1.0, 1: 0.5, 5: 1.0}
    spans = {0: (0, 3), 1: (0, 3), 5: (5, 3)}  # gid 5 starts a new video
    out = fusion.neighbor_boost(scores, spans, boost=0.1, window=2)
    assert out[1] > 0.5          # boosted by neighbour 0
    assert out[5] == 1.0         # nothing near it in its own video


def test_superglobal_preserves_shape_and_improves_true_match():
    rng = np.random.default_rng(0)
    q = rng.normal(size=8).astype(np.float32)
    q /= np.linalg.norm(q)
    # candidates: 3 near-q vectors + 7 random
    near = q[None, :] + 0.1 * rng.normal(size=(3, 8)).astype(np.float32)
    rest = rng.normal(size=(7, 8)).astype(np.float32)
    V = np.concatenate([near, rest])
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    scores = superglobal_rerank(q, V, neighbors=3, qe_top=3)
    assert scores.shape == (10,)
    assert set(np.argsort(-scores)[:3]) == {0, 1, 2}
