"""End-to-end over the synthetic corpus: index → search → stale detection."""

import numpy as np
import pytest

from cvf.data.catalog import KeyframeCatalog
from cvf.index.store import IndexStore
from cvf.search.avs import avs_diversify
from cvf.search.feedback import rocchio
from cvf.search.temporal import trake_search


def test_index_build_and_search(corpus_with_index):
    catalog = KeyframeCatalog(corpus_with_index)
    store = IndexStore(corpus_with_index, "fake")
    assert store.count() == 15

    # query with the exact stored vector of gid 0 → top hit is itself
    v = np.load(store.embedding_path(catalog.ref(0).video_id))[0]
    scores, gids = store.search(v[None, :], topk=5)
    assert int(gids[0][0]) == 0
    assert scores[0][0] > 0.99


def test_stale_index_fails_loud(corpus_with_index, tmp_path):
    catalog = KeyframeCatalog(corpus_with_index)
    store = IndexStore(corpus_with_index, "fake")
    # simulate corpus change: tamper with the stored signature
    meta = store.meta()
    meta["catalog_signature"] = "deadbeef"
    from cvf.utils.io import atomic_write_json

    atomic_write_json(store.meta_path, meta)
    store._index = None
    with pytest.raises(RuntimeError, match="STALE"):
        store.load(catalog)


def test_trake_on_synthetic(corpus_with_index):
    catalog = KeyframeCatalog(corpus_with_index)
    store = IndexStore(corpus_with_index, "fake")
    # craft event vectors equal to frames 2 and 5 of L21_V001 → expect that seq
    vecs = np.load(store.embedding_path("L21_V001"))
    event_vecs = np.stack([vecs[1], vecs[4]]).astype(np.float32)
    out = trake_search(
        event_vecs=event_vecs,
        index_search=store.search,
        embedding_path=store.embedding_path,
        catalog=catalog,
        settings=corpus_with_index,
    )
    assert out
    best = out[0]
    assert best.video_id == "L21_V001"
    assert best.ns == [2, 5]
    assert best.frame_idxs == [200, 500]
    assert best.frame_idxs[0] < best.frame_idxs[1]


def test_avs_diversify_caps_per_video(corpus_with_index):
    from cvf.search.engine import SearchResult

    catalog = KeyframeCatalog(corpus_with_index)
    results = [
        SearchResult(ref=catalog.ref(g), score=1.0 - g * 0.01) for g in range(15)
    ]
    out = avs_diversify(results, per_video_cap=2, min_gap_s=0.0, limit=10)
    from collections import Counter

    top6 = Counter(r.video_id for r in out[:6])
    assert all(c <= 2 for c in top6.values())


def test_rocchio_moves_toward_positives():
    q = np.array([1.0, 0.0], dtype=np.float32)
    pos = np.array([[0.0, 1.0]], dtype=np.float32)
    q2 = rocchio(q, pos)
    assert q2[1] > 0  # pulled toward the positive
    assert abs(np.linalg.norm(q2) - 1.0) < 1e-6
