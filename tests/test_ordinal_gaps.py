"""Keyframe ordinal GAPS (missing jpgs) must not shift embedding-row reads.

The convention everywhere: embedding .npy rows are positional over a video's
catalog rows sorted by n — the row of a frame is ``global_id - video_start``,
NOT ``n - 1``. These tests pin that convention end-to-end.
"""

import numpy as np

from cvf.data.catalog import KeyframeCatalog
from cvf.index.store import IndexStore


def _make_gap(corpus):
    """Remove keyframe 004.jpg of L21_V001 → ordinals [1,2,3,5,6]."""
    kf = corpus.paths.data("keyframes") / "L21_V001" / "004.jpg"
    kf.unlink()


def test_catalog_positions_with_gap(corpus):
    _make_gap(corpus)
    catalog = KeyframeCatalog(corpus)
    df = catalog.build(force=True)
    grp = df[df["video_id"] == "L21_V001"].sort_values("n")
    assert list(grp["n"]) == [1, 2, 3, 5, 6]
    start, count = catalog.video_span("L21_V001")
    assert count == 5
    # positional row of ordinal n=5 is 3 (0-based), i.e. global_id - start
    row_of_5 = grp[grp["n"] == 5]["global_id"].iloc[0] - start
    assert row_of_5 == 3


def test_vector_reads_align_with_gap(corpus):
    _make_gap(corpus)
    catalog = KeyframeCatalog(corpus)
    df = catalog.build(force=True)
    store = IndexStore(corpus, "fake")
    rng = np.random.default_rng(1)
    for vid, grp in df.groupby("video_id"):
        vecs = rng.normal(size=(len(grp), 8)).astype(np.float32)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        store.embedding_path(str(vid)).parent.mkdir(parents=True, exist_ok=True)
        with open(store.embedding_path(str(vid)), "wb") as f:
            np.save(f, vecs)
    store.build(catalog)
    assert store.count() == len(df)

    # FAISS row must equal global_id: querying the stored vector of the frame
    # AFTER the gap (n=5) must return its own global_id as top hit.
    start, _ = catalog.video_span("L21_V001")
    grp = df[df["video_id"] == "L21_V001"].sort_values("n")
    gid_n5 = int(grp[grp["n"] == 5]["global_id"].iloc[0])
    vecs = np.load(store.embedding_path("L21_V001"))
    q = vecs[gid_n5 - start]
    scores, gids = store.search(q[None, :], topk=1)
    assert int(gids[0][0]) == gid_n5
    assert scores[0][0] > 0.999
