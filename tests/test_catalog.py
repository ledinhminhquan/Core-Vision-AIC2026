import pandas as pd

from cvp.data.catalog import KeyframeCatalog


def test_build_and_invariants(corpus):
    catalog = KeyframeCatalog(corpus)
    df = catalog.build()
    assert len(df) == 15
    assert list(df["global_id"]) == list(range(15))
    # sorted by (video_id, n): K01 < L21
    assert df.iloc[0]["video_id"] == "K01_V001"
    # frame_idx comes from map-keyframes (n * 100)
    row = df[(df["video_id"] == "L21_V001") & (df["n"] == 3)].iloc[0]
    assert int(row["frame_idx"]) == 300
    assert bool(row["has_map"])


def test_ref_and_video_span(corpus):
    catalog = KeyframeCatalog(corpus)
    catalog.build()
    start, count = catalog.video_span("L21_V001")
    ref = catalog.ref(start)
    assert ref.video_id == "L21_V001" and ref.n == 1
    assert count == 6


def test_rebuild_idempotent_signature(corpus):
    catalog = KeyframeCatalog(corpus)
    df1 = catalog.build()
    sig1 = catalog.signature()
    catalog2 = KeyframeCatalog(corpus)
    df2 = catalog2.build()  # detects up-to-date, loads
    assert sig1 == catalog2.signature()
    pd.testing.assert_frame_equal(df1, df2)
