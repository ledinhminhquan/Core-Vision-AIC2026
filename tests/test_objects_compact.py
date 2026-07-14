"""Compact parquet objects index (port from Core-Vision_HCMC-AI, adapted).

Pure-CPU synthetic corpus: organiser-format detection JSONs (ALL values are
strings in the real AIC packs — pinned here) folded into one parquet, then
read back through ``CompactObjects`` and compared against the per-file
``ObjectStore``. Also pins the ``ObjectBooster`` auto-preference.
"""

from __future__ import annotations

import json

import pytest

from cvp.config import Settings
from cvp.data.metadata import ObjectStore
from cvp.data.objects_compact import CompactObjects, build_objects_index, index_path

pytest.importorskip("pyarrow")


class _StubCatalog:
    """Duck-typed stand-in: only videos() and video_rows(vid)['n'] are used."""

    def __init__(self, vids_ns: dict[str, list[int]]):
        self._vids = vids_ns

    def videos(self):
        return list(self._vids)

    def video_rows(self, vid):
        import pandas as pd

        return pd.DataFrame({"n": self._vids[vid]})


def _write_json(root, vid, n, dets):
    d = root / vid
    d.mkdir(parents=True, exist_ok=True)
    payload = {
        "detection_scores": [str(s) for (_e, s, _b) in dets],
        "detection_class_entities": [e for (e, _s, _b) in dets],
        "detection_boxes": [[str(x) for x in b] for (_e, _s, b) in dets],
    }
    (d / f"{n:03d}.json").write_text(json.dumps(payload), encoding="utf-8")


@pytest.fixture()
def corpus(tmp_path):
    settings = Settings()
    settings.paths.data_root = tmp_path / "data"
    settings.paths.artifacts_root = tmp_path / "artifacts"
    obj_root = settings.paths.data(settings.paths.objects_dir)
    _write_json(obj_root, "L01_V001", 1, [
        ("Person", 0.91, (0.1, 0.1, 0.9, 0.9)),
        ("Car", 0.45, (0.2, 0.2, 0.8, 0.8)),
        ("Tree", 0.12, (0.0, 0.0, 0.5, 0.5)),     # below 0.3 → dropped at build
    ])
    _write_json(obj_root, "L01_V001", 2, [("Bus", 0.77, (0.3, 0.3, 0.7, 0.7))])
    # n=3 has NO json — lookup must stay total (empty row, not KeyError).
    _write_json(obj_root, "L02_V009", 1, [("Boat", 0.5, (0.1, 0.2, 0.3, 0.4))])
    catalog = _StubCatalog({"L01_V001": [1, 2, 3], "L02_V009": [1]})
    return settings, catalog


def test_build_and_roundtrip_matches_objectstore(corpus):
    settings, catalog = corpus
    out = build_objects_index(settings, catalog)
    assert out.exists() and out == index_path(settings)

    compact, store = CompactObjects(settings), ObjectStore(settings)
    assert compact.available()
    got = compact.get("L01_V001", 1)
    assert [d.entity for d in got] == ["Person", "Car"]           # 0.12 dropped
    ref = [d for d in store.get("L01_V001", 1) if d.score >= 0.3]
    assert [(d.entity, pytest.approx(d.score)) for d in got] == \
           [(d.entity, pytest.approx(d.score)) for d in ref]
    assert got[0].box == pytest.approx((0.1, 0.1, 0.9, 0.9))


def test_missing_json_keyframe_yields_empty_but_total(corpus):
    settings, catalog = corpus
    build_objects_index(settings, catalog)
    compact = CompactObjects(settings)
    assert compact.get("L01_V001", 3) == []          # row exists, no detections
    assert compact.get("L99_V999", 1) == []          # unknown video → empty


def test_reader_without_parquet_degrades_to_empty(corpus):
    settings, _ = corpus
    compact = CompactObjects(settings)
    assert not compact.available()
    assert compact.get("L01_V001", 1) == []


def test_build_is_idempotent_unless_overwrite(corpus):
    settings, catalog = corpus
    out = build_objects_index(settings, catalog)
    stamp = out.stat().st_mtime_ns
    assert build_objects_index(settings, catalog) == out
    assert out.stat().st_mtime_ns == stamp            # untouched without overwrite
    build_objects_index(settings, catalog, overwrite=True)
    assert out.exists()


def test_has_video(corpus):
    settings, catalog = corpus
    build_objects_index(settings, catalog)
    compact = CompactObjects(settings)
    assert compact.has_video("L02_V009") and not compact.has_video("L99_V999")


def test_object_booster_prefers_compact_index(corpus):
    from cvp.search.object_filter import ObjectBooster

    settings, catalog = corpus
    booster = ObjectBooster(settings)
    assert isinstance(booster.store, ObjectStore)     # no parquet yet
    build_objects_index(settings, catalog)
    booster2 = ObjectBooster(settings)
    assert isinstance(booster2.store, CompactObjects)
    # Scoring path works through the compact store (Person constraint).
    from types import SimpleNamespace

    constraints = booster2.parse("một người đứng cạnh xe hơi")
    if constraints:  # vocab includes 'người' → Person
        ref = SimpleNamespace(video_id="L01_V001", n=1, global_id=0)
        assert booster2.score(constraints, ref) > 0.0
