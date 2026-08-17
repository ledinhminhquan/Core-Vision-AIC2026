"""Round-4 review regressions — findings surfaced by auditing against the REAL
2026 Batch-1 drop (verified 2026-08-15) and the official prelim rules PDF.

Each test pins one confirmed finding:
  R4-1  media-info keywords are a python-list-LOOKING STRING in the real data
  R4-2  TRAKE chains with tied frame_idx (614 real tied pairs!) must survive
  R4-3  extract_video must never delete organiser keyframes (map csv missing)
  R4-4  official.py accepts the 'segments' GT spelling + fails loud on TRAKE
        entries with no events; legacy API infers task from the stem
  R4-5  zero-candidate queries return None instead of a 0-byte CSV
  R4-6  service TRAKE with an empty events list → 422, engine guard returns []
  R4-7  notebook builder: nb03 packages only its own files, nb02 staleness
        sees caption growth, nb01 merge copies via tmp+rename
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cvp.config import Settings

REPO = Path(__file__).resolve().parents[1]


# ── R4-1 · media-info keywords: string-typed in the real Batch-1 files ───────
def _store_with(tmp_path, payload):
    from cvp.data.metadata import MediaInfoStore

    s = Settings()
    s.paths.data_root = tmp_path
    (tmp_path / s.paths.media_info_dir).mkdir(parents=True)
    (tmp_path / s.paths.media_info_dir / "L21_V001.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return MediaInfoStore(s)


def test_keywords_python_list_string_is_parsed(tmp_path):
    blob = _store_with(tmp_path, {
        "title": "T", "description": "D",
        "keywords": "['HTV Tin tức', 'chuong trinh 60 giay']",
    }).text_blob("L21_V001")
    assert "HTV Tin tức" in blob and "60 giay" in blob


def test_keywords_genuine_list_still_works(tmp_path):
    blob = _store_with(tmp_path, {"title": "T", "keywords": ["a", "b"]}).text_blob("L21_V001")
    assert "a b" in blob


def test_keywords_plain_comma_string_fallback(tmp_path):
    blob = _store_with(tmp_path, {"title": "T", "keywords": "tin tức, thời sự"}).text_blob("L21_V001")
    assert "tin tức" in blob and "thời sự" in blob


# ── R4-2 · TRAKE tied frame_idx chains ───────────────────────────────────────
def test_monotonize_bumps_ties_and_cascades():
    from cvp.search.temporal import monotonize_frame_idxs

    assert monotonize_frame_idxs([480, 480, 700]) == [480, 481, 700]
    assert monotonize_frame_idxs([480, 480, 481]) == [480, 481, 482]  # cascade
    assert monotonize_frame_idxs([100, 200, 300]) == [100, 200, 300]  # untouched
    assert monotonize_frame_idxs([5]) == [5]
    assert monotonize_frame_idxs([]) == []


def test_monotonize_rejects_genuine_decrease():
    from cvp.search.temporal import monotonize_frame_idxs

    assert monotonize_frame_idxs([300, 200]) is None
    assert monotonize_frame_idxs([100, 100, 99]) is None


# ── R4-3 · extraction must not destroy organiser keyframes ───────────────────
def test_extract_video_refuses_to_delete_unmarked_keyframes(tmp_path):
    from cvp.data.extraction import extract_video

    kf, mp = tmp_path / "keyframes", tmp_path / "map-keyframes"
    vdir = kf / "L21_V001"
    vdir.mkdir(parents=True)
    for i in (1, 2, 3):
        (vdir / f"{i:03d}.jpg").write_bytes(b"organiser-jpg")
    # No map csv, no sentinel: these are organiser files → refuse, keep them.
    # Returns 0 (round-5): nothing extracted, no usable map — callers must not
    # count a refusal as fresh work.
    n = extract_video(tmp_path / "L21_V001.mp4", kf, mp, overwrite=False)
    assert n == 0
    assert sorted(p.name for p in vdir.iterdir()) == ["001.jpg", "002.jpg", "003.jpg"]
    assert (vdir / "001.jpg").read_bytes() == b"organiser-jpg"


def test_extract_video_complete_pair_short_circuits(tmp_path):
    from cvp.data.extraction import extract_video

    kf, mp = tmp_path / "keyframes", tmp_path / "map-keyframes"
    vdir = kf / "L21_V001"
    vdir.mkdir(parents=True)
    mp.mkdir(parents=True)
    for i in (1, 2):
        (vdir / f"{i:03d}.jpg").write_bytes(b"x")
    (mp / "L21_V001.csv").write_text(
        "n,pts_time,fps,frame_idx\n1,0.0,30.0,0\n2,3.0,30.0,90\n", encoding="utf-8")
    assert extract_video(tmp_path / "L21_V001.mp4", kf, mp) == 2


# ── R4-4 · official scoring: segments alias, loud TRAKE guard, stem inference ─
def test_gt_segments_spelling_loads_and_scores():
    from cvp.eval.official import load_ground_truth, score_rows
    import tempfile

    gt = {"query-p1-1-trake": {"task": "trake", "video_id": "L10_V010",
                               "segments": [[95, 105], [145, 155]]}}
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "gt.json"
        p.write_text(json.dumps(gt), encoding="utf-8")
        entry = load_ground_truth(p)["query-p1-1-trake"]
    qs = score_rows("trake", [["L10_V010", "100", "150"]], entry)
    assert qs.final == 1.0


def test_trake_gt_without_events_fails_loud(tmp_path):
    from cvp.eval.official import load_ground_truth

    p = tmp_path / "gt.json"
    p.write_text(json.dumps({"query-p1-2-trake": {
        "task": "trake", "video_id": "L10_V010", "windows": [[95, 105]]}}), encoding="utf-8")
    with pytest.raises(ValueError, match="no usable events"):
        load_ground_truth(p)


def test_legacy_csv_api_infers_task_from_stem(tmp_path):
    from cvp.eval.official import score_submission_csv

    csv_p = tmp_path / "query-p1-3-qa.csv"
    csv_p.write_text("L05_V005,888,WRONG ANSWER\n", encoding="utf-8")
    gt_entry = {"video_id": "L05_V005", "range": [800, 900], "answers": ["màu xanh"]}
    # Task-less GT + qa stem: the wrong answer must NOT be credited (the old
    # KIS default ignored the answer column and returned 1.0).
    assert score_submission_csv(csv_p, gt_entry)["final"] == 0.0
    csv_p.write_text("L05_V005,888,màu xanh\n", encoding="utf-8")
    assert score_submission_csv(csv_p, gt_entry)["final"] == 1.0


def test_legacy_dir_api_infers_task_from_stem(tmp_path):
    from cvp.eval.official import score_submission_dir

    sub = tmp_path / "subs"
    sub.mkdir()
    (sub / "query-p1-3-qa.csv").write_text("L05_V005,888,WRONG\n", encoding="utf-8")
    gt_p = tmp_path / "gt.json"
    gt_p.write_text(json.dumps({"query-p1-3-qa": {
        "video_id": "L05_V005", "range": [800, 900], "answers": ["màu xanh"]}}), encoding="utf-8")
    rep = score_submission_dir(sub, gt_p)
    assert rep["per_query"]["query-p1-3-qa"]["task"] == "qa"
    assert rep["per_query"]["query-p1-3-qa"]["final"] == 0.0


# ── R4-5 · zero-candidate query → None, never a 0-byte CSV ───────────────────
def test_empty_results_return_none_not_empty_csv(tmp_path):
    from cvp.pipeline.run_queries import run_query_file

    class _Eng:
        settings = Settings()

        def search_text(self, q, **kw):
            return []

    qf = tmp_path / "query-9-kis.txt"
    qf.write_text("một cảnh không tồn tại\n", encoding="utf-8")
    out = run_query_file(_Eng(), qf, tmp_path, None)
    assert out is None
    assert not (tmp_path / "query-9-kis.csv").exists()


def test_empty_trake_candidates_return_none(tmp_path):
    from cvp.pipeline.run_queries import run_query_file

    class _Eng:
        settings = Settings()

        def search_trake(self, events, **kw):
            return []

    qf = tmp_path / "query-9-trake.txt"
    qf.write_text("Bối cảnh chung\nE1: chạy đà\nE2: giậm nhảy\n", encoding="utf-8")
    out = run_query_file(_Eng(), qf, tmp_path, None)
    assert out is None
    assert not (tmp_path / "query-9-trake.csv").exists()


# ── R4-6 · empty TRAKE events: service 422 ───────────────────────────────────
def test_service_trake_empty_events_is_422():
    fastapi = pytest.importorskip("fastapi")  # noqa: F841
    from fastapi.testclient import TestClient

    from cvp.service.app import create_app

    class _Eng:
        settings = Settings()

        def search_trake(self, events, **kw):  # pragma: no cover — must not be reached
            raise AssertionError("engine must not be called for empty events")

    client = TestClient(create_app(engine=_Eng()))
    r = client.post("/search/trake", json={"events": []})
    assert r.status_code == 422


# ── R4-7 · notebook builder contracts ────────────────────────────────────────
def test_builder_nb03_packages_only_its_own_files():
    src = (REPO / "notebooks" / "_build_notebooks.py").read_text(encoding="utf-8")
    assert "files=nb03_files" in src
    assert "nb03_files = [p]" in src


def test_builder_nb02_staleness_sees_caption_growth():
    src = (REPO / "notebooks" / "_build_notebooks.py").read_text(encoding="utf-8")
    assert '_caps = sorted(cap_dir.glob("*.json"))' in src
    assert "_stale = _stale_pub or _stale_caps" in src


def test_builder_local_copy_merge_uses_tmp_rename():
    src = (REPO / "notebooks" / "_build_notebooks.py").read_text(encoding="utf-8")
    merge = src.split("PHA 2")[1].split("videos stay on Drive")[0]
    assert '.__tmp' in merge and "tmp_target.rename(target)" in merge
