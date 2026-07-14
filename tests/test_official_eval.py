"""Official scoring formulas — reproduces the organisers' worked examples exactly.

Ports the 27 reference tests from Core-Vision_HCMC-AI (test_eval_official.py)
plus the new dataclass API (score_rows / score_csv / score_run /
load_ground_truth), k-cutoff boundaries, truncation, validation and the CLI.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from cvp.eval.official import (
    K_VALUES,
    QueryScore,
    RunReport,
    final_score,
    infer_task,
    load_ground_truth,
    normalize_answer,
    normalize_task,
    r_at_k,
    r_score_kis,
    r_score_qa,
    r_score_trake,
    range_from_center,
    score_csv,
    score_rows,
    score_run,
    score_submission_csv,
    score_submission_dir,
    trake_gt,
    validate_csv,
    validate_rows,
)

REPO = Path(__file__).resolve().parents[1]

GT_KIS = {"task": "kis", "video_id": "L01_V001", "frame_start": 500, "frame_end": 510}
GT_QA = {"task": "qa", "video_id": "L05_V005", "frame_start": 800, "frame_end": 900,
         "answer": "màu xanh"}
GT_TRAKE = trake_gt("L10_V010", [100, 150, 200, 250], epsilon=5)


# ---------------------------------------------------------------- KIS (slide p8-9)
def test_kis_slide_example():
    assert r_score_kis(("L01_V001", 505), GT_KIS) == 1
    assert r_score_kis(("L01_V001", 600), GT_KIS) == 0


def test_kis_wrong_video_zero():
    assert r_score_kis(("L02_V001", 505), GT_KIS) == 0


def test_kis_range_inclusive():
    assert r_score_kis(("L01_V001", 500), GT_KIS) == 1
    assert r_score_kis(("L01_V001", 510), GT_KIS) == 1
    assert r_score_kis(("L01_V001", 511), GT_KIS) == 0


# ---------------------------------------------------------------- VQA (slide p10-11)
def test_qa_slide_example():
    assert r_score_qa(("L05_V005", 888, "màu xanh"), GT_QA) == 1
    assert r_score_qa(("L05_V005", 888, "màu trắng"), GT_QA) == 0   # wrong answer
    assert r_score_qa(("L06_V007", 888, "màu xanh"), GT_QA) == 0    # wrong video


def test_qa_answer_normalisation():
    assert r_score_qa(("L05_V005", 850, "  Màu   XANH. "), GT_QA) == 1  # case/ws/punct
    assert r_score_qa(("L05_V005", 850, "màu xanh!"), GT_QA) == 1
    assert r_score_qa(("L05_V005", 850, "mau xanh"), GT_QA) == 0        # accents preserved


def test_normalize_answer():
    assert normalize_answer("  Hà   Nội . ") == "hà nội"
    assert normalize_answer("42?") == "42"
    assert normalize_answer("mau xanh") != normalize_answer("màu xanh")


# ---------------------------------------------------------------- TRAKE (slide p12-15)
def test_trake_slide_example_075():
    # GT frames 100/150/200/250, epsilon 5 -> windows [95,105] [145,155] [195,205] [245,255]
    assert GT_TRAKE["events"][0] == {"frame_start": 95, "frame_end": 105}
    # 101 hit, 156 miss, 203 hit, 251 hit -> 3/4
    assert r_score_trake(("L10_V010", [101, 156, 203, 251]), GT_TRAKE) == pytest.approx(0.75)


def test_trake_wrong_video_zero():
    assert r_score_trake(("L10_V011", [101, 151, 201, 251]), GT_TRAKE) == 0.0


def test_trake_partial_and_flat_row():
    # flat CSV-style string row, only 3 of 4 frames submitted: 2 hits / 4 events
    assert r_score_trake(("L10_V010", "100", "150", "999"), GT_TRAKE) == pytest.approx(0.5)


def test_range_from_center():
    assert range_from_center(100, 5) == {"frame_start": 95, "frame_end": 105}


# ---------------------------------------------------------------- R@k / Final (slide p16-20)
def test_final_score_slide_example_074():
    # rank-1 scores 0.5, rank-3 scores 0.8 (best), rank-15 scores 0.6
    scores = [0.0] * 100
    scores[0], scores[2], scores[14] = 0.5, 0.8, 0.6
    assert r_at_k(scores, 1) == pytest.approx(0.5)
    assert r_at_k(scores, 5) == pytest.approx(0.8)
    assert r_at_k(scores, 20) == pytest.approx(0.8)
    assert final_score(scores) == pytest.approx(0.74)


def test_r_at_k_empty_and_short():
    assert r_at_k([], 5) == 0.0
    assert final_score([]) == 0.0
    assert r_at_k([0.5], 100) == 0.5  # fewer rows than k is fine


# ---------------------------------------------------------------- CSV level (legacy dict API)
def test_score_submission_csv_kis(tmp_path):
    p = tmp_path / "query-p1-1-kis.csv"
    p.write_text("L09_V999,1\nL01_V001,505\n", encoding="utf-8")
    res = score_submission_csv(p, GT_KIS)
    assert res["r_at_k"]["1"] == 0.0
    assert res["r_at_k"]["5"] == 1.0
    assert res["best_rank"] == 2
    assert res["final"] == pytest.approx(0.8)  # (0 + 1 + 1 + 1 + 1) / 5


def test_score_submission_csv_trake(tmp_path):
    p = tmp_path / "query-p1-2-trake.csv"
    p.write_text("L10_V010,101,156,203,251\n", encoding="utf-8")
    res = score_submission_csv(p, GT_TRAKE)
    assert res["final"] == pytest.approx(0.75)
    assert res["best_rank"] == 1
    assert res["best_score"] == pytest.approx(0.75)


def test_score_submission_csv_qa(tmp_path):
    p = tmp_path / "query-p1-3-qa.csv"
    p.write_text("L05_V005,888,Màu xanh\n", encoding="utf-8")
    res = score_submission_csv(p, GT_QA)
    assert res["final"] == pytest.approx(1.0)


# ---------------------------------------------------------------- task normalisation
def test_normalize_task_aliases_and_whitespace():
    assert normalize_task("vqa") == "qa"
    assert normalize_task("VQA") == "qa"
    assert normalize_task("qa ") == "qa"
    assert normalize_task("TRAKE\n") == "trake"
    assert normalize_task("tkis") == "kis"
    assert normalize_task("kis-t") == "kis"
    assert normalize_task("textual-kis") == "kis"
    assert normalize_task("zzz") is None
    assert normalize_task("") is None


def test_normalize_task_video_kis_aliases():
    # Fix L5 (review 2026-07-08): hand-labelled Video-KIS GT must score as KIS,
    # not land in `unscored`.
    for alias in ("kis-v", "KIS-V", "kisv", "vkis", "video-kis"):
        assert normalize_task(alias) == "kis"


# ---------------------------------------------------------------- multi-window GT ("ranges")
def test_r_score_kis_multiple_ranges_any_window_hits():
    # Enhancement E5 (review 2026-07-08): AVS-style dev GT can declare several
    # acceptable windows in one video — a row hits when inside ANY of them.
    gt = {"task": "kis", "video_id": "L01_V001", "ranges": [[100, 110], [500, 510]]}
    assert r_score_kis(("L01_V001", 105), gt) == 1
    assert r_score_kis(("L01_V001", 505), gt) == 1
    assert r_score_kis(("L01_V001", 300), gt) == 0
    assert r_score_kis(("L02_V001", 105), gt) == 0
    # per-window dict / center+epsilon spellings inside "ranges" also work
    gt2 = {"video_id": "L01_V001",
           "ranges": [{"frame_start": 10, "frame_end": 20}, {"center": 900, "epsilon": 5}]}
    assert r_score_kis(("L01_V001", 15), gt2) == 1
    assert r_score_kis(("L01_V001", 903), gt2) == 1
    assert r_score_kis(("L01_V001", 50), gt2) == 0


def test_ranges_gt_through_score_rows_and_load_ground_truth(tmp_path):
    gt = {"query-p1-8-kis": {"task": "kis", "video_id": "L01_V001",
                             "ranges": [[100, 110], [500, 510]]}}
    gt_path = tmp_path / "gt.json"
    gt_path.write_text(json.dumps(gt), encoding="utf-8")
    loaded = load_ground_truth(gt_path)  # must NOT raise "no usable frame window"
    entry = loaded["query-p1-8-kis"]
    assert entry["ranges"] == [[100, 110], [500, 510]]
    assert entry["frame_start"] == 100  # first window kept for legacy readers
    qs = score_rows("kis", [["L01_V001", "505"]], entry)
    assert qs.final == pytest.approx(1.0)
    qs_miss = score_rows("kis", [["L01_V001", "300"]], entry)
    assert qs_miss.final == pytest.approx(0.0)


def test_single_range_gt_behaviour_is_unchanged():
    # regression guard: entries WITHOUT "ranges" score exactly as before
    assert r_score_kis(("L01_V001", 505), GT_KIS) == 1
    assert r_score_kis(("L01_V001", 511), GT_KIS) == 0


def test_score_submission_csv_vqa_alias_scores_as_qa(tmp_path):
    # A 'vqa' GT must use the QA formula: wrong answer -> 0, not KIS's 1.
    p = tmp_path / "query-p1-3-qa.csv"
    p.write_text("L05_V005,888,màu trắng\n", encoding="utf-8")
    res = score_submission_csv(p, {**GT_QA, "task": "vqa"})
    assert res["final"] == pytest.approx(0.0)
    p.write_text("L05_V005,888,màu xanh\n", encoding="utf-8")
    assert score_submission_csv(p, {**GT_QA, "task": "VQA"})["final"] == pytest.approx(1.0)


def test_score_submission_csv_unknown_task_raises(tmp_path):
    p = tmp_path / "query-p1-1-kis.csv"
    p.write_text("L01_V001,505\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown task"):
        score_submission_csv(p, {**GT_KIS, "task": "zzz"})


def test_score_submission_csv_qa_missing_gt_answer_raises(tmp_path):
    # An empty submitted answer must never score 1.0 against an empty GT answer.
    p = tmp_path / "query-p1-3-qa.csv"
    p.write_text("L05_V005,888,\n", encoding="utf-8")
    gt_no_answer = {k: v for k, v in GT_QA.items() if k != "answer"}
    with pytest.raises(ValueError, match="missing answer"):
        score_submission_csv(p, gt_no_answer)
    with pytest.raises(ValueError, match="missing answer"):
        score_submission_csv(p, {**GT_QA, "answer": "  . "})  # normalises to empty


# ---------------------------------------------------------------- directory level (legacy)
def test_score_submission_dir_reports_unscored(tmp_path):
    sub = tmp_path / "subs"
    sub.mkdir()
    (sub / "query-p1-1-kis.csv").write_text("L01_V001,505\n", encoding="utf-8")
    (sub / "query-p1-9-qa.csv").write_text("L05_V005,888,màu xanh\n", encoding="utf-8")
    gt = {"query-p1-1-kis": GT_KIS, "query-p1-7-kis": GT_KIS}
    gt_path = tmp_path / "gt.json"
    gt_path.write_text(json.dumps(gt, ensure_ascii=False), encoding="utf-8")

    report = score_submission_dir(sub, gt_path)
    assert report["num_scored"] == 1
    assert report["num_unscored"] == 1
    assert report["per_query"]["query-p1-9-qa"]["status"] == "unscored"
    assert report["per_query"]["query-p1-1-kis"]["status"] == "scored"
    assert report["gt_without_submission"] == ["query-p1-7-kis"]
    # official-style macro counts BOTH GT queries; the missing one scores 0.
    assert report["num_gt"] == 2
    assert report["macro"]["final"] == pytest.approx(0.5)
    assert report["macro_scored"]["final"] == pytest.approx(1.0)  # scored-only diagnostic


def test_score_submission_dir_accepts_mapping(tmp_path):
    sub = tmp_path / "subs"
    sub.mkdir()
    (sub / "query-p1-2-trake.csv").write_text("L10_V010,101,156,203,251\n", encoding="utf-8")
    report = score_submission_dir(sub, {"query-p1-2-trake": GT_TRAKE})
    assert report["macro"]["final"] == pytest.approx(0.75)
    assert report["macro"]["r_at_k"]["100"] == pytest.approx(0.75)


def test_score_submission_dir_macro_over_every_gt_entry(tmp_path):
    # 3 GT queries, only 1 (perfect) submission -> headline macro is 1/3, not 1.0.
    sub = tmp_path / "subs"
    sub.mkdir()
    (sub / "query-p1-1-kis.csv").write_text("L01_V001,505\n", encoding="utf-8")
    gt = {"query-p1-1-kis": GT_KIS, "query-p1-2-kis": GT_KIS, "query-p1-3-qa": GT_QA}
    report = score_submission_dir(sub, gt)
    assert report["num_gt"] == 3
    assert report["num_scored"] == 1
    assert report["macro"]["final"] == pytest.approx(1 / 3)
    assert report["macro"]["r_at_k"]["100"] == pytest.approx(1 / 3)
    assert report["macro_scored"]["final"] == pytest.approx(1.0)


def test_score_submission_dir_unknown_task_unscored_not_kis(tmp_path):
    # 'zzz' must NOT silently fall back to KIS scoring (row would score 1.0).
    sub = tmp_path / "subs"
    sub.mkdir()
    (sub / "query-p1-1-kis.csv").write_text("L01_V001,505\n", encoding="utf-8")
    report = score_submission_dir(sub, {"query-p1-1-kis": {**GT_KIS, "task": "zzz"}})
    row = report["per_query"]["query-p1-1-kis"]
    assert row["status"] == "unscored"
    assert row["reason"].startswith("unknown task")
    assert report["num_scored"] == 0
    assert report["macro"]["final"] == pytest.approx(0.0)


def test_score_submission_dir_normalises_task_strings(tmp_path):
    # Trailing whitespace / case / 'vqa' alias all route to the right formula.
    sub = tmp_path / "subs"
    sub.mkdir()
    (sub / "query-p1-1-kis.csv").write_text("L01_V001,505\n", encoding="utf-8")
    (sub / "query-p1-2-qa.csv").write_text("L05_V005,888,màu trắng\n", encoding="utf-8")
    gt = {
        "query-p1-1-kis": {**GT_KIS, "task": "KIS\n"},
        "query-p1-2-qa": {**GT_QA, "task": "vqa"},  # wrong answer -> 0 under QA rules
    }
    report = score_submission_dir(sub, gt)
    assert report["per_query"]["query-p1-1-kis"] == {
        **report["per_query"]["query-p1-1-kis"], "status": "scored", "task": "kis"}
    assert report["per_query"]["query-p1-1-kis"]["final"] == pytest.approx(1.0)
    qa_row = report["per_query"]["query-p1-2-qa"]
    assert qa_row["task"] == "qa"
    assert qa_row["final"] == pytest.approx(0.0)


def test_score_submission_dir_qa_gt_missing_answer_unscored(tmp_path):
    sub = tmp_path / "subs"
    sub.mkdir()
    (sub / "query-p1-3-qa.csv").write_text("L05_V005,888,\n", encoding="utf-8")
    gt_no_answer = {k: v for k, v in GT_QA.items() if k != "answer"}
    report = score_submission_dir(sub, {"query-p1-3-qa": gt_no_answer})
    row = report["per_query"]["query-p1-3-qa"]
    assert row["status"] == "unscored"
    assert row["reason"] == "gt missing answer"
    assert report["macro"]["final"] == pytest.approx(0.0)


def test_score_submission_dir_missing_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        score_submission_dir(tmp_path / "no-such-dir", {"query-p1-1-kis": GT_KIS})


def test_score_submission_dir_empty_dir_warns(tmp_path, caplog):
    sub = tmp_path / "subs"
    sub.mkdir()
    with caplog.at_level("WARNING", logger="cvp.eval.official"):
        report = score_submission_dir(sub, {"query-p1-1-kis": GT_KIS})
    assert report["macro"]["final"] == pytest.approx(0.0)
    assert report["num_gt"] == 1
    assert any("No *.csv" in r.message for r in caplog.records)


# ================================================================ new: k-cutoff boundaries
@pytest.mark.parametrize("rank,expected", [(1, 1.0), (5, 0.8), (20, 0.6), (50, 0.4), (100, 0.2)])
def test_hit_exactly_at_each_cutoff(rank, expected):
    scores = [0.0] * 100
    scores[rank - 1] = 1.0
    assert final_score(scores) == pytest.approx(expected)


def test_hit_just_past_each_cutoff():
    for rank, expected in [(2, 0.8), (6, 0.6), (21, 0.4), (51, 0.2)]:
        scores = [0.0] * 100
        scores[rank - 1] = 1.0
        assert final_score(scores) == pytest.approx(expected)


def test_duplicate_rows_do_not_inflate_score():
    dup = score_rows("kis", [["L01_V001", "505"]] * 30, GT_KIS)
    single = score_rows("kis", [["L01_V001", "505"]], GT_KIS)
    assert dup.r_at == single.r_at
    assert dup.final == pytest.approx(single.final) == pytest.approx(1.0)


# ================================================================ new: empty / oversized
def test_empty_csv_scores_zero(tmp_path):
    p = tmp_path / "query-p1-1-kis.csv"
    p.write_text("", encoding="utf-8")
    qs = score_csv(p, "kis", GT_KIS)
    assert qs.final == 0.0
    assert qs.num_rows == 0
    assert qs.best_rank is None
    assert qs.r_at == {k: 0.0 for k in K_VALUES}


def test_rows_beyond_100_are_ignored():
    # hit at rank 120: past the official budget -> scores nothing at all.
    rows = [["L09_V999", "1"]] * 119 + [["L01_V001", "505"]]
    qs = score_rows("kis", rows, GT_KIS)
    assert qs.num_rows == 100
    assert qs.final == 0.0
    assert qs.best_rank is None
    # hit exactly at rank 100 still counts for R@100 only.
    rows2 = [["L09_V999", "1"]] * 99 + [["L01_V001", "505"]] + [["L09_V999", "1"]] * 50
    qs2 = score_rows("kis", rows2, GT_KIS)
    assert qs2.final == pytest.approx(0.2)
    assert qs2.best_rank == 100


# ================================================================ new: dataclass API
def test_score_rows_trake_organiser_worked_example():
    qs = score_rows("trake", [["L10_V010", "101", "156", "203", "251"]], GT_TRAKE)
    assert isinstance(qs, QueryScore)
    assert qs.final == pytest.approx(0.75)  # 3 of 4 moments hit
    assert qs.best_score == pytest.approx(0.75)
    assert qs.task == "trake"


def test_score_rows_qa_diacritics_preserved():
    assert score_rows("qa", [["L05_V005", "888", "mau xanh"]], GT_QA).final == 0.0
    assert score_rows("qa", [["L05_V005", "888", "màu xanh"]], GT_QA).final == 1.0


def test_score_rows_multiple_acceptable_answers():
    gt = {k: v for k, v in GT_QA.items() if k != "answer"}
    gt["answers"] = ["màu xanh", "xanh dương"]
    assert score_rows("qa", [["L05_V005", "888", "Xanh dương"]], gt).final == 1.0
    assert score_rows("qa", [["L05_V005", "888", "xanh nhạt"]], gt).final == 0.0


def test_score_rows_unknown_task_raises():
    with pytest.raises(ValueError, match="unknown task"):
        score_rows("zzz", [["L01_V001", "505"]], GT_KIS)


def test_score_rows_avs_alias_scores_as_kis():
    assert score_rows("avs", [["L01_V001", "505"]], GT_KIS).final == pytest.approx(1.0)


def test_score_csv_int_r_at_keys(tmp_path):
    p = tmp_path / "query-p1-1-kis.csv"
    p.write_text("L09_V999,1\nL01_V001,505\n", encoding="utf-8")
    qs = score_csv(p, "kis", GT_KIS)
    assert sorted(qs.r_at) == sorted(K_VALUES)  # int keys
    assert qs.r_at[1] == 0.0
    assert qs.r_at[5] == 1.0
    assert qs.final == pytest.approx(0.8)


# ================================================================ new: GT loading / unify
def test_load_ground_truth_unifies_both_formats(tmp_path):
    gt_json = {
        # canonical cvp format: range / moments / answers; task inferred from stem
        "query-p1-1-kis": {"video_id": "L01_V001", "range": [500, 510]},
        "query-p1-2-qa": {"task": "qa", "video_id": "L05_V005",
                          "center": 850, "epsilon": 50, "answers": ["màu xanh"]},
        "query-p1-3-trake": {"task": "trake", "video_id": "L10_V010",
                             "moments": [[95, 105], [145, 155]]},
        "query-p1-4-trake": {"task": "trake", "video_id": "L10_V010",
                             "centers": [100, 150], "epsilon": 5},
        # Core-Vision_HCMC-AI format: frame_start/frame_end, events, answer
        "query-p1-5-qa": {"task": "qa", "video_id": "L05_V005",
                          "frame_start": 800, "frame_end": 900, "answer": "màu xanh"},
        "query-p1-6-trake": {"task": "trake", "video_id": "L10_V010",
                             "events": [{"frame_start": 95, "frame_end": 105}]},
    }
    p = tmp_path / "gt.json"
    p.write_text(json.dumps(gt_json, ensure_ascii=False), encoding="utf-8")
    gt = load_ground_truth(p)

    assert gt["query-p1-1-kis"]["task"] == "kis"  # inferred from the stem
    assert gt["query-p1-1-kis"]["frame_start"] == 500
    assert r_score_kis(("L01_V001", 505), gt["query-p1-1-kis"]) == 1
    assert gt["query-p1-2-qa"]["frame_start"] == 800  # center 850 - eps 50
    assert r_score_qa(("L05_V005", 888, "màu xanh"), gt["query-p1-2-qa"]) == 1
    assert gt["query-p1-3-trake"]["events"][0] == {"frame_start": 95, "frame_end": 105}
    assert r_score_trake(("L10_V010", [101, 151]), gt["query-p1-4-trake"]) == 1.0
    assert r_score_qa(("L05_V005", 888, "màu xanh"), gt["query-p1-5-qa"]) == 1
    assert gt["query-p1-5-qa"]["answers"] == ["màu xanh"]
    assert r_score_trake(("L10_V010", [100]), gt["query-p1-6-trake"]) == 1.0


def test_load_ground_truth_bad_shape_raises(tmp_path):
    p = tmp_path / "gt.json"
    p.write_text(json.dumps(["not", "an", "object"]), encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        load_ground_truth(p)
    with pytest.raises(FileNotFoundError):
        load_ground_truth(tmp_path / "missing.json")


def test_load_ground_truth_kis_qa_without_window_raises(tmp_path):
    # Malformed GT is operator error: a KIS/QA entry with no usable window
    # would silently score every submission 0 — it must be loud, naming the id.
    p = tmp_path / "gt.json"
    p.write_text(json.dumps({"query-p1-1-kis": {"video_id": "L01_V001"}}), encoding="utf-8")
    with pytest.raises(ValueError, match="query-p1-1-kis"):
        load_ground_truth(p)
    p.write_text(json.dumps(
        {"query-p1-2-qa": {"task": "qa", "video_id": "L05_V005", "answers": ["màu xanh"]}},
        ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="query-p1-2-qa"):
        load_ground_truth(p)
    # unparseable window spellings are just as unusable
    p.write_text(json.dumps(
        {"query-p1-3-kis": {"video_id": "L01_V001", "range": ["not", "ints"]}}),
        encoding="utf-8")
    with pytest.raises(ValueError, match="query-p1-3-kis"):
        load_ground_truth(p)
    # TRAKE entries and well-formed KIS/QA windows are unaffected
    p.write_text(json.dumps({
        "query-p1-4-trake": {"task": "trake", "video_id": "L10_V010",
                             "moments": [[95, 105]]},
        "query-p1-5-kis": {"video_id": "L01_V001", "range": [500, 510]},
    }), encoding="utf-8")
    gt = load_ground_truth(p)
    assert set(gt) == {"query-p1-4-trake", "query-p1-5-kis"}


# ================================================================ new: task inference
def test_infer_task_priority_matches_run_queries():
    from cvp.pipeline.run_queries import infer_task as pipeline_infer_task

    for name in ("query-p1-9-trake.csv", "kis-then-trake.csv", "query-avs-1.csv",
                 "query-2-qa.csv", "query-3-kis.csv", "mystery.csv"):
        assert infer_task(name) == pipeline_infer_task(name)
    assert infer_task("kis-then-trake.csv") == "trake"  # priority, not position
    assert infer_task("query-avs-1.csv") == "avs"
    assert infer_task("mystery.csv") == "kis"


# ================================================================ new: score_run
def _write_run(tmp_path):
    sub = tmp_path / "subs"
    sub.mkdir()
    (sub / "query-p1-1-kis.csv").write_text("L01_V001,505\n", encoding="utf-8")
    (sub / "query-p1-2-qa.csv").write_text("L05_V005,888,màu xanh\n", encoding="utf-8")
    (sub / "query-p1-3-trake.csv").write_text("L10_V010,101,156,203,251\n", encoding="utf-8")
    gt = {"query-p1-1-kis": GT_KIS, "query-p1-2-qa": GT_QA,
          "query-p1-3-trake": GT_TRAKE, "query-p1-4-kis": GT_KIS}
    gt_path = tmp_path / "gt.json"
    gt_path.write_text(json.dumps(gt, ensure_ascii=False), encoding="utf-8")
    return sub, gt_path


def test_score_run_end_to_end(tmp_path):
    sub, gt_path = _write_run(tmp_path)
    report = score_run(sub, gt_path)
    assert isinstance(report, RunReport)
    assert report.num_gt == 4
    assert report.num_scored == 3
    assert report.per_query["query-p1-1-kis"].final == pytest.approx(1.0)
    assert report.per_query["query-p1-3-trake"].final == pytest.approx(0.75)
    assert report.sum_final == pytest.approx(2.75)
    assert report.mean_final == pytest.approx(2.75 / 4)   # missing query counts as 0
    assert report.mean_r_at[100] == pytest.approx(2.75 / 4)
    assert report.by_task["kis"] == pytest.approx(0.5)    # (1.0 + missing 0) / 2
    assert report.by_task["qa"] == pytest.approx(1.0)
    assert report.by_task["trake"] == pytest.approx(0.75)
    assert report.gt_without_submission == ["query-p1-4-kis"]
    # JSON round-trip stays intact
    data = json.loads(json.dumps(report.to_dict(), ensure_ascii=False))
    assert data["mean_final"] == pytest.approx(2.75 / 4)
    assert data["per_query"]["query-p1-3-trake"]["r_at"]["1"] == pytest.approx(0.75)


def test_score_run_infers_task_from_filename(tmp_path):
    sub = tmp_path / "subs"
    sub.mkdir()
    (sub / "query-p1-1-kis.csv").write_text("L01_V001,505\n", encoding="utf-8")
    gt_no_task = {k: v for k, v in GT_KIS.items() if k != "task"}
    report = score_run(sub, {"query-p1-1-kis": gt_no_task})
    assert report.per_query["query-p1-1-kis"].task == "kis"
    assert report.mean_final == pytest.approx(1.0)


def test_score_run_unscored_reasons(tmp_path):
    sub = tmp_path / "subs"
    sub.mkdir()
    (sub / "query-p1-8-kis.csv").write_text("L01_V001,505\n", encoding="utf-8")   # no GT
    (sub / "query-p1-1-kis.csv").write_text("L01_V001,505\n", encoding="utf-8")   # junk task
    report = score_run(sub, {"query-p1-1-kis": {**GT_KIS, "task": "zzz"}})
    assert report.num_scored == 0
    assert report.unscored["query-p1-8-kis"] == "no ground-truth entry"
    assert report.unscored["query-p1-1-kis"].startswith("unknown task")
    assert report.mean_final == pytest.approx(0.0)


def test_score_run_missing_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        score_run(tmp_path / "no-such-dir", {"query-p1-1-kis": GT_KIS})


# ================================================================ new: validation
def test_validate_rows_accepts_writer_style_rows():
    assert validate_rows("kis", [["L01_V001", "505"]]) == []
    assert validate_rows("qa", [["L01_V001", "505", "màu xanh"]]) == []
    assert validate_rows("trake", [["L01_V001", "10", "20", "30"]]) == []
    assert validate_rows("avs", [["L01_V001", "505"]]) == []  # AVS = KIS row format


def test_validate_rows_flags_problems():
    assert validate_rows("kis", [["bad_id", "505"]])                 # bad video id
    assert validate_rows("kis", [["L01_V001", "-5"]])                # negative frame
    assert validate_rows("kis", [["L01_V001", "x"]])                 # non-integer frame
    assert validate_rows("kis", [["L01_V001", "5", "extra"]])        # extra KIS column
    assert validate_rows("qa", [["L01_V001", "5"]])                  # missing answer col
    assert validate_rows("trake", [["L01_V001", "30", "20"]])        # not increasing
    over = [["L01_V001", str(i + 1)] for i in range(101)]
    assert any("101 rows" in p for p in validate_rows("kis", over))  # >100 rows


def test_validate_csv_infers_task_from_filename(tmp_path):
    p = tmp_path / "query-p1-1-trake.csv"
    p.write_text("L01_V001,30,20\n", encoding="utf-8")  # invalid as TRAKE (not increasing)
    assert validate_csv(p)  # inferred trake -> flagged
    assert validate_csv(p, "kis")  # explicit kis -> extra column flagged too


# ================================================================ new: CLI (scripts/40)
def _run_cli(*args: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    return subprocess.run(
        [sys.executable, str(REPO / "scripts" / "40_eval_official.py"), *args],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        cwd=str(REPO), env=env, timeout=120,
    )


def test_cli_scores_and_writes_json(tmp_path):
    sub, gt_path = _write_run(tmp_path)
    out_json = tmp_path / "report.json"
    proc = _run_cli("--submission-dir", str(sub), "--gt", str(gt_path),
                    "--json-out", str(out_json))
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert "run score" in proc.stdout
    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert data["mean_final"] == pytest.approx(2.75 / 4)
    assert data["per_query"]["query-p1-1-kis"]["final"] == pytest.approx(1.0)
    assert data["malformed"] == {}


def test_cli_exits_nonzero_on_malformed_csv(tmp_path):
    sub, gt_path = _write_run(tmp_path)
    (sub / "query-p1-4-kis.csv").write_text("not_a_video,xx\n", encoding="utf-8")
    proc = _run_cli("--submission-dir", str(sub), "--gt", str(gt_path))
    assert proc.returncode == 2, proc.stderr + proc.stdout
    assert "MALFORMED" in proc.stderr
