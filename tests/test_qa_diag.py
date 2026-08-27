"""scripts/67_qa_diag.py — phân loại offline câu QA fail (Nhiệm vụ A đợt 2).

Fixture tự dựng: gt.json (dạng scripts/62) + CSV lượt chạy tí hon phủ đủ mọi
verdict: OK / MOMENT_MISS (mất video & suýt trúng) / FORMAT_MISS (số↔chữ,
ngoặc kép) / ANSWER_MISS (INHERITED, NO_ANSWER) / NO_SUBMISSION. Offline
thuần — không engine, không API.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

GT_VIDEO = "L01_V001"
OTHER_VIDEO = "L09_V009"


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _entry(answers: list[str]) -> dict:
    return {"task": "qa", "video_id": GT_VIDEO, "range": [100, 200],
            "answers": answers}


def test_fallback_constant_matches_run_queries():
    m = _load_script("67_qa_diag")
    from cvp.pipeline.run_queries import QA_FALLBACK_ANSWER

    assert m.QA_FALLBACK == QA_FALLBACK_ANSWER


def test_classify_ok_strict_match():
    m = _load_script("67_qa_diag")
    rows = [[OTHER_VIDEO, "50", "táo"], [GT_VIDEO, "150", "Màu Xanh."]]
    d = m.classify_qa("q", rows, _entry(["màu xanh"]))
    assert d["verdict"] == "OK" and d["best_rank"] == 2


def test_classify_moment_miss_no_video():
    m = _load_script("67_qa_diag")
    rows = [[OTHER_VIDEO, "50", "táo"], [OTHER_VIDEO, "60", "táo"]]
    d = m.classify_qa("q", rows, _entry(["màu xanh"]))
    assert d["verdict"] == "MOMENT_MISS" and "NO_VIDEO" in d["flags"]


def test_classify_moment_miss_near_reports_distance():
    m = _load_script("67_qa_diag")
    rows = [[OTHER_VIDEO, "50", "táo"], [GT_VIDEO, "260", "màu xanh"]]
    d = m.classify_qa("q", rows, _entry(["màu xanh"]))
    assert d["verdict"] == "MOMENT_MISS" and "NEAR_MISS" in d["flags"]
    assert d["min_frame_distance"] == 60 and d["first_video_rank"] == 2


def test_classify_format_miss_number_vs_word_tier2():
    m = _load_script("67_qa_diag")
    rows = [[GT_VIDEO, "150", "sáu"]]
    d = m.classify_qa("q", rows, _entry(["6"]))
    assert d["verdict"] == "FORMAT_MISS" and d["format_tier"] == 2


def test_classify_format_miss_wrapping_quotes_tier1():
    m = _load_script("67_qa_diag")
    rows = [[GT_VIDEO, "150", '"Khung trời mơ ước"']]
    d = m.classify_qa("q", rows, _entry(["Khung trời mơ ước"]))
    assert d["verdict"] == "FORMAT_MISS" and d["format_tier"] == 1


def test_classify_format_miss_accent_only_flagged_tier3():
    m = _load_script("67_qa_diag")
    rows = [[GT_VIDEO, "150", "mau xanh"]]
    d = m.classify_qa("q", rows, _entry(["màu xanh"]))
    assert d["verdict"] == "FORMAT_MISS" and d["format_tier"] == 3
    assert "ACCENT_ONLY" in d["flags"]


def test_classify_answer_miss_inherited_signature():
    # Điểm chết A1: dòng trúng GT mang ĐÚNG answer của dòng top-1 KHÁC video.
    m = _load_script("67_qa_diag")
    rows = [[OTHER_VIDEO, "50", "táo"], [GT_VIDEO, "150", "táo"]]
    d = m.classify_qa("q", rows, _entry(["ổi"]))
    assert d["verdict"] == "ANSWER_MISS" and "INHERITED" in d["flags"]


def test_classify_answer_miss_no_answer():
    m = _load_script("67_qa_diag")
    rows = [[GT_VIDEO, "150", "không rõ"], [GT_VIDEO, "160", "không rõ"]]
    d = m.classify_qa("q", rows, _entry(["ổi"]))
    assert d["verdict"] == "ANSWER_MISS" and "NO_ANSWER" in d["flags"]
    assert d["pct_fallback"] == 1.0


def test_classify_no_submission():
    m = _load_script("67_qa_diag")
    d = m.classify_qa("q", None, _entry(["ổi"]))
    assert d["verdict"] == "NO_SUBMISSION"


def test_classify_quoted_comma_answer_rejoined():
    # Writer quote answer chứa dấu phẩy → csv.reader tách lại thành 1 cell;
    # nhưng CSV tay có thể KHÔNG quote → answer tràn cột. Cả hai phải khớp.
    m = _load_script("67_qa_diag")
    rows = [[GT_VIDEO, "150", "màu đỏ", " trắng"]]      # tràn cột không quote
    d = m.classify_qa("q", rows, _entry(["màu đỏ, trắng"]))
    assert d["verdict"] in ("OK", "FORMAT_MISS")        # nội dung phải được nhận ra


def _write_fixture_run(tmp_path: Path) -> tuple[Path, Path]:
    run = tmp_path / "lab_full"
    run.mkdir()
    (run / "query-p1-1-qa.csv").write_text(
        f"{GT_VIDEO},150,màu xanh\n", encoding="utf-8")             # OK
    (run / "query-p1-2-qa.csv").write_text(
        f"{OTHER_VIDEO},50,táo\n{GT_VIDEO},150,táo\n", encoding="utf-8")  # INHERITED
    (run / "query-p1-3-qa.csv").write_text(
        f"{GT_VIDEO},150,sáu\n", encoding="utf-8")                  # FORMAT_MISS
    gt = {
        "query-p1-1-qa": _entry(["màu xanh"]),
        "query-p1-2-qa": _entry(["ổi"]),
        "query-p1-3-qa": _entry(["6"]),
        "query-p1-4-qa": _entry(["mất tích"]),                      # NO_SUBMISSION
        "query-p1-5-kis": {"task": "kis", "video_id": GT_VIDEO, "range": [1, 2]},
    }
    gt_path = tmp_path / "gt.json"
    gt_path.write_text(json.dumps(gt, ensure_ascii=False), encoding="utf-8")
    return run, gt_path


def test_cli_end_to_end_writes_md_and_json(tmp_path, monkeypatch, capsys):
    m = _load_script("67_qa_diag")
    run, gt_path = _write_fixture_run(tmp_path)
    out_md = tmp_path / "diag.md"
    out_json = tmp_path / "diag.json"
    monkeypatch.setattr(sys, "argv", [
        "67", "--run", str(run), "--gt", str(gt_path),
        "--out", str(out_md), "--json-out", str(out_json)])
    m.main()

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    verdicts = {d["stem"]: d["verdict"] for d in payload["queries"]}
    assert verdicts == {
        "query-p1-1-qa": "OK",
        "query-p1-2-qa": "ANSWER_MISS",
        "query-p1-3-qa": "FORMAT_MISS",
        "query-p1-4-qa": "NO_SUBMISSION",
    }
    assert "query-p1-5-kis" not in verdicts             # câu KIS không bị lôi vào
    md = out_md.read_text(encoding="utf-8")
    assert "FORMAT_MISS" in md and "INHERITED" in md
    console = capsys.readouterr().out
    assert "QA diag: 4 câu" in console


def test_cli_missing_run_fails_loud(tmp_path, monkeypatch):
    import pytest

    m = _load_script("67_qa_diag")
    _run, gt_path = _write_fixture_run(tmp_path)
    monkeypatch.setattr(sys, "argv", [
        "67", "--run", str(tmp_path / "khong_ton_tai"), "--gt", str(gt_path)])
    with pytest.raises(SystemExit, match="không tồn tại"):
        m.main()


def test_cli_missing_queries_dir_fails_loud(tmp_path, monkeypatch):
    # Round-75: --queries gõ nhầm từng làm echo parse biến mất KHÔNG dấu vết.
    import pytest

    m = _load_script("67_qa_diag")
    run, gt_path = _write_fixture_run(tmp_path)
    monkeypatch.setattr(sys, "argv", [
        "67", "--run", str(run), "--gt", str(gt_path),
        "--queries", str(tmp_path / "pack_go_nham")])
    with pytest.raises(SystemExit, match="--queries không tồn tại"):
        m.main()


def test_cli_gt_without_qa_fails_loud(tmp_path, monkeypatch):
    import pytest

    m = _load_script("67_qa_diag")
    run = tmp_path / "run"
    run.mkdir()
    (run / "query-p1-5-kis.csv").write_text(f"{GT_VIDEO},1\n", encoding="utf-8")
    gt_path = tmp_path / "gt.json"
    gt_path.write_text(json.dumps(
        {"query-p1-5-kis": {"task": "kis", "video_id": GT_VIDEO, "range": [1, 2]}}),
        encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["67", "--run", str(run), "--gt", str(gt_path)])
    with pytest.raises(SystemExit, match="không có câu QA"):
        m.main()
