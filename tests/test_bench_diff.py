"""scripts/65_bench_diff.py — bench delta reporter (Nhiệm vụ 3).

Fixtures are hand-built RunReport.to_dict() shapes (the exact JSON the Lab
notebook writes as bench_full.json); everything runs offline on tmp files.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _q(task: str, final: float) -> dict:
    return {"task": task, "final": final,
            "r_at": {"1": final, "5": final, "20": final, "50": final, "100": final},
            "best_rank": 1, "best_score": final, "num_rows": 100}


PREV = {
    "mean_final": 0.50,
    "sum_final": 2.0,
    "mean_r_at": {},
    "by_task": {"kis": 0.60, "trake": 0.20},
    "num_gt": 5,
    "num_scored": 4,
    "per_query": {
        "query-p1-1-kis": _q("kis", 1.00),     # sẽ tụt
        "query-p1-2-kis": _q("kis", 0.20),     # sẽ tăng
        "query-p1-3-trake": _q("trake", 0.20),  # đứng yên
        "query-p1-5-kis": _q("kis", 0.40),     # bản mới KHÔNG chấm được
    },
    "unscored": {},
    "gt_without_submission": ["query-p1-4-qa"],
}

CUR = {
    "mean_final": 0.55,
    "sum_final": 2.2,
    "mean_r_at": {},
    "by_task": {"kis": 0.55, "trake": 0.20, "qa": 0.30},
    "num_gt": 5,
    "num_scored": 4,
    "per_query": {
        "query-p1-1-kis": _q("kis", 0.60),
        "query-p1-2-kis": _q("kis", 0.80),
        "query-p1-3-trake": _q("trake", 0.20),
        "query-p1-4-qa": _q("qa", 0.30),       # mới xuất hiện
    },
    "unscored": {"query-p1-5-kis": "error: bad frame_idx 'x'"},
    "gt_without_submission": [],
}


def test_diff_reports_deltas_notes_and_counts():
    m = _load_script("65_bench_diff")
    diff = m.diff_reports(CUR, PREV)

    assert diff["mean_final"] == {"prev": 0.50, "cur": 0.55, "delta": 0.05}
    assert diff["scored"] == {"prev": "4/5", "cur": "4/5"}
    assert diff["counts"] == {"improved": 1, "worsened": 1, "unchanged": 1,
                              "new": 1, "gone": 1}

    by_task = {row["task"]: row for row in diff["by_task"]}
    assert set(by_task) == {"kis", "trake", "qa"}
    assert by_task["kis"]["delta"] == pytest.approx(-0.05)
    assert by_task["qa"]["prev"] is None and by_task["qa"]["cur"] == 0.30

    q = {row["stem"]: row for row in diff["queries"]}
    assert q["query-p1-1-kis"]["delta"] == pytest.approx(-0.40)
    assert q["query-p1-2-kis"]["delta"] == pytest.approx(0.60)
    assert q["query-p1-3-trake"]["delta"] == 0.0
    # bản mới không chấm được câu p1-5 → Δ tính với 0.0 + ghi rõ lý do unscored
    assert q["query-p1-5-kis"]["cur"] is None
    assert q["query-p1-5-kis"]["delta"] == pytest.approx(-0.40)
    assert "unscored" in q["query-p1-5-kis"]["note"]
    assert "bad frame_idx" in q["query-p1-5-kis"]["note"]
    # câu mới xuất hiện được đánh dấu
    assert q["query-p1-4-qa"]["prev"] is None
    assert "mới" in q["query-p1-4-qa"]["note"]


def test_queries_sorted_regressions_first_within_task():
    m = _load_script("65_bench_diff")
    diff = m.diff_reports(CUR, PREV)
    kis_rows = [row for row in diff["queries"] if row["task"] == "kis"]
    deltas = [row["delta"] for row in kis_rows]
    assert deltas == sorted(deltas)                    # xấu nhất đứng đầu
    assert kis_rows[0]["stem"] in ("query-p1-1-kis", "query-p1-5-kis")


def test_render_markdown_contains_tables_and_marks():
    m = _load_script("65_bench_diff")
    md = m.render_markdown(m.diff_reports(CUR, PREV), "bench_full.json",
                           "bench_full-prev.json")
    assert "**mean_final: 0.500 → 0.550 (+0.050)**" in md
    assert "↑1 ↓1 =1 mới:1 mất:1" in md
    assert "| kis | 0.600 | 0.550 | -0.050 |" in md    # bảng theo task
    assert "## kis (3 câu" in md and "## trake (1 câu" in md and "## qa (1 câu" in md
    assert "| query-p1-2-kis | 0.200 | 0.800 | +0.600 |" in md
    assert "| query-p1-5-kis | 0.400 | — | -0.400 | bản mới unscored: error: bad frame_idx 'x' |" in md
    # per-query của task nào nằm dưới header task đó
    kis_section = md.split("## kis")[1].split("## trake")[0]
    assert "query-p1-1-kis" in kis_section and "query-p1-3-trake" not in kis_section


def test_cli_reads_files_and_writes_out(tmp_path, monkeypatch, capsys):
    m = _load_script("65_bench_diff")
    lab = tmp_path / "lab"
    lab.mkdir()
    (lab / "bench_full.json").write_text(json.dumps(CUR, ensure_ascii=False), encoding="utf-8")
    (lab / "bench_full-prev.json").write_text(json.dumps(PREV, ensure_ascii=False), encoding="utf-8")
    out_md = tmp_path / "BENCH_DIFF.md"

    monkeypatch.setattr(sys, "argv", ["65", "--dir", str(lab), "--out", str(out_md)])
    m.main()
    printed = capsys.readouterr().out
    assert "mean_final: 0.500 → 0.550 (+0.050)" in printed
    text = out_md.read_text(encoding="utf-8")
    assert text.startswith("# Bench diff — bench_full.json vs bench_full-prev.json")
    assert "query-p1-2-kis" in text


def test_cli_missing_prev_fails_with_friendly_message(tmp_path, monkeypatch):
    m = _load_script("65_bench_diff")
    cur = tmp_path / "bench_full.json"
    cur.write_text(json.dumps(CUR), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [
        "65", "--cur", str(cur), "--prev", str(tmp_path / "bench_full-prev.json")])
    with pytest.raises(SystemExit, match="bench_full-prev"):
        m.main()


def test_cli_corrupt_json_fails_loud(tmp_path, monkeypatch):
    m = _load_script("65_bench_diff")
    cur = tmp_path / "bench_full.json"
    prev = tmp_path / "bench_full-prev.json"
    cur.write_text("{broken", encoding="utf-8")
    prev.write_text(json.dumps(PREV), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["65", "--cur", str(cur), "--prev", str(prev)])
    with pytest.raises(SystemExit, match="hỏng"):
        m.main()


def test_diff_handles_empty_reports():
    m = _load_script("65_bench_diff")
    diff = m.diff_reports({}, {})
    assert diff["mean_final"]["delta"] == 0.0
    assert diff["queries"] == [] and diff["by_task"] == []
    md = m.render_markdown(diff)
    assert "mean_final: 0.000 → 0.000 (+0.000)" in md
