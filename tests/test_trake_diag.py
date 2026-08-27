"""scripts/66_trake_diag.py — định lượng H1/H2/H6 TRAKE (Nhiệm vụ B đợt 2).

Fixture tự dựng: map-keyframes CSV nhỏ + gt.json + CSV lượt chạy tí hon.
Offline thuần — không engine, không API.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

VIDEO = "L01_V001"
# Lưới keyframe: frame gốc 100, 200, ..., 1000 (gap đều 100 frame).
GRID = [n * 100 for n in range(1, 11)]
# Event windows: trúng lưới (100) / lọt khe lưới (710-730) / trúng lưới (900).
EVENTS = [[95, 105], [710, 730], [880, 920]]


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _write_map(dir_: Path, video: str = VIDEO, frames: list[int] = GRID) -> Path:
    dir_.mkdir(parents=True, exist_ok=True)
    with open(dir_ / f"{video}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["n", "pts_time", "fps", "frame_idx"])
        for i, fr in enumerate(frames, start=1):
            w.writerow([i, fr / 25.0, 25.0, fr])
    return dir_


# ── H1: trần lưới (DP strictly-increasing) ───────────────────────────────────


def test_grid_ceiling_hits_and_gap_miss():
    m = _load_script("66_trake_diag")
    got = m.grid_ceiling(GRID, [tuple(e) for e in EVENTS])
    assert got["per_event"] == [True, False, True]     # event 2 lọt khe lưới
    assert got["hits"] == 2 and got["ceiling"] == pytest.approx(2 / 3)
    assert got["feasible"]


def test_grid_ceiling_ordering_constraint_shared_window():
    # Hai event cùng cửa sổ chỉ chứa MỘT keyframe → strictly-increasing chỉ
    # cho 1 hit dù per-event đều "có keyframe".
    m = _load_script("66_trake_diag")
    got = m.grid_ceiling(GRID, [(95, 105), (95, 105)])
    assert got["per_event"] == [True, True]
    assert got["hits"] == 1 and got["ceiling"] == 0.5


def test_grid_ceiling_reversed_windows_cannot_both_hit():
    # GT event 2 nằm TRƯỚC event 1 trong video → thứ tự tăng chặn 1 trong 2.
    m = _load_script("66_trake_diag")
    got = m.grid_ceiling(GRID, [(895, 905), (95, 105)])
    assert got["hits"] == 1 and got["ceiling"] == 0.5


def test_grid_ceiling_infeasible_when_fewer_keyframes_than_events():
    # Round-73: dòng nộp NGẮN vẫn hợp lệ và được chấm zip với các event đầu —
    # trần là DP trên p_count event đầu (mẫu số vẫn K), không phải 0.0.
    m = _load_script("66_trake_diag")
    got = m.grid_ceiling([100], [(95, 105), (195, 205)])
    assert got["ceiling"] == 0.5 and not got["feasible"]
    got_miss = m.grid_ceiling([500], [(95, 105), (195, 205)])
    assert got_miss["ceiling"] == 0.0 and not got_miss["feasible"]
    got_empty = m.grid_ceiling([], [(95, 105), (195, 205)])
    assert got_empty["ceiling"] == 0.0 and not got_empty["feasible"]


def test_median_grid_gap():
    m = _load_script("66_trake_diag")
    assert m.median_grid_gap(GRID) == 100.0
    assert m.median_grid_gap([5]) is None


# ── diagnose_query: H1 + H2 + H6 trên fixture đầy đủ ────────────────────────


def test_diagnose_query_full_pipeline(tmp_path):
    m = _load_script("66_trake_diag")
    map_dir = _write_map(tmp_path / "map")
    entry = {"task": "trake", "video_id": VIDEO, "moments": EVENTS}
    rows = [
        ["L09_V009", "1", "2", "3"],            # dòng 1: sai video → 0 điểm
        [VIDEO, "100", "200", "900"],            # dòng 2: trúng event 1+3 → 2/3
    ]
    d = m.diagnose_query("query-p1-4-trake", entry, map_dir, rows)
    assert d["h1_ceiling"] == pytest.approx(2 / 3)
    assert d["median_gap"] == 100.0
    assert d["h2_video_in_rows"] and d["h2_first_rank"] == 2
    assert d["h6_row1"] == 0.0
    assert d["h6_best_row"] == pytest.approx(0.6667, abs=1e-4)
    assert d["h6_best_rank"] == 2
    assert d["h6_benefit"] == pytest.approx(0.6667, abs=1e-4)
    assert d["h6_headroom"] == pytest.approx(0.0, abs=1e-4)   # best đã chạm trần lưới


def test_diagnose_query_missing_map_notes_h1(tmp_path):
    m = _load_script("66_trake_diag")
    entry = {"task": "trake", "video_id": "L99_V999", "moments": EVENTS}
    d = m.diagnose_query("q", entry, tmp_path, None)
    assert "h1_ceiling" not in d and "thiếu map-keyframes" in d["h1_note"]
    assert "h2_video_in_rows" not in d                       # không --run → bỏ H2/H6


# ── verdicts ─────────────────────────────────────────────────────────────────


def test_verdicts_thresholds():
    m = _load_script("66_trake_diag")
    assert "H1: ĐÚNG" in m.verdict_h1([0.2, 0.3])
    assert "H1: SAI" in m.verdict_h1([0.95, 1.0])
    assert "H1: MỘT PHẦN" in m.verdict_h1([0.6, 0.7])
    assert "KHÔNG ĐỦ DỮ LIỆU" in m.verdict_h1([])
    assert "H2: ĐÚNG" in m.verdict_h2([True, False, False])
    assert "H2: SAI" in m.verdict_h2([True, True, True])
    assert "H6: ĐÚNG" in m.verdict_h6([0.2, 0.3], [0.1])
    v = m.verdict_h6([0.0], [0.0])
    assert "gần như không cứu điểm" in v and "CHẠM trần lưới" in v


# ── CLI end-to-end ───────────────────────────────────────────────────────────


def _write_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    map_dir = _write_map(tmp_path / "map")
    gt = {"query-p1-4-trake": {"task": "trake", "video_id": VIDEO, "moments": EVENTS},
          "query-p1-1-kis": {"task": "kis", "video_id": VIDEO, "range": [1, 2]}}
    gt_path = tmp_path / "gt.json"
    gt_path.write_text(json.dumps(gt, ensure_ascii=False), encoding="utf-8")
    run = tmp_path / "run"
    run.mkdir()
    (run / "query-p1-4-trake.csv").write_text(
        f"L09_V009,1,2,3\n{VIDEO},100,200,900\n", encoding="utf-8")
    return map_dir, gt_path, run


def test_cli_end_to_end_writes_md_and_json(tmp_path, monkeypatch, capsys):
    m = _load_script("66_trake_diag")
    map_dir, gt_path, run = _write_fixture(tmp_path)
    out_md, out_json = tmp_path / "diag.md", tmp_path / "diag.json"
    monkeypatch.setattr(sys, "argv", [
        "66", "--gt", str(gt_path), "--map-dir", str(map_dir), "--run", str(run),
        "--out", str(out_md), "--json-out", str(out_json)])
    m.main()

    md = out_md.read_text(encoding="utf-8")
    assert "## H1" in md and "## H2" in md and "## H6" in md
    assert "✓✗✓" in md                                        # bảng per-event
    assert "0.667" in md
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert set(payload["verdicts"]) == {"h1", "h2", "h6"}
    assert payload["queries"][0]["stem"] == "query-p1-4-trake"
    assert "TRAKE diag: 1 câu" in capsys.readouterr().out


def test_cli_without_run_skips_h2_h6(tmp_path, monkeypatch):
    m = _load_script("66_trake_diag")
    map_dir, gt_path, _run = _write_fixture(tmp_path)
    out_md = tmp_path / "diag.md"
    monkeypatch.setattr(sys, "argv", [
        "66", "--gt", str(gt_path), "--map-dir", str(map_dir), "--out", str(out_md)])
    m.main()
    md = out_md.read_text(encoding="utf-8")
    assert "H2/H6: bỏ qua" in md and "## H2" not in md


def test_cli_fails_loud_on_bad_inputs(tmp_path, monkeypatch):
    m = _load_script("66_trake_diag")
    map_dir, gt_path, _run = _write_fixture(tmp_path)
    monkeypatch.setattr(sys, "argv", [
        "66", "--gt", str(gt_path), "--map-dir", str(tmp_path / "khong_co")])
    with pytest.raises(SystemExit, match="không tồn tại"):
        m.main()


def test_cli_fails_loud_on_missing_run_dir(tmp_path, monkeypatch):
    # Round-75: --run gõ nhầm từng cho load_run trả {} ÂM THẦM → H2/H6 báo
    # "CSV rỗng" sai lệch. Giờ phải nổ ngay.
    m = _load_script("66_trake_diag")
    map_dir, gt_path, _run = _write_fixture(tmp_path)
    monkeypatch.setattr(sys, "argv", [
        "66", "--gt", str(gt_path), "--map-dir", str(map_dir),
        "--run", str(tmp_path / "run_go_nham")])
    with pytest.raises(SystemExit, match="--run không tồn tại"):
        m.main()

    gt_no_trake = tmp_path / "gt2.json"
    gt_no_trake.write_text(json.dumps(
        {"query-p1-1-kis": {"task": "kis", "video_id": VIDEO, "range": [1, 2]}}),
        encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [
        "66", "--gt", str(gt_no_trake), "--map-dir", str(map_dir)])
    with pytest.raises(SystemExit, match="không có câu TRAKE"):
        m.main()
