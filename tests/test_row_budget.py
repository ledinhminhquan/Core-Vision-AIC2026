"""search.row_strategy / cvp.pipeline.row_budget (Nhiệm vụ D đợt 2).

Offline thuần: đầu ranking giữ NGUYÊN VĂN, đuôi được thay bằng variant frame
lân cận quanh các video top (round-robin), off = bit-identical hành vi cũ.
"""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from cvp.config import Settings
from cvp.pipeline.row_budget import catalog_grid_fn, diversify_tail, variant_frames
from cvp.pipeline.run_queries import run_query_file

GRID = [n * 100 for n in range(1, 11)]        # 100, 200, ..., 1000


# ── variant_frames ───────────────────────────────────────────────────────────


def test_variant_frames_midpoints_first_then_grid_neighbors():
    assert variant_frames(500, GRID, max_variants=4) == [450, 550, 400, 600]
    assert variant_frames(500, GRID, max_variants=2) == [450, 550]


def test_variant_frames_edges_and_off_grid():
    assert variant_frames(100, GRID) == [150, 200]     # không có keyframe trước
    assert variant_frames(1000, GRID) == [950, 900]    # không có keyframe sau
    assert variant_frames(450, GRID) == []             # frame ngoài lưới → chịu
    assert variant_frames(500, None) == []
    assert variant_frames(500, GRID, max_variants=0) == []


# ── diversify_tail ───────────────────────────────────────────────────────────


def _grid_fn(video_id: str):
    return GRID if video_id in ("A", "B") else None


BASE_ROWS = [
    ("A", 500), ("A", 600), ("B", 300),       # head (head_keep=3)
    ("C", 400), ("A", 900),                   # đuôi gốc
]


def test_diversify_head_verbatim_roundrobin_and_tail_kept():
    out = diversify_tail(list(BASE_ROWS), _grid_fn, head_keep=3, budget=10,
                         variants_per_anchor=2)
    assert out[:3] == BASE_ROWS[:3]                        # đầu nguyên văn
    # anchors = A@500, B@300 (video phân biệt, theo rank); round-robin j=0 rồi
    # j=1; trần variant = (10-3)//2 = 3 dòng.
    assert out[3:6] == [("A", 450), ("B", 250), ("A", 550)]
    assert out[6:] == [("C", 400), ("A", 900)]             # đuôi gốc theo sau
    assert len(out) <= 10


def test_diversify_skips_variants_already_in_base_rows():
    rows = list(BASE_ROWS) + [("A", 550)]                  # 550 đã có trong bài
    out = diversify_tail(rows, _grid_fn, head_keep=3, budget=10,
                         variants_per_anchor=2)
    variants = out[3:6]
    assert ("A", 550) not in variants                      # không nhân đôi
    assert variants == [("A", 450), ("B", 250), ("B", 350)]


def test_diversify_qa_payload_rides_from_anchor():
    rows = [("A", 500, "sáu"), ("B", 300, "màu đỏ"), ("A", 900, "sáu")]
    out = diversify_tail(rows, _grid_fn, head_keep=2, budget=10,
                         variants_per_anchor=1)
    assert out[:2] == rows[:2]
    assert ("A", 450, "sáu") in out and ("B", 250, "màu đỏ") in out


def test_diversify_budget_cap_and_variant_share():
    # budget 6, head 3 → đuôi 3, variant tối đa (6-3)//2 = 1 dòng.
    out = diversify_tail(list(BASE_ROWS), _grid_fn, head_keep=3, budget=6,
                         variants_per_anchor=4)
    assert len(out) == 6
    assert out[3] == ("A", 450)                            # đúng 1 variant
    assert out[4:] == [("C", 400), ("A", 900)]


def test_diversify_no_grids_or_short_list_returns_same_object():
    rows = list(BASE_ROWS)
    assert diversify_tail(rows, lambda v: None, head_keep=3, budget=10) is rows
    short = [("A", 500)]
    assert diversify_tail(short, _grid_fn, head_keep=3, budget=10) is short
    assert diversify_tail(rows, _grid_fn, head_keep=3, budget=3) is rows


# ── catalog_grid_fn ──────────────────────────────────────────────────────────


def test_catalog_grid_fn_reads_sorted_grid_and_caches():
    calls = []

    def _load():
        calls.append(1)
        return pd.DataFrame({"video_id": ["A", "A", "A", "B"],
                             "frame_idx": [300, 100, 200, 700]})

    fn = catalog_grid_fn(SimpleNamespace(load=_load))
    assert fn("A") == [100, 200, 300]
    assert fn("A") == [100, 200, 300]
    assert len(calls) == 1                                 # cache per video
    assert fn("Z") is None
    assert catalog_grid_fn(None)("A") is None


def test_catalog_grid_fn_survives_broken_load():
    def _boom():
        raise RuntimeError("manifest hỏng")

    fn = catalog_grid_fn(SimpleNamespace(load=_boom))
    assert fn("A") is None                                 # warn, không crash


# ── wiring run_query_file (KIS + QA), off = y cũ ─────────────────────────────


def _result(vid: str, frame: int, gid: int):
    ref = SimpleNamespace(global_id=gid, video_id=vid, n=frame // 100,
                          frame_idx=frame, pts_time=frame / 25.0,
                          path=f"/fake/{vid}/{frame}.jpg")
    return SimpleNamespace(ref=ref, score=1.0 - 0.01 * gid, signals={},
                           video_id=vid, frame_idx=frame, global_id=gid)


class _StubEngine:
    def __init__(self, results, settings):
        self.settings = settings
        self._results = results
        self.catalog = SimpleNamespace(load=lambda: pd.DataFrame({
            "video_id": ["L01_V001"] * 10 + ["L02_V002"] * 10,
            "frame_idx": GRID + GRID,
        }))

    def search_text(self, query, topk=None, display_k=None):
        return list(self._results)


def _results_two_videos():
    return [_result("L01_V001", 500, 0), _result("L02_V002", 300, 1),
            _result("L01_V001", 900, 2), _result("L02_V002", 800, 3)]


def test_run_query_file_kis_diversify_on(tmp_path):
    settings = Settings()
    settings.search.row_strategy = "diversify_tail"
    settings.search.row_strategy_head = 2
    settings.search.row_strategy_variants = 2
    engine = _StubEngine(_results_two_videos(), settings)
    qf = tmp_path / "query-p1-1-kis.txt"
    qf.write_text("một cảnh\n", encoding="utf-8")
    out = run_query_file(engine, qf, tmp_path)
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert lines[:2] == ["L01_V001,500", "L02_V002,300"]   # đầu nguyên văn
    assert lines[2:6] == ["L01_V001,450", "L02_V002,250",  # round-robin variant
                          "L01_V001,550", "L02_V002,350"]
    assert lines[6:] == ["L01_V001,900", "L02_V002,800"]   # đuôi gốc


def test_run_query_file_kis_off_is_bit_identical(tmp_path):
    engine = _StubEngine(_results_two_videos(), Settings())   # legacy mặc định
    qf = tmp_path / "query-p1-1-kis.txt"
    qf.write_text("một cảnh\n", encoding="utf-8")
    out = run_query_file(engine, qf, tmp_path)
    assert out.read_text(encoding="utf-8").strip().splitlines() == [
        "L01_V001,500", "L02_V002,300", "L01_V001,900", "L02_V002,800"]


def test_run_query_file_qa_variants_carry_answers_and_times_survive(tmp_path):
    settings = Settings()
    settings.search.row_strategy = "diversify_tail"
    settings.search.row_strategy_head = 2
    settings.search.row_strategy_variants = 1

    class _Vqa:
        def answer_group(self, question, image_paths, context=""):
            return "sáu" if "L01_V001" in image_paths[0] else "màu đỏ"

    engine = _StubEngine(_results_two_videos(), settings)
    qf = tmp_path / "query-p1-2-qa.txt"
    qf.write_text("Cảnh đếm đồ vật. Hỏi có mấy cái?\n", encoding="utf-8")
    times: dict[str, list[float]] = {}
    out = run_query_file(engine, qf, tmp_path, vqa=_Vqa(), top1_times=times)
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert lines[:2] == ["L01_V001,500,sáu", "L02_V002,300,màu đỏ"]
    assert "L01_V001,450,sáu" in lines and "L02_V002,250,màu đỏ" in lines
    assert times["query-p1-2-qa"] == [500 / 25.0]          # row 1 không đổi → times giữ


def test_run_query_file_no_catalog_warns_loud_and_keeps_rows(tmp_path, caplog):
    # Round-75: knob bật trên engine KHÔNG có catalog phải cảnh báo TO thay vì
    # âm thầm chạy như legacy (bài học 4 bản vá audit đợt 1).
    import logging

    settings = Settings()
    settings.search.row_strategy = "diversify_tail"
    engine = _StubEngine(_results_two_videos(), settings)
    del engine.catalog                                      # stub không catalog
    qf = tmp_path / "query-p1-1-kis.txt"
    qf.write_text("một cảnh\n", encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        out = run_query_file(engine, qf, tmp_path)
    assert any("KHÔNG có catalog" in r.message for r in caplog.records)
    assert out.read_text(encoding="utf-8").strip().splitlines() == [
        "L01_V001,500", "L02_V002,300", "L01_V001,900", "L02_V002,800"]


def test_run_query_file_avs_never_diversified(tmp_path):
    settings = Settings()
    settings.search.row_strategy = "diversify_tail"

    class _AvsEngine(_StubEngine):
        def search_avs(self, query, **kw):
            return list(self._results)

    engine = _AvsEngine(_results_two_videos(), settings)
    qf = tmp_path / "query-p1-3-avs.txt"
    qf.write_text("một cảnh\n", encoding="utf-8")
    out = run_query_file(engine, qf, tmp_path)
    assert out.read_text(encoding="utf-8").strip().splitlines() == [
        "L01_V001,500", "L02_V002,300", "L01_V001,900", "L02_V002,800"]
