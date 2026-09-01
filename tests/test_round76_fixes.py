"""Round-76: pre-battle hardening for tonight's round (đợt 2).

The 6-agent battle-readiness audit of nb03 confirmed the config stack, but
found the battle notebook lacked the 2-lane assert that nb04's bench has had
since round-71 — the engine degrades gracefully when a lane fails to load, so
a metadata-lazy VM could fight the whole night on HALF the ensemble with only
a buried log warning. nb03's engine cell now hard-asserts
member_names == ["finetuned", "metaclip2"] (when ENGINE_MODEL is "ensemble")
with the remedy in the message: fresh VM, Run all again.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _engine_cell() -> str:
    nb = json.loads((REPO / "notebooks" / "03_test_system.ipynb")
                    .read_text(encoding="utf-8"))
    cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    return next(s for s in cells if "engine = SearchEngine(settings)" in s)


def test_r76_battle_engine_asserts_both_lanes():
    cell = _engine_cell()
    # round-82: the expected member list follows LINEUP (battle = 2 lanes,
    # diverse = 3) — the assert compares against `_want`
    assert "== _want" in cell and '["finetuned", "metaclip2"]' in cell
    assert "ra trận thiếu lane" in cell          # r82 wraps the message across literals
    # the gate fires after the engine is built and before the ready print
    assert cell.index("engine = SearchEngine(settings)") \
        < cell.index("== _want") \
        < cell.index("engine ready in")


def test_r76_assert_scoped_to_ensemble_mode():
    cell = _engine_cell()
    assert 'if ENGINE_MODEL == "ensemble":\n    _want = (' in cell
    # single-lane fallback modes (finetuned-only) stay usable in an emergency
    assert 'ENGINE_MODEL   = "ensemble"' in cell


def test_r76_missing_gemini_key_warns_before_engine_build():
    cell = _engine_cell()
    i_warn = cell.index("KHÔNG có GEMINI_API_KEY")
    assert cell.index('QUERY_PROVIDER == "gemini" and not') < i_warn
    assert i_warn < cell.index("engine = SearchEngine(settings)")


def test_r76_tuned_weights_read_is_nudged_and_loud_when_missing():
    cell = _engine_cell()
    i_nudge = cell.index("list(_tw.parent.iterdir())")
    i_warn = cell.index("KHÔNG thấy tuning/best_weights.json")
    i_use = cell.index("if _tw.exists():\n    import json")
    assert i_nudge < i_warn < i_use


def test_r76_staging_screams_on_missing_read_hot_dirs():
    # shared staging cell (nb03 + nb04): a missing Drive store must be LOUD —
    # a silently skipped text_index means OCR/ASR/caption quietly score 0.
    for name in ("03_test_system.ipynb", "04_lab_artifacts.ipynb"):
        nb = json.loads((REPO / "notebooks" / name).read_text(encoding="utf-8"))
        cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
        cell = next(s for s in cells if "_READ_HOT" in s)
        assert "_missing.append(_d)" in cell, name
        assert "THIẾU" in cell and "ĐỔI MÁY ẢO MỚI" in cell, name
