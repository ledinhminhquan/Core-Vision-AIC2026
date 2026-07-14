"""Round-3 adversarial-review fixes (2026-07-12) — regression pins.

Covers: UTF-8 script bootstrap (M-R3-2), eval_model lazy imports (L-R3-4),
notebook-builder fixes (M-R3-1 ffmpeg-python, L-R3-7 WiSE-FT α=1.0, L-R3-8
pip-check regex, C-R3-3 cell ids), Streamlit source-level guards (M-R3-3,
L-R3-1/2/3, C-R3-2 — streamlit isn't a test dependency, so the app is pinned
statically + via compile), the extraction density knob and scripts/20 --gt.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
APP_SRC = (REPO / "app" / "streamlit_app.py").read_text(encoding="utf-8")


def _run_help(script: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPTS / script), "--help"],
        capture_output=True, cwd=str(REPO), timeout=120,
    )


# ── M-R3-2: piped stdout must survive non-ASCII help/prints ──────────────────


def test_scripts_help_survives_piped_stdout():
    """capture_output pipes stdout → pre-fix this died with UnicodeEncodeError
    (cp1252) on every script whose docstring contains '→'."""
    for script in ("00_build_catalog.py", "20_run_queries.py", "23_dump_signals.py",
                   "25_auto_agent.py", "eval_model.py"):
        proc = _run_help(script)
        assert proc.returncode == 0, f"{script} --help failed:\n{proc.stderr.decode(errors='replace')}"


def test_bootstrap_reconfigures_non_utf8_streams():
    src = (SCRIPTS / "_bootstrap.py").read_text(encoding="utf-8")
    assert "reconfigure" in src and "utf-8" in src


# ── L-R3-4: eval_model keeps the lazy-import convention ──────────────────────


def test_eval_model_has_no_module_level_heavy_imports():
    tree = ast.parse((SCRIPTS / "eval_model.py").read_text(encoding="utf-8"))
    top_level = {
        node.module if isinstance(node, ast.ImportFrom) else alias.name
        for node in tree.body
        for alias in (node.names if isinstance(node, (ast.Import, ast.ImportFrom)) else [])
        if isinstance(node, (ast.Import, ast.ImportFrom))
    }
    heavy = {m for m in top_level if m and (
        m.startswith("cvp.training") or m.startswith("cvp.models") or m in ("numpy", "torch"))}
    assert not heavy, f"module-level heavy imports break torch-less --help: {heavy}"


# ── notebook-builder fixes ───────────────────────────────────────────────────


def _load_builder():
    spec = importlib.util.spec_from_file_location(
        "builder_for_round3_tests", REPO / "notebooks" / "_build_notebooks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_wiseft_alphas_include_raw_tuned_guard():
    # L-R3-7: α=1.0 = raw tuned tower → wiseft_best can never score below it.
    mod = _load_builder()
    assert "WISEFT_ALPHAS       = [0.4, 0.5, 0.6, 1.0]" in Path(
        REPO / "notebooks" / "_build_notebooks.py").read_text(encoding="utf-8")
    assert mod is not None


def test_nb01_installs_ffmpeg_python_with_transnet():
    # M-R3-1: transnetv2-pytorch predict_video hard-requires ffmpeg-python.
    nb = json.loads((REPO / "notebooks" / "01_build_artifacts_colab.ipynb")
                    .read_text(encoding="utf-8"))
    src = "".join("".join(c["source"]) for c in nb["cells"])
    assert '"transnetv2-pytorch", "ffmpeg-python"' in src


def test_pip_check_regex_matches_both_real_pip_formats():
    # L-R3-8: pip emits 'requires X, which is not installed.' AND
    # 'has requirement X, but you have Y'. The old regex missed the second.
    builder_src = (REPO / "notebooks" / "_build_notebooks.py").read_text(encoding="utf-8")
    m = re.search(r're\.findall\(\s*r"(.+?)", out\)', builder_src)
    assert m, "pip-check findall not found in builder"
    pattern = m.group(1)
    missing = "somepkg 1.0 requires easyocr, which is not installed."
    conflict = "somepkg 1.0 has requirement transformers>=5.0, but you have transformers 4.57.1."
    assert re.findall(pattern, missing) == ["easyocr"]
    assert re.findall(pattern, conflict) == ["transformers>=5.0"]


def test_generated_notebooks_have_unique_deterministic_cell_ids():
    # C-R3-3: nbformat 4.5 requires a cell 'id'; ids must be unique and stable.
    for name in ("01_build_artifacts_colab", "02_train_vi_encoder_H100", "03_test_system"):
        nb = json.loads((REPO / "notebooks" / f"{name}.ipynb").read_text(encoding="utf-8"))
        ids = [c.get("id") for c in nb["cells"]]
        assert all(ids), f"{name}: cell without id"
        assert len(set(ids)) == len(ids), f"{name}: duplicate cell ids"
        assert all(re.fullmatch(r"[a-f0-9]{12}", i) for i in ids)


# ── Streamlit app guards (static pins — streamlit is not a test dep) ─────────


def test_app_compiles():
    compile(APP_SRC, "streamlit_app.py", "exec")


def test_app_export_uses_task_provenance_guard():
    # M-R3-3: exports must not auto-append another tab's ranking.
    assert "results_task" in APP_SRC
    assert "_current_results_for" in APP_SRC
    assert APP_SRC.count("_set_results(") >= 6      # all search sites route through it


def test_app_export_stamp_is_date_qualified_and_collision_proof():
    # L-R3-3: HHMMSS-only stamps overwrote same-second / cross-day exports.
    assert '%Y%m%d-%H%M%S' in APP_SRC
    assert "while path.exists()" in APP_SRC


def test_app_vqa_suggestions_only_render_for_qa_tab():
    # L-R3-1: stale 💡 captions must not appear on KIS/AVS/KIS-C grids.
    assert 'task == "qa" and st.session_state.vqa_suggestions.get' in APP_SRC


def test_app_kisc_search_expires_feedback_marks():
    # L-R3-2: marks from an old KIS query must not refine the KIS-C query.
    idx = APP_SRC.find('_set_results(engine.search_text(turn.query')
    assert idx != -1
    tail = APP_SRC[idx: idx + 400]
    assert "marks_pos" in tail and "set(), set()" in tail


def test_app_sidebar_counters_render_after_tab_handlers():
    # C-R3-2: the basket-counter sidebar block sits AFTER the tabs.
    assert APP_SRC.index("st.tabs(") < APP_SRC.index("🧺 Baskets")


# ── enhancements ─────────────────────────────────────────────────────────────


def test_extraction_density_knob_wired():
    from cvp.config import Settings
    from cvp.data.extraction import _pick_frames

    s = Settings()
    assert s.extraction.shot_positions == [0.15, 0.50, 0.85]
    dense = _pick_frames([(0, 100)], positions=(0.1, 0.3, 0.5, 0.7, 0.9))
    assert dense == [10, 30, 50, 70, 90]
    sparse = _pick_frames([(0, 100)])
    assert len(dense) > len(sparse)


def test_run_queries_script_has_gt_flag():
    proc = _run_help("20_run_queries.py")
    assert proc.returncode == 0 and b"--gt" in proc.stdout


# ── round-4 verification fixes ───────────────────────────────────────────────


def test_qa_split_ignores_lowercase_conversational_hoi():
    # Round-4: lowercase "hỏi" is an ordinary verb — a marker-less description
    # containing "hỏi đường" must NOT be truncated at it.
    from cvp.pipeline.run_queries import parse_query_lines, split_qa_line

    line = "Người phụ nữ mặc áo xanh dừng lại hỏi đường cảnh sát giao thông ở ngã tư nào?"
    desc, question = split_qa_line(line)
    assert desc == line and question == line          # no marker → passthrough
    text, q = parse_query_lines("qa", [line])
    assert "cảnh sát giao thông" in text              # retrieval keeps the scene detail
    # capital marker still splits:
    desc2, q2 = split_qa_line("Các em học hỏi trong lớp. Hỏi lớp có mấy em?")
    assert q2.startswith("Hỏi lớp") and desc2.endswith("trong lớp")


def test_dump_signals_mirrors_trake_prepend_knob():
    src = (SCRIPTS / "23_dump_signals.py").read_text(encoding="utf-8")
    assert "event_context" in src and "prepend_context=prepend" in src


def test_nb01_transnet_install_includes_future_and_verifies_import():
    nb = json.loads((REPO / "notebooks" / "01_build_artifacts_colab.ipynb")
                    .read_text(encoding="utf-8"))
    src = "".join("".join(c["source"]) for c in nb["cells"])
    assert '"transnetv2-pytorch", "ffmpeg-python", "future"' in src
    assert "_transnet_ready()" in src                 # verify-then-report, not rc-based


def test_extraction_config_rejects_empty_or_out_of_range_positions():
    import pytest as _pytest

    from cvp.config import ExtractionCfg

    with _pytest.raises(ValueError, match="must not be empty"):
        ExtractionCfg(shot_positions=[])
    with _pytest.raises(ValueError, match="within"):
        ExtractionCfg(shot_positions=[0.5, 1.5])
    assert ExtractionCfg(shot_positions=[0.1, 0.9]).shot_positions == [0.1, 0.9]


def test_drive_setup_env_block_uses_powershell_form():
    src = (REPO / "docs" / "DRIVE_SETUP.md").read_text(encoding="utf-8")
    assert '$env:CVP_PATHS__DATA_ROOT' in src
    assert 'set CVP_PATHS__DATA_ROOT=D:' not in src   # broken cmd form removed
