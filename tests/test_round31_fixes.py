"""Round-31: score-improvement levers after trial submission #1 (6.4 public).

- nb03 engine cell gains VLM_RERANK (Gemini listwise rerank of the top-24,
  UIT CVPRW'25 +10% H@1 — now affordable with paid billing) and enables the
  zero-extra-API low-confidence RRF retry.
- RUN_PACK gains REZIP_ONLY: validate + re-zip the pack's CSVs without
  re-searching — the closing step after a human curates rankings in the UI.
- The UI can export STRAIGHT into the pack under the organiser stem
  (sidebar fields), with a task-suffix guard so a QA-tab export can never
  overwrite a -kis file.
"""

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BUILDER = (REPO / "notebooks" / "_build_notebooks.py").read_text(encoding="utf-8")
APP = (REPO / "app" / "streamlit_app.py").read_text(encoding="utf-8")


def test_engine_cell_has_boost_knobs():
    frag = BUILDER.split("NB3_ENGINE = r")[1].split("NB3_QUERIES")[0]
    assert "VLM_RERANK = True" in frag
    assert 'os.environ["CVP_SEARCH__VLM_RERANK"]' in frag
    assert 'os.environ["CVP_SEARCH__LOW_CONFIDENCE_RETRY"] = "true"' in frag


def test_run_pack_rezip_mode():
    frag = BUILDER.split("NB3_RUN_PACK = r")[1].split("NB3_SCORE_GT")[0]
    assert "REZIP_ONLY" in frag
    assert "validate_file" in frag           # curated CSVs still validated
    assert "has_errors" in frag              # zip refused on errors
    assert 'glob("query-*.csv")' in frag     # only organiser-stem files ride


def test_ui_pack_export_guard():
    assert "def _pack_target" in APP
    assert "infer_task" in APP
    assert "pack_stem_input" in APP and "pack_dir_input" in APP
    # every writer call goes through the target resolver now
    assert '_export_path(out_dir, "kis")' not in APP
    assert '_dst("kis")' in APP and '_dst("qa")' in APP and '_dst("trake")' in APP
    # wrong-task overwrite is refused, not warned
    assert "chặn ghi nhầm task" in APP


def test_engine_cell_has_cross_rerank_knob():
    """Round-33: last unused weapon — Qwen3-VL cross-encoder knob (default
    False until the trial leaderboard measures it; fail-open by design)."""
    frag = BUILDER.split("NB3_ENGINE = r")[1].split("NB3_QUERIES")[0]
    assert "CROSS_RERANK" in frag
    assert '"qwen_reranker" if CROSS_RERANK else "none"' in frag
