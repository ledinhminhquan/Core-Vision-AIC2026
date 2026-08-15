"""Round-6 review regressions — auditing the round-5 diff, the Streamlit UI,
and a contest-day simulation.

  R6-1  self-written map csvs stay re-extractable via the .selfmade marker
        (the round-5 map guard deadlocked K-batch dense re-extraction);
        --overwrite is plumbed through extract_missing/scripts/01
  R6-2  inverted SINGLE 'range'/frame_start+frame_end GT windows fail loud
  R6-3  TRAKE _texts_for consumes expansions for English-only lanes
  R6-4  every Gemini surface has a wall-clock timeout (agent/VQA/rerank)
  R6-5  UI: grid gated per owning tab, PIL upload guarded, QA all-blank
        answers refuse to export
  R6-6  run_query_folder fails loud on a typo'd dir; scripts/20 lists DROPPED
  R6-7  packager keeps an append-only history + ledger of every build
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from cvp.config import Settings

REPO = Path(__file__).resolve().parents[1]
APP_SRC = (REPO / "app" / "streamlit_app.py").read_text(encoding="utf-8")


# ── R6-1 · selfmade marker unlocks legitimate re-extraction ──────────────────
def test_selfmade_map_csv_is_re_extractable(tmp_path):
    from cvp.data.extraction import extract_video

    kf, mp = tmp_path / "keyframes", tmp_path / "map-keyframes"
    mp.mkdir(parents=True)
    (mp / "K01_V001.csv").write_text("n,pts_time,fps,frame_idx\n1,0.0,25.0,7\n", encoding="utf-8")
    (mp / "K01_V001.csv.selfmade").touch()          # OUR csv, marker survives
    # Guards must let this through to actual extraction — which then fails on
    # the nonexistent video, PROVING the refusal did not fire.
    with pytest.raises(RuntimeError, match="Cannot open video"):
        extract_video(tmp_path / "K01_V001.mp4", kf, mp, overwrite=False)


def test_unmarked_map_csv_still_refused(tmp_path):
    from cvp.data.extraction import extract_video

    kf, mp = tmp_path / "keyframes", tmp_path / "map-keyframes"
    mp.mkdir(parents=True)
    organiser = "n,pts_time,fps,frame_idx\n1,0.0,30.0,0\n"
    (mp / "L21_V001.csv").write_text(organiser, encoding="utf-8")
    assert extract_video(tmp_path / "L21_V001.mp4", kf, mp, overwrite=False) == 0
    assert (mp / "L21_V001.csv").read_text(encoding="utf-8") == organiser


def test_extract_missing_accepts_overwrite_kw(tmp_path):
    from cvp.data.extraction import extract_missing

    s = Settings()
    s.paths.data_root = tmp_path                     # no videos dir → 0, but the
    assert extract_missing(s, overwrite=True) == 0   # kwarg must exist (scripts/01)


def test_scripts01_has_overwrite_flag():
    src = (REPO / "scripts" / "01_extract_keyframes.py").read_text(encoding="utf-8")
    assert "--overwrite" in src and "overwrite=args.overwrite" in src


# ── R6-2 · inverted single-window GT fails loud ──────────────────────────────
def _load(tmp_path, stem, entry):
    from cvp.eval.official import load_ground_truth

    p = tmp_path / "gt.json"
    p.write_text(json.dumps({stem: entry}), encoding="utf-8")
    return load_ground_truth(p)


def test_gt_inverted_single_range_raises(tmp_path):
    with pytest.raises(ValueError, match="inverted"):
        _load(tmp_path, "query-p1-1-kis",
              {"video_id": "L21_V001", "range": [510, 500]})


def test_gt_inverted_frame_start_end_raises(tmp_path):
    with pytest.raises(ValueError, match="inverted"):
        _load(tmp_path, "query-p1-1-kis",
              {"video_id": "L21_V001", "frame_start": 510, "frame_end": 500})


def test_gt_valid_single_range_still_loads(tmp_path):
    gt = _load(tmp_path, "query-p1-1-kis",
               {"video_id": "L21_V001", "range": [500, 510]})
    assert "query-p1-1-kis" in gt


# ── R6-3 · TRAKE english fallback includes expansions ────────────────────────
def test_trake_texts_for_consumes_expansions():
    src = (REPO / "src" / "cvp" / "search" / "engine.py").read_text(encoding="utf-8")
    assert "p.expansions[0] if p.expansions else p.original" in src


# ── R6-4 · Gemini wall-clock timeouts everywhere ─────────────────────────────
def test_generate_with_fallback_times_out_to_next_model():
    from cvp.search.vqa import generate_with_fallback

    class _Resp:
        text = "ok-from-fast-model"

    class _Models:
        def generate_content(self, model, contents):
            if model == "slow":
                time.sleep(3)
            return _Resp()

    class _Client:
        models = _Models()

    out = generate_with_fallback(_Client(), ["slow", "fast"], "q", timeout_s=0.3)
    assert out == "ok-from-fast-model"


def test_gemini_wall_timeout_floor():
    from cvp.search.vqa import gemini_wall_timeout

    assert gemini_wall_timeout(Settings()) >= 30.0


def test_make_gemini_client_requires_key(monkeypatch):
    from cvp.search.vqa import make_gemini_client

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        make_gemini_client(Settings())


def test_no_bare_gemini_clients_left():
    # Every genai.Client in src/ must carry the timeout discipline — only the
    # two factories (query_processor's own + vqa.make_gemini_client) may call
    # the constructor directly.
    import re as _re

    hits = []
    for py in (REPO / "src").rglob("*.py"):
        for i, line in enumerate(py.read_text(encoding="utf-8").splitlines(), 1):
            if _re.search(r"genai\.Client\(", line):
                hits.append(f"{py.name}:{i}")
    assert sorted({h.split(":")[0] for h in hits}) == ["query_processor.py", "vqa.py"], hits


# ── R6-5 · UI hardening (source pins — streamlit apps aren't importable here) ─
def test_app_grids_are_gated_per_tab():
    assert APP_SRC.count("_grid_results_for(") >= 4          # helper + 3 call sites
    assert 'in ("kis", "kisc") else []' in APP_SRC           # KIS tab gate
    assert "render_result_grid(st.session_state.results" not in APP_SRC


def test_app_kisv_upload_is_guarded():
    assert "Ảnh không đọc được" in APP_SRC
    upload_block = APP_SRC.split("kisv_go")[1].split("Export KIS CSV")[0]
    assert "try:" in upload_block and "except Exception" in upload_block


def test_app_qa_export_refuses_all_blank_answers():
    assert "dòng QA thiếu answer chấm 0 điểm" in APP_SRC


# ── R6-6 · loud query-dir failures + DROPPED summary ─────────────────────────
def test_run_query_folder_raises_on_missing_dir(tmp_path):
    from cvp.pipeline.run_queries import run_query_folder

    with pytest.raises(FileNotFoundError, match="query dir"):
        run_query_folder(Settings(), tmp_path / "nope", tmp_path / "out")


def test_scripts20_prints_dropped_stems():
    src = (REPO / "scripts" / "20_run_queries.py").read_text(encoding="utf-8")
    assert "DROPPED" in src and "produced NO CSV" in src


# ── R6-7 · packager audit trail ──────────────────────────────────────────────
def test_package_codabench_archives_history_and_ledger(tmp_path):
    from cvp.submission.packager import has_errors, package_codabench

    sub = tmp_path / "subs"
    sub.mkdir()
    csv_p = sub / "query-p1-1-kis.csv"
    csv_p.write_text("L21_V001,505\n", encoding="utf-8")
    zip_p = sub / "submission.zip"
    issues = package_codabench(sub, zip_p, package_name="submission", files=[csv_p])
    assert not has_errors(issues) and zip_p.exists()
    history = sorted((sub / "history").glob("*.zip"))
    ledger = sub / "submissions_log.jsonl"
    assert len(history) == 1 and ledger.exists()
    # Repackage: same content → history may dedupe by sha, ledger always appends.
    package_codabench(sub, zip_p, package_name="submission", files=[csv_p])
    lines = ledger.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["files"] == ["query-p1-1-kis.csv"]
