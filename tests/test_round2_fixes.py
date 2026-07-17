"""Round-2 adversarial review regressions (22 confirmed findings, all fixed).

CPU-only. Each test names its finding; see the round-2 report for the full
evidence trail.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cvp.config import Settings

REPO = Path(__file__).resolve().parents[1]


# ── C1: warm-cache warms the EXACT round-time strings ────────────────────────
def _load_warm_script():
    scripts = REPO / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    spec = importlib.util.spec_from_file_location(
        "warm_cache_script", scripts / "51_warm_cache.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_warm_cache_trake_yields_parsed_events(tmp_path):
    mod = _load_warm_script()
    qf = tmp_path / "query-p1-16-trake.txt"
    qf.write_text("Đoạn video múa lân, tìm các sự kiện sau:\n"
                  "E1: Lân quay vòng trên cột.\n"
                  "E2: Bốn chân chạm đất.\n", encoding="utf-8")
    texts = mod.round_time_texts(qf)
    # Header dropped, prefixes stripped, ONE string per event — exactly what
    # run_query_file feeds the processor at round time.
    assert texts == ["Lân quay vòng trên cột.", "Bốn chân chạm đất."]


def test_warm_cache_qa_yields_description_only(tmp_path):
    mod = _load_warm_script()
    qf = tmp_path / "query-p1-15-qa.txt"
    qf.write_text("Đoạn video về một chương trình từ thiện tại Khánh Hòa. "
                  "Hỏi xã này có tên là gì?\n", encoding="utf-8")
    from cvp.pipeline.run_queries import parse_query_lines

    expected, _q = parse_query_lines(
        "qa", ["Đoạn video về một chương trình từ thiện tại Khánh Hòa. "
               "Hỏi xã này có tên là gì?"])
    assert mod.round_time_texts(qf) == [expected]
    assert "Hỏi" not in mod.round_time_texts(qf)[0]


def test_warm_cache_kis_matches_runtime_join(tmp_path):
    mod = _load_warm_script()
    qf = tmp_path / "query-p2-4-kis.txt"
    qf.write_text("đoạn một. \n\nđoạn hai.\n", encoding="utf-8")
    assert mod.round_time_texts(qf) == ["đoạn một. đoạn hai."]  # single space


# ── C16: one dense lane failing must not kill text search ────────────────────
def test_dense_scores_degrades_to_surviving_lanes():
    from cvp.models.query_processor import ProcessedQuery
    from cvp.search.engine import SearchEngine

    class _OkModel:
        key, multilingual = "ok", True

        def encode_text(self, texts):
            return np.ones((len(texts), 4), dtype=np.float32)

    class _BoomModel(_OkModel):
        key = "dead"

        def encode_text(self, texts):
            raise RuntimeError("CUDA OOM")

    class _Store:
        def search(self, vecs, k):
            n = vecs.shape[0]
            return (np.tile(np.asarray([[0.9, 0.8]], dtype=np.float32), (n, 1)),
                    np.tile(np.asarray([[7, 8]]), (n, 1)))

    eng = object.__new__(SearchEngine)
    eng.settings = Settings()
    eng.settings.search.rerank = False
    eng.members = [(_BoomModel(), _Store()), (_OkModel(), _Store())]
    eng.member_weights = [0.55, 0.45]
    dense = SearchEngine._dense_scores(eng, ProcessedQuery(original="q"), topk=5)
    assert set(dense) == {7, 8} and dense[7] > dense[8]   # surviving lane answered

    eng.members = [(_BoomModel(), _Store())]
    eng.member_weights = [1.0]
    assert SearchEngine._dense_scores(eng, ProcessedQuery(original="q"), topk=5) == {}


# ── C17: weights/members length mismatch fails LOUD ──────────────────────────
def test_ensemble_weights_length_mismatch_raises():
    with pytest.raises(Exception, match="must match 1:1"):
        Settings(embedding={"ensemble_members": ["siglip2", "openclip", "jina"],
                            "ensemble_weights": [0.5, 0.5]})


# ── C18: nearest() degrades when embeddings folder was not synced ────────────
def test_nearest_returns_empty_on_missing_embeddings(tmp_path):
    from cvp.search.engine import SearchEngine

    eng = object.__new__(SearchEngine)
    eng.catalog = SimpleNamespace(ref=lambda gid: SimpleNamespace(
        global_id=gid, video_id="L01_V001"))
    eng.primary_store = SimpleNamespace(
        embedding_path=lambda vid: tmp_path / "missing" / f"{vid}.npy")
    assert SearchEngine.nearest(eng, 5) == []


# ── C19: DRES evaluation id reaches the submit URL ───────────────────────────
def test_dres_url_uses_evaluation_id():
    from cvp.submission.dres_client import DresClient

    c = DresClient("https://dres.example.org", evaluation_id="EVAL42")
    assert c._url(c.submit_path) == "https://dres.example.org/api/v2/submit/EVAL42"
    c2 = DresClient("https://dres.example.org")           # id not yet announced
    assert c2._url(c2.submit_path) == "https://dres.example.org/api/v2/submit"
    assert Settings().submission.dres_evaluation_id == ""


# ── C20: writer-skipped top row drops the recorded DRES timestamps ───────────
def test_top1_times_dropped_when_writer_skips_top_row(tmp_path):
    from cvp.pipeline.run_queries import run_query_file

    class _Eng:
        settings = Settings()

        def search_text(self, q, **kw):
            bad = SimpleNamespace(video_id="NOT_A_VIDEO_ID", frame_idx=100,
                                  global_id=0, signals={},
                                  ref=SimpleNamespace(pts_time=4.0, n=1,
                                                      global_id=0, path="x"),
                                  score=0.9)
            good = SimpleNamespace(video_id="L01_V001", frame_idx=200,
                                   global_id=1, signals={},
                                   ref=SimpleNamespace(pts_time=8.0, n=2,
                                                       global_id=1, path="y"),
                                   score=0.8)
            return [bad, good]

    qf = tmp_path / "query-1-kis.txt"
    qf.write_text("một cảnh nào đó\n", encoding="utf-8")
    times: dict[str, list[float]] = {}
    out = run_query_file(_Eng(), qf, tmp_path, None, top1_times=times)
    assert out is not None
    rows = out.read_text(encoding="utf-8").strip().splitlines()
    assert rows and rows[0].startswith("L01_V001")   # writer dropped the bad row
    # The 4.0s timestamp belonged to the DROPPED candidate — it must NOT ride
    # into DRES paired with L01_V001's frame.
    assert qf.stem not in times


def test_top1_times_kept_when_top_row_survives(tmp_path):
    from cvp.pipeline.run_queries import run_query_file

    class _Eng:
        settings = Settings()

        def search_text(self, q, **kw):
            return [SimpleNamespace(video_id="L01_V001", frame_idx=200,
                                    global_id=1, signals={},
                                    ref=SimpleNamespace(pts_time=8.0, n=2,
                                                        global_id=1, path="y"),
                                    score=0.8)]

    qf = tmp_path / "query-2-kis.txt"
    qf.write_text("một cảnh nào đó\n", encoding="utf-8")
    times: dict[str, list[float]] = {}
    run_query_file(_Eng(), qf, tmp_path, None, top1_times=times)
    assert times[qf.stem] == [8.0]


# ── C21: DRES verdict surfaced (ACCEPTED ≠ CORRECT) ──────────────────────────
def test_dres_result_carries_wrong_verdict():
    from cvp.submission.dres_client import DresClient

    res = DresClient._result(200, {"status": True, "submission": "wrong"}, "{}")
    assert res.ok is True and res.verdict == "WRONG"
    res2 = DresClient._result(200, {"status": True}, "{}")
    assert res2.verdict == ""


# ── C2/C3/C4: app wiring greps (same style as the round-3 app guards) ────────
APP_SRC = (REPO / "app" / "streamlit_app.py").read_text(encoding="utf-8")


def test_app_image_search_expires_marks_and_last_query():
    idx = APP_SRC.find("engine.search_image(img")
    assert idx != -1
    tail = APP_SRC[idx: idx + 600]
    assert 'last_query = ""' in tail and "set(), set()" in tail


def test_app_episodic_log_is_per_session_and_rotates():
    assert "st.session_state.episodic_log" in APP_SRC          # not cache_resource
    assert 'st.session_state.pop("episodic_log", None)' in APP_SRC  # reset rotates


def test_app_assistant_receives_episodic_summary():
    assert "episodic_summary=get_episodic_log().recent_summary()" in APP_SRC


# ── C8: qwen_reranker backend dependency is declared ─────────────────────────
def test_rerank_extra_and_colab_requirement_declared():
    py = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    assert "sentence-transformers" in py and "rerank =" in py
    req = (REPO / "requirements-colab.txt").read_text(encoding="utf-8")
    assert "sentence-transformers" in req and "matplotlib" in req   # C7 too
