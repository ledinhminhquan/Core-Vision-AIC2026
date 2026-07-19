"""Round-3 adversarial review regressions (28 confirmed findings, all fixed).

CPU-only. Each test names its finding (R3-C*); the HIGH one (C5: nprobe keyed
off config instead of the on-disk index) gets an empirical recall check.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from cvp.config import Settings
from cvp.eval.official import (
    _entry_targets,
    _row_hits_target,
    coverage_at_k,
    score_rows,
    score_run,
)

REPO = Path(__file__).resolve().parents[1]


# ── R3-C5/C6/C8 (HIGH+MED): knobs follow the ON-DISK index, structure change rebuilds
def test_ivf_index_keeps_recall_under_wrong_config_type(corpus_with_index):
    from cvp.data.catalog import KeyframeCatalog
    from cvp.index.store import IndexStore

    settings = corpus_with_index
    catalog = KeyframeCatalog(settings)
    rng = np.random.default_rng(11)
    queries = rng.normal(size=(5, 16)).astype(np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)

    def _top(store, k=10):
        _s, gids = store.search(queries, k)
        return [set(int(g) for g in row if g >= 0) for row in gids]

    exact = _top(IndexStore(settings, "fake"))

    settings.index.type = "ivf"
    settings.index.ivf_nlist = 4
    settings.index.ivf_nprobe = 4
    ivf = IndexStore(settings, "fake")
    ivf.build(catalog, force=True)

    # THE C5 scenario: the on-disk index is IVF but the config says flatip —
    # nprobe must still be applied to what the index ACTUALLY IS (recall used
    # to collapse to FAISS's default nprobe=1 silently).
    settings.index.type = "flatip"
    misconfigured = IndexStore(settings, "fake")
    recall = np.mean([len(a & b) / len(a) for a, b in zip(exact, _top(misconfigured))])
    assert recall >= 0.9, f"IVF-on-disk under flatip config: recall {recall:.2f}"

    # C6: flipping the config type must REBUILD, not report 'up-to-date'.
    settings.index.type = "flatip"
    rebuilt = IndexStore(settings, "fake")
    rebuilt.build(catalog, force=False)          # structure changed → rebuild
    assert rebuilt.meta().get("index_type") == "flatip"


def test_hnsw_efsearch_reapplied_on_load(corpus_with_index):
    from cvp.data.catalog import KeyframeCatalog
    from cvp.index.store import IndexStore

    settings = corpus_with_index
    catalog = KeyframeCatalog(settings)
    settings.index.type = "hnsw"
    store = IndexStore(settings, "fake")
    store.build(catalog, force=True)
    settings.index.hnsw_ef_search = 77           # retune WITHOUT rebuild (C8)
    fresh = IndexStore(settings, "fake")
    index = fresh.load()
    assert index.hnsw.efSearch == 77
    settings.index.type = "flatip"
    IndexStore(settings, "fake").build(catalog, force=False)


def test_ivf_nlist_larger_than_corpus_fails_loud(corpus_with_index):
    from cvp.data.catalog import KeyframeCatalog
    from cvp.index.store import IndexStore

    settings = corpus_with_index
    settings.index.type = "ivf"
    settings.index.ivf_nlist = 10_000            # corpus has only 15 vectors
    store = IndexStore(settings, "fake")
    with pytest.raises(RuntimeError, match="ivf_nlist"):
        store.build(KeyframeCatalog(settings), force=True)
    settings.index.type = "flatip"
    settings.index.ivf_nlist = 4096
    IndexStore(settings, "fake").build(KeyframeCatalog(settings), force=True)


# ── R3-C1/C17/C27: by_task carries the [avs] bucket ─────────────────────────
def test_score_run_groups_targets_entries_under_avs(tmp_path):
    sub = tmp_path / "subs"
    sub.mkdir()
    (sub / "query-1-kis.csv").write_text("L01_V001, 505\n", encoding="utf-8")
    (sub / "query-2-avs.csv").write_text("L01_V001, 105\n", encoding="utf-8")
    gt = tmp_path / "gt.json"
    gt.write_text(json.dumps({
        "query-1-kis": {"task": "kis", "video_id": "L01_V001", "range": [500, 510]},
        "query-2-avs": {"task": "avs", "targets": [
            {"video_id": "L01_V001", "range": [100, 110]},
            {"video_id": "L02_V002", "range": [1, 5]},
        ]},
    }), encoding="utf-8")
    report = score_run(sub, gt)
    assert set(report.by_task) == {"kis", "avs"}
    assert report.by_task["kis"] == pytest.approx(1.0)      # NOT contaminated
    assert report.by_task["avs"] == pytest.approx(0.5)      # 1 of 2 targets
    assert report.per_query["query-2-avs"].task == "avs"


# ── R3-C2: targets beats the QA-answer gate ──────────────────────────────────
def test_targets_entry_under_qa_stem_scores_as_coverage(tmp_path):
    sub = tmp_path / "subs"
    sub.mkdir()
    (sub / "query-3-qa.csv").write_text("L02_V002, 150\n", encoding="utf-8")
    gt = tmp_path / "gt.json"
    gt.write_text(json.dumps({
        "query-3-qa": {"targets": [{"video_id": "L02_V002", "range": [100, 200]}]},
    }), encoding="utf-8")
    report = score_run(sub, gt)
    assert "query-3-qa" not in report.unscored              # no 'gt missing answer'
    assert report.per_query["query-3-qa"].final == pytest.approx(1.0)
    qs = score_rows("qa", [["L02_V002", "150"]],
                    {"targets": [{"video_id": "L02_V002", "range": [100, 200]}]})
    assert qs.task == "avs" and qs.final == pytest.approx(1.0)


# ── R3-C3: raw docstring spellings accepted by the public API ────────────────
def test_row_hits_target_accepts_raw_spellings():
    assert _row_hits_target(["V", "150"], {"video_id": "V", "range": [100, 200]})
    assert _row_hits_target(["V", "95"], {"video_id": "V", "center": 100, "epsilon": 10})
    assert coverage_at_k([["V", "150"]],
                         [{"video_id": "V", "range": [100, 200]}], 1) == 1.0


# ── R3-C4: partially-malformed target windows raise ──────────────────────────
def test_entry_targets_raises_on_dropped_window():
    with pytest.raises(ValueError, match="unparseable"):
        _entry_targets({"targets": [
            {"video_id": "A", "ranges": [[100, 110], ["x", "y"]]},
        ]})


# ── R3-C9/C12: diff tool full-row semantics + loud missing folder ────────────
def _load_diff():
    import importlib.util
    import sys

    scripts = REPO / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    spec = importlib.util.spec_from_file_location(
        "diff_submissions_script2", scripts / "41_diff_submissions.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_diff_detects_qa_answer_and_trake_tail_changes(tmp_path):
    mod = _load_diff()
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(), b.mkdir()
    (a / "q-qa.csv").write_text('L01_V001, 100, "màu đỏ"\n', encoding="utf-8")
    (b / "q-qa.csv").write_text('L01_V001, 100, "màu xanh"\n', encoding="utf-8")
    (a / "q-trake.csv").write_text("L01_V001, 100, 200, 300\n", encoding="utf-8")
    (b / "q-trake.csv").write_text("L01_V001, 100, 200, 999\n", encoding="utf-8")
    recs = {r["stem"]: r for r in mod.diff_stems(a, b)}
    assert recs["q-qa"]["status"] == "TOP1-CHANGED"        # answer edit visible
    assert recs["q-trake"]["status"] == "TOP1-CHANGED"     # tail frame visible


def test_diff_raises_on_missing_folder(tmp_path):
    mod = _load_diff()
    (tmp_path / "real").mkdir()
    with pytest.raises(SystemExit, match="does not exist"):
        mod.diff_stems(tmp_path / "typo", tmp_path / "real")


# ── R3-C26: auto-agent surfaces dropped queries ──────────────────────────────
def test_auto_report_tracks_failed_queries(tmp_path):
    from cvp.pipeline.auto_agent import run_auto

    qdir = tmp_path / "q"
    qdir.mkdir()
    (qdir / "query-1-kis.txt").write_text("một cảnh\n", encoding="utf-8")
    (qdir / "query-2-kis.txt").write_text("", encoding="utf-8")    # empty → dropped

    class _Eng:
        settings = Settings()

        def search_text(self, q, **kw):
            from types import SimpleNamespace

            ref = SimpleNamespace(global_id=0, video_id="L01_V001", n=1,
                                  frame_idx=100, pts_time=4.0, path="x")
            return [SimpleNamespace(ref=ref, score=0.9, signals={},
                                    video_id="L01_V001", frame_idx=100, global_id=0)]

    report = run_auto(qdir, tmp_path / "out", Settings(), submit=False,
                      engine_factory=lambda s: _Eng(), vqa=None)
    assert len(report.written) == 1
    assert "query-2-kis" in report.failed
    assert report.ok and not report.complete               # short pack ≠ complete


# ── R3-C14/C25/C28: unzip routing + wrapper heuristics (builder source) ──────
def test_guess_dest_specific_families_win_over_generic():
    import re

    src = (REPO / "notebooks" / "_build_notebooks.py").read_text(encoding="utf-8")
    m = re.search(r"def guess_dest\(zname: str\):.*?return None", src, re.DOTALL)
    ns = {"ZIP_DEST": {k: k for k in
                       ("keyframes", "videos", "clip-features", "map-keyframes",
                        "media-info", "objects")}}
    exec(m.group(0), ns)  # noqa: S102 — builder cell source, trusted
    guess = ns["guess_dest"]
    assert guess("keyframe-map-aic26-b1.zip") == "map-keyframes"   # C25 core case
    assert guess("Map_Keyframes_b2.zip") == "map-keyframes"
    assert guess("Keyframes_L21.zip") == "keyframes"
    assert guess("Videos_K08.zip") == "videos"


def test_unzip_wrapper_walk_markers_present():
    src = (REPO / "notebooks" / "_build_notebooks.py").read_text(encoding="utf-8")
    assert "_VID_DIR_RE" in src                    # payload dirs never stripped (C14)
    assert "giữ nguyên" in src                     # merge-never-clobber path (C28)
    assert "merged" in src and "new item(s) from Drive into local" in src  # C15
