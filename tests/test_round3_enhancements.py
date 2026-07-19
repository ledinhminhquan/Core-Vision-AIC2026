"""Round-3 enhancements: AVS coverage scorer, ANN recall gate, run-diff, 2026-proof unzip.

CPU-only; the ANN test builds tiny IVF/HNSW indexes over the synthetic corpus.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import numpy as np
import pytest

from cvp.eval.official import (
    coverage_at_k,
    load_ground_truth,
    score_rows,
)

REPO = Path(__file__).resolve().parents[1]


# ── AVS coverage scoring (GT `targets` form) ─────────────────────────────────
def _targets():
    return [
        {"video_id": "L01_V001", "ranges": [[100, 110]]},
        {"video_id": "L01_V001", "ranges": [[500, 510]]},
        {"video_id": "L07_V003", "ranges": [[40, 60]]},
    ]


def test_coverage_at_k_counts_distinct_targets():
    rows = [("L01_V001", "105"),      # hits target 1
            ("L01_V001", "105"),      # duplicate — must NOT double-count
            ("L09_V009", "999"),      # miss
            ("L07_V003", "50")]       # hits target 3
    assert coverage_at_k(rows, _targets(), 1) == pytest.approx(1 / 3)
    assert coverage_at_k(rows, _targets(), 4) == pytest.approx(2 / 3)
    assert coverage_at_k([], _targets(), 100) == 0.0


def test_score_rows_avs_targets_path():
    rows = [["L01_V001", "105"], ["L07_V003", "50"], ["L01_V001", "505"]]
    qs = score_rows("avs", rows, {"targets": _targets()})
    assert qs.task == "avs"
    assert qs.r_at[1] == pytest.approx(1 / 3)
    assert qs.r_at[5] == pytest.approx(1.0)       # all 3 targets inside top-5
    assert qs.best_rank == 1
    # final = mean of coverage@k, k∈{1,5,20,50,100}
    assert qs.final == pytest.approx((1 / 3 + 1.0 + 1.0 + 1.0 + 1.0) / 5)


def test_score_rows_without_targets_unchanged_kis_path():
    qs = score_rows("avs", [["L01_V001", "505"]],
                    {"video_id": "L01_V001", "range": [500, 510]})
    assert qs.task == "kis" and qs.final == 1.0    # legacy proxy path intact


def test_load_ground_truth_accepts_targets_and_rejects_bad_ones(tmp_path):
    import json

    good = tmp_path / "gt.json"
    good.write_text(json.dumps({
        "query-1-avs": {"task": "avs", "targets": [
            {"video_id": "L01_V001", "range": [1, 5]},
            {"video_id": "L02_V002", "center": 100, "epsilon": 10},
        ]}}), encoding="utf-8")
    gt = load_ground_truth(good)
    assert len(gt["query-1-avs"]["targets"]) == 2
    assert gt["query-1-avs"]["targets"][1]["ranges"] == [[90, 110]]

    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({
        "q": {"task": "avs", "targets": [{"video_id": "L01_V001"}]}  # no window
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="no usable frame window"):
        load_ground_truth(bad)

    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps({"q": {"task": "avs", "targets": []}}),
                     encoding="utf-8")
    with pytest.raises(ValueError, match="non-empty"):
        load_ground_truth(empty)


# ── ANN scale-out gate: IVF/HNSW recall vs exact flatip ──────────────────────
def test_ivf_and_hnsw_recall_against_flatip(corpus_with_index):
    from cvp.data.catalog import KeyframeCatalog
    from cvp.index.store import IndexStore

    settings = corpus_with_index
    catalog = KeyframeCatalog(settings)
    rng = np.random.default_rng(7)
    queries = rng.normal(size=(5, 16)).astype(np.float32)  # conftest DIM = 16
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)

    def _top(store, k=10):
        _s, gids = store.search(queries, k)
        return [set(int(g) for g in row if g >= 0) for row in gids]

    exact = _top(IndexStore(settings, "fake"))

    # Tiny corpus → tiny IVF cells; probe EVERY cell so recall is deterministic.
    settings.index.type = "ivf"
    settings.index.ivf_nlist = 4
    settings.index.ivf_nprobe = 4
    ivf = IndexStore(settings, "fake")
    ivf.build(catalog, force=True)
    ivf_recall = np.mean([len(a & b) / len(a) for a, b in zip(exact, _top(ivf))])
    assert ivf_recall >= 0.9, f"IVF recall vs exact dropped to {ivf_recall:.2f}"

    settings.index.type = "hnsw"
    hnsw = IndexStore(settings, "fake")
    hnsw.build(catalog, force=True)
    hnsw_recall = np.mean([len(a & b) / len(a) for a, b in zip(exact, _top(hnsw))])
    assert hnsw_recall >= 0.9, f"HNSW recall vs exact dropped to {hnsw_recall:.2f}"

    # Restore the exact index for any later fixture consumer.
    settings.index.type = "flatip"
    IndexStore(settings, "fake").build(catalog, force=True)


# ── scripts/41: submission-run diff ──────────────────────────────────────────
def _load_diff_script():
    scripts = REPO / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    spec = importlib.util.spec_from_file_location(
        "diff_submissions_script", scripts / "41_diff_submissions.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_diff_stems_reports_top1_overlap_and_missing(tmp_path):
    mod = _load_diff_script()
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(), b.mkdir()
    (a / "q1.csv").write_text("L01_V001, 100\nL01_V001, 200\n", encoding="utf-8")
    (b / "q1.csv").write_text("L01_V001, 100\nL02_V002, 999\n", encoding="utf-8")
    (a / "q2.csv").write_text("L01_V001, 100\n", encoding="utf-8")
    (b / "q2.csv").write_text("L03_V003, 5\n", encoding="utf-8")
    (a / "only_a.csv").write_text("L01_V001, 1\n", encoding="utf-8")

    recs = {r["stem"]: r for r in mod.diff_stems(a, b, k=20)}
    assert recs["q1"]["status"] == "same-top1"
    assert recs["q1"]["overlap_k"] == pytest.approx(0.5)
    assert recs["q2"]["status"] == "TOP1-CHANGED"
    assert recs["only_a"]["status"] == "only-in-A"


# ── 2026-proof zip routing in notebook 01 ────────────────────────────────────
def _extract_guess_dest():
    src = (REPO / "notebooks" / "_build_notebooks.py").read_text(encoding="utf-8")
    m = re.search(r"def guess_dest\(zname: str\):.*?\n\n_unknown_zips", src, re.DOTALL)
    assert m, "guess_dest block not found in NB1_UNZIP"
    ns = {"ZIP_DEST": {k: k for k in
                       ("keyframes", "videos", "clip-features", "map-keyframes",
                        "media-info", "objects")}}
    exec(m.group(0).rsplit("\n\n", 1)[0], ns)  # noqa: S102 — builder cell source, trusted
    return ns["guess_dest"]


@pytest.mark.parametrize("zname,dest", [
    ("Keyframes_L21.zip", "keyframes"),
    ("Keyframes_M05_aic26.zip", "keyframes"),          # 2026 letters
    ("keyframe_b3.zip", "keyframes"),                  # singular + underscore
    ("Videos_K08.zip", "videos"),
    ("video_batch_2.zip", "videos"),
    ("clip_features-32-aic26-b1.zip", "clip-features"),
    ("Map Keyframes aic26.zip", "map-keyframes"),
    ("media_info-aic26.zip", "media-info"),
    ("metadata-aic26-b1.zip", "media-info"),
    ("objects-aic26-b1.zip", "objects"),
    ("mystery-package.zip", None),                     # must be flagged, not lost
])
def test_unzip_routing_handles_2026_spellings(zname, dest):
    guess = _extract_guess_dest()
    assert guess(zname) == dest


def test_unzip_cell_warns_loudly_on_unknown_zips():
    src = (REPO / "notebooks" / "_build_notebooks.py").read_text(encoding="utf-8")
    assert "_unknown_zips" in src and "KHÔNG NHẬN DIỆN" in src
    assert "DATASET_INGESTION.md" in src               # points the operator at the mapping
