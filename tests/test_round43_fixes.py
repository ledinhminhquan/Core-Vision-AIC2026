"""Round-43: reference-GT builder + run-ensemble merge, MEASURED offline.

With gt.json built from the shared 19.8/23 trial reference, offline
mean_final tracked the live public leaderboard almost perfectly
(0.537/0.589/0.650/0.641 for public 7.2/7.6/8.2/8.4) — so the trial pack is
now a real tuning bench. On that bench, RRF-merging the two best same-config
runs scored 0.6587, beating BOTH parents; merging a weak run with a strong
one (0.5978) lost to the strong parent — merge noise-siblings only.
"""

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_build_entry_kis_qa_trake():
    m = _load("62_build_gt_from_reference")
    kis = m.build_entry("query-p1-7-kis",
                        [["V1", "100"], ["V1", "160"], ["V2", "999"]],
                        kis_eps=50, trake_eps=10, top_frames=5)
    assert kis == {"task": "kis", "video_id": "V1",
                   "ranges": [[50, 150], [110, 210]]}   # V2 row excluded
    qa = m.build_entry("query-p1-3-qa",
                       [["V1", "100", "30 kg"], ["V2", "5", "sai"],
                        ["V1", "120", "30 kg"], ["V1", "130", "ba mươi kg"]],
                       kis_eps=10, trake_eps=10, top_frames=2)
    assert qa["answers"] == ["30 kg", "ba mươi kg"]     # top-video answers only
    tr = m.build_entry("query-p1-16-trake", [["V9", "50", "70", "90"]],
                       kis_eps=99, trake_eps=5, top_frames=5)
    assert tr == {"task": "trake", "video_id": "V9",
                  "moments": [[45, 55], [65, 75], [85, 95]]}


def test_rrf_merge_semantics():
    m = _load("63_ensemble_runs")
    a = [["V1", "10", "đáp A"], ["V2", "20", "x"]]
    b = [["V1", "10", "đáp B"], ["V3", "30", "y"]]
    out = m.rrf_merge(a, b, k=60, task="qa")
    assert out[0][:2] == ["V1", "10"]        # agreed row wins the top
    assert out[0][2] == "đáp A"              # answer from the better-ranked run
    assert {tuple(r[:2]) for r in out} == {("V1", "10"), ("V2", "20"), ("V3", "30")}
    # TRAKE identity is the whole sequence — different frames never dedupe
    t = m.rrf_merge([["V1", "1", "2"]], [["V1", "1", "3"]], k=60, task="trake")
    assert len(t) == 2


def test_r44_lab_notebook_and_tuned_weight_adoption():
    """Round-44: nb04 'Lab' gates every artifact upgrade behind the bench, and
    nb03's engine cell auto-loads Lab-tuned fusion weights from Drive."""
    src = (REPO / "notebooks" / "_build_notebooks.py").read_text(encoding="utf-8")
    assert 'write_nb("04_lab_artifacts.ipynb"' in src
    for knob in ("RUN_GT", "RUN_BENCH_FULL", "RUN_TUNE", "RUN_METACLIP",
                 "RUN_ASR_LARGE"):
        assert knob in src
    eng = src.split("NB3_ENGINE = r")[1].split("NB3_QUERIES")[0]
    assert "best_weights.json" in eng
    assert "CVP_SEARCH__WEIGHTS__" in eng
