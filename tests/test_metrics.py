import numpy as np

from cvf.eval.metrics import (
    KisGroundTruth,
    qualifier_score,
    retrieval_metrics,
    score_kis_submission,
    score_trake_submission,
)


def test_qualifier_score_rank1():
    assert qualifier_score([True] + [False] * 99) == 1.0


def test_qualifier_score_rank_bands():
    # first hit at rank 6 → misses k=1,5; hits k=20,50,100 → 3/5
    flags = [False] * 5 + [True] + [False] * 94
    assert abs(qualifier_score(flags) - 0.6) < 1e-9


def test_qualifier_score_no_hit():
    assert qualifier_score([False] * 100) == 0.0


def test_kis_segment_scoring():
    gt = KisGroundTruth("L21_V001", 100, 200)
    rows = [("L21_V002", 150), ("L21_V001", 150)]  # hit at rank 2
    assert abs(score_kis_submission(rows, gt) - 0.8) < 1e-9


def test_trake_all_events_must_hit():
    gt_segments = [(10, 20), (30, 40)]
    ok_row = ("K01_V001", [15, 35])
    bad_row = ("K01_V001", [15, 99])
    assert score_trake_submission([ok_row], "K01_V001", gt_segments) == 1.0
    assert score_trake_submission([bad_row], "K01_V001", gt_segments) == 0.0


def test_retrieval_metrics_perfect_alignment():
    vecs = np.eye(8, dtype=np.float32)
    m = retrieval_metrics(vecs, vecs, ks=(1, 5))
    assert m["R@1"] == 1.0 and m["MedR"] == 1.0
