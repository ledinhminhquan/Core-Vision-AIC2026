"""Round-5 review regressions — executing the code against the REAL Batch-1
bytes plus an adversarial audit of the round-4 fixes themselves.

  R5-1  extract_video must never overwrite an ORGANISER map csv (the round-4
        guard only protected the jpgs); refusals return 0, not a fake count
  R5-2  write_qa dedups on the FULL row — distinct answers at one locus survive
  R5-3  legacy score_submission_dir honors the coverage(targets) answer exemption
  R5-4  partially-broken GT windows fail LOUD (extends R3-C4 to ranges/events)
  R5-5  English-only ensemble lanes are skipped when no translation exists
  R5-6  stale/absent persisted text_index warns instead of degrading silently
"""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace

import numpy as np
import pytest

from cvp.config import Settings


# ── R5-1 · organiser map csv must survive self-extraction ────────────────────
ORGANISER_CSV = "n,pts_time,fps,frame_idx\n1,0.0,30.0,0\n2,3.0,30.0,90\n"


def test_extract_video_refuses_to_overwrite_organiser_map_csv(tmp_path):
    from cvp.data.extraction import extract_video

    kf, mp = tmp_path / "keyframes", tmp_path / "map-keyframes"
    mp.mkdir(parents=True)
    (mp / "L21_V001.csv").write_text(ORGANISER_CSV, encoding="utf-8")
    # Official map csv present, keyframes dir ABSENT (Keyframes zip still
    # uploading) — extraction must refuse instead of clobbering the csv.
    n = extract_video(tmp_path / "L21_V001.mp4", kf, mp, overwrite=False)
    assert n == 0
    assert (mp / "L21_V001.csv").read_text(encoding="utf-8") == ORGANISER_CSV


def test_extract_missing_does_not_count_refused_videos(tmp_path, monkeypatch):
    from cvp.data import extraction

    s = Settings()
    s.paths.data_root = tmp_path
    (tmp_path / s.paths.videos_dir).mkdir(parents=True)
    (tmp_path / s.paths.videos_dir / "L21_V001.mp4").write_bytes(b"fake")
    vdir = tmp_path / s.paths.keyframes_dir / "L21_V001"
    vdir.mkdir(parents=True)
    (vdir / "001.jpg").write_bytes(b"organiser")
    # Organiser jpgs, no map csv → refusal path returns 0 → count stays 0.
    assert extraction.extract_missing(s) == 0
    assert (vdir / "001.jpg").read_bytes() == b"organiser"


# ── R5-2 · QA dedup keeps distinct answers at the same locus ─────────────────
def test_write_qa_keeps_distinct_answers_same_locus(tmp_path):
    from cvp.submission.writer import write_qa

    p = write_qa(tmp_path / "q.csv", [
        ("L05_V005", 888, "màu xanh"),
        ("L05_V005", 888, "màu trắng"),   # different answer → MUST survive
        ("L05_V005", 888, "màu xanh"),    # true duplicate → collapses
        ("L05_V005", 888, "Màu Xanh"),    # casefold duplicate → collapses
    ])
    rows = p.read_text(encoding="utf-8").strip().splitlines()
    assert len(rows) == 2
    assert rows[0].endswith("màu xanh") and rows[1].endswith("màu trắng")


# ── R5-3 · legacy dir scorer: coverage entries need no answer ────────────────
def test_score_submission_dir_scores_qa_targets_entry(tmp_path):
    from cvp.eval.official import score_submission_dir

    sub = tmp_path / "subs"
    sub.mkdir()
    (sub / "query-p1-7-qa.csv").write_text("L05_V005,850\n", encoding="utf-8")
    gt_p = tmp_path / "gt.json"
    gt_p.write_text(json.dumps({"query-p1-7-qa": {
        "task": "qa",
        "targets": [{"video_id": "L05_V005", "range": [800, 900]}],
    }}), encoding="utf-8")
    rep = score_submission_dir(sub, gt_p)
    q = rep["per_query"]["query-p1-7-qa"]
    assert q["status"] == "scored", q          # was: unscored "gt missing answer"
    assert q["final"] == 1.0


# ── R5-4 · partially-broken GT fails loud ────────────────────────────────────
def _load(tmp_path, entry):
    from cvp.eval.official import load_ground_truth

    p = tmp_path / "gt.json"
    p.write_text(json.dumps({"query-p1-1-kis": entry}), encoding="utf-8")
    return load_ground_truth(p)


def test_gt_partial_ranges_typo_raises(tmp_path):
    with pytest.raises(ValueError, match="unparseable"):
        _load(tmp_path, {"video_id": "L01_V001",
                         "ranges": [[100, 200], "oops"]})


def test_gt_inverted_range_raises(tmp_path):
    with pytest.raises(ValueError, match="inverted"):
        _load(tmp_path, {"video_id": "L01_V001", "ranges": [[200, 100]]})


def test_gt_partial_trake_event_raises(tmp_path):
    from cvp.eval.official import load_ground_truth

    p = tmp_path / "gt.json"
    p.write_text(json.dumps({"query-p1-2-trake": {
        "task": "trake", "video_id": "L10_V010",
        "events": [[95, 105], {"bad": "spelling"}],
    }}), encoding="utf-8")
    with pytest.raises(ValueError, match="unparseable"):
        load_ground_truth(p)


def test_gt_valid_multi_range_still_loads(tmp_path):
    gt = _load(tmp_path, {"video_id": "L01_V001",
                          "ranges": [[100, 200], [300, 400]]})
    assert "query-p1-1-kis" in gt


# ── R5-5 · English-only lane skipped without translation ─────────────────────
def _fake_engine(models):
    class _Store:
        def search(self, vecs, k):
            n = len(vecs)
            return (np.ones((n, 2), dtype=np.float32),
                    np.tile(np.array([0, 1]), (n, 1)))

    s = Settings()
    s.search.rerank = False
    return SimpleNamespace(members=[(m, _Store()) for m in models],
                           member_weights=[1.0 / len(models)] * len(models),
                           settings=s)


class _Model:
    def __init__(self, key, multilingual, calls):
        self.key, self.multilingual, self._calls = key, multilingual, calls

    def encode_text(self, texts):
        self._calls.append((self.key, list(texts)))
        return np.ones((len(texts), 4), dtype=np.float32)


def test_dense_scores_skips_english_only_lane_without_translation():
    from cvp.models.query_processor import ProcessedQuery
    from cvp.search.engine import SearchEngine

    calls: list = []
    fake = _fake_engine([_Model("openclip", False, calls),
                         _Model("siglip2", True, calls)])
    out = SearchEngine._dense_scores(fake, ProcessedQuery(original="áo đỏ"), 2)
    assert out                                          # multilingual lane served
    assert {k for k, _ in calls} == {"siglip2"}         # EN-only lane never ran
    calls.clear()
    SearchEngine._dense_scores(
        fake, ProcessedQuery(original="áo đỏ", translation="a red shirt"), 2)
    assert {k for k, _ in calls} == {"openclip", "siglip2"}


def test_dense_scores_single_english_lane_still_serves():
    # Deliberate single-lane EN config without translation: noisy beats empty.
    from cvp.models.query_processor import ProcessedQuery
    from cvp.search.engine import SearchEngine

    calls: list = []
    fake = _fake_engine([_Model("openclip", False, calls)])
    out = SearchEngine._dense_scores(fake, ProcessedQuery(original="áo đỏ"), 2)
    assert out and calls


def test_has_english_flag():
    from cvp.models.query_processor import ProcessedQuery

    assert not ProcessedQuery(original="áo đỏ").has_english()
    assert ProcessedQuery(original="x", translation="red").has_english()
    assert ProcessedQuery(original="x", enhanced="a red shirt").has_english()
    assert ProcessedQuery(original="x", expansions=["red top"]).has_english()


# ── R5-6 · stale text_index warns loudly ─────────────────────────────────────
def test_text_signals_warns_on_stale_index(tmp_path, caplog):
    from cvp.search.text_signals import TextSignals

    s = Settings()
    s.paths.artifacts_root = tmp_path      # no text_index here → mismatch path
    ts = TextSignals(s, SimpleNamespace(signature=lambda: "sig-abc"))
    with caplog.at_level(logging.WARNING, logger="cvp.search.text_signals"):
        assert ts._persisted_enabled() is False
    assert any("STALE" in r.message or "ABSENT" in r.message for r in caplog.records)


def test_doctor_reports_text_index_freshness_key():
    import inspect

    from cvp.pipeline import ingest

    src = inspect.getsource(ingest.doctor)
    assert "text_index_fresh" in src
