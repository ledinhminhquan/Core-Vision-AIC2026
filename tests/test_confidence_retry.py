"""Low-confidence reformulation retry for the batch/auto path (2026 upgrade).

CPU-only stubs. Pins: the margin-based confidence metric, RRF merging with
primary-first object identity, the off-by-default gate, cached-expansion reuse
and every failure path returning the original ranking.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from cvp.config import Settings
from cvp.pipeline.run_queries import (
    maybe_retry_low_confidence,
    ranking_confidence,
    rrf_merge_results,
)


def _r(gid: int, score: float):
    ref = SimpleNamespace(global_id=gid, video_id="L01_V001", n=gid + 1,
                          frame_idx=100 * (gid + 1), pts_time=float(gid),
                          path=f"/fake/{gid:03d}.jpg")
    return SimpleNamespace(ref=ref, score=score, signals={},
                           video_id=ref.video_id, frame_idx=ref.frame_idx,
                           global_id=gid)


# ── confidence metric ────────────────────────────────────────────────────────
def test_confidence_high_when_top_towers_over_flat_tail():
    scores = [1.0] + [0.1] * 49
    assert ranking_confidence(scores) == pytest.approx(0.9)


def test_confidence_low_on_plateau():
    # Near-identical fused scores = the query failed to discriminate — the
    # magnitude-relative gap must flag it no matter where the median sits.
    assert ranking_confidence([0.500001 - i * 1e-9 for i in range(50)]) <= 0.01


def test_confidence_zero_for_short_or_degenerate_lists():
    assert ranking_confidence([1.0] * 5) == 0.0          # too short
    assert ranking_confidence([0.7] * 20) == 0.0         # constant


# ── RRF merge ────────────────────────────────────────────────────────────────
def test_rrf_merge_keeps_primary_objects_and_boosts_agreement():
    a = [_r(1, 0.9), _r(2, 0.8), _r(3, 0.7)]
    b = [_r(2, 0.95), _r(9, 0.5)]
    merged = rrf_merge_results([a, b], k=60)
    gids = [m.ref.global_id for m in merged]
    assert gids[0] == 2                        # appears in both → wins
    assert set(gids) == {1, 2, 3, 9}
    assert merged[0] is a[1]                   # primary list's object survives


def test_rrf_merge_respects_limit():
    a = [_r(i, 1.0 - i * 0.01) for i in range(10)]
    assert len(rrf_merge_results([a], limit=4)) == 4


# ── gated retry ──────────────────────────────────────────────────────────────
class _Engine:
    def __init__(self, primary, alternates=None, expansions=("cảnh quay khác",)):
        self.settings = Settings()
        self._primary = primary
        self._alt = alternates or {}
        self.searched: list[str] = []
        self.query_processor = SimpleNamespace(process=lambda text: SimpleNamespace(
            enhanced="mô tả tăng cường chi tiết", expansions=list(expansions)))

    def search_text(self, text, **kw):
        self.searched.append(text)
        return list(self._alt.get(text, self._primary))


def _flat(n=50):
    return [_r(i, 0.5 - i * 1e-9) for i in range(n)]


def test_retry_disabled_by_default():
    eng = _Engine(_flat())
    results = _flat()
    assert maybe_retry_low_confidence(eng, "câu truy vấn", results) is results
    assert eng.searched == []                  # no extra searches happened


def test_retry_skipped_when_confident():
    eng = _Engine(_flat())
    eng.settings.search.low_confidence_retry = True
    confident = [_r(0, 1.0)] + [_r(i, 0.1) for i in range(1, 50)]
    assert maybe_retry_low_confidence(eng, "q", confident) is confident


def test_retry_merges_reformulations_on_flat_ranking():
    alt = {"mô tả tăng cường chi tiết": [_r(77, 0.9)],
           "cảnh quay khác": [_r(77, 0.8), _r(88, 0.7)]}
    eng = _Engine(_flat(), alternates=alt)
    eng.settings.search.low_confidence_retry = True
    out = maybe_retry_low_confidence(eng, "câu truy vấn gốc", _flat())
    gids = [r.ref.global_id for r in out]
    assert 77 in gids                          # reformulation hit merged in
    assert len(out) == 50                      # capped at the original length
    assert len(eng.searched) == 2              # both alternates searched


def test_retry_survives_processor_failure():
    eng = _Engine(_flat())
    eng.settings.search.low_confidence_retry = True
    eng.query_processor = SimpleNamespace(
        process=lambda text: (_ for _ in ()).throw(RuntimeError("no network")))
    results = _flat()
    assert maybe_retry_low_confidence(eng, "q", results) is results


def test_retry_noop_when_expansions_equal_query():
    eng = _Engine(_flat(), expansions=())
    eng.settings.search.low_confidence_retry = True
    eng.query_processor = SimpleNamespace(process=lambda text: SimpleNamespace(
        enhanced="  q  ", expansions=[]))
    results = _flat()
    assert maybe_retry_low_confidence(eng, "q", results) is results


def test_new_retry_settings_defaults():
    s = Settings()
    assert s.search.low_confidence_retry is False
    assert s.search.low_confidence_threshold == pytest.approx(0.25)


# ── review findings C2/C19/C20 regressions ───────────────────────────────────
def test_confidence_is_order_invariant():
    # Rerankers reorder rows WITHOUT updating .score — the metric must sort.
    decisive = [1.0] + [0.1] * 49
    shuffled = [0.1] * 25 + [1.0] + [0.1] * 24
    assert ranking_confidence(shuffled) == ranking_confidence(decisive)


def test_retry_prefers_search_prepared_over_search_text():
    alt = {"mô tả tăng cường chi tiết": [_r(77, 0.9)], "cảnh quay khác": [_r(88, 0.7)]}
    eng = _Engine(_flat(), alternates=alt)
    eng.settings.search.low_confidence_retry = True
    prepared_calls = []

    def search_prepared(text, **kw):
        prepared_calls.append(text)
        return list(alt.get(text, []))

    eng.search_prepared = search_prepared
    out = maybe_retry_low_confidence(eng, "câu truy vấn gốc", _flat())
    # The alts went through the processor BYPASS, not the Gemini-bound path.
    assert len(prepared_calls) == 2 and eng.searched == []
    assert 77 in [r.ref.global_id for r in out]
