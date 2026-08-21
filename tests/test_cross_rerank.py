"""Pairwise cross-encoder rerank stage (Unified-IMMR recipe, port + upgrade).

All tests are pure CPU with stub scorers — no transformers, no downloads. The
contract pinned here: head-only reordering, tail untouched, blended min-max
scores, ``signals["cross"]`` attribution, and EVERY failure path returning the
input ranking unchanged (a reranker must never sink a live query).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import cvp.search.cross_rerank as cr
from cvp.config import Settings


def _results(n: int, start_score: float = 1.0) -> list[SimpleNamespace]:
    out = []
    for i in range(n):
        out.append(SimpleNamespace(
            ref=SimpleNamespace(path=f"/fake/{i:03d}.jpg", global_id=i),
            score=start_score - i * 0.01,
            signals={},
        ))
    return out


class _StubReranker:
    def __init__(self, scores):
        self._scores = scores
        self.calls: list[tuple[str, list[str]]] = []

    def score(self, query, paths):
        self.calls.append((query, list(paths)))
        s = self._scores
        return np.asarray(s(paths) if callable(s) else s, dtype=np.float32)


@pytest.fixture(autouse=True)
def _reset_singleton():
    cr._RERANKER, cr._RERANKER_KEY = None, None
    yield
    cr._RERANKER, cr._RERANKER_KEY = None, None


def _settings(**search_kw) -> Settings:
    s = Settings()
    for k, v in search_kw.items():
        setattr(s.search, k, v)
    return s


def test_none_backend_is_passthrough():
    results = _results(5)
    assert cr.cross_rerank(results, "q", _settings(reranker="none")) is results


def test_unknown_backend_is_passthrough():
    results = _results(5)
    s = _settings()
    s.search.reranker = "bogus_backend"
    assert cr.cross_rerank(results, "q", s) is results


def test_reorders_head_only_and_sets_cross_signal(monkeypatch):
    results = _results(10)
    # Cross scores strictly reversed: last head row is the best match.
    stub = _StubReranker(lambda paths: list(range(len(paths))))
    monkeypatch.setattr(cr, "_get_reranker", lambda settings: stub)
    out = cr.cross_rerank(results, "một người mặc áo đỏ",
                          _settings(reranker="blip2_itm", rerank_topk=4,
                                    rerank_weight=1.0))
    # Head (4) fully reversed by cross score; tail (6) untouched in order.
    assert [r.ref.global_id for r in out[:4]] == [3, 2, 1, 0]
    assert [r.ref.global_id for r in out[4:]] == [4, 5, 6, 7, 8, 9]
    assert all("cross" in r.signals for r in out[:4])
    assert all("cross" not in r.signals for r in out[4:])
    # The reranker saw exactly the head's paths, in original rank order.
    assert stub.calls[0][1] == [f"/fake/{i:03d}.jpg" for i in range(4)]


def test_blend_weight_zero_keeps_fused_order(monkeypatch):
    results = _results(6)
    stub = _StubReranker(lambda paths: list(range(len(paths))))  # reversed pref
    monkeypatch.setattr(cr, "_get_reranker", lambda settings: stub)
    out = cr.cross_rerank(results, "q",
                          _settings(reranker="blip2_itm", rerank_topk=6,
                                    rerank_weight=0.0))
    assert [r.ref.global_id for r in out] == list(range(6))  # fused score wins


def test_scorer_exception_is_passthrough(monkeypatch):
    results = _results(5)

    class _Boom:
        def score(self, query, paths):
            raise RuntimeError("cuda OOM")

    monkeypatch.setattr(cr, "_get_reranker", lambda settings: _Boom())
    out = cr.cross_rerank(results, "q", _settings(reranker="blip2_itm"))
    assert out is results and all("cross" not in r.signals for r in results)


def test_wrong_score_shape_is_passthrough(monkeypatch):
    results = _results(5)
    stub = _StubReranker([0.1, 0.2])  # too few scores
    monkeypatch.setattr(cr, "_get_reranker", lambda settings: stub)
    assert cr.cross_rerank(results, "q", _settings(reranker="blip2_itm")) is results


def test_single_candidate_is_passthrough(monkeypatch):
    results = _results(1)
    monkeypatch.setattr(cr, "_get_reranker",
                        lambda settings: pytest.fail("must not build for <2 rows"))
    assert cr.cross_rerank(results, "q", _settings(reranker="blip2_itm")) is results


def test_failed_build_disables_for_session(monkeypatch):
    built = []

    class _FailingBackend:
        def __init__(self, settings):
            built.append(1)
            raise RuntimeError("no such model")

    monkeypatch.setitem(cr._BACKENDS, "blip2_itm", _FailingBackend)
    s = _settings(reranker="blip2_itm")
    results = _results(5)
    assert cr.cross_rerank(results, "q", s) is results
    assert cr.cross_rerank(results, "q", s) is results
    assert len(built) == 1                     # built once, then cached as disabled


def test_backend_switch_rebuilds(monkeypatch):
    class _A:
        def __init__(self, settings): ...
        def score(self, q, p): return np.zeros(len(p), dtype=np.float32)

    class _B(_A): ...

    monkeypatch.setitem(cr._BACKENDS, "blip2_itm", _A)
    monkeypatch.setitem(cr._BACKENDS, "qwen_reranker", _B)
    assert isinstance(cr._get_reranker(_settings(reranker="blip2_itm")), _A)
    assert isinstance(cr._get_reranker(_settings(reranker="qwen_reranker")), _B)


def test_minmax_constant_scores_are_safe():
    assert not np.any(np.isnan(cr._minmax(np.asarray([2.0, 2.0, 2.0]))))
    assert cr._minmax(np.asarray([2.0, 2.0])).tolist() == [0.0, 0.0]


def test_settings_yaml_declares_reranker_keys():
    # yaml ⟷ pydantic sync is covered by the config test; pin the NEW keys too.
    s = Settings()
    assert s.search.reranker == "none"
    assert s.search.rerank_topk == 100
    assert s.search.blip2_itm_id == "Salesforce/blip2-itm-vit-g"
    assert s.search.qwen_reranker_id == "Qwen/Qwen3-VL-Reranker-8B"


# ── review findings C3/C17/C4 regressions ────────────────────────────────────
def test_model_id_change_rebuilds_and_clears_failed_latch(monkeypatch):
    built = []

    class _Backend:
        def __init__(self, settings):
            built.append(settings.search.blip2_itm_id)
            if settings.search.blip2_itm_id == "bad/model":
                raise RuntimeError("404 model not found")

        def score(self, q, p):
            return np.zeros(len(p), dtype=np.float32)

    monkeypatch.setitem(cr._BACKENDS, "blip2_itm", _Backend)
    s = _settings(reranker="blip2_itm")
    s.search.blip2_itm_id = "bad/model"
    assert cr._get_reranker(s) is None            # failed build → disabled
    assert cr._get_reranker(s) is None and built == ["bad/model"]  # latched
    s2 = _settings(reranker="blip2_itm")
    s2.search.blip2_itm_id = "good/model"
    assert cr._get_reranker(s2) is not None       # corrected id → REBUILD
    assert built == ["bad/model", "good/model"]


def test_score_ce_zeroes_unreadable_images(monkeypatch):
    from cvp.search.cross_rerank import QwenVLReranker

    rr = object.__new__(QwenVLReranker)
    rr.batch_size = 8

    class _CE:
        def predict(self, pairs, batch_size=8):
            return [0.9] * len(pairs)             # every READABLE pair scores high

    rr._ce = _CE()
    monkeypatch.setattr(cr, "load_rgb",
                        lambda p: None if "corrupt" in str(p) else object())
    scores = rr._score_ce("query", ["/a.jpg", "/corrupt.jpg", "/b.jpg"])
    assert scores.tolist() == [pytest.approx(0.9), 0.0, pytest.approx(0.9)]
