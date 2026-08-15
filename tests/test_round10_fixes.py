"""Round-10 regressions — the first run at the REAL 177k-row Batch-1 scale
(latency-gate breach) plus config-matrix interaction findings.

  R10-1  hot-path batching: cached embedding mmaps, batched catalog.refs /
         video_ids, one-shot temporal-boost vector fetch
  R10-2  AVS MMR relevance follows list RANK (rerank promotions survive)
  R10-3  low-confidence retry alts skip the cross/VLM rerank stack
  R10-4  config fails loud: all-zero fusion weights, min_gap>max_gap,
         multi_query_agg typos, rerank_weight out of [0,1]
  R10-5  neighbor_boost breaks out of an oversized window
  R10-6  one shared round-time query loader across runner/warm-cache/dumps
  R10-7  a 0-frame re-extraction never lands a header-only map csv
"""

from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cvp.config import Settings

REPO = Path(__file__).resolve().parents[1]


# ── R10-1 · hot-path batching ────────────────────────────────────────────────
def test_vectors_mmap_is_cached(tmp_path):
    from cvp.index.store import IndexStore

    s = Settings()
    s.paths.artifacts_root = tmp_path
    store = IndexStore(s, "fake")
    store.embed_dir.mkdir(parents=True, exist_ok=True)
    np.save(store.embedding_path("L21_V001"), np.ones((3, 4), dtype=np.float32))
    a = store.vectors_mmap("L21_V001")
    b = store.vectors_mmap("L21_V001")
    assert a is b                       # same handle, no re-open per query
    src = (REPO / "src" / "cvp" / "index" / "store.py").read_text(encoding="utf-8")
    assert "_mmap_cache.clear()" in src  # invalidated on rebuild


def test_catalog_refs_batch_matches_singles(corpus):
    from cvp.data.catalog import KeyframeCatalog

    cat = KeyframeCatalog(corpus)
    cat.build()
    gids = [0, 2, 1]
    batch = cat.refs(gids)
    singles = [cat.ref(g) for g in gids]
    assert [r.global_id for r in batch] == [r.global_id for r in singles]
    assert [r.frame_idx for r in batch] == [r.frame_idx for r in singles]
    assert [r.path for r in batch] == [r.path for r in singles]
    assert cat.video_ids(gids) == [r.video_id for r in singles]
    assert cat.refs([]) == [] and cat.video_ids([]) == []


def test_temporal_boost_uses_one_batched_fetch():
    src = (REPO / "src" / "cvp" / "search" / "engine.py").read_text(encoding="utf-8")
    block = src.split("def _maybe_temporal_boost")[1].split("def ")[0]
    assert block.count("self._vectors_for(") == 1   # ONE call, not per-candidate
    assert "uniq_rows" in block


# ── R10-2 · AVS MMR follows list rank ────────────────────────────────────────
def _res(i, score, vid="L21_V001", t=None):
    ref = SimpleNamespace(global_id=i, video_id=vid, pts_time=float(t if t is not None else i * 30))
    return SimpleNamespace(ref=ref, score=score, video_id=vid,
                           frame_idx=i * 100, global_id=i, signals={})


def test_avs_mmr_relevance_follows_list_order_not_stale_score():
    from cvp.search.avs import avs_diversify

    # Reranked list: order is truth, .score is the STALE pre-rerank value —
    # the first element carries the LOWEST score on purpose.
    results = [_res(0, 0.1), _res(1, 0.5), _res(2, 0.9), _res(3, 0.7)]
    vecs = np.eye(4, dtype=np.float32)     # orthogonal → no novelty penalty
    picked = avs_diversify(results, limit=4, per_video_cap=10, min_gap_s=0.0,
                           mmr_lambda=0.7, cand_vecs=vecs)
    assert picked[0] is results[0]         # head = list head, NOT max-score row


# ── R10-3 · retry alts skip the rerank stack ─────────────────────────────────
def test_retry_passes_skip_rerank():
    from cvp.pipeline.run_queries import maybe_retry_low_confidence

    calls: list[dict] = []

    class _Eng:
        settings = Settings()
        settings.search.low_confidence_retry = True
        settings.search.low_confidence_threshold = 0.99   # always retry

        class query_processor:  # noqa: N801 — attribute stub
            @staticmethod
            def process(q):
                return SimpleNamespace(enhanced="alt one", expansions=["alt two"])

        def search_prepared(self, text, **kw):
            calls.append(kw)
            return [_res(9, 0.5)]

    flat = [_res(i, 0.5) for i in range(5)]               # zero-spread → low conf
    maybe_retry_low_confidence(_Eng(), "truy vấn", flat)
    assert calls and all(kw.get("skip_rerank") is True for kw in calls)


def test_finalize_signature_has_skip_rerank():
    src = (REPO / "src" / "cvp" / "search" / "engine.py").read_text(encoding="utf-8")
    assert "skip_rerank: bool = False" in src
    assert "if skip_rerank:" in src


# ── R10-4 · loud config validation ───────────────────────────────────────────
def test_all_zero_fusion_weights_fail_loud():
    from cvp.config import FusionWeights

    with pytest.raises(ValueError, match="ALL fusion weights"):
        FusionWeights(visual=0, ocr=0, asr=0, caption=0, metadata=0, object=0)
    with pytest.raises(ValueError, match=">= 0"):
        FusionWeights(visual=-1)
    FusionWeights(visual=1.0, ocr=0)      # partial zeros stay legal


def test_temporal_gap_inversion_fails_loud():
    from cvp.config import TemporalCfg

    with pytest.raises(ValueError, match="min_gap_s"):
        TemporalCfg(min_gap_s=10.0, max_gap_s=5.0)
    with pytest.raises(ValueError, match=">= 0"):
        TemporalCfg(min_gap_s=-1.0)
    TemporalCfg(min_gap_s=0.0, max_gap_s=150.0)


def test_multi_query_agg_typo_fails_loud():
    from cvp.config import QueryCfg

    with pytest.raises(Exception):
        QueryCfg(multi_query_agg="avg")
    QueryCfg(multi_query_agg="mean")


def test_rerank_weight_bounded():
    from cvp.config import SearchCfg

    with pytest.raises(Exception):
        SearchCfg(rerank_weight=1.5)
    SearchCfg(rerank_weight=1.0)


# ── R10-5 · neighbor_boost oversized window ──────────────────────────────────
def test_neighbor_boost_breaks_out_of_huge_window():
    from cvp.search.fusion import neighbor_boost

    scores = {i: 1.0 for i in range(100)}
    spans = {i: (0, 100) for i in range(100)}
    t0 = time.perf_counter()
    out = neighbor_boost(scores, spans, boost=0.1, window=10_000_000)
    assert time.perf_counter() - t0 < 2.0          # was candidates×window iterations
    assert out[1] > scores[1]                      # boost still applied


# ── R10-6 · one shared round-time loader ─────────────────────────────────────
def test_shared_query_loader_everywhere():
    from cvp.pipeline.run_queries import load_query_lines  # noqa: F401

    for rel in ("scripts/51_warm_cache.py", "scripts/23_dump_signals.py"):
        src = (REPO / rel).read_text(encoding="utf-8")
        assert "load_query_lines" in src, rel
        assert 'read_text(encoding="utf-8-sig").splitlines() if ln.strip()' not in src, rel


def test_load_query_lines_strips_invisible(tmp_path):
    from cvp.pipeline.run_queries import load_query_lines

    p = tmp_path / "q.txt"
    p.write_text("header\n​E1: a\n\nE2: b\n", encoding="utf-8")
    assert load_query_lines(p) == ["header", "E1: a", "E2: b"]


# ── R10-7 · zero-frame extraction never lands a header-only csv ──────────────
def test_zero_frame_extraction_keeps_previous_map():
    src = (REPO / "src" / "cvp" / "data" / "extraction.py").read_text(encoding="utf-8")
    guard = src.split("if not rows:")[1].split("map_dir.mkdir")[0]
    assert "keeping the previous map" in guard and "return 0" in guard
