"""End-to-end SearchEngine tests on the synthetic corpus with a fake encoder.

Exercises the full online path (dense → SuperGlobal → BM25/objects → fusion →
neighbor boost → ranking → AVS/feedback) without GPU, network, or real models —
the integration layer that unit tests can't see (design weakness #13).
"""

from __future__ import annotations

import numpy as np
import pytest

import cvf.search.engine as engine_mod
from cvf.config import Settings
from cvf.search.engine import SearchEngine


class FakeModel:
    """Deterministic text/image encoder aligned with the stored fake embeddings."""

    key = "fake"
    multilingual = True
    dim = 16

    def __init__(self, settings: Settings, target_vec: np.ndarray | None = None):
        self._target = target_vec

    def encode_text(self, texts: list[str]) -> np.ndarray:
        rng = np.random.default_rng(abs(hash(tuple(texts))) % (2**32))
        out = rng.normal(size=(len(texts), self.dim)).astype(np.float32)
        if self._target is not None:
            out[0] = self._target  # first variant points at the planted frame
        return out / np.linalg.norm(out, axis=1, keepdims=True)

    def encode_image(self, images) -> np.ndarray:
        rng = np.random.default_rng(7)
        out = rng.normal(size=(len(images), self.dim)).astype(np.float32)
        return out / np.linalg.norm(out, axis=1, keepdims=True)


@pytest.fixture()
def engine(corpus_with_index: Settings, monkeypatch) -> SearchEngine:
    settings = corpus_with_index
    settings.embedding.model = "fake"
    settings.query.provider = "none"
    settings.search.vlm_rerank = False

    from cvf.data.catalog import KeyframeCatalog
    from cvf.index.store import IndexStore

    catalog = KeyframeCatalog(settings)
    catalog.load()
    store = IndexStore(settings, "fake")
    vec = np.load(store.embedding_path("L21_V002"), mmap_mode="r")[2].astype(np.float32)

    monkeypatch.setattr(engine_mod, "build_model", lambda s, name: FakeModel(s, target_vec=vec))
    monkeypatch.setattr(engine_mod, "index_key_for", lambda name: "fake")
    return SearchEngine(settings)


def test_search_text_finds_planted_frame(engine: SearchEngine):
    results = engine.search_text("hai người phụ nữ nấu ăn", display_k=15)
    assert results, "engine returned no results"
    top = results[0]
    # the planted target: L21_V002 keyframe n=3 (row 2)
    assert top.video_id == "L21_V002"
    assert top.frame_idx == 300  # n=3 → frame_idx = n*100 per conftest map
    assert results == sorted(results, key=lambda r: -r.score)
    assert "visual" in top.signals


def test_search_text_debug_returns_signal_maps(engine: SearchEngine):
    results, dump = engine.search_text_debug("bản tin 60 giây", display_k=10)
    assert results and "visual" in dump and len(dump["visual"]) > 0
    # metadata BM25 should light up for this query (title matches L21_V001)
    if "metadata" in dump:
        gids = set(dump["metadata"])
        vids = {engine.catalog.ref(g).video_id for g in gids}
        assert "L21_V001" in vids


def test_search_avs_diversifies(engine: SearchEngine):
    results = engine.search_avs("người", limit=10)
    assert results
    from collections import Counter

    per_video = Counter(r.video_id for r in results[:6])
    assert max(per_video.values()) <= engine.settings.search.avs_per_video_cap + 1


def test_feedback_reranks_toward_positive(engine: SearchEngine):
    base = engine.search_text("một cảnh bất kỳ", display_k=15)
    target_gid = base[-1].global_id
    fed = engine.search_with_feedback("một cảnh bất kỳ", positive_gids=[target_gid])
    assert fed
    base_rank = next(i for i, r in enumerate(base) if r.global_id == target_gid)
    fed_ranks = [i for i, r in enumerate(fed) if r.global_id == target_gid]
    assert fed_ranks and fed_ranks[0] <= base_rank


def test_nearest_and_temporal_neighbors(engine: SearchEngine):
    results = engine.search_text("người", display_k=5)
    gid = results[0].global_id
    near = engine.nearest(gid, k=5)
    assert near and all(r.global_id != gid for r in near)
    ctx = engine.temporal_neighbors(gid, window=2)
    assert 1 <= len(ctx) <= 5
    assert all(r.video_id == results[0].video_id for r in ctx)


def test_search_trake_returns_monotone_sequences(engine: SearchEngine):
    cands = engine.search_trake(["sự kiện một", "sự kiện hai"], max_results=10)
    for c in cands:
        assert list(c.frame_idxs) == sorted(c.frame_idxs)
        assert len(set(c.frame_idxs)) == len(c.frame_idxs)
        assert list(c.ns) == sorted(set(c.ns))
