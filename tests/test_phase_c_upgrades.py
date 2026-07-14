"""Tests for the Phase-C engine-side upgrades: multi-variant SuperGlobal,
zero-row guarding, MMR AVS diversification, and VLM-rerank ordering."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest

from cvp.search.avs import avs_diversify
from cvp.search.superglobal import superglobal_rerank
from cvp.search.vlm_rerank import _parse_scores


def _norm(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)


# ── SuperGlobal ──────────────────────────────────────────────────────────────


def test_superglobal_accepts_single_and_stacked_queries():
    rng = np.random.default_rng(0)
    V = _norm(rng.normal(size=(30, 16)).astype(np.float32))
    q = V[0]
    single = superglobal_rerank(q, V)
    stacked = superglobal_rerank(q[None, :], V)
    assert single.shape == (30,)
    np.testing.assert_allclose(single, stacked, atol=1e-6)


def test_superglobal_multi_variant_is_max_fused():
    """A hit only variant #2 finds must not be demoted below its single-variant score."""
    rng = np.random.default_rng(1)
    V = _norm(rng.normal(size=(40, 16)).astype(np.float32))
    q1, q2 = V[3], V[27]
    s1 = superglobal_rerank(q1, V)
    s2 = superglobal_rerank(q2, V)
    both = superglobal_rerank(np.stack([q1, q2]), V)
    np.testing.assert_allclose(both, np.maximum(s1, s2), atol=1e-6)
    # the variant-2 target keeps its top score under multi-variant reranking
    assert both[27] >= s2[27] - 1e-6


def test_superglobal_zero_rows_score_zero_and_do_not_pollute():
    rng = np.random.default_rng(2)
    V = _norm(rng.normal(size=(12, 8)).astype(np.float32))
    V[5] = 0.0  # unreadable frame
    q = V[0]
    scores = superglobal_rerank(q, V, neighbors=4, qe_top=4)
    assert scores[5] == 0.0
    # neighbours of the query row are refined only with valid vectors: removing
    # the zero row entirely must not change any other score.
    V2 = np.delete(V, 5, axis=0)
    scores2 = superglobal_rerank(q, V2, neighbors=4, qe_top=4)
    np.testing.assert_allclose(np.delete(scores, 5), scores2, atol=1e-5)


# ── AVS MMR ──────────────────────────────────────────────────────────────────


@dataclass
class _Ref:
    global_id: int
    video_id: str
    pts_time: float


@dataclass
class _Result:
    ref: _Ref
    score: float
    signals: dict = field(default_factory=dict)

    @property
    def video_id(self) -> str:
        return self.ref.video_id

    @property
    def global_id(self) -> int:
        return self.ref.global_id


def _mk(i, vid, t, score):
    return _Result(ref=_Ref(global_id=i, video_id=vid, pts_time=t), score=score)


def test_avs_mmr_prefers_novel_over_near_duplicate():
    # r0 best; r1 = near-duplicate of r0 in ANOTHER video; r2 = distinct scene.
    results = [
        _mk(0, "L01_V001", 10.0, 1.00),
        _mk(1, "L02_V001", 50.0, 0.99),
        _mk(2, "L03_V001", 90.0, 0.60),
    ]
    dup = _norm(np.array([[1, 0, 0], [0.999, 0.04, 0], [0, 1, 0]], dtype=np.float32))
    picked = avs_diversify(results, cand_vecs=dup, mmr_lambda=0.5, limit=2)
    assert [r.global_id for r in picked[:2]] == [0, 2]
    # pure-relevance mode keeps score order
    picked_rel = avs_diversify(results, cand_vecs=dup, mmr_lambda=1.0, limit=2)
    assert [r.global_id for r in picked_rel[:2]] == [0, 1]


def test_avs_mmr_respects_video_cap_and_gap_and_backfills():
    results = [
        _mk(0, "L01_V001", 0.0, 1.0),
        _mk(1, "L01_V001", 3.0, 0.9),   # < min_gap from r0 → deferred
        _mk(2, "L01_V001", 30.0, 0.8),
        _mk(3, "L01_V001", 60.0, 0.7),  # over per-video cap (2) → deferred
        _mk(4, "L02_V001", 0.0, 0.1),
    ]
    vecs = _norm(np.eye(5, 6, dtype=np.float32))
    picked = avs_diversify(
        results, per_video_cap=2, min_gap_s=10.0, limit=5, cand_vecs=vecs, mmr_lambda=0.7
    )
    ids = [r.global_id for r in picked]
    # constrained picks first (0, 2 from L01 + 4 from L02), then backfill 1, 3
    assert ids[:3] == [0, 2, 4]
    assert set(ids) == {0, 1, 2, 3, 4}


def test_avs_without_vectors_matches_legacy_greedy():
    results = [
        _mk(0, "L01_V001", 0.0, 1.0),
        _mk(1, "L01_V001", 30.0, 0.9),
        _mk(2, "L02_V001", 0.0, 0.8),
    ]
    picked = avs_diversify(results, per_video_cap=1, min_gap_s=10.0, limit=3)
    assert [r.global_id for r in picked] == [0, 2, 1]


def test_avs_mmr_never_exceeds_limit():
    """Regression: MMR path used to return limit+1 when constrained picks
    filled the limit exactly and passed-over rows remained."""
    results = [_mk(i, f"L0{i + 1}_V001", 0.0, 1.0 - i * 0.1) for i in range(4)]
    vecs = _norm(np.eye(4, 6, dtype=np.float32))
    for lam in (0.5, 0.7, 1.0):
        picked = avs_diversify(
            results, per_video_cap=1, min_gap_s=5.0, limit=2, cand_vecs=vecs, mmr_lambda=lam
        )
        assert len(picked) == 2, f"lambda={lam} returned {len(picked)} rows"


def test_embedder_model_tag_marker_blocks_checkpoint_mixing(tmp_path):
    from cvp.config import Settings
    from cvp.index.embedder import _check_model_tag
    from cvp.index.store import IndexStore
    from cvp.utils.io import read_json

    settings = Settings.model_validate({
        "paths": {"data_root": str(tmp_path / "d"), "artifacts_root": str(tmp_path / "a")},
    })
    store = IndexStore(settings, "openclip")

    class _M:
        key = "openclip"
        model_tag = "PE-Core-bigG-14-448"

    _check_model_tag(store, _M(), overwrite=False)  # first run writes the marker
    assert read_json(store.embed_dir / "model_tag.json")["model_tag"] == "PE-Core-bigG-14-448"
    _check_model_tag(store, _M(), overwrite=False)  # same tag → fine

    class _Other:
        key = "openclip"
        model_tag = "dfn5b-vit-h-14-378-quickgelu"

    with pytest.raises(RuntimeError, match="produced by"):
        _check_model_tag(store, _Other(), overwrite=False)
    # overwrite path re-stamps the marker
    _check_model_tag(store, _Other(), overwrite=True)
    assert read_json(store.embed_dir / "model_tag.json")["model_tag"] == "dfn5b-vit-h-14-378-quickgelu"


# ── VLM rerank parsing ───────────────────────────────────────────────────────


def test_parse_scores_strict_json_and_fenced():
    assert _parse_scores('{"scores": [1, 2, 3]}', 3) == [1.0, 2.0, 3.0]
    assert _parse_scores('```json\n{"scores": [0, 10]}\n```', 2) == [0.0, 10.0]
    assert _parse_scores('{"scores": [1, 2]}', 3) is None      # wrong length
    assert _parse_scores("not json", 2) is None
    assert _parse_scores('{"scores": [-5, 99]}', 2) == [0.0, 10.0]  # clamped


def test_vlm_rerank_reorders_head_only(monkeypatch):
    from cvp.config import Settings
    from cvp.search import vlm_rerank as vr

    settings = Settings()
    settings.search.vlm_rerank_provider = "gemini"
    settings.search.vlm_rerank_topk = 3
    results = [
        _mk(0, "L01_V001", 0.0, 1.0),
        _mk(1, "L01_V001", 10.0, 0.9),
        _mk(2, "L01_V001", 20.0, 0.8),
        _mk(3, "L01_V001", 30.0, 0.7),  # tail — must stay in place
    ]
    for r in results:
        r.ref.path = "unused.jpg"  # type: ignore[attr-defined]
    monkeypatch.setattr(vr, "_gemini_scores", lambda q, p, s: [2.0, 9.0, 5.0])
    out = vr.vlm_rerank(results, "query", settings)
    assert [r.global_id for r in out] == [1, 2, 0, 3]
    assert out[0].signals["vlm"] == 9.0


def test_vlm_rerank_failure_keeps_order(monkeypatch):
    from cvp.config import Settings
    from cvp.search import vlm_rerank as vr

    settings = Settings()
    settings.search.vlm_rerank_provider = "gemini"
    results = [_mk(0, "L01_V001", 0.0, 1.0), _mk(1, "L01_V001", 10.0, 0.9)]
    for r in results:
        r.ref.path = "unused.jpg"  # type: ignore[attr-defined]

    def boom(q, p, s):
        raise RuntimeError("no api key")

    monkeypatch.setattr(vr, "_gemini_scores", boom)
    out = vr.vlm_rerank(results, "query", settings)
    assert [r.global_id for r in out] == [0, 1]
