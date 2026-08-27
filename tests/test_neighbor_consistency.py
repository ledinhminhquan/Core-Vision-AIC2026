"""search.neighbor_consistency_boost (Nhiệm vụ C đợt 2): cao nguyên thắng gai.

Unit test công thức (window/decay/chuẩn hóa/biên video) + integration trên
SearchEngine thật với corpus synthetic (fake encoder, không GPU/network):
knob off = bit-identical từng (gid, score); knob on = hàm được gọi đúng chỗ.
"""

from __future__ import annotations

import numpy as np
import pytest

import cvp.search.engine as engine_mod
from cvp.config import Settings
from cvp.search.engine import SearchEngine
from cvp.search.fusion import neighbor_consistency_boost

W, WIN, DECAY = 0.15, 2, 0.5
DENOM = DECAY + DECAY ** 2          # window 2


def _lift(top: float, support: float) -> float:
    return W * top * (support / DENOM)


# ── công thức: cao nguyên vs gai ─────────────────────────────────────────────

PLATEAU_SPIKE = {10: 0.80, 11: 0.85, 12: 0.80, 50: 0.90}
SPANS = {10: (10, 3), 11: (10, 3), 12: (10, 3), 50: (50, 1)}


def test_plateau_center_beats_lone_spike_when_on():
    out = neighbor_consistency_boost(dict(PLATEAU_SPIKE), SPANS,
                                     weight=W, window=WIN, decay=DECAY)
    top = 0.90
    # Tâm cao nguyên: d=1 hai phía 0.80; d=2 cả hai phía vượt biên span → dừng.
    support_center = DECAY * ((0.80 + 0.80) / 2) / top
    assert out[11] == pytest.approx(0.85 + _lift(top, support_center))
    # Gai đơn độc: span 1 frame → không hàng xóm → GIỮ NGUYÊN giá trị.
    assert out[50] == 0.90
    ranked = sorted(out, key=lambda g: -out[g])
    assert ranked[0] == 11                       # cao nguyên vượt gai 0.90
    # Vai cao nguyên: d=1 chỉ phía trong (0.85), d=2 với tới mép kia (0.80).
    support_edge = (DECAY * ((0.85 + 0.0) / 2) + DECAY ** 2 * ((0.80 + 0.0) / 2)) / top
    assert out[10] == pytest.approx(0.80 + _lift(top, support_edge))
    assert out[10] < out[50]


def test_off_returns_identical_object_bit_identical():
    scores = dict(PLATEAU_SPIKE)
    assert neighbor_consistency_boost(scores, SPANS, weight=0.0) is scores
    assert neighbor_consistency_boost(scores, SPANS, weight=-1.0) is scores
    assert neighbor_consistency_boost(scores, SPANS, weight=W, window=0) is scores
    assert neighbor_consistency_boost({}, {}, weight=W) == {}
    zeros = {1: 0.0, 2: 0.0}
    assert neighbor_consistency_boost(zeros, {1: (1, 2), 2: (1, 2)}, weight=W) is zeros


def test_no_bleed_across_video_boundary():
    # Video A = gid 0-1, video B = gid 2-3: gid 1 và 2 kề số nhưng khác video.
    scores = {0: 0.5, 1: 0.9, 2: 0.9, 3: 0.5}
    spans = {0: (0, 2), 1: (0, 2), 2: (2, 2), 3: (2, 2)}
    out = neighbor_consistency_boost(dict(scores), spans,
                                     weight=W, window=WIN, decay=DECAY)
    top = 0.9
    # gid 1: chỉ hàng xóm trái (gid 0) trong span — gid 2 KHÔNG được tính.
    assert out[1] == pytest.approx(0.9 + _lift(top, DECAY * ((0.5 + 0.0) / 2) / top))
    # gid 2: chỉ hàng xóm phải (gid 3) trong span — gid 1 KHÔNG được tính.
    assert out[2] == pytest.approx(0.9 + _lift(top, DECAY * ((0.0 + 0.5) / 2) / top))


def test_decay_shapes_near_vs_far_neighbor_weight():
    # support được chuẩn hóa theo Σdecay^d nên decay là tham số HÌNH DẠNG:
    # hàng xóm GẦN mạnh + hàng xóm XA yếu → decay nhỏ (tập trung gần) phải
    # cho lift LỚN hơn decay=1.0 (chia đều cho cả hàng xóm xa yếu).
    scores = {10: 0.2, 11: 0.8, 12: 0.85, 13: 0.8, 14: 0.2}
    spans = {i: (10, 5) for i in range(10, 15)}
    even = neighbor_consistency_boost(dict(scores), spans, weight=W, window=2, decay=1.0)
    near = neighbor_consistency_boost(dict(scores), spans, weight=W, window=2, decay=0.5)
    assert near[12] > even[12] > scores[12]


def test_missing_span_leaves_candidate_untouched():
    scores = {1: 0.5, 2: 0.9, 3: 0.5}
    spans = {1: (1, 3), 3: (1, 3)}               # gid 2 không có span
    out = neighbor_consistency_boost(dict(scores), spans,
                                     weight=W, window=WIN, decay=DECAY)
    assert out[2] == 0.9                          # không span → không boost
    assert out[1] > 0.5 and out[3] > 0.5          # hàng xóm vẫn được tính


# ── integration: SearchEngine thật trên corpus synthetic ────────────────────


class _FakeModel:
    key = "fake"
    multilingual = True
    dim = 16

    def __init__(self, settings: Settings):
        pass

    def encode_text(self, texts):
        rng = np.random.default_rng(abs(hash(tuple(texts))) % (2 ** 32))
        out = rng.normal(size=(len(texts), self.dim)).astype(np.float32)
        return out / np.linalg.norm(out, axis=1, keepdims=True)

    def encode_image(self, images):
        rng = np.random.default_rng(7)
        out = rng.normal(size=(len(images), self.dim)).astype(np.float32)
        return out / np.linalg.norm(out, axis=1, keepdims=True)


@pytest.fixture()
def engine(corpus_with_index: Settings, monkeypatch) -> SearchEngine:
    settings = corpus_with_index
    settings.embedding.model = "fake"
    settings.query.provider = "none"
    settings.search.vlm_rerank = False
    monkeypatch.setattr(engine_mod, "build_model", lambda s, name: _FakeModel(s))
    monkeypatch.setattr(engine_mod, "index_key_for", lambda name: "fake")
    return SearchEngine(settings)


def test_engine_off_bit_identical_and_on_invokes_boost(engine, monkeypatch):
    calls = []
    real = engine_mod.fusion.neighbor_consistency_boost

    def _spy(scores, spans, **kw):
        calls.append(dict(kw))
        return real(scores, spans, **kw)

    monkeypatch.setattr(engine_mod.fusion, "neighbor_consistency_boost", _spy)
    q = "hai người phụ nữ nấu ăn"

    off_1 = [(r.global_id, r.score) for r in engine.search_text(q, display_k=15)]
    assert calls == []                            # off → đường boost không được gọi

    engine.settings.search.neighbor_consistency_boost = 0.15
    on = [(r.global_id, r.score) for r in engine.search_text(q, display_k=15)]
    assert len(calls) == 1
    assert calls[0] == {"weight": 0.15, "window": 2, "decay": 0.5}
    assert on                                     # ranking hợp lệ khi knob bật

    engine.settings.search.neighbor_consistency_boost = 0.0
    off_2 = [(r.global_id, r.score) for r in engine.search_text(q, display_k=15)]
    assert off_2 == off_1                         # bit-identical từng (gid, score)
    assert len(calls) == 1
