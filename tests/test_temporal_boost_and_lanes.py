"""Temporal-context boost (Vortex recipe) + the 2026 optional lanes.

Pure CPU: regex split, neighbour-row math, boost blending, registry wiring and
the fail-fast Matryoshka validation of the jina lane (no downloads).
"""

from __future__ import annotations

import pytest

from cvp.config import Settings
from cvp.models.registry import index_key_for, known_models, resolve_constructor
from cvp.search.temporal_boost import (
    apply_context_boost,
    neighbor_rows,
    split_temporal_query,
)


# ── query splitting ──────────────────────────────────────────────────────────
def test_split_sau_khi_context_is_before():
    q = "người đàn ông ngã xuống đường sau khi chiếc xe tải màu đỏ vượt đèn đỏ"
    parts = split_temporal_query(q)
    assert parts is not None and parts.direction == "before"
    assert parts.target.startswith("người đàn ông")
    assert parts.context.startswith("chiếc xe tải")


def test_split_truoc_khi_context_is_after():
    q = "các vận động viên xếp hàng chào khán giả trước khi cuộc đua xe đạp bắt đầu"
    parts = split_temporal_query(q)
    assert parts is not None and parts.direction == "after"
    assert "cuộc đua" in parts.context


def test_split_ngay_sau_khi_variant():
    q = "khoảnh khắc pháo hoa bùng nổ trên bầu trời ngay sau khi đồng hồ điểm giao thừa"
    parts = split_temporal_query(q)
    assert parts is not None and parts.direction == "before"


def test_split_no_marker_returns_none():
    assert split_temporal_query("một người phụ nữ mặc áo dài đỏ đứng trước chợ Bến Thành") is None


def test_split_short_tail_does_not_trigger():
    # "sau khi đó" carries no embeddable context — must not split.
    assert split_temporal_query("đoàn diễu hành đi qua lễ đài sau khi đó") is None


def test_split_english_markers():
    parts = split_temporal_query("a man falls off the stage after the singer finishes the song")
    assert parts is not None and parts.direction == "before"


# ── neighbour rows ───────────────────────────────────────────────────────────
def test_neighbor_rows_before_clipped_to_video_start():
    assert neighbor_rows(gid=102, span=(100, 200), direction="before", window=5) == [100, 101]


def test_neighbor_rows_after_clipped_to_video_end():
    assert neighbor_rows(gid=198, span=(100, 200), direction="after", window=5) == [199, 200]


def test_neighbor_rows_never_includes_self_or_crosses_span():
    rows = neighbor_rows(gid=150, span=(100, 200), direction="before", window=3)
    assert rows == [147, 148, 149] and 150 not in rows


# ── boost blending ───────────────────────────────────────────────────────────
def test_apply_context_boost_moves_only_scored_candidates():
    fused = {1: 0.9, 2: 0.8, 3: 0.7}
    out = apply_context_boost(fused, {2: 5.0, 3: 1.0}, weight=0.5)
    assert out[1] == pytest.approx(0.9)          # untouched
    assert out[2] == pytest.approx(0.8 + 0.5)    # max ctx → +weight
    assert out[3] == pytest.approx(0.7 + 0.0)    # min ctx → +0
    # boost can reorder: candidate 2 now outranks 1.
    assert out[2] > out[1]


def test_apply_context_boost_noop_on_empty_or_constant():
    fused = {1: 0.5, 2: 0.4}
    assert apply_context_boost(fused, {}, 0.5) == fused
    assert apply_context_boost(fused, {1: 3.0, 2: 3.0}, 0.5) == fused
    assert apply_context_boost(fused, {1: 9.0}, 0.0) == fused


# ── 2026 lanes registry wiring ───────────────────────────────────────────────
def test_registry_knows_new_lanes():
    models = known_models()
    assert "jina" in models and "metaclip2" in models
    # lazy resolution never imports heavy deps:
    assert callable(resolve_constructor("jina"))
    assert callable(resolve_constructor("metaclip2"))
    # each owns its own index space (unlike finetuned → siglip2):
    assert index_key_for("jina") == "jina"
    assert index_key_for("metaclip2") == "metaclip2"


def test_jina_dim_validation_fails_fast_without_downloads():
    from cvp.models.jina_clip import JinaClipModel

    s = Settings()
    s.embedding.jina_dim = 32          # below the Matryoshka floor
    with pytest.raises(ValueError, match="Matryoshka"):
        JinaClipModel(s)
    s.embedding.jina_dim = 4096        # above the ceiling
    with pytest.raises(ValueError, match="Matryoshka"):
        JinaClipModel(s)


def test_new_settings_defaults_are_wired():
    s = Settings()
    assert s.embedding.jina_id == "jinaai/jina-clip-v2"
    assert s.embedding.metaclip2_id == "facebook/metaclip-2-worldwide-huge-quickgelu"
    assert s.search.temporal_boost is False
    assert s.query.gemini_model == "gemini-3.5-flash-lite"
    assert s.query.gemini_model_fallbacks[-1] == "gemini-flash-latest"
    assert s.vqa.frames_per_answer == 3


# ── review finding C1: catalog span convention (first_row, COUNT) ────────────
def test_neighbor_rows_from_video_span_after_direction_works():
    from cvp.search.temporal_boost import neighbor_rows_from_video_span

    # catalog.video_span returns (first_row, COUNT). Before the fix, passing it
    # raw made 'after' a silent no-op for every video beyond the first.
    rows = neighbor_rows_from_video_span(5100, (5000, 300), "after", 12)
    assert rows == list(range(5101, 5113))


def test_neighbor_rows_from_video_span_never_leaks_across_video_end():
    from cvp.search.temporal_boost import neighbor_rows_from_video_span

    # Video = rows [0..299]; gid near the end must clip at 299 — row 300
    # belongs to the NEXT video.
    rows = neighbor_rows_from_video_span(295, (0, 300), "after", 12)
    assert rows == [296, 297, 298, 299]
    assert 300 not in rows


def test_neighbor_rows_from_video_span_before_matches_plain():
    from cvp.search.temporal_boost import neighbor_rows, neighbor_rows_from_video_span

    assert neighbor_rows_from_video_span(150, (100, 101), "before", 3) == \
        neighbor_rows(150, (100, 200), "before", 3)
