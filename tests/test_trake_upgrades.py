"""TRAKE upgrade knobs (session 2026-08-27) — every default must reproduce the
old behaviour bit-for-bit, every knob must be provably useful on a synthetic
corpus where the right answer is planted:

* ``temporal.caption_signal_weight`` — dense-caption BM25 blended into the DP
  similarity matrix (bước hành động × frame) rescues a chain the dense lane
  ranks second; weight 0 (default) never calls the scorer and returns the
  baseline ranking unchanged.
* ``temporal.event_query_variants: all`` — per-event max over query variants
  finds a chain only the SECOND variant sees; pure-noise variants do not
  change the legacy winner.
* ``temporal.pool_context: prepend`` — pooling-only vectors rescue a video the
  event vectors alone would never pool.
* ``temporal.submit_strategy: jitter`` — the head of the ranking is preserved
  verbatim, extra rows densify frames BETWEEN keyframes, everything stays
  strictly increasing and deduped.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cvp.config import Settings
from cvp.search.temporal import (
    JITTER_HEAD,
    TrakeCandidate,
    _normalized_caption_by_vid,
    _reduce_variants,
    caption_scorer_from_signals,
    expand_candidates_jitter,
    jitter_frame_variants,
    trake_search,
)

DIM = 16
REPO = Path(__file__).resolve().parents[1]


def _basis(i: int) -> np.ndarray:
    v = np.zeros(DIM, dtype=np.float32)
    v[i] = 1.0
    return v


def _mix(cos: float, main: int, junk: int) -> np.ndarray:
    """Unit vector with EXACT cosine ``cos`` against basis ``main``."""
    v = cos * _basis(main) + float(np.sqrt(1.0 - cos * cos)) * _basis(junk)
    return v.astype(np.float32)


def _build_store(settings: Settings, name: str, by_vid: dict[str, np.ndarray]):
    """Custom IndexStore with EXPLICIT per-video vectors (planted layouts)."""
    from cvp.data.catalog import KeyframeCatalog
    from cvp.index.store import IndexStore

    catalog = KeyframeCatalog(settings)
    store = IndexStore(settings, name)
    for vid, vecs in by_vid.items():
        store.embedding_path(vid).parent.mkdir(parents=True, exist_ok=True)
        np.save(store.embedding_path(vid), vecs.astype(np.float32))
    store.build(catalog)
    return catalog, store


def _junk_rows(n: int, start_basis: int) -> list[np.ndarray]:
    return [_basis(start_basis + (i % 3)) for i in range(n)]


# ── config defaults: every new knob must default to the OLD behaviour ────────


def test_new_temporal_knobs_default_to_legacy_behaviour():
    t = Settings().temporal
    assert t.event_query_variants == "original"
    assert t.pool_context == "none"
    assert t.caption_signal_weight == 0.0
    assert t.submit_strategy == "legacy"
    assert t.jitter_videos == 4


def test_settings_yaml_carries_the_new_knobs_with_legacy_defaults():
    text = (REPO / "configs" / "settings.yaml").read_text(encoding="utf-8")
    assert "event_query_variants: original" in text
    assert "pool_context: none" in text
    assert "caption_signal_weight: 0.0" in text
    assert "submit_strategy: legacy" in text
    assert "jitter_videos: 4" in text


# ── _reduce_variants ─────────────────────────────────────────────────────────


def test_reduce_variants_identity_is_a_noop():
    sim = np.arange(12, dtype=np.float64).reshape(4, 3)
    out = _reduce_variants(sim, [0, 1, 2], 3)
    assert out is sim  # fast path: the classic single-text-per-event layout


def test_reduce_variants_takes_per_event_max():
    sim = np.array([[0.1, 0.9, 0.3],
                    [0.8, 0.2, 0.4]])
    out = _reduce_variants(sim, [0, 0, 1], 2)
    assert out.shape == (2, 2)
    assert out[0].tolist() == [0.9, 0.3]
    assert out[1].tolist() == [0.8, 0.4]


# ── caption step signal ──────────────────────────────────────────────────────


def _caption_layout(settings: Settings):
    """Dense lane prefers the DECOY chain (0.9 vs 0.85) — captions must rescue.

    True video L21_V001: events at n=2 and n=5 with cosine 0.85.
    Decoy L21_V002: events at n=1 and n=3 with cosine 0.90.
    """
    by_vid = {
        "L21_V001": np.stack([
            _basis(8), _mix(0.85, 0, 12), _basis(9),
            _basis(10), _mix(0.85, 1, 13), _basis(11),
        ]),
        "L21_V002": np.stack([
            _mix(0.90, 0, 12), _basis(8), _mix(0.90, 1, 13), _basis(9), _basis(10),
        ]),
        "K01_V001": np.stack(_junk_rows(4, 8)),
    }
    catalog, store = _build_store(settings, "trake_cap", by_vid)
    event_vecs = np.stack([_basis(0), _basis(1)])
    return catalog, store, event_vecs


def _true_chain_captions(video_id: str) -> np.ndarray | None:
    """Raw caption-BM25 stub: only the TRUE video has caption hits."""
    if video_id != "L21_V001":
        return None
    m = np.zeros((6, 2), dtype=np.float64)
    m[1, 0] = 3.0   # n=2 row ↔ event 1
    m[4, 1] = 2.5   # n=5 row ↔ event 2
    return m


def test_caption_signal_rescues_chain_dense_ranks_second(corpus_with_index):
    catalog, store, event_vecs = _caption_layout(corpus_with_index)

    baseline = trake_search(event_vecs, store.search, store.embedding_path,
                            catalog, corpus_with_index)
    assert baseline and baseline[0].video_id == "L21_V002"  # dense alone: decoy

    corpus_with_index.temporal.caption_signal_weight = 0.5
    boosted = trake_search(event_vecs, store.search, store.embedding_path,
                           catalog, corpus_with_index,
                           caption_scorer=_true_chain_captions)
    assert boosted and boosted[0].video_id == "L21_V001"
    assert boosted[0].ns == [2, 5]
    assert boosted[0].frame_idxs == [200, 500]


def test_caption_weight_zero_never_calls_scorer_and_matches_baseline(corpus_with_index):
    catalog, store, event_vecs = _caption_layout(corpus_with_index)
    calls: list[str] = []

    def counting_scorer(video_id: str):
        calls.append(video_id)
        return _true_chain_captions(video_id)

    baseline = trake_search(event_vecs, store.search, store.embedding_path,
                            catalog, corpus_with_index)
    with_scorer = trake_search(event_vecs, store.search, store.embedding_path,
                               catalog, corpus_with_index,
                               caption_scorer=counting_scorer)
    assert not calls  # default weight 0.0 → the scorer must never run
    assert [(c.video_id, c.ns) for c in with_scorer] == \
           [(c.video_id, c.ns) for c in baseline]
    assert [c.score for c in with_scorer] == pytest.approx([c.score for c in baseline])


def test_caption_signal_absent_artifacts_is_a_noop(corpus_with_index):
    """Scorer returns None everywhere (no caption artifacts) → baseline ranking."""
    catalog, store, event_vecs = _caption_layout(corpus_with_index)
    corpus_with_index.temporal.caption_signal_weight = 0.5
    out = trake_search(event_vecs, store.search, store.embedding_path,
                       catalog, corpus_with_index, caption_scorer=lambda vid: None)
    baseline_settings = Settings.model_validate(corpus_with_index.model_dump())
    baseline_settings.temporal.caption_signal_weight = 0.0
    baseline = trake_search(event_vecs, store.search, store.embedding_path,
                            catalog, baseline_settings)
    assert [(c.video_id, c.ns) for c in out] == [(c.video_id, c.ns) for c in baseline]
    assert [c.score for c in out] == pytest.approx([c.score for c in baseline])


def test_normalized_caption_dead_events_contribute_zero():
    raw = {"A": np.array([[0.0, 5.0], [0.0, 1.0]]),
           "B": np.array([[0.0, 3.0]])}
    out = _normalized_caption_by_vid(
        ["A", "B"], {"A": 2, "B": 1}, lambda vid: raw.get(vid), 2)
    # event 0 is dead (all zero) → 0 everywhere, NOT the dense 0.5 neutral
    assert out["A"][:, 0].tolist() == [0.0, 0.0]
    assert out["B"][:, 0].tolist() == [0.0]
    # event 1 min-maxes over the POOL: max 5 → 1.0, min 1 → 0.0, 3 → 0.5
    assert out["A"][:, 1].tolist() == pytest.approx([1.0, 0.0])
    assert out["B"][:, 1].tolist() == pytest.approx([0.5])


def test_caption_scorer_from_signals_reads_planted_captions(corpus):
    """End-to-end factory: caption artifacts → persisted BM25 → (rows, K) matrix."""
    from cvp.data.catalog import KeyframeCatalog
    from cvp.index.text_store import TextIndexStore
    from cvp.search.text_signals import TextSignals, collect_field_documents
    from cvp.utils.io import atomic_write_json

    catalog = KeyframeCatalog(corpus)
    catalog.build()
    # NOTE: tokenize_vi FOLDS diacritics ("đầu"/"dầu" → "dau"), so the two
    # captions must share no folded token — cross-talk asserts below rely on it.
    atomic_write_json(
        corpus.paths.art("captions") / "L21_V001.json",
        {"n_to_caption": {"2": "người thợ cắt nấm tươi",
                          "5": "xe máy chạy trong mưa lớn"}},
    )
    sig = catalog.signature()
    keys, texts = collect_field_documents(corpus, catalog, "caption")
    TextIndexStore.build("caption", keys, texts, corpus.paths.artifacts_root, sig)

    ts = TextSignals(corpus, catalog)
    scorer = caption_scorer_from_signals(ts, catalog, ["cắt nấm", "xe máy"])

    m = scorer("L21_V001")
    assert m is not None and m.shape == (6, 2)
    assert m[1, 0] > 0 and m[4, 1] > 0          # n=2 ↔ event 1, n=5 ↔ event 2
    assert m[1, 1] == 0 and m[4, 0] == 0        # no cross-talk
    assert m[0, 0] == 0 and m[2, 0] == 0
    assert scorer("K01_V001") is None           # video without captions → None


def test_score_field_unknown_field_is_empty(corpus):
    from cvp.data.catalog import KeyframeCatalog
    from cvp.search.text_signals import TextSignals

    catalog = KeyframeCatalog(corpus)
    catalog.build()
    ts = TextSignals(corpus, catalog)
    assert ts.score_field("metadata", "bản tin", []) == {}  # video-keyed → not a frame field
    assert ts.score_field("nope", "bản tin", []) == {}


# ── event query variants ─────────────────────────────────────────────────────


def _variant_layout(settings: Settings):
    """Chain (n=1 → n=4) in L21_V001 visible ONLY to event 1's SECOND variant."""
    by_vid = {
        "L21_V001": np.stack([
            _basis(0), _basis(10), _basis(11), _basis(1), _basis(12), _basis(10),
        ]),
        "L21_V002": np.stack(_junk_rows(5, 10)),
        "K01_V001": np.stack(_junk_rows(4, 10)),
    }
    return _build_store(settings, "trake_var", by_vid)


def test_variants_find_chain_only_second_variant_sees(corpus_with_index):
    catalog, store = _variant_layout(corpus_with_index)
    # Legacy: one text per event — event 1's only vector (e7) matches nothing.
    legacy_vecs = np.stack([_basis(7), _basis(1)])
    legacy = trake_search(legacy_vecs, store.search, store.embedding_path,
                          catalog, corpus_with_index)
    legacy_best = legacy[0] if legacy else None

    # Variants: event 1 carries [e7 (miss), e0 (hit)], event 2 stays e1.
    variant_vecs = np.stack([_basis(7), _basis(0), _basis(1)])
    out = trake_search(variant_vecs, store.search, store.embedding_path,
                       catalog, corpus_with_index, event_variant_map=[0, 0, 1])
    assert out
    best = out[0]
    assert best.video_id == "L21_V001"
    assert best.ns == [1, 4]
    assert best.frame_idxs == [100, 400]
    if legacy_best is not None:  # provably better than the single-variant run
        assert best.score > legacy_best.score + 0.3


def test_noise_variants_do_not_change_the_legacy_winner(corpus_with_index):
    """Extra orthogonal variants (match nothing) must leave the ranking as-is."""
    catalog, store = _variant_layout(corpus_with_index)
    clean_vecs = np.stack([_basis(0), _basis(1)])
    legacy = trake_search(clean_vecs, store.search, store.embedding_path,
                          catalog, corpus_with_index)
    noisy_vecs = np.stack([_basis(0), _basis(15), _basis(1), _basis(15)])
    noisy = trake_search(noisy_vecs, store.search, store.embedding_path,
                         catalog, corpus_with_index,
                         event_variant_map=[0, 0, 1, 1])
    assert legacy and noisy
    assert (noisy[0].video_id, noisy[0].ns) == (legacy[0].video_id, legacy[0].ns)
    assert noisy[0].score == pytest.approx(legacy[0].score)


def test_variant_map_with_gap_does_not_crash(corpus_with_index):
    """A variant map skipping an event index (defensive: malformed caller) must
    yield empty evidence for that event, not crash np.concatenate([])."""
    catalog, store = _variant_layout(corpus_with_index)
    vecs = np.stack([_basis(0), _basis(1)])
    out = trake_search(vecs, store.search, store.embedding_path,
                       catalog, corpus_with_index,
                       event_variant_map=[0, 2])   # event 1 has NO variant rows
    for c in out:  # chains still span 3 events, all strictly increasing
        assert len(c.frame_idxs) == 3
        assert all(b > a for a, b in zip(c.frame_idxs, c.frame_idxs[1:]))


def test_identity_variant_map_equals_legacy_call(corpus_with_index):
    catalog, store = _variant_layout(corpus_with_index)
    vecs = np.stack([_basis(0), _basis(1)])
    legacy = trake_search(vecs, store.search, store.embedding_path,
                          catalog, corpus_with_index)
    mapped = trake_search(vecs, store.search, store.embedding_path,
                          catalog, corpus_with_index, event_variant_map=[0, 1])
    assert [(c.video_id, c.ns) for c in mapped] == [(c.video_id, c.ns) for c in legacy]
    assert [c.score for c in mapped] == pytest.approx([c.score for c in legacy])


def test_ensemble_member_variants_rescue_the_chain(corpus_with_index):
    """Variant maps flow through the ensemble path (_pooled_event_hits +
    _ensemble_video_sims): member A's second variant finds the chain even
    though member B sees nothing."""
    from cvp.index.store import IndexStore

    catalog, store_a = _variant_layout(corpus_with_index)
    blind = {vid: np.stack(_junk_rows(n, 13))
             for vid, n in (("L21_V001", 6), ("L21_V002", 5), ("K01_V001", 4))}
    store_b = IndexStore(corpus_with_index, "trake_var_b")
    for vid, vecs in blind.items():
        store_b.embedding_path(vid).parent.mkdir(parents=True, exist_ok=True)
        np.save(store_b.embedding_path(vid), vecs.astype(np.float32))
    store_b.build(catalog)

    out = trake_search(
        np.stack([_basis(7), _basis(1)]), store_a.search, store_a.embedding_path,
        catalog, corpus_with_index,
        event_vecs_by_member={
            "seer": np.stack([_basis(7), _basis(0), _basis(1)]),
            "blind": np.stack([_basis(7), _basis(1)]),
        },
        variant_maps_by_member={"seer": [0, 0, 1], "blind": [0, 1]},
        member_weights={"seer": 0.6, "blind": 0.4},
        stores_by_member={"seer": store_a, "blind": store_b},
    )
    assert out
    assert out[0].video_id == "L21_V001"
    assert out[0].ns == [1, 4]


# ── pool-context vectors ─────────────────────────────────────────────────────


def test_pool_vectors_rescue_video_the_event_vectors_never_pool(corpus_with_index):
    """max_videos=2: the true video ranks 3rd on event-vec pooling and is cut;
    pool vectors (header-context encodings) put it back in the pool while the
    DP still scores the bare event vectors."""
    # Distinct junk bases per video: the pool queries below are STORED vectors,
    # so a shared junk axis would leak similarity across videos.
    true_e1, true_e2 = _mix(0.6, 0, 12), _mix(0.6, 1, 13)
    by_vid = {
        "L21_V001": np.stack([
            _basis(8), true_e1, _basis(9), _basis(10), true_e2, _basis(11),
        ]),
        "L21_V002": np.stack([
            _mix(0.9, 0, 14), _basis(8), _mix(0.9, 1, 14), _basis(9), _basis(10),
        ]),
        "K01_V001": np.stack([
            _mix(0.85, 0, 15), _basis(8), _mix(0.85, 1, 15), _basis(9),
        ]),
    }
    catalog, store = _build_store(corpus_with_index, "trake_pool", by_vid)
    corpus_with_index.temporal.max_videos = 2
    event_vecs = np.stack([_basis(0), _basis(1)])

    without = trake_search(event_vecs, store.search, store.embedding_path,
                           catalog, corpus_with_index)
    assert without and all(c.video_id != "L21_V001" for c in without)

    pool_vecs = np.stack([true_e1, true_e2])  # what "<header>. <event>" would hit
    with_pool = trake_search(event_vecs, store.search, store.embedding_path,
                             catalog, corpus_with_index, pool_event_vecs=pool_vecs)
    assert any(c.video_id == "L21_V001" for c in with_pool)


# ── jitter submit strategy ───────────────────────────────────────────────────


def test_jitter_frame_variants_properties():
    fidx = np.array([100, 200, 300, 400, 500, 600], dtype=np.int64)
    variants = jitter_frame_variants([1, 3], fidx)
    assert variants, "no variants generated"
    assert variants[0] == [150, 350]            # all-early comes first
    assert [200, 400] not in variants           # base tuple excluded
    for v in variants:
        assert all(b > a for a, b in zip(v, v[1:])), f"not increasing: {v}"
        assert v[0] >= 0
    assert len({tuple(v) for v in variants}) == len(variants)  # deduped
    assert [150, 400] in variants and [200, 350] in variants   # early singles
    assert [250, 450] in variants                              # all-late
    assert [100, 400] in variants and [200, 500] in variants   # prev/next singles


def test_jitter_frame_variants_boundary_rows_do_not_crash():
    fidx = np.array([100, 200, 300], dtype=np.int64)
    variants = jitter_frame_variants([0, 2], fidx)  # first/last keyframes
    for v in variants:
        assert all(b > a for a, b in zip(v, v[1:]))
        assert v[0] >= 0
    # adjacent rows: colliding variants must be dropped, not emitted broken
    for v in jitter_frame_variants([1, 2], fidx):
        assert all(b > a for a, b in zip(v, v[1:]))


def _fake_candidates(vid: str, n_rows: int, base_score: float,
                     ns=(2, 5)) -> list[TrakeCandidate]:
    """Distinct legacy-style candidates whose frames stay MULTIPLES of 100 —
    the tests detect jitter rows by ``frame % 100 != 0`` (mid-keyframe)."""
    out = []
    for i in range(n_rows):
        out.append(TrakeCandidate(
            video_id=vid, ns=list(ns),
            frame_idxs=[ns[0] * 100, (ns[1] + i) * 100],
            pts_times=[ns[0] * 4.0, ns[1] * 4.0],
            score=base_score - 0.01 * i, per_event=[base_score, base_score],
        ))
    return out


def _rows_df(n_frames: int):
    import pandas as pd

    return pd.DataFrame({
        "n": np.arange(1, n_frames + 1, dtype=np.int64),
        "frame_idx": np.arange(1, n_frames + 1, dtype=np.int64) * 100,
        "pts_time": np.arange(1, n_frames + 1, dtype=np.float64) * 4.0,
        "global_id": np.arange(n_frames, dtype=np.int64),
    })


def test_expand_candidates_jitter_layout_and_dedupe():
    ranked = (_fake_candidates("L21_V001", 5, 0.9)
              + _fake_candidates("L21_V002", 4, 0.8, ns=(1, 3))
              + _fake_candidates("K01_V001", 3, 0.7))
    rows_by_vid = {"L21_V001": _rows_df(6), "L21_V002": _rows_df(5), "K01_V001": _rows_df(4)}
    cfg = Settings.model_validate({"temporal": {"jitter_videos": 2}}).temporal

    out = expand_candidates_jitter(ranked, rows_by_vid, cfg, max_results=100)
    # Head preserved verbatim → R@1/R@5 can never regress vs legacy.
    assert [(c.video_id, tuple(c.frame_idxs)) for c in out[:JITTER_HEAD]] == \
           [(c.video_id, tuple(c.frame_idxs)) for c in ranked[:JITTER_HEAD]]
    # Jitter blocks contain BETWEEN-keyframe frames for the top 2 videos only.
    jitter_rows = [c for c in out if any(f % 100 != 0 for f in c.frame_idxs)]
    assert jitter_rows and {c.video_id for c in jitter_rows} == {"L21_V001", "L21_V002"}
    # No duplicates, everything strictly increasing, legacy tail still present.
    keys = [(c.video_id, tuple(c.frame_idxs)) for c in out]
    assert len(keys) == len(set(keys))
    for c in out:
        assert all(b > a for a, b in zip(c.frame_idxs, c.frame_idxs[1:]))
    assert any(c.video_id == "K01_V001" for c in out)
    assert len(out) > len(ranked)

    capped = expand_candidates_jitter(ranked, rows_by_vid, cfg, max_results=10)
    assert len(capped) == 10


def test_trake_search_jitter_keeps_top1_and_densifies(corpus_with_index):
    catalog, store = _variant_layout(corpus_with_index)
    event_vecs = np.stack([_basis(0), _basis(1)])
    legacy = trake_search(event_vecs, store.search, store.embedding_path,
                          catalog, corpus_with_index)

    corpus_with_index.temporal.submit_strategy = "jitter"
    jittered = trake_search(event_vecs, store.search, store.embedding_path,
                            catalog, corpus_with_index)
    assert legacy and jittered
    # top-1 row identical → R@1 unchanged by construction
    assert (jittered[0].video_id, jittered[0].frame_idxs) == \
           (legacy[0].video_id, legacy[0].frame_idxs)
    assert len(jittered) > len(legacy)
    assert any(f % 100 != 0 for c in jittered for f in c.frame_idxs)
    for c in jittered:
        assert all(b > a for a, b in zip(c.frame_idxs, c.frame_idxs[1:]))
    keys = [(c.video_id, tuple(c.frame_idxs)) for c in jittered]
    assert len(keys) == len(set(keys))


# ── header split + runner wiring ─────────────────────────────────────────────


ORGANISER_LINES = [
    "Đoạn video múa lân, tìm các sự kiện sau:",
    "E1: Lân quay vòng trên cột",
    "E2: Bốn chân chạm đất",
]


def test_split_trake_query_extracts_header_and_events():
    from cvp.pipeline.run_queries import split_trake_query

    header, events = split_trake_query(ORGANISER_LINES)
    assert header == "Đoạn video múa lân, tìm các sự kiện sau"
    assert events == ["Lân quay vòng trên cột", "Bốn chân chạm đất"]


def test_split_trake_query_plain_format_has_no_header():
    from cvp.pipeline.run_queries import split_trake_query

    plain = ["sự kiện một", "sự kiện hai"]
    assert split_trake_query(plain) == ("", plain)


def test_parse_trake_events_unchanged_by_refactor():
    from cvp.pipeline.run_queries import parse_trake_events

    assert parse_trake_events(ORGANISER_LINES) == \
        ["Lân quay vòng trên cột", "Bốn chân chạm đất"]
    assert parse_trake_events(ORGANISER_LINES, prepend_context=True) == [
        "Đoạn video múa lân, tìm các sự kiện sau. Lân quay vòng trên cột",
        "Đoạn video múa lân, tìm các sự kiện sau. Bốn chân chạm đất",
    ]
    plain = ["một", "hai"]
    assert parse_trake_events(plain) == plain


class _Cand:
    video_id = "L21_V001"
    frame_idxs = [100, 200]
    pts_times = [4.0, 8.0]


class _PoolContextEngine:
    """Stub engine WITH the pool_context knob on — must receive the header."""

    def __init__(self):
        self.settings = SimpleNamespace(temporal=SimpleNamespace(
            event_context="none", pool_context="prepend"))
        self.seen: dict = {}

    def search_trake(self, events, max_results=100, *, context=None):
        self.seen = {"events": list(events), "context": context}
        return [_Cand()]


class _LegacyEngine:
    """Old-style stub WITHOUT the context kwarg — default config must not
    pass one (backwards compatibility with every existing engine stub)."""

    settings = SimpleNamespace(temporal=SimpleNamespace(
        event_context="none", pool_context="none"))

    def __init__(self):
        self.calls = 0

    def search_trake(self, events, max_results=100):
        self.calls += 1
        return [_Cand()]


def _write_query(tmp_path: Path) -> Path:
    qf = tmp_path / "query-p9-1-trake.txt"
    qf.write_text("\n".join(ORGANISER_LINES), encoding="utf-8")
    return qf


def test_run_query_file_passes_header_only_when_knob_is_on(tmp_path):
    from cvp.pipeline.run_queries import run_query_file

    engine = _PoolContextEngine()
    out = run_query_file(engine, _write_query(tmp_path), tmp_path)
    assert out is not None
    assert engine.seen["events"] == ["Lân quay vòng trên cột", "Bốn chân chạm đất"]
    assert engine.seen["context"] == "Đoạn video múa lân, tìm các sự kiện sau"

    legacy = _LegacyEngine()
    out2 = run_query_file(legacy, _write_query(tmp_path), tmp_path)
    assert out2 is not None and legacy.calls == 1  # no kwargs → old stubs fine
