"""Persisted BM25 text index: parity with in-memory rank-bm25, candidate restriction."""

from __future__ import annotations

import pytest

from cvf.data.catalog import KeyframeCatalog
from cvf.index.text_store import TextIndexStore
from cvf.search.text_signals import ALL_FIELDS, TextSignals, collect_field_documents
from cvf.utils.io import atomic_write_json

QUERIES = [
    "bản tin 60 giây buổi sáng",
    "người phụ nữ nấu ăn trong bếp",
    "phóng viên tường thuật trực tiếp",
    "tin tức thời sự",
    "60 60 giây",              # duplicated query token — BM25 counts per occurrence
    "hoàn toàn không liên quan xyz",
]


def _plant_artifacts(corpus) -> None:
    """OCR + captions + ASR docs over the synthetic 3-video corpus."""
    atomic_write_json(
        corpus.paths.art("ocr") / "L21_V001.json",
        {"n_to_text": {"3": "BẢN TIN 60 GIÂY chào buổi sáng", "5": "thời tiết hà nội hôm nay"}},
    )
    atomic_write_json(
        corpus.paths.art("ocr") / "K01_V001.json",
        {"n_to_text": {"1": "bản tin thể thao"}},
    )
    atomic_write_json(
        corpus.paths.art("captions") / "L21_V002.json",
        {"n_to_caption": {"2": "một người phụ nữ nấu ăn trong bếp", "4": "món ngon trên bàn ăn"}},
    )
    # pts_time of frame n is 4·n seconds (frame_idx = 100·n at 25 fps).
    atomic_write_json(
        corpus.paths.art("asr") / "L21_V001.json",
        {"segments": [
            {"start": 6.0, "end": 10.5, "text": "phóng viên tường thuật trực tiếp"},
            {"start": 18.0, "end": 22.0, "text": "dự báo thời tiết ngày mai"},
        ]},
    )


@pytest.fixture()
def corpus_txt(corpus):
    catalog = KeyframeCatalog(corpus)
    catalog.build()
    _plant_artifacts(corpus)
    return corpus, catalog


def _build_index(corpus, catalog, signature: str | None = None) -> dict:
    sig = signature or catalog.signature()
    built = {}
    for field in ALL_FIELDS:
        collected = collect_field_documents(corpus, catalog, field)
        if collected is None:
            continue
        keys, texts = collected
        built[field] = TextIndexStore.build(field, keys, texts, corpus.paths.artifacts_root, sig)
    return built


# ── parity ───────────────────────────────────────────────────────────────────


def test_persisted_scores_match_inmemory(corpus_txt):
    corpus, catalog = corpus_txt
    candidates = [catalog.ref(g) for g in range(len(catalog))]

    ts_mem = TextSignals(corpus, catalog)
    assert ts_mem._persisted_enabled() is False  # nothing persisted yet

    built = _build_index(corpus, catalog)
    assert set(built) == {"ocr", "asr", "caption", "metadata"}

    ts_per = TextSignals(corpus, catalog)
    assert ts_per._persisted_enabled() is True

    for q in QUERIES:
        mem = ts_mem.score_candidates(q, candidates)
        per = ts_per.score_candidates(q, candidates)
        # same fields fire (empty dicts tolerated on either side)
        assert {k for k, v in mem.items() if v} == {k for k, v in per.items() if v}
        for field, mem_scores in mem.items():
            per_scores = per.get(field, {})
            assert set(mem_scores) == set(per_scores), (q, field)
            for gid, s in mem_scores.items():
                assert abs(s - per_scores[gid]) < 1e-6, (q, field, gid)
    assert ts_per._fields == {}  # persisted path never built an in-memory corpus


def test_persisted_roundtrip_and_missing_field(corpus_txt):
    corpus, catalog = corpus_txt
    built = _build_index(corpus, catalog)
    loaded = TextIndexStore.load("ocr", corpus.paths.artifacts_root)
    assert loaded is not None
    assert loaded.n_docs == built["ocr"].n_docs == 3
    assert loaded.avgdl == pytest.approx(built["ocr"].avgdl)
    assert TextIndexStore.load("nonexistent", corpus.paths.artifacts_root) is None
    meta = TextIndexStore.read_meta(corpus.paths.artifacts_root)
    assert meta["signature"] == catalog.signature()
    assert meta["fields"]["ocr"]["n_docs"] == 3


# ── zero-overlap / sparsity ──────────────────────────────────────────────────


def test_zero_overlap_scores_zero(corpus_txt):
    corpus, catalog = corpus_txt
    _build_index(corpus, catalog)
    ts = TextSignals(corpus, catalog)
    candidates = [catalog.ref(g) for g in range(len(catalog))]

    scores = ts.score_candidates("bản tin 60 giây", candidates)
    ocr = scores["ocr"]
    # L21_V001 n=5 has OCR text sharing NO query term → absent (score 0)
    gid_n5 = int(catalog.video_rows("L21_V001").iloc[4]["global_id"])
    assert gid_n5 not in ocr
    # L21_V001 n=3 matches all four terms and must beat K01_V001 n=1 (two terms)
    gid_n3 = int(catalog.video_rows("L21_V001").iloc[2]["global_id"])
    gid_k1 = int(catalog.video_rows("K01_V001").iloc[0]["global_id"])
    assert ocr[gid_n3] > ocr[gid_k1] > 0

    # query with zero corpus overlap → nothing fires anywhere
    none = ts.score_candidates("zzz qqq www", candidates)
    assert all(not v for v in none.values())


# ── staleness / fallback ─────────────────────────────────────────────────────


def test_stale_signature_falls_back_to_inmemory(corpus_txt):
    corpus, catalog = corpus_txt
    _build_index(corpus, catalog, signature="0123456789abcdef")  # wrong corpus
    ts = TextSignals(corpus, catalog)
    assert ts._persisted_enabled() is False
    candidates = [catalog.ref(g) for g in range(len(catalog))]
    scores = ts.score_candidates("bản tin 60 giây", candidates)
    assert scores["ocr"]                      # fallback still delivers signal
    assert "ocr" in ts._fields                # ... via the in-memory path
    assert ts._persisted == {}                # persisted files never loaded


def test_field_missing_from_persisted_build_falls_back(corpus_txt):
    corpus, catalog = corpus_txt
    sig = catalog.signature()
    for field in ("ocr", "metadata"):  # older build: captions/asr artifacts came later
        keys, texts = collect_field_documents(corpus, catalog, field)
        TextIndexStore.build(field, keys, texts, corpus.paths.artifacts_root, sig)
    ts = TextSignals(corpus, catalog)
    assert ts._persisted_enabled() is True
    candidates = [catalog.ref(g) for g in range(len(catalog))]
    scores = ts.score_candidates("người phụ nữ nấu ăn", candidates)
    assert scores["caption"]           # served by the per-field in-memory fallback
    assert "caption" in ts._fields
    assert "ocr" not in ts._fields     # persisted fields stayed persisted


def test_meta_reset_on_signature_change(corpus_txt):
    corpus, catalog = corpus_txt
    keys, texts = collect_field_documents(corpus, catalog, "ocr")
    TextIndexStore.build("ocr", keys, texts, corpus.paths.artifacts_root, "sig_a")
    TextIndexStore.build("metadata", *collect_field_documents(corpus, catalog, "metadata"),
                         corpus.paths.artifacts_root, "sig_b")
    meta = TextIndexStore.read_meta(corpus.paths.artifacts_root)
    assert meta["signature"] == "sig_b"
    assert set(meta["fields"]) == {"metadata"}  # sig_a fields dropped from the manifest


# ── candidate restriction ────────────────────────────────────────────────────


class _Recorder(dict):
    def __init__(self, base):
        super().__init__(base)
        self.touched = set()

    def get(self, k, default=None):
        self.touched.add(k)
        return super().get(k, default)


def test_persisted_path_touches_only_candidates(corpus_txt, monkeypatch):
    corpus, catalog = corpus_txt
    _build_index(corpus, catalog)
    ts = TextSignals(corpus, catalog)

    # persisted path must never re-collect documents from artifact JSONs
    import cvf.search.text_signals as tsig

    def _boom(*_a, **_k):
        raise AssertionError("corpus scan attempted on the persisted path")

    monkeypatch.setattr(tsig, "collect_field_documents", _boom)

    cand_gids = [
        int(catalog.video_rows("L21_V001").iloc[2]["global_id"]),
        int(catalog.video_rows("K01_V001").iloc[0]["global_id"]),
    ]
    candidates = [catalog.ref(g) for g in cand_gids]
    scores = ts.score_candidates("bản tin 60 giây", candidates)
    assert scores["ocr"]
    assert ts._fields == {}  # no in-memory corpus was built

    # doc-stat lookups hit candidate ids (and candidate video ids) only
    recorders = {}
    for name, f in ts._persisted.items():
        if f is not None and f.n_docs:
            recorders[name] = f.docs = _Recorder(f.docs)
    ts.score_candidates("bản tin 60 giây thời tiết", candidates)
    allowed = set(cand_gids) | {c.video_id for c in candidates}
    for name, rec in recorders.items():
        assert rec.touched <= allowed, name


def test_empty_built_field_skipped_without_rescan(corpus, monkeypatch):
    catalog = KeyframeCatalog(corpus)
    catalog.build()
    sig = catalog.signature()
    root = corpus.paths.artifacts_root
    # ocr built over token-less docs; asr/caption built empty; metadata real
    TextIndexStore.build("ocr", [0, 1], ["!!!", "..."], root, sig)
    TextIndexStore.build("asr", [], [], root, sig)
    TextIndexStore.build("caption", [], [], root, sig)
    TextIndexStore.build("metadata", *collect_field_documents(corpus, catalog, "metadata"), root, sig)

    ts = TextSignals(corpus, catalog)
    assert ts._persisted_enabled() is True

    import cvf.search.text_signals as tsig
    monkeypatch.setattr(
        tsig, "collect_field_documents",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("rescan attempted")),
    )
    candidates = [catalog.ref(g) for g in range(len(catalog))]
    scores = ts.score_candidates("tin tức thời sự", candidates)
    assert set(scores) <= {"metadata"}  # empty fields skipped silently, no rescan
    assert scores.get("metadata")
