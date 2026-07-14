
from cvf.config import load_settings
from cvf.data.catalog import KeyframeCatalog
from cvf.search.object_filter import ObjectBooster
from cvf.search.text_signals import TextSignals
from cvf.utils.io import atomic_write_json


def test_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("CVF_SEARCH__TOPK", "42")
    monkeypatch.setenv("CVF_SEARCH__WEIGHTS__OCR", "0.9")
    monkeypatch.setenv("CVF_PATHS__DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("CVF_SETTINGS", str(tmp_path / "nonexistent.yaml"))
    s = load_settings()
    assert s.search.topk == 42
    assert abs(s.search.weights.ocr - 0.9) < 1e-9
    assert str(tmp_path) in str(s.paths.data_root)


def test_object_constraint_parsing(corpus):
    booster = ObjectBooster(corpus)
    cons = booster.parse("có 2 người ở bên trái và một chiếc xe máy")
    entities = {c.entity for c in cons}
    assert "Person" in entities and "Motorcycle" in entities
    person = next(c for c in cons if c.entity == "Person")
    assert person.count == 2
    assert person.position == "left"


def test_object_scoring(corpus):
    catalog = KeyframeCatalog(corpus)
    catalog.build()
    booster = ObjectBooster(corpus)
    cons = booster.parse("2 người")  # synthetic objects: 2 persons per frame
    ref = catalog.ref(0)
    assert booster.score(cons, ref) == 1.0


def test_text_signals_ocr_and_metadata(corpus):
    catalog = KeyframeCatalog(corpus)
    catalog.build()
    # plant an OCR artifact for one frame
    atomic_write_json(
        corpus.paths.art("ocr") / "L21_V001.json",
        {"n_to_text": {"3": "BẢN TIN 60 GIÂY chào buổi sáng"}},
    )
    ts = TextSignals(corpus, catalog)
    candidates = [catalog.ref(g) for g in range(15)]
    scores = ts.score_candidates("bản tin 60 giây", candidates)
    assert "ocr" in scores
    gid = catalog.video_rows("L21_V001").iloc[2]["global_id"]
    assert int(gid) in scores["ocr"]
    assert scores["ocr"][int(gid)] > 0
    # metadata: only L21_V001's title mentions "bản tin 60 giây"
    assert "metadata" in scores and len(scores["metadata"]) > 0
    meta_vids = {catalog.ref(g).video_id for g in scores["metadata"]}
    assert "L21_V001" in meta_vids


def test_text_signals_persisted_index_path(corpus):
    """Same scenario as above, but scored from the persisted text_index."""
    from cvf.index.text_store import TextIndexStore
    from cvf.search.text_signals import ALL_FIELDS, collect_field_documents

    catalog = KeyframeCatalog(corpus)
    catalog.build()
    atomic_write_json(
        corpus.paths.art("ocr") / "L21_V001.json",
        {"n_to_text": {"3": "BẢN TIN 60 GIÂY chào buổi sáng"}},
    )
    sig = catalog.signature()
    for field in ALL_FIELDS:
        collected = collect_field_documents(corpus, catalog, field)
        if collected is not None:
            TextIndexStore.build(field, collected[0], collected[1], corpus.paths.artifacts_root, sig)

    ts = TextSignals(corpus, catalog)
    assert ts._persisted_enabled() is True
    candidates = [catalog.ref(g) for g in range(15)]
    scores = ts.score_candidates("bản tin 60 giây", candidates)
    gid = int(catalog.video_rows("L21_V001").iloc[2]["global_id"])
    assert scores["ocr"][gid] > 0
    assert "metadata" in scores and len(scores["metadata"]) > 0
    # ocr/metadata came from the persisted index; asr/caption artifacts don't
    # exist here, so their fallback probe cached None — no BM25 corpus built.
    assert all(v is None for v in ts._fields.values())
    assert "ocr" not in ts._fields and "metadata" not in ts._fields
