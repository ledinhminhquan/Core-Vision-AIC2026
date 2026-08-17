"""Round-13 preflight regressions — hardening the UNEXECUTED nb01 path.

  R13-1  EOFError (0-byte npy) joins every np.load guard: resume re-embeds
         instead of crashing (store.missing_videos, embedder dst-check,
         temporal, build_dataset)
  R13-2  read_json returns the caller's default on a torn/0-byte marker
  R13-3  aux sweeps fail LOUD on systemic breakage (first-video all-fail /
         5 consecutive ASR failures) instead of writing empty artifacts
  R13-4  nb01 cell 9 isolates each stage; cell 11 re-raises collected errors
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


REPO = Path(__file__).resolve().parents[1]


# ── R13-1 · EOFError guards ──────────────────────────────────────────────────
def test_missing_videos_treats_empty_npy_as_missing(corpus):
    from cvp.data.catalog import KeyframeCatalog
    from cvp.index.store import IndexStore

    catalog = KeyframeCatalog(corpus)
    df = catalog.build()
    store = IndexStore(corpus, "fake")
    store.embed_dir.mkdir(parents=True, exist_ok=True)
    counts = df.groupby("video_id")["n"].count()
    vids = sorted(map(str, counts.index))
    for vid in vids:
        np.save(store.embedding_path(vid),
                np.ones((int(counts[vid]), 8), dtype=np.float32))
    assert store.missing_videos(catalog) == []
    store.embedding_path(vids[0]).write_bytes(b"")      # Drive-FUSE corruption
    assert store.missing_videos(catalog) == [vids[0]]   # re-embed, not crash


def test_ingest_dst_check_survives_empty_dst(tmp_path, corpus):
    from cvp.data.catalog import KeyframeCatalog
    from cvp.index.embedder import ingest_provided_features
    from cvp.index.store import IndexStore

    catalog = KeyframeCatalog(corpus)
    df = catalog.build()
    feat_dir = Path(str(corpus.paths.data_root)) / corpus.paths.clip_features_dir
    feat_dir.mkdir(parents=True, exist_ok=True)
    counts = df.groupby("video_id")["n"].count()
    for vid in map(str, counts.index):
        np.save(feat_dir / f"{vid}.npy",
                np.random.rand(int(counts[vid]), 8).astype(np.float32))
    store = IndexStore(corpus, "provided_clip32")
    store.embed_dir.mkdir(parents=True, exist_ok=True)
    vid0 = sorted(map(str, counts.index))[0]
    store.embedding_path(vid0).write_bytes(b"")         # corrupt already-ingested dst
    done = ingest_provided_features(corpus, catalog)    # must NOT raise EOFError
    assert done == len(counts)                           # vid0 re-ingested too
    assert np.load(store.embedding_path(vid0)).shape[0] == int(counts[vid0])


def test_all_np_load_guards_include_eoferror():
    for rel in ("src/cvp/index/store.py", "src/cvp/index/embedder.py",
                "src/cvp/search/temporal.py", "src/cvp/training/build_dataset.py"):
        src = (REPO / rel).read_text(encoding="utf-8")
        assert "except (OSError, ValueError)" not in src, rel
        assert "EOFError" in src, rel


# ── R13-2 · torn marker json → default ───────────────────────────────────────
def test_read_json_default_on_torn_marker(tmp_path):
    from cvp.utils.io import read_json

    p = tmp_path / "meta.json"
    p.write_bytes(b"")                                   # 0-byte marker
    assert read_json(p, default={}) == {}
    p.write_text('{"half":', encoding="utf-8")           # torn write
    assert read_json(p, default={}) == {}
    with pytest.raises(json.JSONDecodeError):
        read_json(p)                                     # no default → still loud


# ── R13-3 · systemic-failure detection in the aux sweeps ─────────────────────
def test_ocr_systemic_failure_raises(corpus, monkeypatch):
    from cvp.auxindex import ocr
    from cvp.data.catalog import KeyframeCatalog

    catalog = KeyframeCatalog(corpus)
    catalog.build()

    class _Boom:
        def read(self, path):
            raise RuntimeError("v5 drift")

    monkeypatch.setattr(ocr, "_build_engine", lambda s: _Boom())
    with pytest.raises(RuntimeError, match="systemic failure"):
        ocr.ocr_all_keyframes(corpus, catalog)


def test_captioner_systemic_failure_raises(corpus, monkeypatch):
    from cvp.auxindex import captioner
    from cvp.data.catalog import KeyframeCatalog

    catalog = KeyframeCatalog(corpus)
    catalog.build()

    class _Boom:
        def caption(self, path):
            raise RuntimeError("v5 drift")

    monkeypatch.setattr(captioner, "VinternCaptioner", lambda s: _Boom())
    with pytest.raises(RuntimeError, match="systemic failure"):
        captioner.caption_all_keyframes(corpus, catalog)


def test_asr_five_consecutive_failures_raise():
    src = (REPO / "src" / "cvp" / "auxindex" / "asr.py").read_text(encoding="utf-8")
    assert "consecutive_failures >= 5" in src and "systemic failure" in src


# ── R13-4 · nb01 stage isolation + final surfacing ───────────────────────────
def test_builder_isolates_aux_stages_and_surfaces_errors():
    src = (REPO / "notebooks" / "_build_notebooks.py").read_text(encoding="utf-8")
    assert "_aux_errors" in src and "def _aux_stage(" in src
    assert 'globals().get("_aux_errors")' in src          # cell 11 re-raise
    assert "aux stage đã FAIL" in src
