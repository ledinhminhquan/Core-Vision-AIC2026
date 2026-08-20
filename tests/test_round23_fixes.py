"""Round-23 — nb02/nb03 preflight (54-agent workflow) against the REAL nb01
artifacts: 4 confirmed defects closed before the first competition round.

  R23-1  objects.parquet was never built → ObjectBooster fell back to per-file
         Drive-FUSE reads per query. nb01 cell 10 + nb03 now build it once.
  R23-2  engine model_tag mismatch (torn cache → silent checkpoint fallback)
         now DISABLES the lane loudly at startup instead of warning + failing
         every query quietly.
  R23-3  siglip2 loads survive a torn shared cache via local re-download.
  R23-4  trainer resume falls back to older checkpoints on torn files and
         restores virgin adapter weights on total failure.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def test_engine_disables_mismatched_lane_loudly():
    src = (REPO / "src" / "cvp" / "search" / "engine.py").read_text(encoding="utf-8")
    frag = src.split("built_tag and live_tag and built_tag != live_tag")[1]
    assert "continue" in frag.split("self.members.append")[0]   # lane skipped
    assert "model_tag mismatch" in frag                          # listed in errors
    assert "lane DISABLED" in frag                               # loud log


def test_engine_mismatch_lane_skipped_functionally(corpus, monkeypatch):
    import numpy as np

    from cvp.data.catalog import KeyframeCatalog
    from cvp.index.store import IndexStore
    from cvp.search import engine as eng

    catalog = KeyframeCatalog(corpus)
    df = catalog.build()
    counts = df.groupby("video_id")["n"].count()
    store = IndexStore(corpus, "fake")
    store.embed_dir.mkdir(parents=True, exist_ok=True)
    for vid in map(str, counts.index):
        np.save(store.embedding_path(vid),
                np.ones((int(counts[vid]), 8), dtype=np.float32))
    store.build(catalog, model_tag="checkpoint-A")

    class _WrongModel:
        key = "fake"
        model_tag = "checkpoint-B"                               # drifted load

    monkeypatch.setattr(eng, "build_model", lambda s, n: _WrongModel())
    monkeypatch.setattr(eng, "index_key_for", lambda n: "fake")
    monkeypatch.setattr(corpus.embedding, "model", "fake", raising=False)
    with pytest.raises(RuntimeError, match="No ensemble member"):
        eng.SearchEngine(corpus)                                 # sole lane dropped


def test_siglip2_uses_resilient_loader():
    src = (REPO / "src" / "cvp" / "models" / "siglip2.py").read_text(encoding="utf-8")
    assert "resilient_from_pretrained" in src


def test_resilient_from_pretrained_falls_back_to_local(monkeypatch, tmp_path):
    pytest.importorskip("huggingface_hub")
    import huggingface_hub

    from cvp.models.hf_compat import resilient_from_pretrained

    calls = []

    def load(target):
        calls.append(target)
        if target == "google/siglip2":
            raise OSError("torn safetensors header")
        return f"loaded:{target}"

    monkeypatch.setattr(huggingface_hub, "snapshot_download",
                        lambda mid, cache_dir=None: str(tmp_path / "snap"),
                        raising=False)
    out = resilient_from_pretrained(load, "google/siglip2")
    assert out == f"loaded:{tmp_path / 'snap'}"
    assert calls == ["google/siglip2", str(tmp_path / "snap")]


def test_notebooks_build_objects_parquet():
    src = (REPO / "notebooks" / "_build_notebooks.py").read_text(encoding="utf-8")
    assert src.count("build_objects_index(settings, catalog)") >= 2   # nb01 + nb03
    assert "NB3_OBJECTS" in src


def test_trainer_resume_falls_back_and_restores_virgin():
    src = (REPO / "src" / "cvp" / "training" / "lit_trainer.py").read_text(encoding="utf-8")
    assert "for ckpt in reversed(dirs):" in src
    assert "def _resume_from" in src
    assert "_virgin" in src                                       # fresh means fresh
