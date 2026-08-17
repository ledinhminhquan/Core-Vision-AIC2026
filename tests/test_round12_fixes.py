"""Round-12 regressions — second live nb01 lesson: Drive-FUSE empty files can
hit ANY copied artifact, not just map csvs.

  R12-1  ingest_provided_features skips a corrupt/empty .npy with a warning
         instead of dying with EOFError after hours of GPU work
  R12-2  nb01 cell 6 heals ALL small-file artifacts (map csvs, clip-features
         npys, media-info, objects) from their source zips
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[1]


def test_ingest_skips_corrupt_npy(tmp_path, corpus):
    from cvp.data.catalog import KeyframeCatalog
    from cvp.index.embedder import ingest_provided_features

    catalog = KeyframeCatalog(corpus)
    df = catalog.build()
    feat_dir = Path(str(corpus.paths.data_root)) / corpus.paths.clip_features_dir
    feat_dir.mkdir(parents=True, exist_ok=True)
    vids = sorted(map(str, df["video_id"].unique()))
    counts = df.groupby("video_id")["n"].count()
    # first video: EMPTY npy (the Drive-FUSE corruption); rest: valid features
    (feat_dir / f"{vids[0]}.npy").write_bytes(b"")
    for vid in vids[1:]:
        arr = np.random.rand(int(counts[vid]), 8).astype(np.float32)
        np.save(feat_dir / f"{vid}.npy", arr)
    done = ingest_provided_features(corpus, catalog)   # must NOT raise EOFError
    assert done == len(vids) - 1


def test_builder_heals_all_artifact_families():
    src = (REPO / "notebooks" / "_build_notebooks.py").read_text(encoding="utf-8")
    assert "_HEAL_SPECS" in src
    for sub in ("map-keyframes", "clip-features-32", "media-info", "objects"):
        assert f'("{sub}"' in src, sub
    assert "_bad_npy" in src and "_bad_csv" in src


def test_embedder_guard_message_mentions_zip():
    src = (REPO / "src" / "cvp" / "index" / "embedder.py").read_text(encoding="utf-8")
    assert "corrupt/empty" in src and "EOFError" in src
