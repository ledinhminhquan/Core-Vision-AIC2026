"""Shared fixtures: a tiny synthetic AIC-style corpus (no GPU, no network)."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from cvp.config import Settings  # noqa: E402

DIM = 16
VIDEOS = {"L21_V001": 6, "L21_V002": 5, "K01_V001": 4}


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


@pytest.fixture()
def corpus(tmp_path: Path) -> Settings:
    """data/ + artifacts/ for 3 videos, 15 keyframes, fake 16-d embeddings."""
    data = tmp_path / "data"
    art = tmp_path / "artifacts"
    settings = Settings.model_validate({
        "paths": {"data_root": str(data), "artifacts_root": str(art)},
        "index": {"type": "flatip"},
    })

    for vid, n_frames in VIDEOS.items():
        kf_dir = data / "keyframes" / vid
        kf_dir.mkdir(parents=True)
        rows = []
        for n in range(1, n_frames + 1):
            img = Image.new("RGB", (32, 32), color=(n * 20 % 255, 80, 120))
            img.save(kf_dir / f"{n:03d}.jpg")
            fps = 25.0
            frame_idx = n * 100
            rows.append((n, frame_idx / fps, fps, frame_idx))
        map_dir = data / "map-keyframes"
        map_dir.mkdir(parents=True, exist_ok=True)
        with open(map_dir / f"{vid}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["n", "pts_time", "fps", "frame_idx"])
            w.writerows(rows)

        titles = {
            "L21_V001": ("Bản tin 60 giây sáng", ["60 giây", "HTV"]),
            "L21_V002": ("Món ngon mỗi ngày", ["ẩm thực", "nấu ăn"]),
            "K01_V001": ("Lan tỏa năng lượng tích cực", ["tuổi trẻ"]),
        }
        title, keywords = titles[vid]
        mi_dir = data / "media-info"
        mi_dir.mkdir(parents=True, exist_ok=True)
        (mi_dir / f"{vid}.json").write_text(
            json.dumps({
                "title": title,
                "description": "tin tức thời sự thành phố",
                "keywords": keywords,
                "watch_url": "https://youtube.com/watch?v=x",
            }),
            encoding="utf-8",
        )

        obj_dir = data / "objects" / vid
        obj_dir.mkdir(parents=True, exist_ok=True)
        for n in range(1, n_frames + 1):
            (obj_dir / f"{n:03d}.json").write_text(
                json.dumps({
                    "detection_class_entities": ["Person", "Person", "Car"],
                    "detection_scores": [0.9, 0.8, 0.7],
                    "detection_boxes": [[0.1, 0.1, 0.5, 0.4], [0.1, 0.5, 0.5, 0.9], [0.6, 0.2, 0.9, 0.8]],
                }),
                encoding="utf-8",
            )
    return settings


@pytest.fixture()
def corpus_with_index(corpus: Settings) -> Settings:
    """corpus + built catalog, fake embeddings and FAISS index for 'fake' model."""
    pytest.importorskip("faiss", reason="install the [search] extra for index tests")
    from cvp.data.catalog import KeyframeCatalog
    from cvp.index.store import IndexStore

    catalog = KeyframeCatalog(corpus)
    df = catalog.build()
    store = IndexStore(corpus, "fake")
    rng = _rng(0)
    for vid, grp in df.groupby("video_id"):
        vecs = rng.normal(size=(len(grp), DIM)).astype(np.float32)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        store.embedding_path(str(vid)).parent.mkdir(parents=True, exist_ok=True)
        np.save(store.embedding_path(str(vid)), vecs)
    store.build(catalog)
    return corpus
