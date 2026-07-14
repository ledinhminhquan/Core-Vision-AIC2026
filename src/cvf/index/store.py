"""FAISS index build / load / search, one index per embedding backend.

Layout (per model key, e.g. ``siglip2``):

    artifacts/indexes/{model_key}/kf.faiss     the FAISS index (row == global_id)
    artifacts/indexes/{model_key}/meta.json    dim, count, catalog signature

Embeddings live in ``artifacts/embeddings/{model_key}/{video_id}.npy`` with one
row per keyframe ordinal, so the index is streamed video-by-video in catalog
order — FAISS row ids equal catalog ``global_id`` by construction, and RAM
never holds more than one video's vectors during a build.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from cvf.config import Settings
from cvf.data.catalog import KeyframeCatalog
from cvf.utils.io import atomic_write_json, read_json

log = logging.getLogger(__name__)


def _faiss():
    import faiss  # local import: keeps faiss optional for non-search tooling

    return faiss


class IndexStore:
    def __init__(self, settings: Settings, model_key: str):
        self.settings = settings
        self.model_key = model_key
        self.dir = settings.paths.art("indexes", model_key)
        self.index_path = self.dir / "kf.faiss"
        self.meta_path = self.dir / "meta.json"
        self.embed_dir = settings.paths.art("embeddings", model_key)
        self._index = None

    # ── build ────────────────────────────────────────────────────────────

    def embedding_path(self, video_id: str) -> Path:
        return self.embed_dir / f"{video_id}.npy"

    def missing_videos(self, catalog: KeyframeCatalog, expected_dim: int | None = None) -> list[str]:
        """Videos with no (or wrong-shaped) embedding file.

        ``expected_dim`` also flags dim mismatches — the 'openclip' lane can
        resolve to different checkpoints (PE-Core-bigG 1280-d vs DFN5B 1024-d)
        on different machines, and a silent mix would only explode at
        ``index.add`` (or worse, not at all).
        """
        missing = []
        df = catalog.load()
        for vid, cnt in df.groupby("video_id", sort=True)["n"].count().items():
            p = self.embedding_path(str(vid))
            if not p.exists():
                missing.append(str(vid))
                continue
            try:
                shape = np.load(p, mmap_mode="r").shape
            except (OSError, ValueError):
                missing.append(str(vid))
                continue
            if shape[0] != int(cnt):
                log.warning("Embedding count mismatch for %s: %s rows vs %d keyframes", vid, shape[0], cnt)
                missing.append(str(vid))
            elif expected_dim and len(shape) == 2 and shape[1] != expected_dim:
                log.warning(
                    "Embedding dim mismatch for %s: %d vs expected %d — "
                    "different checkpoint produced these vectors; re-embed",
                    vid, shape[1], expected_dim,
                )
                missing.append(str(vid))
        return missing

    def build(self, catalog: KeyframeCatalog, force: bool = False,
              model_tag: str | None = None) -> None:
        """Stream per-video embeddings (catalog order) into a fresh FAISS index."""
        faiss = _faiss()
        if self.index_path.exists() and not force and not self.is_stale(catalog):
            log.info("[%s] index up-to-date (%d vectors)", self.model_key, self.count())
            return

        missing = self.missing_videos(catalog)
        if missing:
            raise RuntimeError(
                f"[{self.model_key}] {len(missing)} videos lack embeddings "
                f"(first: {missing[:5]}). Run the embed step first."
            )

        df = catalog.load()
        videos = df["video_id"].unique().tolist()  # manifest order == sorted order
        first = np.load(self.embedding_path(videos[0]), mmap_mode="r")
        dim = int(first.shape[1])

        cfg = self.settings.index
        if cfg.type == "flatip":
            index = faiss.IndexFlatIP(dim)
        elif cfg.type == "ivf":
            quantizer = faiss.IndexFlatIP(dim)
            index = faiss.IndexIVFFlat(quantizer, dim, cfg.ivf_nlist, faiss.METRIC_INNER_PRODUCT)
        elif cfg.type == "hnsw":
            index = faiss.IndexHNSWFlat(dim, cfg.hnsw_m, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efSearch = cfg.hnsw_ef_search
        else:
            raise ValueError(f"Unknown index type: {cfg.type}")

        if cfg.type == "ivf":
            # Train on a subsample streamed from the corpus.
            rng = np.random.default_rng(0)
            sample_vids = list(rng.permutation(videos))[: max(1, len(videos) // 10)]
            train = np.concatenate(
                [np.asarray(np.load(self.embedding_path(v), mmap_mode="r"), dtype=np.float32) for v in sample_vids]
            )
            index.train(train)
            del train

        total = 0
        for vid in videos:
            vecs = np.asarray(np.load(self.embedding_path(vid)), dtype=np.float32)
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            index.add(np.ascontiguousarray(vecs / norms))
            total += len(vecs)
        if total != len(df):
            raise RuntimeError(f"Vector count {total} != catalog rows {len(df)} — aborting index write")

        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.index_path.with_suffix(".faiss.tmp")
        faiss.write_index(index, str(tmp))
        tmp.replace(self.index_path)
        meta = {
            "model_key": self.model_key,
            "dim": dim,
            "count": total,
            "index_type": cfg.type,
            "catalog_signature": catalog.signature(),
        }
        if model_tag:
            meta["model_tag"] = model_tag  # which checkpoint produced these vectors
        atomic_write_json(self.meta_path, meta)
        self._index = None
        log.info("[%s] index built: %d vectors, dim %d", self.model_key, total, dim)

    # ── load / search ────────────────────────────────────────────────────

    def meta(self) -> dict:
        return read_json(self.meta_path, default={}) or {}

    def count(self) -> int:
        return int(self.meta().get("count", 0))

    def dim(self) -> int:
        return int(self.meta().get("dim", 0))

    def is_stale(self, catalog: KeyframeCatalog) -> bool:
        return self.meta().get("catalog_signature") != catalog.signature()

    def load(self, catalog: KeyframeCatalog | None = None):
        if self._index is None:
            faiss = _faiss()
            if not self.index_path.exists():
                raise FileNotFoundError(
                    f"[{self.model_key}] index missing: {self.index_path}. Build it first."
                )
            if catalog is not None and self.is_stale(catalog):
                raise RuntimeError(
                    f"[{self.model_key}] index is STALE vs the current corpus — rebuild before searching. "
                    "(A stale index returns wrong global_ids, which become wrong frame_idx submissions.)"
                )
            index = faiss.read_index(str(self.index_path))
            if self.settings.index.type == "ivf":
                index.nprobe = self.settings.index.ivf_nprobe
            if self.settings.index.use_gpu:
                try:
                    res = faiss.StandardGpuResources()
                    index = faiss.index_cpu_to_gpu(res, 0, index)
                except (AttributeError, RuntimeError) as e:
                    log.warning("faiss-gpu unavailable (%s) — staying on CPU", e)
            self._index = index
        return self._index

    def search(self, query_vecs: np.ndarray, topk: int) -> tuple[np.ndarray, np.ndarray]:
        """(Q, dim) L2-normalized queries → (scores, global_ids), each (Q, topk)."""
        index = self.load()
        q = np.ascontiguousarray(np.asarray(query_vecs, dtype=np.float32))
        if q.ndim == 1:
            q = q[None, :]
        return index.search(q, topk)
