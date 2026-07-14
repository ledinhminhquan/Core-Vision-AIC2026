"""Assemble the Vietnamese caption↔keyframe training set.

Inputs (produced by the offline pipeline on the SAME corpus):
    artifacts/captions/{vid}.json                    Vintern Vietnamese captions
    artifacts/embeddings/siglip2/{vid}.npy           frozen image-tower embeddings
    artifacts/train_data/public/extra_*.parquet      optional public channels
                                                     (scripts/12, auto-merged)

Curation:
    * caption length gate (15–400 chars),
    * near-duplicate frame suppression inside each video (embedding cosine
      > 0.97 vs the last kept frame — news repeats anchor shots constantly),
    * VIDEO-level train/val split (~3% val) so no frame of a val video ever
      leaks into train.

Outputs:
    artifacts/train_data/embeds.npy    (N, dim) float16 image embeddings
    artifacts/train_data/meta.parquet  video_id | n | caption | caption_en | split
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from cvf.config import Settings
from cvf.data.catalog import KeyframeCatalog
from cvf.index.store import IndexStore
from cvf.utils.io import read_json

log = logging.getLogger(__name__)

MIN_CAPTION_CHARS = 15
MAX_CAPTION_CHARS = 400
NEAR_DUP_COSINE = 0.97
VAL_FRACTION = 0.03


def _split_for(video_id: str) -> str:
    h = int(hashlib.sha1(video_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "val" if h < VAL_FRACTION else "train"


def _load_extra_parquet(path: Path, expected_dim: int | None) -> tuple[list[dict], np.ndarray]:
    """Read one public-channel parquet (see training/public_datasets.py).

    Returns (meta_rows, embeds). Raises ``ValueError`` when the embedding dim
    does not match the corpus channel — mixing embedding spaces silently would
    corrupt training.
    """
    df = pd.read_parquet(path)
    cap_col = "caption" if "caption" in df.columns else "caption_vi"
    required = {"embed", cap_col, "video_id", "split"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path}: extra parquet is missing columns {sorted(missing)}")
    if df.empty:
        return [], np.zeros((0, expected_dim or 0), np.float16)
    embeds = np.stack([np.asarray(e, dtype=np.float32) for e in df["embed"]])
    if expected_dim is not None and embeds.shape[1] != expected_dim:
        raise ValueError(
            f"{path}: embed dim {embeds.shape[1]} != corpus dim {expected_dim} — "
            "rebuild the parquet with the same frozen image tower (scripts/12)."
        )
    en = df["caption_en"].fillna("").astype(str) if "caption_en" in df.columns else [""] * len(df)
    rows = [
        {
            "video_id": str(vid), "n": 0, "caption": str(cap).strip(),
            "caption_en": str(cap_en).strip(), "split": str(split),
        }
        for vid, cap, cap_en, split in zip(df["video_id"], df[cap_col], en, df["split"])
    ]
    return rows, embeds.astype(np.float16)


def build_training_set(
    settings: Settings,
    model_key: str = "siglip2",
    extra_parquets: Sequence[str | os.PathLike] | None = None,
) -> tuple[int, int]:
    """Returns (train_rows, val_rows).

    Args:
        extra_parquets: public-channel parquets to merge (embed + caption[+_en]
            + video_id + split columns, built by scripts/12). ``None`` (default)
            auto-merges every ``train_data/public/*.parquet``; pass ``[]`` to
            disable merging.
    """
    catalog = KeyframeCatalog(settings)
    df = catalog.load()
    store = IndexStore(settings, model_key)
    cap_dir = settings.paths.art("captions")
    out_dir = settings.paths.art("train_data")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    embeds: list[np.ndarray] = []
    for vid, grp in df.groupby("video_id", sort=True):
        cap_path = cap_dir / f"{vid}.json"
        if not cap_path.is_file():
            continue
        n_to_caption = (read_json(cap_path, default={}) or {}).get("n_to_caption") or {}
        if not n_to_caption:
            continue
        try:
            vecs = np.asarray(np.load(store.embedding_path(str(vid))), dtype=np.float32)
        except (OSError, ValueError):
            log.warning("No embeddings for %s — skipped", vid)
            continue
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vecs = vecs / norms
        split = _split_for(str(vid))
        # Embedding rows are POSITIONAL over the catalog group sorted by n
        # (gaps in filenames are legal) — never index with n-1 directly.
        ns_sorted = sorted(int(x) for x in grp["n"])
        n_to_row = {n: i for i, n in enumerate(ns_sorted)}
        last_kept: np.ndarray | None = None
        for n in sorted(int(k) for k in n_to_caption):
            caption = str(n_to_caption[str(n)]).strip()
            if not (MIN_CAPTION_CHARS <= len(caption) <= MAX_CAPTION_CHARS):
                continue
            row_idx = n_to_row.get(n, -1)
            if not (0 <= row_idx < len(vecs)):
                continue
            v = vecs[row_idx]
            if last_kept is not None and float(v @ last_kept) > NEAR_DUP_COSINE:
                continue
            last_kept = v
            rows.append({"video_id": str(vid), "n": n, "caption": caption,
                         "caption_en": "", "split": split})
            embeds.append(v.astype(np.float16))

    # Merge public caption channels (human-written VI + optional EN anchors).
    corpus_dim = int(embeds[0].shape[0]) if embeds else None
    if extra_parquets is None:
        public_dir = settings.paths.art("train_data", "public")
        extra_parquets = sorted(public_dir.glob("*.parquet")) if public_dir.is_dir() else []
    for pq in extra_parquets:
        extra_rows, extra_embeds = _load_extra_parquet(Path(pq), corpus_dim)
        if not extra_rows:
            continue
        rows.extend(extra_rows)
        embeds.extend(extra_embeds)
        log.info("Merged %d public caption rows from %s", len(extra_rows), pq)

    if not rows:
        raise RuntimeError("No training rows assembled — run captioning + embedding first.")

    meta = pd.DataFrame(rows)
    arr = np.stack(embeds)
    tmp_npy = out_dir / "embeds.npy.tmp"
    with open(tmp_npy, "wb") as f:  # np.save appends ".npy" to bare paths
        np.save(f, arr)
    tmp_npy.replace(out_dir / "embeds.npy")
    tmp_pq = out_dir / "meta.parquet.tmp"
    meta.to_parquet(tmp_pq, index=False)
    tmp_pq.replace(out_dir / "meta.parquet")

    n_train = int((meta["split"] == "train").sum())
    n_val = int((meta["split"] == "val").sum())
    log.info("Training set: %d train / %d val pairs, dim=%d", n_train, n_val, arr.shape[1])
    return n_train, n_val
