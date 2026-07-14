"""Public Vietnamese caption datasets → EXTRA training parquet channels.

WHY: ``build_training_set`` pairs Vintern-GENERATED captions with corpus
keyframes only, so the LiT fine-tune can overfit Vintern's phrasing instead of
learning Vietnamese. These builders add human-written Vietnamese captions from
public datasets as extra channels (different domains, different caption
style), merged via the ``extra_parquets`` argument of
:func:`cvp.training.build_dataset.build_training_set` — domain diversity
against caption-style overfitting, per the winning-recipe research (ViCLIP-OT
trains a Vietnamese CLIP on exactly these sets).

Verified Hugging Face dataset ids (checked 2026-07 via the HF hub API — both
expose image + per-image ``caption_vi`` list columns through the parquet
viewer):

- ``ktvic``    → ``ai-enthusiasm-community/KTVIC`` (arXiv:2401.08100):
  ~4.3k daily-life images / ~21.6k VI captions; splits train + test.
- ``uit_viic`` → ``ai-enthusiasm-community/UIT-ViIC`` (arXiv:2002.00175,
  CC-BY-4.0 sports subset of MS-COCO): ~3.6k images / ~18k VI captions;
  splits train + validation.

If the hub is unreachable, pass ``local_dir`` pointing at a
``datasets.save_to_disk`` copy of the same dataset instead.

Output parquet schema — the ``extra_parquets`` contract of
``training/build_dataset.py``: an ``embed`` column holding the frozen
image-tower vector (SAME model + dim as the corpus keyframe embeds, else the
merge raises) plus meta columns caption / caption_en / video_id / source /
domain / split. Public test/validation splits map to split="val" so they also
enrich the retrieval eval with human-caption queries.

Resumable: image embeddings are checkpointed per split (atomic .npy under
``artifacts/cache/public_datasets``) after every encode batch, and a finished
parquet is skipped entirely unless ``overwrite=True``. Checkpoints are keyed
by dataset + split + embedding model (key and dim in the filename, plus a
json sidecar verified on resume), so switching the embedding model can never
silently resume from another model's vectors.
"""

from __future__ import annotations

import io
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

import numpy as np
import pandas as pd

from cvp.config import Settings, load_settings
from cvp.utils.io import atomic_write_bytes, atomic_write_json, read_json

if TYPE_CHECKING:  # pragma: no cover - typing only
    from cvp.models.base import EmbeddingModel

log = logging.getLogger(__name__)

# Columns of the extra_parquets contract (embed first, then the meta columns
# build_training_set folds into train_data/meta.parquet).
EXTRA_PARQUET_COLUMNS = ("embed", "caption", "caption_en", "video_id",
                         "source", "domain", "split")


@dataclass(frozen=True)
class PublicDatasetSpec:
    hf_id: str
    splits: tuple[tuple[str, str], ...]  # (hf split name, our split name)
    domain: str
    paper: str


PUBLIC_DATASETS: dict[str, PublicDatasetSpec] = {
    "ktvic": PublicDatasetSpec(
        hf_id="ai-enthusiasm-community/KTVIC",
        splits=(("train", "train"), ("test", "val")),
        domain="daily_life",
        paper="arXiv:2401.08100",
    ),
    "uit_viic": PublicDatasetSpec(
        hf_id="ai-enthusiasm-community/UIT-ViIC",
        splits=(("train", "train"), ("validation", "val")),
        domain="sports",
        paper="arXiv:2002.00175",
    ),
}


def _filter_captions(captions: Iterable[str] | None,
                     min_words: int = 3, max_words: int = 45) -> list[str]:
    """Whitespace-normalise, drop out-of-length captions, dedupe (keep order).

    One consistent caption-quality policy across every training channel.
    """
    seen: set[str] = set()
    out: list[str] = []
    for cap in captions or []:
        txt = " ".join(str(cap).split()).strip()
        n = len(txt.split())
        if not (min_words <= n <= max_words):
            continue
        key = txt.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(txt)
    return out


def _load_split(spec: PublicDatasetSpec, hf_split: str,
                local_dir: str | Path | None = None):
    """Load one HF split (lazy import — ``datasets`` is a Colab-side dependency)."""
    try:
        from datasets import load_dataset, load_from_disk
    except ImportError as e:
        # NOT a network problem — say so, or callers report it as one.
        raise ImportError(
            "the 'datasets' package is required for the public caption channels — "
            "pip install 'datasets>=2.19' (this is a missing dependency, not a "
            "network/hub failure)"
        ) from e

    if local_dir is not None:
        return load_from_disk(str(local_dir))[hf_split]
    return load_dataset(spec.hf_id, split=hf_split)


def _save_npy_atomic(path: Path, arr: np.ndarray) -> None:
    buf = io.BytesIO()
    np.save(buf, arr)
    atomic_write_bytes(path, buf.getvalue())


def _fs_safe(text: str) -> str:
    """Filesystem-safe token for cache filenames (model ids contain '/', ':')."""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", str(text)).strip("-.") or "model"


def _checkpoint_paths(cache_dir: Path, name: str, hf_split: str,
                      model: "EmbeddingModel") -> tuple[Path, Path]:
    """Model-fingerprinted checkpoint pair: .npy vectors + .json sidecar.

    The model key/dim in the filename keeps checkpoints of different embedding
    models apart (switching models restarts the encode instead of resuming
    from the OLD model's vectors — silent mixed-space corruption); the sidecar
    lets resume verify the fingerprint even if a checkpoint was renamed.
    """
    stem = f"{name}_{hf_split}_{_fs_safe(model.key)}_{int(model.dim)}_embeds"
    return cache_dir / f"{stem}.npy", cache_dir / f"{stem}.json"


def default_public_parquet_path(name: str, settings: Settings | None = None) -> Path:
    """Where scripts/12 writes each extra parquet.

    Under ``train_data/public/`` so scripts/11 and the Colab training notebook
    (which auto-merge every parquet found in that folder) pick it up.
    """
    settings = settings or load_settings()
    return settings.paths.art("train_data", "public", f"extra_{name}.parquet")


def _build_split_frame(name: str, spec: PublicDatasetSpec, ds, hf_split: str,
                       our_split: str, model: "EmbeddingModel", batch_size: int,
                       cache_dir: Path, min_words: int, max_words: int) -> pd.DataFrame:
    """Encode one split's images (checkpointed) and expand to per-caption rows.

    Row alignment on resume is purely caption-determined: an undecodable image
    is embedded as a blank placeholder (never silently dropped), so checkpoint
    row K always pairs with the K-th caption-valid dataset row.
    """
    from PIL import Image
    from tqdm import tqdm

    ck, sidecar = _checkpoint_paths(cache_dir, name, hf_split, model)
    fingerprint = dict(model_key=str(model.key), dim=int(model.dim),
                       dataset=name, split=hf_split)
    blocks: list[np.ndarray] = []
    n_done = 0
    if ck.exists():
        found = read_json(sidecar, default=None) if sidecar.exists() else None
        if found != fingerprint:
            raise RuntimeError(
                f"{name}:{hf_split}: embedding checkpoint {ck} was written by a "
                f"different embedding model/config (sidecar says {found}, this run "
                f"is {fingerprint}) — resuming would mix embedding spaces. "
                f"Delete {ck} (and {sidecar}) and rerun."
            )
        cached = np.load(ck).astype(np.float32)
        n_done = len(cached)
        if n_done:
            blocks.append(cached)
        log.info("Resuming %s:%s from checkpoint (%d images embedded)", name, hf_split, n_done)

    metas: list[tuple[str, list[str]]] = []  # (image_uid, captions) per kept row
    pending: list = []

    def flush() -> None:
        if not pending:
            return
        emb = np.asarray(model.encode_image(pending), dtype=np.float32)
        blocks.append(emb)
        pending.clear()
        atomic_write_json(sidecar, fingerprint)  # sidecar first: every .npy has one
        _save_npy_atomic(ck, np.concatenate(blocks, axis=0))

    kept = 0
    for row in tqdm(ds, desc=f"{name}:{hf_split}"):
        caps = _filter_captions(row.get("caption_vi"), min_words, max_words)
        if not caps:
            continue
        metas.append((str(row.get("image_uid") or kept), caps))
        if kept >= n_done:  # not covered by the checkpoint yet → embed it
            try:
                img = row["image"].convert("RGB")
            except Exception as e:  # pragma: no cover - corrupt image
                log.warning("Undecodable image in %s:%s (row %d): %s — using placeholder",
                            name, hf_split, kept, e)
                img = Image.new("RGB", (64, 64))
            pending.append(img)
            if len(pending) >= batch_size:
                flush()
        kept += 1
    flush()

    embeds = (np.concatenate(blocks, axis=0) if blocks
              else np.zeros((0, int(model.dim)), np.float32))
    if len(embeds) != len(metas):
        raise RuntimeError(
            f"{name}:{hf_split}: checkpoint has {len(embeds)} rows but the dataset "
            f"yields {len(metas)} caption-valid rows — the filter params or the "
            f"dataset changed since the checkpoint was written. Delete {ck} and rerun."
        )

    records: list[dict] = []
    for (uid, caps), emb in zip(metas, embeds):
        vid = f"{name}:{uid}"
        for cap in caps:
            records.append(dict(embed=emb, caption=cap, caption_en="",
                                video_id=vid, source=name, domain=spec.domain,
                                split=our_split))
    return pd.DataFrame(records, columns=list(EXTRA_PARQUET_COLUMNS))


def build_public_parquet(
    name: str,
    settings: Settings | None = None,
    model: "EmbeddingModel | None" = None,
    out_path: str | Path | None = None,
    batch_size: int = 64,
    overwrite: bool = False,
    local_dir: str | Path | None = None,
    min_words: int = 3,
    max_words: int = 45,
) -> Path:
    """Build one extra training parquet from a public VI caption dataset.

    ``model`` must be the SAME frozen image tower used for the corpus
    keyframes (default: ``cvp.models.build_model(settings)``);
    ``build_training_set`` enforces the dim match at merge time.
    """
    if name not in PUBLIC_DATASETS:
        raise ValueError(f"Unknown public dataset {name!r} (known: {sorted(PUBLIC_DATASETS)})")
    spec = PUBLIC_DATASETS[name]
    settings = settings or load_settings()
    out_path = Path(out_path) if out_path else default_public_parquet_path(name, settings)
    if out_path.exists() and not overwrite:
        log.info("%s parquet already exists → %s (use overwrite=True to rebuild)", name, out_path)
        return out_path

    if model is None:
        from cvp.models.registry import build_model

        model = build_model(settings)
    cache_dir = settings.paths.art("cache", "public_datasets")
    cache_dir.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    for hf_split, our_split in spec.splits:
        ds = _load_split(spec, hf_split, local_dir)
        frames.append(_build_split_frame(name, spec, ds, hf_split, our_split,
                                         model, max(1, int(batch_size)),
                                         cache_dir, min_words, max_words))
    df = pd.concat(frames, ignore_index=True)
    if df.empty:
        raise RuntimeError(f"{name} ({spec.hf_id}) produced 0 caption rows — dataset schema changed?")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, out_path)
    for hf_split, _ in spec.splits:  # checkpoints are folded into the parquet now
        for p in _checkpoint_paths(cache_dir, name, hf_split, model):
            p.unlink(missing_ok=True)
    log.info("%s: %d caption rows (%d images, dim=%d) → %s",
             name, len(df), df["video_id"].nunique(),
             int(np.asarray(df["embed"].iloc[0]).shape[-1]), out_path)
    return out_path


def build_all_public_parquets(
    names: Iterable[str] | None = None,
    settings: Settings | None = None,
    model: "EmbeddingModel | None" = None,
    batch_size: int = 64,
    overwrite: bool = False,
) -> list[Path]:
    """Build every requested public parquet, loading the embed model at most once.

    Skips already-built parquets WITHOUT touching the model, so a fully-built
    run is instant (and works offline / without GPU).
    """
    settings = settings or load_settings()
    paths: list[Path] = []
    for name in (list(names) if names is not None else sorted(PUBLIC_DATASETS)):
        target = default_public_parquet_path(name, settings)
        if target.exists() and not overwrite:
            log.info("%s already built → %s", name, target)
            paths.append(target)
            continue
        if model is None:
            from cvp.models.registry import build_model

            model = build_model(settings)
        paths.append(build_public_parquet(name, settings=settings, model=model,
                                          batch_size=batch_size, overwrite=overwrite))
    return paths
