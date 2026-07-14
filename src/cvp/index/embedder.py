"""Offline keyframe embedding: keyframes → artifacts/embeddings/{model}/{vid}.npy.

Resumable at video granularity: a video with a correctly-shaped .npy is
skipped, so a Colab disconnect costs at most one video of work. For
``provided_clip32`` the organiser's precomputed ``clip-features-32`` packs are
ingested directly (fp16 → fp32, L2-normalized) — no GPU needed at all.
"""

from __future__ import annotations

import logging

import numpy as np

from cvp.config import Settings
from cvp.data.catalog import KeyframeCatalog
from cvp.index.store import IndexStore
from cvp.models.base import EmbeddingModel, l2_normalize
from cvp.utils.images import load_rgb_batch

log = logging.getLogger(__name__)


def _save_atomic(path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".npy.tmp")
    # np.save appends ".npy" to any filename that lacks it — write through a
    # file handle so the tmp file keeps its exact name and replace() works.
    with open(tmp, "wb") as f:
        np.save(f, arr)
    tmp.replace(path)


def ingest_provided_features(settings: Settings, catalog: KeyframeCatalog) -> int:
    """Copy organiser clip-features-32 into the provided_clip32 embedding store."""
    store = IndexStore(settings, "provided_clip32")
    src_root = settings.paths.data(settings.paths.clip_features_dir)
    if not src_root.is_dir():
        raise FileNotFoundError(f"clip-features-32 folder not found: {src_root}")
    df = catalog.load()
    done = 0
    for vid, cnt in df.groupby("video_id", sort=True)["n"].count().items():
        dst = store.embedding_path(str(vid))
        if dst.exists():
            try:
                if np.load(dst, mmap_mode="r").shape[0] == int(cnt):
                    continue
            except (OSError, ValueError):
                pass
        src = src_root / f"{vid}.npy"
        if not src.is_file():
            log.warning("No provided features for %s — embed it with a real model instead", vid)
            continue
        vecs = np.asarray(np.load(src), dtype=np.float32)
        if vecs.shape[0] != int(cnt):
            log.warning(
                "Provided features for %s have %d rows but catalog has %d keyframes — skipped",
                vid, vecs.shape[0], cnt,
            )
            continue
        _save_atomic(dst, l2_normalize(vecs))
        done += 1
    log.info("Ingested provided features for %d videos", done)
    return done


def _check_model_tag(store: IndexStore, model: EmbeddingModel, overwrite: bool) -> None:
    """Refuse to resume into a folder produced by a DIFFERENT checkpoint.

    The 'openclip' lane resolves to one of several checkpoints (PE-Core-bigG →
    PE-Core-L → DFN5B) depending on hub reachability. Session A embedding half
    the corpus with one and session B resuming with another would mix spaces —
    a crash at index time if dims differ, silent garbage if they coincide.
    """
    from cvp.utils.io import atomic_write_json, read_json

    tag = getattr(model, "model_tag", None) or model.key
    marker = store.embed_dir / "model_tag.json"
    existing = (read_json(marker, default={}) or {}).get("model_tag")
    if existing and existing != tag and not overwrite:
        raise RuntimeError(
            f"[{model.key}] embeddings folder was produced by {existing!r} but this "
            f"session loaded {tag!r}. Re-run with overwrite/FORCE_EMBED to re-embed "
            "everything with the current checkpoint, or restore access to the "
            "original checkpoint."
        )
    store.embed_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(marker, {"model_tag": tag})


def embed_all_keyframes(
    model: EmbeddingModel,
    settings: Settings,
    catalog: KeyframeCatalog,
    overwrite: bool = False,
) -> int:
    """Encode every keyframe with ``model``; skip videos already done."""
    store = IndexStore(settings, model.key)
    _check_model_tag(store, model, overwrite)
    df = catalog.load()
    todo = (
        [str(v) for v in df["video_id"].unique()]
        if overwrite
        else store.missing_videos(catalog)
    )
    if not todo:
        log.info("[%s] embeddings complete for all %d videos", model.key, df["video_id"].nunique())
        return 0

    log.info("[%s] embedding %d videos ...", model.key, len(todo))
    by_video = {vid: grp.sort_values("n") for vid, grp in df.groupby("video_id", sort=False)}
    done = 0
    for vid in todo:
        grp = by_video[vid]
        paths = [catalog.resolve_path(str(p)) for p in grp["path"]]
        images, kept = load_rgb_batch(paths)
        vecs = model.encode_image(images) if images else np.zeros((0, model.dim), dtype=np.float32)
        if len(kept) != len(paths):
            # Keep row alignment: unreadable frames get zero vectors (never match).
            full = np.zeros((len(paths), vecs.shape[1] if len(vecs) else model.dim), dtype=np.float32)
            for row, src_i in enumerate(kept):
                full[src_i] = vecs[row]
            vecs = full
            log.warning("[%s] %s: %d unreadable frames zero-filled", model.key, vid, len(paths) - len(kept))
        _save_atomic(store.embedding_path(vid), vecs.astype(np.float32))
        done += 1
        if done % 10 == 0 or done == len(todo):
            log.info("[%s] %d/%d videos embedded", model.key, done, len(todo))
    return done
