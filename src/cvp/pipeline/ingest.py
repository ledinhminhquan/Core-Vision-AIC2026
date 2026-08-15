"""Idempotent offline build: (extract) → catalog → embed → index → aux.

One entry point safely handles both first-time builds and incremental dataset
drops (organisers release data in waves). Every stage skips finished work, so
re-running after adding zips or after a Colab disconnect only does the delta.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from cvp.config import Settings
from cvp.data.catalog import KeyframeCatalog
from cvp.data.extraction import extract_missing
from cvp.index.embedder import embed_all_keyframes, ingest_provided_features
from cvp.index.store import IndexStore
from cvp.models.registry import build_model

log = logging.getLogger(__name__)


@dataclass
class IngestReport:
    extracted_videos: int = 0
    embedded: dict[str, int] = field(default_factory=dict)
    indexed: list[str] = field(default_factory=list)
    keyframes: int = 0
    videos: int = 0


def member_names(settings: Settings) -> list[str]:
    """Embedding-space names that need offline embed+index work.

    ``finetuned`` searches the siglip2 space (LiT keeps the image tower
    frozen), so it maps to siglip2 for offline builds.
    """
    names = (
        list(settings.embedding.ensemble_members)
        if settings.embedding.model == "ensemble"
        else [settings.embedding.model]
    )
    from cvp.models.registry import index_key_for

    out: list[str] = []
    for n in names:
        k = index_key_for(n)
        if k not in out:
            out.append(k)
    return out


def run_ingest(settings: Settings, extract: bool = True, force_index: bool = False) -> IngestReport:
    report = IngestReport()

    if extract:
        report.extracted_videos = extract_missing(settings)

    catalog = KeyframeCatalog(settings)
    df = catalog.build()
    report.keyframes = len(df)
    report.videos = df["video_id"].nunique()

    for name in member_names(settings):
        model_tag = None
        if name == "provided_clip32":
            report.embedded[name] = ingest_provided_features(settings, catalog)
        else:
            model = build_model(settings, name)
            report.embedded[name] = embed_all_keyframes(model, settings, catalog)
            model_tag = getattr(model, "model_tag", None) or model.key
            del model  # free VRAM before the next member loads
            _empty_cuda_cache()
        store = IndexStore(settings, name)
        store.build(catalog, force=force_index, model_tag=model_tag)
        report.indexed.append(name)

    log.info(
        "Ingest complete: %d keyframes / %d videos; embedded=%s",
        report.keyframes, report.videos, report.embedded,
    )
    return report


def _empty_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def doctor(settings: Settings) -> dict:
    """Corpus health report: coverage of every artifact per video + staleness."""
    catalog = KeyframeCatalog(settings)
    df = catalog.load()
    videos = [str(v) for v in df["video_id"].unique()]
    counts = df.groupby("video_id")["n"].count()

    def _coverage(art: str, suffix: str = ".json") -> int:
        d = settings.paths.art(art)
        return sum(1 for v in videos if (d / f"{v}{suffix}").exists()) if d.is_dir() else 0

    report: dict = {
        "keyframes": int(len(df)),
        "videos": len(videos),
        "with_map_keyframes": int(df.groupby("video_id")["has_map"].any().sum()),
        "ocr_videos": _coverage("ocr"),
        "asr_videos": _coverage("asr"),
        "caption_videos": _coverage("captions"),
        "members": {},
    }
    for name in member_names(settings):
        store = IndexStore(settings, name)
        missing = store.missing_videos(catalog)
        report["members"][name] = {
            "embedded_videos": len(videos) - len(missing),
            "missing_videos": missing[:10],
            "index_exists": store.index_path.exists(),
            "index_stale": store.is_stale(catalog) if store.index_path.exists() else None,
            "index_count": store.count(),
        }
    try:
        from cvp.index.text_store import TextIndexStore

        report["text_index_fresh"] = TextIndexStore.signature_matches(
            settings.paths.artifacts_root, catalog.signature()
        )
    except Exception:  # noqa: BLE001 — health report must never crash on a probe
        report["text_index_fresh"] = False
    no_map = counts.index.difference(df[df["has_map"]]["video_id"].unique()).tolist()
    if no_map:
        report["videos_without_map"] = no_map[:20]
    return report
