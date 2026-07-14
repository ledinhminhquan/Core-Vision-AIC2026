"""Image loading helpers (robust to truncated competition JPEGs)."""

from __future__ import annotations

import logging
from pathlib import Path

from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True  # organiser zips occasionally truncate
log = logging.getLogger(__name__)


def load_rgb(path: str | Path) -> Image.Image | None:
    try:
        with Image.open(path) as im:
            return im.convert("RGB")
    except (OSError, ValueError) as e:
        log.warning("Unreadable image %s: %s", path, e)
        return None


def load_rgb_batch(paths: list[str | Path]) -> tuple[list[Image.Image], list[int]]:
    """Load a batch; returns (images, kept_indices) — silently drops unreadables."""
    images: list[Image.Image] = []
    kept: list[int] = []
    for i, p in enumerate(paths):
        im = load_rgb(p)
        if im is not None:
            images.append(im)
            kept.append(i)
    return images, kept
