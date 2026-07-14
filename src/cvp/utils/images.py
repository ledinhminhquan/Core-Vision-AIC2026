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


# ── Perceptual hashing (map-keyframes reconstruction, AVS dedup) ─────────────
def dhash(img: Image.Image, hash_size: int = 8) -> int:
    """Difference hash → 64-bit int. Cheap, dependency-free, good for dedup."""
    import numpy as np

    g = img.convert("L").resize((hash_size + 1, hash_size), Image.LANCZOS)
    a = np.asarray(g, dtype=np.int16)
    diff = a[:, 1:] > a[:, :-1]
    bits = 0
    for v in diff.flatten():
        bits = (bits << 1) | int(v)
    return bits


def hamming(a: int, b: int) -> int:
    """Hamming distance between two 64-bit perceptual hashes."""
    return bin(a ^ b).count("1")


def is_near_duplicate(h: int, kept_hashes: list[int], threshold: int = 5) -> bool:
    """True if hash ``h`` is within ``threshold`` Hamming bits of any kept hash."""
    return any(hamming(h, k) <= threshold for k in kept_hashes)
