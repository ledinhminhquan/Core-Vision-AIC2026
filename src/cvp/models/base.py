"""Embedding model contract + device/dtype resolution.

Every backend maps images and texts into ONE shared space and returns
L2-normalized ``float32 (N, dim)`` arrays, so inner product == cosine and any
backend can drop into FAISS + fusion unchanged.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from PIL import Image


def resolve_device(pref: str = "auto") -> str:
    if pref != "auto":
        return pref
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def resolve_dtype(pref: str, device: str):
    import torch

    if pref == "bf16":
        return torch.bfloat16
    if pref == "fp16":
        return torch.float16
    if pref == "fp32":
        return torch.float32
    # auto
    if device == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def l2_normalize(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


class EmbeddingModel(ABC):
    """Bi-encoder into a shared image/text space."""

    #: unique key — names the embeddings/index folders (filesystem-safe)
    key: str = "base"
    #: embedding dimensionality (filled on load)
    dim: int = 0
    #: True if the TEXT tower understands Vietnamese natively
    multilingual: bool = False

    @abstractmethod
    def encode_image(self, images: list[Image.Image]) -> np.ndarray:
        """(N, dim) float32, L2-normalized."""

    @abstractmethod
    def encode_text(self, texts: list[str]) -> np.ndarray:
        """(N, dim) float32, L2-normalized."""

    def encode_one_text(self, text: str) -> np.ndarray:
        return self.encode_text([text])[0]
