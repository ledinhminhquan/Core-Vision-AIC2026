"""Jina CLIP v2 lane — jinaai/jina-clip-v2 via 🤗 transformers (trust_remote_code).

WHY this lane exists: jina-clip-v2 is natively multilingual (89 languages incl.
Vietnamese), so raw queries embed WITHOUT the VI→EN translation hop and its
recall loss. As an ensemble member it adds diversity: different training data
from SigLIP-2/PE-Core → different failure modes → better fusion. (Ported from
Core-Vision_HCMC-AI, where it was the second retrieval channel.)

Matryoshka: trained with Matryoshka representation learning — embeddings can be
truncated to any dim in [64, 1024] (then re-normalised) for a graceful
accuracy/cost trade-off; ``embedding.jina_dim`` picks the operating point.

LICENSE NOTE: jina-clip-v2 weights are CC-BY-NC-4.0 — fine for the competition
(non-commercial research/evaluation) but NOT for commercial use.
"""

from __future__ import annotations

import logging

import numpy as np
from PIL import Image

from cvp.config import Settings
from cvp.models.base import EmbeddingModel, l2_normalize, resolve_device, resolve_dtype

log = logging.getLogger(__name__)

# Matryoshka training covers this closed range; outside it the truncated
# prefix was never optimised and retrieval quality collapses.
TRUNCATE_MIN, TRUNCATE_MAX = 64, 1024


class JinaClipModel(EmbeddingModel):
    multilingual = True

    def __init__(self, settings: Settings):
        cfg = settings.embedding
        # Validate BEFORE any heavy import/download so misconfiguration fails fast.
        if not (TRUNCATE_MIN <= int(cfg.jina_dim) <= TRUNCATE_MAX):
            raise ValueError(
                f"embedding.jina_dim must be in [{TRUNCATE_MIN}, {TRUNCATE_MAX}] "
                f"(jina-clip-v2 Matryoshka range), got {cfg.jina_dim}"
            )

        import torch  # noqa: F401 — heavy imports stay out of module scope
        from transformers import AutoModel

        self.key = "jina"
        self.device = resolve_device(cfg.device)
        self.dtype = resolve_dtype(cfg.dtype, self.device)
        self.dim = int(cfg.jina_dim)
        self.batch_size = cfg.batch_size
        self.model_tag = cfg.jina_id

        log.info("Loading Jina-CLIP %s (dim=%d)", cfg.jina_id, self.dim)
        # trust_remote_code: jina-clip-v2 ships its own modelling code exposing
        # encode_text / encode_image convenience methods.
        self.model = AutoModel.from_pretrained(
            cfg.jina_id, trust_remote_code=True, torch_dtype=self.dtype
        ).to(self.device).eval()

    def _post(self, feats) -> np.ndarray:
        # encode_* returns numpy by default, but remote-code revisions may hand
        # back torch tensors; coerce either way. Re-enforce L2 normalisation —
        # remote-code behaviour is not under our control and FAISS IP == cosine
        # must hold.
        if hasattr(feats, "cpu"):
            feats = feats.float().cpu().numpy()
        return l2_normalize(np.asarray(feats, dtype=np.float32))

    def encode_image(self, images: list[Image.Image]) -> np.ndarray:
        import torch

        out = []
        for i in range(0, len(images), self.batch_size):
            batch = images[i:i + self.batch_size]
            with torch.no_grad():
                feats = self.model.encode_image(batch, truncate_dim=self.dim)
            out.append(self._post(feats))
        return np.concatenate(out, axis=0) if out else np.zeros((0, self.dim), np.float32)

    def encode_text(self, texts: list[str]) -> np.ndarray:
        import torch

        out = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            with torch.no_grad():
                feats = self.model.encode_text(batch, truncate_dim=self.dim)
            out.append(self._post(feats))
        return np.concatenate(out, axis=0) if out else np.zeros((0, self.dim), np.float32)
