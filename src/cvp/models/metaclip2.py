"""MetaCLIP 2 lane — facebook/metaclip-2-worldwide-* via 🤗 transformers.

WHY this lane exists: MetaCLIP 2 ("A Worldwide Scaling Recipe", 2025) is
trained on 300+ languages / 29B pairs and holds the multilingual retrieval
SOTA among open two-tower models as of mid-2026 (XM3600 I2T 64.3%, above
mSigLIP and SigLIP-2). It reads Vietnamese natively, making it a strong
optional diversity member next to siglip2 (different data, different failure
modes → better fusion). Plain CLIP-style API in transformers ≥4.56
(``MetaClip2Model`` resolves through ``AutoModel``) — our shared floor is 4.57.
"""

from __future__ import annotations

import logging

import numpy as np
from PIL import Image

from cvp.config import Settings
from cvp.models.hf_compat import feature_tensor
from cvp.models.base import EmbeddingModel, l2_normalize, resolve_device, resolve_dtype

log = logging.getLogger(__name__)


class MetaClip2Model(EmbeddingModel):
    multilingual = True

    def __init__(self, settings: Settings):
        import torch  # noqa: F401 — heavy imports stay out of module scope
        from transformers import AutoModel, AutoProcessor

        cfg = settings.embedding
        self.key = "metaclip2"
        self.device = resolve_device(cfg.device)
        self.dtype = resolve_dtype(cfg.dtype, self.device)
        self.batch_size = cfg.batch_size
        self.model_tag = cfg.metaclip2_id

        log.info("Loading MetaCLIP 2 %s", cfg.metaclip2_id)
        from cvp.models.hf_compat import resilient_from_pretrained

        # Round-90: phiên nb09 850660ee mất lane này vì "[Errno 5] Input/output
        # error" đọc HF cache trên Drive — cùng cơ chế tải lại về đĩa cục bộ
        # như siglip2 / qwen_embed (round-22 / verify-R23).
        self.model = resilient_from_pretrained(
            lambda mid: AutoModel.from_pretrained(mid, torch_dtype=self.dtype),
            cfg.metaclip2_id).to(self.device).eval()
        self.processor = resilient_from_pretrained(
            AutoProcessor.from_pretrained, cfg.metaclip2_id)

        probe = self.encode_text(["probe"])
        self.dim = int(probe.shape[1])

    def encode_image(self, images: list[Image.Image]) -> np.ndarray:
        import torch

        feats = []
        with torch.no_grad():
            for i in range(0, len(images), self.batch_size):
                inputs = self.processor(
                    images=images[i:i + self.batch_size], return_tensors="pt"
                )
                pixel_values = inputs["pixel_values"].to(self.device, self.dtype)
                out = self.model.get_image_features(pixel_values=pixel_values)
                feats.append(feature_tensor(out).float().cpu().numpy())
        return l2_normalize(np.concatenate(feats, axis=0)) if feats else \
            np.zeros((0, self.dim), np.float32)

    def encode_text(self, texts: list[str]) -> np.ndarray:
        import torch

        cfg_max = self.model.config.text_config.max_position_embeddings
        feats = []
        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                inputs = self.processor(
                    text=texts[i:i + self.batch_size], padding=True,
                    truncation=True, max_length=cfg_max, return_tensors="pt",
                ).to(self.device)
                out = self.model.get_text_features(**inputs)
                feats.append(feature_tensor(out).float().cpu().numpy())
        return l2_normalize(np.concatenate(feats, axis=0)) if feats else \
            np.zeros((0, getattr(self, "dim", 0) or 0), np.float32)
