"""M-CLIP backend (optional ensemble member): multilingual XLM-R text tower
aligned to OpenCLIP ViT-L/14 images. Older than SigLIP-2 but adds diversity —
its Vietnamese behaviour differs enough to rescue some queries in fusion.
"""

from __future__ import annotations

import logging

import numpy as np
from PIL import Image

from cvf.config import Settings
from cvf.models.base import EmbeddingModel, l2_normalize, resolve_device

log = logging.getLogger(__name__)


class MClipModel(EmbeddingModel):
    multilingual = True

    def __init__(self, settings: Settings):
        import open_clip
        import torch
        from multilingual_clip import pt_multilingual_clip
        from transformers import AutoTokenizer

        cfg = settings.embedding
        self.key = "mclip"
        self.device = resolve_device(cfg.device)
        self.batch_size = cfg.batch_size

        log.info("Loading M-CLIP %s", cfg.mclip_id)
        self.text_model = pt_multilingual_clip.MultilingualCLIP.from_pretrained(cfg.mclip_id)
        self.text_model = self.text_model.to(self.device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.mclip_id)

        model, _, preprocess = open_clip.create_model_and_transforms(
            cfg.mclip_image_arch, pretrained=cfg.mclip_image_pretrained
        )
        self.image_model = model.to(self.device).eval()
        self.preprocess = preprocess

        with torch.no_grad():
            probe = self.encode_text(["probe"])
        self.dim = int(probe.shape[1])

    def encode_image(self, images: list[Image.Image]) -> np.ndarray:
        import torch

        feats = []
        with torch.no_grad():
            for i in range(0, len(images), self.batch_size):
                batch = torch.stack([self.preprocess(im) for im in images[i : i + self.batch_size]]).to(self.device)
                out = self.image_model.encode_image(batch)
                feats.append(out.float().cpu().numpy())
        return l2_normalize(np.concatenate(feats, axis=0))

    def encode_text(self, texts: list[str]) -> np.ndarray:
        import torch

        feats = []
        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i : i + self.batch_size]
                out = self.text_model.forward(batch, self.tokenizer)
                feats.append(out.float().cpu().numpy())
        return l2_normalize(np.concatenate(feats, axis=0))
