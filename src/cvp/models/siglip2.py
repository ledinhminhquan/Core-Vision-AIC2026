"""SigLIP-2 backend (default): multilingual, reads Vietnamese natively.

``google/siglip2-so400m-patch16-384`` — 1152-d shared space, trained
multilingually (WebLI); the strongest open zero-shot retrieval encoder that
also accepts raw Vietnamese text, which makes it the safety net when
translation is offline. The `finetuned` variant grafts a Vietnamese
LiT-tuned text tower on top of the same (frozen) image tower, so image
embeddings — and the FAISS index — are reused unchanged.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PIL import Image

from cvp.config import Settings
from cvp.models.hf_compat import feature_tensor
from cvp.models.base import EmbeddingModel, l2_normalize, resolve_device, resolve_dtype

log = logging.getLogger(__name__)


class SigLIP2Model(EmbeddingModel):
    multilingual = True

    def __init__(self, settings: Settings, finetuned: bool = False):
        import torch
        from transformers import AutoModel, AutoProcessor

        cfg = settings.embedding
        self.key = "finetuned" if finetuned else "siglip2"
        self.device = resolve_device(cfg.device)
        self.dtype = resolve_dtype(cfg.dtype, self.device)
        self.batch_size = cfg.batch_size
        self.text_max_length = cfg.text_max_length

        base_id = settings.finetuned.base_id if finetuned else cfg.siglip2_id
        log.info("Loading %s (%s, %s, %s)", base_id, self.key, self.device, self.dtype)
        from cvp.models.hf_compat import resilient_from_pretrained

        # verify-R23: the competition-primary lane must survive a torn
        # Drive-cache read (local re-download fallback, như ASR round-22).
        self.model = resilient_from_pretrained(
            lambda mid: AutoModel.from_pretrained(mid, torch_dtype=self.dtype),
            base_id).to(self.device).eval()
        self.processor = resilient_from_pretrained(
            AutoProcessor.from_pretrained, base_id)

        if finetuned:
            self._load_finetuned_text_tower(self._resolve_checkpoint(settings))

        with torch.no_grad():
            probe = self.encode_text(["probe"])
        self.dim = int(probe.shape[1])

    @staticmethod
    def _resolve_checkpoint(settings: Settings) -> Path:
        """Find the exported checkpoint whether the config path is absolute,
        CWD-relative, or artifacts-relative (Colab exports go to Drive)."""
        raw = Path(settings.finetuned.checkpoint)
        candidates = [raw]
        if not raw.is_absolute():
            candidates.append(settings.paths.artifacts_root / raw)
            # config default "./artifacts/checkpoints/X" → artifacts_root/"checkpoints/X"
            parts = raw.parts
            if "artifacts" in parts:
                idx = parts.index("artifacts")
                candidates.append(settings.paths.artifacts_root.joinpath(*parts[idx + 1:]))
        for c in candidates:
            if (c / "text_tower.safetensors").is_file() or (c / "model.safetensors").is_file():
                return c
        return raw  # let the loader raise its descriptive error

    def _load_finetuned_text_tower(self, ckpt: Path) -> None:
        """Graft a LiT-tuned text tower; image tower stays the base one."""
        from safetensors.torch import load_file

        candidates = [ckpt / "text_tower.safetensors", ckpt / "model.safetensors"]
        path = next((p for p in candidates if p.is_file()), None)
        if path is None:
            raise FileNotFoundError(
                f"Fine-tuned checkpoint not found under {ckpt} "
                "(expected text_tower.safetensors — produced by the training notebook)."
            )
        state = load_file(str(path))
        # Accept either bare text-tower keys or full-model keys.
        text_state = {}
        for k, v in state.items():
            if k.startswith("text_model."):
                text_state[k[len("text_model."):]] = v
            elif not k.startswith(("vision_model.", "logit_")):
                text_state[k] = v
        missing, unexpected = self.model.text_model.load_state_dict(text_state, strict=False)
        if missing:
            log.warning("Fine-tuned text tower: %d missing keys (first: %s)", len(missing), missing[:3])
        self.model.to(self.device, dtype=self.dtype)
        log.info("Grafted fine-tuned Vietnamese text tower from %s", path)

    # ── encoding ─────────────────────────────────────────────────────────

    def encode_image(self, images: list[Image.Image]) -> np.ndarray:
        import torch

        feats = []
        with torch.no_grad():
            for i in range(0, len(images), self.batch_size):
                batch = images[i : i + self.batch_size]
                inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
                if "pixel_values" in inputs:
                    inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)
                out = self.model.get_image_features(**inputs)
                feats.append(feature_tensor(out).float().cpu().numpy())
        return l2_normalize(np.concatenate(feats, axis=0))

    def encode_text(self, texts: list[str]) -> np.ndarray:
        import torch

        feats = []
        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i : i + self.batch_size]
                # SigLIP was trained with fixed-length padded text (64 tokens).
                inputs = self.processor(
                    text=batch, padding="max_length", max_length=self.text_max_length,
                    truncation=True, return_tensors="pt",
                ).to(self.device)
                out = self.model.get_text_features(**inputs)
                feats.append(feature_tensor(out).float().cpu().numpy())
        return l2_normalize(np.concatenate(feats, axis=0))
