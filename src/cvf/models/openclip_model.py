"""OpenCLIP backend — English-space encoders for the ensemble.

Two roles:
* ``openclip``       — the big English tower used as the second ensemble
                       member on translated/enhanced queries. Tries the
                       configured hf-hub checkpoints first (Meta Perception
                       Encoder, ``timm/PE-Core-*`` — best open dual encoder
                       2026), falling back to the classic arch/pretrained
                       pair (Apple DFN5B ViT-H/14) when a hub load fails.
* ``provided_clip32``— OpenAI CLIP ViT-B/32, dimension-compatible (512) with
                       the organiser's precomputed ``clip-features-32`` packs,
                       so a full index exists minutes after download.

The embedding dim depends on which checkpoint actually loaded, so ``dim`` is
resolved lazily (probe on first access) and ``model_tag`` names the resolved
checkpoint for artifact metadata.
"""

from __future__ import annotations

import logging
import re

import numpy as np
from PIL import Image

from cvf.config import Settings
from cvf.models.base import EmbeddingModel, l2_normalize, resolve_device, resolve_dtype

log = logging.getLogger(__name__)

# A load attempt is ("hub", "hf-hub:org/name") or ("arch", (arch, pretrained)).
Attempt = tuple[str, object]


def _load_attempts(
    hub_ids: list[str], arch: str, pretrained: str
) -> list[Attempt]:
    """Ordered checkpoint attempts: every hub id first, arch/pretrained last."""
    attempts: list[Attempt] = [("hub", h) for h in hub_ids]
    attempts.append(("arch", (arch, pretrained)))
    return attempts


def _tag_for_attempt(attempt: Attempt) -> str:
    """Human/metadata-friendly checkpoint name for a load attempt.

    ``("hub", "hf-hub:timm/PE-Core-bigG-14-448")`` → ``PE-Core-bigG-14-448``;
    ``("arch", ("ViT-H-14-378-quickgelu", "dfn5b"))`` → ``dfn5b-vit-h-14-378-quickgelu``.
    """
    kind, spec = attempt
    if kind == "hub":
        return str(spec).split("/")[-1]
    arch, pretrained = spec  # type: ignore[misc]
    return re.sub(r"[^A-Za-z0-9.]+", "-", f"{pretrained}-{arch}").strip("-").lower()


def _load_with_fallback(attempts: list[Attempt], loader):
    """Try ``loader(attempt)`` in order; return ``(attempt, result)`` of the
    first success. Logs a warning naming the fallback on each failure and
    raises ``RuntimeError`` only when every attempt failed."""
    last_err: Exception | None = None
    for i, attempt in enumerate(attempts):
        try:
            return attempt, loader(attempt)
        except Exception as e:  # hub loads fail in many ways: net, disk, deps
            last_err = e
            if i + 1 < len(attempts):
                log.warning(
                    "OpenCLIP load failed for %s (%s: %s) — falling back to %s",
                    _tag_for_attempt(attempt), type(e).__name__, e,
                    _tag_for_attempt(attempts[i + 1]),
                )
    raise RuntimeError(
        f"All OpenCLIP checkpoints failed to load (last error: {last_err})"
    ) from last_err


class OpenClipModel(EmbeddingModel):
    multilingual = False  # English tower — feed it translated queries

    def __init__(self, settings: Settings, arch: str | None = None,
                 pretrained: str | None = None, key: str | None = None):
        cfg = settings.embedding
        self.arch = arch or cfg.openclip_arch
        self.pretrained = pretrained or cfg.openclip_pretrained
        self.key = key or "openclip"
        self.device = resolve_device(cfg.device)
        self.batch_size = cfg.batch_size
        # open_clip handles autocast poorly on CPU — keep fp32 there.
        self.dtype = resolve_dtype(cfg.dtype, self.device) if self.device == "cuda" else None
        self._precision = (
            "fp32" if self.dtype is None
            else ("bf16" if str(self.dtype) == "torch.bfloat16" else "fp16")
        )

        # Explicit arch/pretrained (e.g. provided_clip32) pins one checkpoint;
        # the default 'openclip' lane walks hub ids first, DFN5B last.
        if arch is not None or pretrained is not None:
            attempts = [("arch", (self.arch, self.pretrained))]
        else:
            attempts = _load_attempts(list(cfg.openclip_hub_ids), self.arch, self.pretrained)

        loaded, (model, preprocess, tokenizer) = _load_with_fallback(
            attempts, self._create_from_attempt
        )
        self._model_tag = _tag_for_attempt(loaded)
        self.model = model.to(self.device).eval()
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        self._dim: int | None = None
        log.info("OpenCLIP ready: %s (%s, %s)", self._model_tag, self.key, self.device)

    def _create_from_attempt(self, attempt: Attempt):
        """Instantiate (model, preprocess, tokenizer) for one load attempt."""
        import open_clip

        kind, spec = attempt
        log.info("Loading OpenCLIP %s (%s)", _tag_for_attempt(attempt), self.device)
        if kind == "hub":
            model, preprocess = open_clip.create_model_from_pretrained(
                str(spec), precision=self._precision
            )
            tokenizer = open_clip.get_tokenizer(str(spec))
        else:
            arch, pretrained = spec  # type: ignore[misc]
            model, _, preprocess = open_clip.create_model_and_transforms(
                arch, pretrained=pretrained, precision=self._precision
            )
            tokenizer = open_clip.get_tokenizer(arch)
        return model, preprocess, tokenizer

    @property
    def dim(self) -> int:
        """Embedding width — probed lazily (depends on the loaded checkpoint)."""
        if self._dim is None:
            self._dim = int(self.encode_text(["probe"]).shape[1])
        return self._dim

    @property
    def model_tag(self) -> str:
        """The checkpoint that actually loaded, e.g. ``PE-Core-bigG-14-448``."""
        return self._model_tag

    def encode_image(self, images: list[Image.Image]) -> np.ndarray:
        import torch

        feats = []
        with torch.no_grad():
            for i in range(0, len(images), self.batch_size):
                batch = torch.stack([self.preprocess(im) for im in images[i : i + self.batch_size]])
                batch = batch.to(self.device, dtype=self.dtype) if self.dtype else batch.to(self.device)
                out = self.model.encode_image(batch)
                feats.append(out.float().cpu().numpy())
        return l2_normalize(np.concatenate(feats, axis=0))

    def encode_text(self, texts: list[str]) -> np.ndarray:
        import torch

        feats = []
        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                toks = self.tokenizer(texts[i : i + self.batch_size]).to(self.device)
                out = self.model.encode_text(toks)
                feats.append(out.float().cpu().numpy())
        return l2_normalize(np.concatenate(feats, axis=0))


def provided_clip32_model(settings: Settings) -> OpenClipModel:
    """OpenAI CLIP ViT-B/32 — matches the organiser's 512-d feature packs."""
    return OpenClipModel(settings, arch="ViT-B-32", pretrained="openai", key="provided_clip32")
