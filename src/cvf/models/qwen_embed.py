"""Qwen3-VL-Embedding backend — native-Vietnamese multimodal lane (optional).

``Qwen/Qwen3-VL-Embedding-2B`` turns an MLLM into a dual-tower embedder: the
last-layer hidden state of the final ([EOS]) token is the semantic vector
(per the official Qwen3VLEmbedder). Best native-vi zero-shot retrieval in its
class (32.13 R@1 UIT-OpenViIC), but heavy at index time — an *optional*
ensemble member, off by default.

Matryoshka (MRL) output: the model supports 64–2048 dims; vectors are
truncated to ``embedding.qwen_embed_dim`` and re-normalized, trading a little
recall for index size/speed. Text QUERIES are instruction-prefixed
(``embedding.qwen_embed_instruction``) via the chat template's system turn;
DOCUMENT (image) embeddings carry no instruction — matching the official
Qwen3VLEmbedder convention where only the query side is instruction-aware.

Requires ``transformers>=4.57`` (first release with the Qwen3-VL classes).
"""

from __future__ import annotations

import logging

import numpy as np
from PIL import Image

from cvf.config import Settings
from cvf.models.base import EmbeddingModel, l2_normalize, resolve_device

log = logging.getLogger(__name__)


def _truncate_and_renorm(arr: np.ndarray, dim: int) -> np.ndarray:
    """MRL truncation: keep the first ``dim`` dims, then L2-renormalize.

    Matryoshka-trained embeddings pack coarse-to-fine information front to
    back, so a prefix slice is a valid lower-dim embedding once re-normalized.
    ``dim <= 0`` or ``dim >= width`` keeps the full width (still returns
    normalized float32).
    """
    arr = np.asarray(arr, dtype=np.float32)
    if 0 < dim < arr.shape[-1]:
        arr = arr[..., :dim]
    return l2_normalize(arr)


class QwenEmbedModel(EmbeddingModel):
    multilingual = True  # reads Vietnamese natively — no translation needed

    def __init__(self, settings: Settings, key: str | None = None):
        import torch
        from transformers import AutoModel, AutoProcessor

        cfg = settings.embedding
        self.key = key or "qwen_embed"
        self.model_id = cfg.qwen_embed_id
        self.instruction = cfg.qwen_embed_instruction
        self.mrl_dim = int(cfg.qwen_embed_dim)
        self.batch_size = cfg.batch_size
        self.device = resolve_device(cfg.device)
        # bf16 on capable CUDA, fp32 elsewhere (MLLM towers are fragile in fp16 on CPU).
        self.dtype = (
            torch.bfloat16
            if self.device == "cuda" and torch.cuda.is_bf16_supported()
            else torch.float32
        )

        log.info("Loading %s (%s, %s, MRL dim %d)", self.model_id, self.device, self.dtype, self.mrl_dim)
        self.model = (
            AutoModel.from_pretrained(self.model_id, torch_dtype=self.dtype, trust_remote_code=True)
            .to(self.device)
            .eval()
        )
        self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        self._dim: int | None = None

    @property
    def dim(self) -> int:
        """Embedding width after MRL truncation — probed lazily."""
        if self._dim is None:
            self._dim = int(self.encode_text(["probe"]).shape[1])
        return self._dim

    @property
    def model_tag(self) -> str:
        """Resolved model identity for artifact metadata, e.g. ``Qwen3-VL-Embedding-2B-mrl1024``."""
        return f"{self.model_id.split('/')[-1]}-mrl{self.mrl_dim}"

    # ── internals ─────────────────────────────────────────────────────────

    def _messages(self, *, text: str | None = None, image: Image.Image | None = None,
                  with_instruction: bool = True) -> list[dict]:
        """One chat-template conversation; the instruction system turn is only
        added for QUERIES (``with_instruction=True``) — documents (keyframe
        images) are encoded plain, per the official Qwen3VLEmbedder usage."""
        content: list[dict] = []
        if image is not None:
            content.append({"type": "image", "image": image})
        if text is not None:
            content.append({"type": "text", "text": text})
        messages: list[dict] = []
        if with_instruction:
            messages.append(
                {"role": "system", "content": [{"type": "text", "text": self.instruction}]}
            )
        messages.append({"role": "user", "content": content})
        return messages

    def _embed_batch(self, conversations: list[list[dict]],
                     images: list[Image.Image] | None = None) -> np.ndarray:
        """Forward one batch; pool the last valid ([EOS]) token's hidden state."""
        import torch

        texts = [
            self.processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=False)
            for conv in conversations
        ]
        kwargs: dict = {"text": texts, "padding": True, "return_tensors": "pt"}
        if images:
            kwargs["images"] = images
        inputs = self.processor(**kwargs).to(self.device)
        with torch.no_grad():
            out = self.model(**inputs)
        hidden = getattr(out, "last_hidden_state", None)
        if hidden is None:
            hidden = out.hidden_states[-1]
        # Last non-pad position per row, robust to either padding side
        # (official Qwen3VLEmbedder pooling: flip the mask, argmax).
        mask = inputs["attention_mask"]
        last = mask.shape[1] - 1 - mask.flip(dims=[1]).float().argmax(dim=1)
        pooled = hidden[torch.arange(hidden.shape[0], device=hidden.device), last]
        return pooled.float().cpu().numpy()

    # ── EmbeddingModel contract ───────────────────────────────────────────

    def encode_image(self, images: list[Image.Image]) -> np.ndarray:
        if not images:
            return np.zeros((0, self.mrl_dim), dtype=np.float32)
        feats = []
        for i in range(0, len(images), self.batch_size):
            batch = images[i : i + self.batch_size]
            # Documents are embedded WITHOUT the instruction turn (queries only).
            convs = [self._messages(image=im, with_instruction=False) for im in batch]
            feats.append(self._embed_batch(convs, images=batch))
        return _truncate_and_renorm(np.concatenate(feats, axis=0), self.mrl_dim)

    def encode_text(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.mrl_dim), dtype=np.float32)
        feats = []
        for i in range(0, len(texts), self.batch_size):
            convs = [self._messages(text=t) for t in texts[i : i + self.batch_size]]
            feats.append(self._embed_batch(convs))
        return _truncate_and_renorm(np.concatenate(feats, axis=0), self.mrl_dim)
