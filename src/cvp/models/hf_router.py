"""Round-96: Hugging Face Inference Providers as an INDEPENDENT vendor lane.

Verified 05/09/2026 (docs + router probes, see memory): ``router.huggingface.co``
is OpenAI-compatible, charges the provider's own price with no markup, and
serves e.g. ``Qwen/Qwen3-VL-235B-A22B-Instruct`` (novita, deepinfra),
``Qwen/Qwen3-VL-30B-A3B-Instruct`` (novita, deepinfra, featherless-ai),
``zai-org/GLM-5.3-Flash`` (6 providers), ``google/gemma-4-31B-it`` (5 providers).
A model id may pin a provider: ``org/model:novita``. HF PRO ($9/month) gives a
small monthly credit; pay-as-you-go credits must be pre-purchased.

Usage inside the repo: a chain entry ``hf:<model[:provider]>`` anywhere a Gemini
model id is accepted (``query.gemini_model_fallbacks``, ``vqa.answer_model``,
``search.vlm_rerank_model``). :class:`cvp.models.gemini_keys.RotatingGeminiClient`
dispatches the prefix here, so every call site (enhancement, VQA strips, VLM
rerank) gets the lane for free and the circuit breaker tracks it as its own
model. Needs the Colab secret ``HF_TOKEN``; without it ``hf:`` entries are
dropped from every chain (``available()``).

``huggingface_hub.InferenceClient`` waits forever by default — the timeout is
mandatory here (the 45–90 s wall clock of the caller).
"""
from __future__ import annotations

import base64
import io
import logging
import os
import threading
from types import SimpleNamespace

log = logging.getLogger(__name__)

PREFIX = "hf:"
_LOCK = threading.Lock()
_CLIENTS: dict[float, object] = {}


def token() -> str:
    return (os.environ.get("HF_TOKEN") or "").strip()


def available() -> bool:
    return bool(token())


def is_hf_id(model_id: str) -> bool:
    return str(model_id or "").startswith(PREFIX)


def strip_prefix(model_id: str) -> str:
    return model_id[len(PREFIX):] if is_hf_id(model_id) else model_id


def _client(timeout_s: float):
    from huggingface_hub import InferenceClient

    with _LOCK:
        c = _CLIENTS.get(timeout_s)
        if c is None:
            c = InferenceClient(token=token(), timeout=float(timeout_s))
            _CLIENTS[timeout_s] = c
        return c


def image_part(img, *, max_side: int = 768, quality: int = 75) -> dict:
    """PIL image → OpenAI-style ``image_url`` part with a base64 JPEG data URL."""
    im = img.convert("RGB") if img.mode != "RGB" else img
    if max(im.size) > max_side:
        im = im.copy()
        im.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=int(quality))
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}


def build_messages(text: str, images: list) -> list[dict]:
    content: list[dict] = [{"type": "text", "text": text}]
    content.extend(image_part(img) for img in images)
    return [{"role": "user", "content": content}]


def split_contents(contents) -> tuple[str, list]:
    """Gemini-style ``contents`` (str | [str | PIL, ...]) → (text, images)."""
    if isinstance(contents, str):
        return contents, []
    texts = [c for c in contents if isinstance(c, str)]
    images = [c for c in contents if not isinstance(c, str)]
    return "\n".join(texts), images


def generate(model_id: str, contents, *, timeout_s: float = 45.0,
             max_tokens: int = 512) -> SimpleNamespace:
    """One chat completion through the HF router; returns an object with ``.text``
    like ``genai`` responses so call sites need no branch."""
    if not available():
        raise RuntimeError("hf: lane needs the HF_TOKEN secret (chưa cấu hình)")
    model = strip_prefix(model_id)
    text, images = split_contents(contents)
    client = _client(float(timeout_s))
    resp = client.chat_completion(model=model, messages=build_messages(text, images),
                                  max_tokens=int(max_tokens), temperature=0.0)
    choice = (resp.choices or [None])[0]
    out = ""
    if choice is not None:
        msg = getattr(choice, "message", None)
        out = getattr(msg, "content", None) or ""
    return SimpleNamespace(text=str(out).strip(), model=model)
