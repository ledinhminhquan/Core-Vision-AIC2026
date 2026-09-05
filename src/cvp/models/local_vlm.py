"""Round-96: ONE local vision-language model shared by the QA fallback and the
VLM-rerank fallback (``vqa.local_backend = "hf_auto"``, id from ``vqa.local_hf_id``).

Plan B khi Gemini bão (sơ tuyển 3, 04/09/2026: 80–90 % cuộc gọi 503/504 suốt 8 giờ):
một VLM open-weights nạp lười lên GPU đang có (A100 80 GB còn ~58 GiB sau
SigLIP2 + MetaCLIP-2 + Qwen3-VL-Reranker-8B). Đọc nhiều ảnh trong MỘT lượt
(strip QA 3–6 khung; chấm điểm từng khung cho rerank). Chat template chuẩn HF
(``AutoModelForImageTextToText`` + ``AutoProcessor``) nên id nào cũng chạy được
— id KHÔNG được đoán: để trống thì fail loud, chỉ khai id đã kiểm chứng.

Singleton theo (model_id): nạp một lần cho cả tiến trình; mọi lượt sinh được
tuần tự hóa (một GPU model, một caller) — pool QA song song gọi vào đây an toàn.
"""
from __future__ import annotations

import logging
import threading
import time

log = logging.getLogger(__name__)

_LOCK = threading.RLock()
_MODEL: tuple | None = None          # (model_id, model, processor, device)


def local_vlm_id(settings) -> str:
    return str(getattr(settings.vqa, "local_hf_id", "") or "").strip()


def is_configured(settings) -> bool:
    return (getattr(settings.vqa, "local_backend", "vintern") == "hf_auto"
            and bool(local_vlm_id(settings)))


def get_local_vlm(model_id: str):
    """Load (once) and return ``(model, processor, device)`` for ``model_id``."""
    global _MODEL
    if not model_id:
        raise RuntimeError(
            "vqa.local_backend='hf_auto' nhưng vqa.local_hf_id trống — khai id model "
            "đã kiểm chứng (nb03: LOCAL_VLM_ID) rồi chạy lại; không đoán id.")
    with _LOCK:
        if _MODEL is not None and _MODEL[0] == model_id:
            return _MODEL[1:]
        if _MODEL is not None:                   # another id → free the GPU copy first
            log.info("Local VLM switch %s → %s: unloading the old model", _MODEL[0], model_id)
            _MODEL = None
            import gc

            gc.collect()
            try:
                import torch as _t

                _t.cuda.empty_cache()
            except Exception:  # noqa: BLE001 — best effort
                pass
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        from cvp.models.hf_compat import resilient_from_pretrained

        t0 = time.time()
        log.info("Loading local hf_auto VLM %s (plan B khi Gemini bão)", model_id)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        model = resilient_from_pretrained(
            lambda p: AutoModelForImageTextToText.from_pretrained(p, torch_dtype=dtype),
            model_id).to(device).eval()
        processor = resilient_from_pretrained(lambda p: AutoProcessor.from_pretrained(p), model_id)
        _MODEL = (model_id, model, processor, device)
        log.info("Local VLM %s ready on %s in %.0fs", model_id, device, time.time() - t0)
        return _MODEL[1:]


def generate(model_id: str, images: list, prompt: str, *, max_new_tokens: int = 64) -> str:
    """One chat turn: ``images`` (PIL, in order) + ``prompt`` → decoded text."""
    if not images:
        raise RuntimeError("local VLM: no readable image")
    with _LOCK:
        model, processor, device = get_local_vlm(model_id)
        import torch

        content = [{"type": "image", "image": img} for img in images]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        # enable_thinking=False: Qwen3.5-class templates think by default (minutes
        # of hidden tokens per answer); templates without the variable ignore it.
        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt", enable_thinking=False).to(device)
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=int(max_new_tokens), do_sample=False)
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        return str(processor.decode(new_tokens, skip_special_tokens=True)).strip()


def unload() -> None:
    """Free the GPU copy (tests / VRAM-tight sessions)."""
    global _MODEL
    with _LOCK:
        _MODEL = None
        try:
            import gc
            import torch
            gc.collect()
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 — best effort
            pass
