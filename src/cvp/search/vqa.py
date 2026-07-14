"""VQA assistant for the QA task: propose an answer from the top frames.

The operator stays in charge — this fills a *suggested* answer next to each
candidate frame; a human confirms or edits before export (VLM answers are not
trusted blindly). Providers: Gemini (best; needs GEMINI_API_KEY) or a local
Vintern-1B (offline fallback, Vietnamese-tuned).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from cvp.config import Settings
from cvp.constants import MAX_QA_ANSWER_CHARS
from cvp.utils.images import load_rgb

log = logging.getLogger(__name__)

_VQA_PROMPT = (
    "Bạn đang xem một khung hình từ video tin tức Việt Nam. "
    "Trả lời NGẮN GỌN câu hỏi sau (chỉ đáp án, tối đa 100 ký tự, không giải thích):\n{question}"
)

# Multi-frame strip (round-2026 upgrade): the real 2025 "math in video" QA
# needed reading SEVERAL consecutive frames — one frame never carries the whole
# problem statement. One call sees the whole strip, so cross-frame consistency
# is free.
_VQA_STRIP_PROMPT = (
    "Bạn đang xem {n} khung hình LIÊN TIẾP theo thứ tự thời gian từ CÙNG MỘT "
    "cảnh trong video tin tức Việt Nam. Kết hợp thông tin từ TẤT CẢ các khung "
    "hình (chữ trên màn hình có thể trải dài qua nhiều khung) và trả lời NGẮN "
    "GỌN câu hỏi sau (chỉ đáp án, tối đa 100 ký tự, không giải thích):\n{question}"
)


@dataclass
class VqaAnswer:
    global_id: int
    answer: str
    provider: str


def gemini_model_chain(settings: Settings, primary: str) -> list[str]:
    """Primary model + the query section's fallback ids (deduped, in order).

    Model ids churn mid-season (previews get retired) — every Gemini call site
    (query enhancement, VQA, VLM rerank) walks the same chain so a stale id
    degrades to the next Gemini model instead of failing the feature
    (review finding C9: docs promised this for VQA/rerank too).
    """
    fallbacks = list(getattr(settings.query, "gemini_model_fallbacks", []))
    return [primary] + [m for m in fallbacks if m != primary]


def generate_with_fallback(client, models: list[str], contents) -> str:
    """One generate_content call, trying each model id until one answers."""
    last: Exception | None = None
    for model_id in models:
        try:
            resp = client.models.generate_content(model=model_id, contents=contents)
            return (resp.text or "").strip()
        except Exception as e:  # noqa: BLE001 — try the next model id
            last = e
            log.warning("Gemini model %r failed (%s) — trying next fallback", model_id, e)
    raise last if last else RuntimeError("no Gemini model succeeded")


class VqaAssistant:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.cfg = settings.vqa
        self._gemini_client = None
        self._local = None

    # ── providers ────────────────────────────────────────────────────────

    def _ask_gemini(self, image_path: str, question: str) -> str:
        from google import genai

        if self._gemini_client is None:
            api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            if not api_key:
                raise RuntimeError("GEMINI_API_KEY not set")
            self._gemini_client = genai.Client(api_key=api_key)
        img = load_rgb(image_path)
        if img is None:
            raise RuntimeError(f"Unreadable image: {image_path}")
        return generate_with_fallback(
            self._gemini_client,
            gemini_model_chain(self.settings, self.cfg.gemini_model),
            [_VQA_PROMPT.format(question=question), img],
        )

    def _ask_local(self, image_path: str, question: str) -> str:
        """Vintern-1B (InternVL family) — loaded lazily, cached."""
        import torch
        from transformers import AutoModel, AutoTokenizer

        if self._local is None:
            model_id = self.cfg.local_model
            log.info("Loading local VQA model %s", model_id)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.bfloat16 if device == "cuda" else torch.float32
            model = AutoModel.from_pretrained(
                model_id, torch_dtype=dtype, trust_remote_code=True
            ).to(device).eval()
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=False)
            self._local = (model, tokenizer, device, dtype)
        model, tokenizer, device, dtype = self._local

        from cvp.auxindex.vintern_preprocess import load_image_tiles

        pixel_values = load_image_tiles(image_path, max_num=self.settings.caption.max_tiles)
        pixel_values = pixel_values.to(device=device, dtype=dtype)
        prompt = "<image>\n" + _VQA_PROMPT.format(question=question)
        gen_cfg = dict(max_new_tokens=64, do_sample=False, num_beams=2)
        answer = model.chat(tokenizer, pixel_values, prompt, gen_cfg)
        return str(answer).strip()

    def _ask_gemini_strip(self, image_paths: list[str], question: str) -> str:
        from google import genai

        if self._gemini_client is None:
            api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            if not api_key:
                raise RuntimeError("GEMINI_API_KEY not set")
            self._gemini_client = genai.Client(api_key=api_key)
        imgs = []
        for p in image_paths:
            img = load_rgb(p)
            if img is not None:
                img.thumbnail((768, 768))
                imgs.append(img)
        if not imgs:
            raise RuntimeError("no readable frame in the strip")
        return generate_with_fallback(
            self._gemini_client,
            gemini_model_chain(self.settings, self.cfg.gemini_model),
            [_VQA_STRIP_PROMPT.format(n=len(imgs), question=question), *imgs],
        )

    # ── public ───────────────────────────────────────────────────────────

    def answer_group(self, question: str, image_paths: list[str]) -> str:
        """ONE answer for a temporal strip of frames from one candidate group.

        Gemini sees the whole strip in a single call (multi-frame evidence —
        fixes the 2025 "math in video" failure mode where the problem statement
        spans several consecutive frames). Degrades to the local single-frame
        model on the strip's middle frame, then to "" (caller falls back).
        """
        paths = [p for p in image_paths if p][: max(1, int(self.cfg.frames_per_answer))]
        if not paths:
            return ""
        if self.cfg.provider == "gemini":
            try:
                return self._ask_gemini_strip(paths, question)[:MAX_QA_ANSWER_CHARS]
            except Exception as e:  # noqa: BLE001 — degrade to local model
                log.warning("Gemini strip-VQA failed (%s) — trying local model", e)
        if self.cfg.provider in ("gemini", "vintern"):
            try:
                middle = paths[len(paths) // 2]
                return self._ask_local(middle, question)[:MAX_QA_ANSWER_CHARS]
            except Exception as e:  # noqa: BLE001
                log.warning("Local VQA failed: %s", e)
        return ""

    def suggest(self, question: str, frames: list[tuple[int, str]]) -> list[VqaAnswer]:
        """frames: [(global_id, image_path)] — returns one suggestion per frame."""
        out: list[VqaAnswer] = []
        for gid, path in frames[: self.cfg.top_frames]:
            answer, provider = "", "none"
            if self.cfg.provider == "gemini":
                try:
                    answer, provider = self._ask_gemini(path, question), "gemini"
                except Exception as e:  # noqa: BLE001 — degrade to local model
                    log.warning("Gemini VQA failed (%s) — trying local model", e)
            if not answer and self.cfg.provider in ("gemini", "vintern"):
                try:
                    answer, provider = self._ask_local(path, question), "vintern"
                except Exception as e:  # noqa: BLE001
                    log.warning("Local VQA failed: %s", e)
            if answer:
                out.append(VqaAnswer(
                    global_id=gid, answer=answer[:MAX_QA_ANSWER_CHARS], provider=provider,
                ))
        return out
