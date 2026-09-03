"""VQA assistant for the QA task: propose an answer from the top frames.

The operator stays in charge — this fills a *suggested* answer next to each
candidate frame; a human confirms or edits before export (VLM answers are not
trusted blindly). Providers: Gemini (best; needs GEMINI_API_KEY) or a local
Vintern-1B (offline fallback, Vietnamese-tuned).
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass

from cvp.config import Settings
from cvp.constants import MAX_QA_ANSWER_CHARS
from cvp.utils.images import load_rgb

log = logging.getLogger(__name__)

_VQA_PROMPT = (
    "Bạn đang xem một khung hình từ video tiếng Việt. "
    "Trả lời NGẮN GỌN câu hỏi sau (chỉ đáp án, tối đa 100 ký tự, không giải thích):\n{question}"
)

# Multi-frame strip (round-2026 upgrade): the real 2025 "math in video" QA
# needed reading SEVERAL consecutive frames — one frame never carries the whole
# problem statement. One call sees the whole strip, so cross-frame consistency
# is free.
_VQA_STRIP_PROMPT = (
    "Bạn đang xem {n} khung hình LIÊN TIẾP theo thứ tự thời gian từ CÙNG MỘT "
    "cảnh trong video tiếng Việt. Kết hợp thông tin từ TẤT CẢ các khung "
    "hình (chữ trên màn hình có thể trải dài qua nhiều khung) và trả lời NGẮN "
    "GỌN câu hỏi sau (chỉ đáp án, tối đa 100 ký tự, không giải thích):\n{question}"
)


# Round-77 (bài học q19 sơ tuyển 2: model diễn đạt lại câu thơ, lệch GT đúng
# MỘT chữ): khi vqa.exact_transcription bật, mọi prompt QA yêu cầu chép
# NGUYÊN VĂN chữ/số hiển thị thay vì paraphrase.
_EXACT_SUFFIX = (
    "\nNếu đáp án là chữ/số hiển thị trong video, hãy chép CHÍNH XÁC nguyên "
    "văn từng chữ, từng dấu như trên màn hình — không thêm bớt từ, không "
    "diễn đạt lại."
)


def _exact_suffix(cfg) -> str:
    return _EXACT_SUFFIX if getattr(cfg, "exact_transcription", False) else ""


def _with_context(prompt: str, context: str) -> str:
    """Prepend the ASR transcript around the candidate moment.

    Buổi tập huấn 4: đây là Q&A chứ không phải VQA — câu hỏi có thể dựa trên
    ÂM THANH (lời thoại) chứ không chỉ hình ảnh. Model phải được đọc thoại.
    """
    if not context:
        return prompt
    return (f"Lời thoại trong đoạn video quanh khoảnh khắc này (nhận dạng từ "
            f'âm thanh, có thể sai chính tả): "{context}"\n{prompt}')


def asr_context(settings: Settings, video_id: str, pts_time: float,
                window_s: float = 20.0) -> str:
    """Transcript text overlapping [pts_time ± window_s] from artifacts/asr."""
    from cvp.utils.io import read_json

    p = settings.paths.art("asr") / f"{video_id}.json"
    if not p.exists():
        return ""
    segs = (read_json(p, default={}) or {}).get("segments") or []
    lo, hi = pts_time - window_s, pts_time + window_s
    texts = [str(s.get("text", "")).strip() for s in segs
             if s.get("text") and float(s.get("end", 0)) >= lo
             and float(s.get("start", 0)) <= hi]
    return " ".join(t for t in texts if t)[:800]


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


def make_gemini_client(settings: Settings):
    """genai.Client with the same HTTP timeout discipline as QueryProcessor.

    A hung request on a flaky venue network must never freeze the caller — in
    the Streamlit UI that thread IS the session, and the only operator escape
    (browser refresh) wipes every basket/hint/timer (round-6 HIGH).
    """
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set")
    from google import genai

    # Same budget as the LARGEST wall cap in play: round-46 live bug — the
    # client HTTP deadline was built from the 45s generic wall, so the QA
    # answer path's 90s Pro budget never applied (server returned 504 at the
    # HTTP deadline long before the Pro model finished thinking).
    budget = max(gemini_wall_timeout(settings),
                 float(getattr(settings.vqa, "answer_timeout_s", 0.0)))
    try:
        # google-genai HttpOptions.timeout is in MILLISECONDS.
        return genai.Client(api_key=api_key,
                            http_options={"timeout": int(budget * 1000)})
    except TypeError:  # older google-genai without http_options
        return genai.Client(api_key=api_key)


def gemini_wall_timeout(settings: Settings) -> float:
    """Per-attempt hard wall-clock cap for image-carrying Gemini calls.

    3× the text-query timeout, floored at 30 s: image payloads are legitimately
    slower than text, but an unbounded hang is never acceptable.
    """
    return max(30.0, float(getattr(settings.query, "timeout_s", 15.0)) * 3)


def generate_with_fallback(client, models: list[str], contents,
                           timeout_s: float | None = None,
                           economical: bool = False) -> str:
    """One generate_content call, trying each model id until one answers.

    ``timeout_s`` adds a hard per-attempt wall-clock cap (on expiry the model
    id is treated as failed and the next fallback is tried) — the HTTP-level
    timeout alone cannot stop a stalled read on every transport.
    ``economical=True`` applies :func:`economical_config`; a config the server
    rejects falls back to a config-less retry of the SAME model before the
    chain moves on (a billing knob must never cost the feature).
    """
    from cvp.models.query_processor import (_call_with_timeout, config_rejected,
                                            economical_config)

    last: Exception | None = None
    for model_id in models:
        cfgs: list = [None]
        if economical:
            ec = economical_config(model_id)
            if ec is not None:
                cfgs = [ec, None]      # economical first, plain as the rescue
        for cfg in cfgs:
            try:
                def _do(mid=model_id, c=cfg):
                    if c is not None:
                        return client.models.generate_content(
                            model=mid, contents=contents, config=c)
                    return client.models.generate_content(model=mid, contents=contents)

                resp = _call_with_timeout(_do, timeout_s) if timeout_s else _do()
                return (resp.text or "").strip()
            except Exception as e:  # noqa: BLE001 — next config, then next model
                last = e
                log.warning("Gemini model %r%s failed (%s) — trying next",
                            model_id, " (economical)" if cfg is not None else "", e)
                if cfg is not None and not config_rejected(e):
                    break  # round-89: transient (429/5xx/timeout) → next MODEL,
                           # never a full-price config-less retry of the same one
            if cfg is None:
                break  # plain attempt done — move to the next model id
    raise last if last else RuntimeError("no Gemini model succeeded")


class VqaAssistant:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.cfg = settings.vqa
        self._gemini_client = None
        self._local = None

    # ── providers ────────────────────────────────────────────────────────

    # Audit r82: with vqa.parallel_calls > 1 several groups call this object
    # from pool threads. The lazy local model (Vintern / hf_auto) is a single
    # GPU resource — its load AND its use are serialized; the lazy Gemini
    # client init is double-checked so only one instance is built.
    _LOCK = threading.RLock()

    def _ask_gemini(self, image_path: str, question: str, context: str = "") -> str:
        if self._gemini_client is None:
            with self._LOCK:
                if self._gemini_client is None:
                    self._gemini_client = make_gemini_client(self.settings)
        img = load_rgb(image_path)
        if img is None:
            raise RuntimeError(f"Unreadable image: {image_path}")
        return generate_with_fallback(
            self._gemini_client,
            gemini_model_chain(self.settings, self.cfg.gemini_model),
            [_with_context(_VQA_PROMPT.format(question=question)
                           + _exact_suffix(self.cfg), context), img],
            timeout_s=gemini_wall_timeout(self.settings),
            economical=True,      # round-45: suggests are advisory — cheap tier
        )

    def _ask_local(self, image_path: str, question: str, context: str = "") -> str:
        """Local VQA fallback — Vintern (mặc định) hoặc một VLM chat-template
        bất kỳ qua ``vqa.local_backend="hf_auto"`` (round-79: Qwen3.5-class).
        """
        with self._LOCK:      # one GPU model, one caller at a time
            if getattr(self.cfg, "local_backend", "vintern") == "hf_auto":
                return self._ask_local_hf_auto(image_path, question, context)
            return self._ask_local_vintern(image_path, question, context)

    def _ask_local_hf_auto(self, image_path: str, question: str,
                           context: str = "") -> str:
        """Round-79: fallback local thế-hệ-2026 qua chat template chuẩn HF.

        Nghiên cứu 29/08: Qwen3.5-9B/27B (201 ngôn ngữ, Apache-2.0) vượt xa
        Vintern-1B khi Gemini sập — nạp lười CHỈ lúc cần, id lấy từ
        ``vqa.local_hf_id`` (bắt buộc khai — không đoán id chưa kiểm chứng).
        """
        model_id = str(getattr(self.cfg, "local_hf_id", "") or "").strip()
        if not model_id:
            raise RuntimeError(
                "vqa.local_backend='hf_auto' nhưng vqa.local_hf_id trống — "
                "khai id model (vd bản Qwen3.5 đã kiểm chứng) rồi chạy lại.")
        import torch

        if self._local is not None and self._local[0] != "hf_auto":
            self._local = None      # audit r79: đổi backend giữa phiên → nạp lại
        if self._local is None:
            from transformers import AutoModelForImageTextToText, AutoProcessor

            log.info("Loading local hf_auto VQA model %s", model_id)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.bfloat16 if device == "cuda" else torch.float32
            model = AutoModelForImageTextToText.from_pretrained(
                model_id, torch_dtype=dtype).to(device).eval()
            processor = AutoProcessor.from_pretrained(model_id)
            self._local = ("hf_auto", model, processor, device)
        _tag, model, processor, device = self._local
        img = load_rgb(image_path)
        if img is None:
            raise RuntimeError(f"Unreadable image: {image_path}")
        prompt = _with_context(
            _VQA_PROMPT.format(question=question) + _exact_suffix(self.cfg), context)
        messages = [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": prompt}]}]
        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt").to(device)
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=64, do_sample=False)
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        answer = processor.decode(new_tokens, skip_special_tokens=True)
        return str(answer).strip()

    def _ask_local_vintern(self, image_path: str, question: str,
                           context: str = "") -> str:
        """Vintern-1B (InternVL family) — loaded lazily, cached."""
        import torch
        from transformers import AutoModel

        if self._local is not None and self._local[0] != "vintern":
            self._local = None      # audit r79: đổi backend giữa phiên → nạp lại
        if self._local is None:
            model_id = self.cfg.local_model
            log.info("Loading local VQA model %s", model_id)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.bfloat16 if device == "cuda" else torch.float32
            from cvp.models.hf_compat import ensure_remote_code_compat, load_tokenizer

            ensure_remote_code_compat()
            model = AutoModel.from_pretrained(
                model_id, torch_dtype=dtype, trust_remote_code=True
            ).to(device).eval()
            tokenizer = load_tokenizer(model_id)
            self._local = ("vintern", model, tokenizer, device, dtype)
        _tag, model, tokenizer, device, dtype = self._local

        from cvp.auxindex.vintern_preprocess import load_image_tiles

        pixel_values = load_image_tiles(image_path, max_num=self.settings.caption.max_tiles)
        pixel_values = pixel_values.to(device=device, dtype=dtype)
        prompt = "<image>\n" + _with_context(
            _VQA_PROMPT.format(question=question) + _exact_suffix(self.cfg), context)
        gen_cfg = dict(max_new_tokens=64, do_sample=False, num_beams=2)
        answer = model.chat(tokenizer, pixel_values, prompt, gen_cfg)
        return str(answer).strip()

    def _ask_gemini_strip(self, image_paths: list[str], question: str,
                          context: str = "") -> str:
        if self._gemini_client is None:
            with self._LOCK:
                if self._gemini_client is None:
                    self._gemini_client = make_gemini_client(self.settings)
        imgs = []
        for p in image_paths:
            img = load_rgb(p)
            if img is not None:
                img.thumbnail((768, 768))
                imgs.append(img)
        if not imgs:
            raise RuntimeError("no readable frame in the strip")
        # Round-41: QA answers ride the Pro model with its own (longer) wall
        # cap — Pro thinks before answering; the chain degrades to Flash so a
        # slow/retired Pro id costs one timeout, never the answer.
        primary = (getattr(self.cfg, "answer_model", "") or self.cfg.gemini_model)
        wall = max(float(getattr(self.cfg, "answer_timeout_s", 0.0)),
                   gemini_wall_timeout(self.settings))
        chain = gemini_model_chain(self.settings, primary)
        # Round-89 (audit): vqa.gemini_model is the FIRST Flash rescue for QA
        # whatever the shared fallback order — 3.7 stays battle-proven right
        # behind Pro; 3.8 rides behind it.
        rescue = self.cfg.gemini_model
        if rescue and rescue != primary:
            chain = [primary, rescue] + [m for m in chain[1:] if m != rescue]
        return generate_with_fallback(
            self._gemini_client, chain,
            [_with_context(_VQA_STRIP_PROMPT.format(n=len(imgs), question=question)
                           + _exact_suffix(self.cfg), context), *imgs],
            timeout_s=wall,
        )

    # ── public ───────────────────────────────────────────────────────────

    def answer_group_votes(self, question: str, image_paths: list[str],
                           context: str = "") -> list[str]:
        """RAW self-consistency ballots for one strip (≤ ``self_consistency``).

        The vote-collection loop of :meth:`answer_group`, exposed so the batch
        QA path can pool ballots ACROSS several strips of the same moment
        (knob ``vqa.answer_neighbor_frames``) before taking one majority.
        Gemini-only: non-gemini providers return [] (the caller falls back to
        :meth:`answer_group`, which owns the local-model degradation).
        """
        paths = [p for p in image_paths if p][: max(1, int(self.cfg.frames_per_answer))]
        if not paths or self.cfg.provider != "gemini":
            return []
        votes = max(1, int(getattr(self.cfg, "self_consistency", 1)))
        got: list[str] = []
        for v in range(votes):
            try:
                a = self._ask_gemini_strip(paths, question, context)
                if a and a.strip():
                    got.append(a[:MAX_QA_ANSWER_CHARS])
            except Exception as e:  # noqa: BLE001 — a lost vote, not a lost group
                log.warning("Gemini strip-VQA vote %d/%d failed (%s)", v + 1, votes, e)
        return got

    def answer_group(self, question: str, image_paths: list[str],
                     context: str = "") -> str:
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
            # Round-40 self-consistency: N answers, majority wins. One sample
            # flip-flopped '300 kg' ↔ '30 kg' between live runs; votes don't.
            got = self.answer_group_votes(question, paths, context)
            if got:
                if getattr(self.cfg, "answer_canonicalize", False):
                    # Round-73 knob: count "2"/"hai"/"02" as ONE candidate —
                    # format variants split the raw-string majority otherwise.
                    from cvp.search.answer_norm import majority_vote

                    return majority_vote(got, canonicalize=True).answer
                from collections import Counter

                best = Counter(g.strip().casefold() for g in got).most_common(1)[0][0]
                return next(g for g in got if g.strip().casefold() == best)
            log.warning("Gemini strip-VQA produced no answer — trying local model")
        if self.cfg.provider in ("gemini", "vintern"):
            try:
                middle = paths[len(paths) // 2]
                return self._ask_local(middle, question, context)[:MAX_QA_ANSWER_CHARS]
            except Exception as e:  # noqa: BLE001
                log.warning("Local VQA failed: %s", e)
        return ""

    def suggest(self, question: str, frames: list[tuple[int, str]],
                context: str = "") -> list[VqaAnswer]:
        """frames: [(global_id, image_path)] — returns one suggestion per frame."""
        out: list[VqaAnswer] = []
        for gid, path in frames[: self.cfg.top_frames]:
            answer, provider = "", "none"
            if self.cfg.provider == "gemini":
                try:
                    answer, provider = self._ask_gemini(path, question, context), "gemini"
                except Exception as e:  # noqa: BLE001 — degrade to local model
                    log.warning("Gemini VQA failed (%s) — trying local model", e)
            if not answer and self.cfg.provider in ("gemini", "vintern"):
                try:
                    answer = self._ask_local(path, question, context)
                    provider = getattr(self.cfg, "local_backend", "vintern")
                except Exception as e:  # noqa: BLE001
                    log.warning("Local VQA failed: %s", e)
            if answer:
                out.append(VqaAnswer(
                    global_id=gid, answer=answer[:MAX_QA_ANSWER_CHARS], provider=provider,
                ))
        return out
