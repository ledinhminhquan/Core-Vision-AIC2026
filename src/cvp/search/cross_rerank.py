"""Pairwise cross-encoder re-rank of the fused head (Unified-IMMR recipe).

WHY: bi-encoder retrieval (SigLIP/PE-Core + FAISS) encodes each modality
independently, so it misses fine-grained query↔frame interactions (counts,
spatial relations, small on-screen text). A cross-encoder attends over the
image and the query *jointly* and fixes exactly that — the AIC-2025 recipe
"dual-embedding retrieve → BLIP-2 ITM rerank of the fused top-100" scored
76.4/88 (Unified-IMMR). It runs only on the top ``search.rerank_topk``
candidates because the joint forward pass is ~100× a FAISS probe.

Two local backends (``search.reranker``):

* ``blip2_itm``     — ``Blip2ForImageTextRetrieval`` with the ITM head; softmax
  index 1 of the 2-way classifier = "text matches image" probability.
* ``qwen_reranker`` — ``Qwen/Qwen3-VL-Reranker-2B`` (Jan 2026, Apache-2.0), the
  first purpose-built OPEN multimodal cross-encoder. Tried first through
  ``sentence_transformers.CrossEncoder`` (the model-card API), then through
  the Qwen yes/no-token protocol on ``AutoModelForImageTextToText``.
  ⚠ Smoke-test this lane on Colab before a competition round (docs/EVALUATION).

Both are complementary to the LISTWISE VLM rerank (``vlm_rerank.py``): the
cross-encoder scores each (query, frame) pair in isolation with a calibrated
head, the VLM judges the head of the ranking holistically. Blend with the
fused score: ``final = (1-w)·minmax(fused) + w·minmax(cross)`` over the head
only — the tail is untouched, so a reranker failure can cost at most
``rerank_topk`` positions. Every failure path returns the input unchanged.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

import numpy as np

from cvp.config import Settings
from cvp.utils.images import load_rgb

if TYPE_CHECKING:  # circular-import guard — engine imports this module
    from cvp.search.engine import SearchResult

log = logging.getLogger(__name__)

# BLIP-2's text tower is BERT-based; never feed it more tokens than this even
# if the tokenizer reports a sentinel model_max_length (e.g. 1e30).
_MAX_TEXT_LEN_CAP = 512


class Blip2ItmReranker:
    """Scores (query, keyframe) pairs with the BLIP-2 ITM head.

    Contract: ``score(query, image_paths) -> float32 array (len(paths),)``,
    higher = better. Unreadable images get 0.0 instead of raising, so one
    corrupt keyframe can never kill a live competition query.
    """

    def __init__(self, settings: Settings) -> None:
        import torch  # noqa: F401 — heavy import kept out of module scope
        from transformers import AutoProcessor, Blip2ForImageTextRetrieval

        from cvp.models.base import resolve_device, resolve_dtype

        sc = settings.search
        self.device = resolve_device(settings.embedding.device)
        self.dtype = resolve_dtype(settings.embedding.dtype, self.device)
        self.batch_size = max(1, int(sc.rerank_batch_size))

        self.model = Blip2ForImageTextRetrieval.from_pretrained(
            sc.blip2_itm_id, torch_dtype=self.dtype).to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(sc.blip2_itm_id)

        tok = getattr(self.processor, "tokenizer", None)
        max_len = getattr(tok, "model_max_length", _MAX_TEXT_LEN_CAP) or _MAX_TEXT_LEN_CAP
        self.text_max_length = int(min(max_len, _MAX_TEXT_LEN_CAP))
        log.info("BLIP-2 ITM reranker ready: %s device=%s dtype=%s batch=%d",
                 sc.blip2_itm_id, self.device, self.dtype, self.batch_size)

    def score(self, query: str, image_paths: list[str]) -> np.ndarray:
        import torch

        scores = np.zeros(len(image_paths), dtype=np.float32)
        images: list = []
        ok_idx: list[int] = []
        for i, path in enumerate(image_paths):
            img = load_rgb(path)
            if img is None:
                log.warning("Unreadable keyframe %s; rerank score 0.0.", path)
                continue
            images.append(img)
            ok_idx.append(i)

        for start in range(0, len(images), self.batch_size):
            batch = images[start:start + self.batch_size]
            inputs = self.processor(
                images=batch, text=[query] * len(batch), return_tensors="pt",
                padding=True, truncation=True, max_length=self.text_max_length,
            )
            inputs = {
                k: (v.to(self.device, self.dtype)
                    if torch.is_floating_point(v) else v.to(self.device))
                for k, v in inputs.items()
            }
            with torch.no_grad():
                out = self.model(**inputs, use_image_text_matching_head=True)
            # ITM head → (batch, 2) logits; softmax index 1 = match probability.
            probs = torch.softmax(out.logits_per_image.float(), dim=1)[:, 1]
            for j, s in zip(ok_idx[start:start + len(batch)],
                            probs.cpu().numpy().astype(np.float32)):
                scores[j] = s
        return scores


_QWEN_RERANK_INSTRUCTION = (
    "Given a search query, judge whether the video keyframe matches the query."
)


class QwenVLReranker:
    """Qwen3-VL-Reranker pairwise scorer (2B default, 8B when VRAM allows).

    Backend A (preferred): ``sentence_transformers.CrossEncoder`` — the API the
    model card documents. Backend B: manual Qwen yes/no-token scoring on
    ``AutoModelForImageTextToText`` (the documented text-Qwen3-Reranker
    protocol, applied through the VL chat template).
    """

    def __init__(self, settings: Settings) -> None:
        sc = settings.search
        self.model_id = sc.qwen_reranker_id
        self.batch_size = max(1, int(sc.rerank_batch_size))
        self._ce = None            # sentence-transformers CrossEncoder
        self._manual = None        # (model, processor, yes_id, no_id, device)
        try:
            from sentence_transformers import CrossEncoder

            self._ce = CrossEncoder(self.model_id, trust_remote_code=True)
            log.info("Qwen3-VL reranker ready via sentence-transformers: %s", self.model_id)
            return
        except Exception as e:  # noqa: BLE001 — fall through to manual backend
            log.info("sentence-transformers backend unavailable (%s); trying manual", e)
        self._init_manual(settings)

    def _init_manual(self, settings: Settings) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        from cvp.models.base import resolve_device, resolve_dtype

        device = resolve_device(settings.embedding.device)
        dtype = resolve_dtype(settings.embedding.dtype, device)
        model = AutoModelForImageTextToText.from_pretrained(
            self.model_id, torch_dtype=dtype, trust_remote_code=True).to(device).eval()
        processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        tok = processor.tokenizer
        yes_id = tok.convert_tokens_to_ids("yes")
        no_id = tok.convert_tokens_to_ids("no")
        if yes_id is None or no_id is None or yes_id == tok.unk_token_id:
            raise RuntimeError("yes/no tokens not found in Qwen reranker tokenizer")
        self._manual = (model, processor, yes_id, no_id, device)
        log.info("Qwen3-VL reranker ready via yes/no protocol: %s device=%s",
                 self.model_id, device)
        _ = torch  # keep import referenced

    def score(self, query: str, image_paths: list[str]) -> np.ndarray:
        if self._ce is not None:
            return self._score_ce(query, image_paths)
        return self._score_manual(query, image_paths)

    def _score_ce(self, query: str, image_paths: list[str]) -> np.ndarray:
        # Same contract as the BLIP-2 lane: unreadable keyframes score 0.0 —
        # feeding the CrossEncoder a placeholder would give a corrupt frame an
        # arbitrary (possibly boosting) score (review finding C4).
        scores = np.zeros(len(image_paths), dtype=np.float32)
        pairs, ok_idx = [], []
        for i, p in enumerate(image_paths):
            img = load_rgb(p)
            if img is None:
                log.warning("Unreadable keyframe %s; rerank score 0.0.", p)
                continue
            pairs.append([query, img])
            ok_idx.append(i)
        if pairs:
            preds = np.asarray(self._ce.predict(pairs, batch_size=self.batch_size),
                               dtype=np.float32).reshape(-1)
            for i, s in zip(ok_idx, preds):
                scores[i] = s
        return scores

    def _score_manual(self, query: str, image_paths: list[str]) -> np.ndarray:
        import torch

        model, processor, yes_id, no_id, device = self._manual
        scores = np.zeros(len(image_paths), dtype=np.float32)
        for i, path in enumerate(image_paths):
            img = load_rgb(path)
            if img is None:
                continue
            img.thumbnail((448, 448))
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text":
                        f"<Instruct>: {_QWEN_RERANK_INSTRUCTION}\n"
                        f"<Query>: {query}\n"
                        "Does the image match the query? Answer only yes or no."},
                ],
            }]
            try:
                inputs = processor.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=True,
                    return_dict=True, return_tensors="pt").to(device)
                with torch.no_grad():
                    logits = model(**inputs).logits[0, -1]
                pair = torch.stack([logits[no_id], logits[yes_id]]).float()
                scores[i] = torch.softmax(pair, dim=0)[1].item()
            except Exception as e:  # noqa: BLE001 — one bad frame must not sink the batch
                log.warning("Qwen rerank failed on %s (%s); score 0.0", path, e)
        return scores


_BACKENDS = {"blip2_itm": Blip2ItmReranker, "qwen_reranker": QwenVLReranker}

_RERANKER = None      # lazy singleton (or False after a failed build)
_RERANKER_KEY = None  # full identity the singleton was built for
_BUILD_LOCK = threading.Lock()


def _identity_key(settings: Settings) -> tuple:
    """Full identity of the reranker a Settings object asks for.

    Keyed on backend + model id + device (not just the backend name): an
    in-process A/B (scripts/26-style) or a notebook that edits settings must
    get a REBUILD, not the previous configuration's weights; and correcting a
    bad model id must clear the failed-build latch (review findings C3/C17).
    """
    sc = settings.search
    model_id = sc.blip2_itm_id if sc.reranker == "blip2_itm" else sc.qwen_reranker_id
    return (sc.reranker, model_id, settings.embedding.device)


def _get_reranker(settings: Settings):
    global _RERANKER, _RERANKER_KEY
    key = _identity_key(settings)
    # Double-checked locking: model builds take tens of seconds — a thundering
    # herd of first requests (service/Streamlit) must build exactly ONE copy.
    if _RERANKER_KEY == key and _RERANKER is not None:
        return _RERANKER or None
    with _BUILD_LOCK:
        if _RERANKER_KEY != key:
            _RERANKER, _RERANKER_KEY = None, key
        if _RERANKER is None:
            try:
                _RERANKER = _BACKENDS[settings.search.reranker](settings)
            except Exception as e:  # noqa: BLE001 — optional stage must never sink the engine
                log.warning("Cross-encoder reranker %r failed to build (%s) — "
                            "DISABLED until the configuration changes", key, e)
                _RERANKER = False
    return _RERANKER or None


def _minmax(arr: np.ndarray) -> np.ndarray:
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-12:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def cross_rerank(
    results: list["SearchResult"],
    query: str,
    settings: Settings,
    topk: int | None = None,
) -> list["SearchResult"]:
    """Re-order the top-``rerank_topk`` rows by blended cross-encoder score."""
    sc = settings.search
    if sc.reranker in ("", "none"):
        return results
    if sc.reranker not in _BACKENDS:
        log.warning("Unknown search.reranker %r — skipping", sc.reranker)
        return results
    topk = topk or sc.rerank_topk
    head = results[:topk]
    if len(head) < 2:
        return results
    reranker = _get_reranker(settings)
    if reranker is None:
        return results
    try:
        cross = np.asarray(reranker.score(query, [r.ref.path for r in head]),
                           dtype=np.float32)
    except Exception as e:  # noqa: BLE001 — reranking must never sink the query
        log.warning("Cross-encoder rerank failed (%s) — keeping original order", e)
        return results
    if cross.shape != (len(head),):
        log.warning("Cross-encoder returned %s scores for %d candidates — skipping",
                    cross.shape, len(head))
        return results
    fused = _minmax(np.asarray([r.score for r in head], dtype=np.float32))
    blended = (1.0 - sc.rerank_weight) * fused + sc.rerank_weight * _minmax(cross)
    order = sorted(range(len(head)), key=lambda i: (-float(blended[i]), i))
    reranked = [head[i] for i in order]
    for r, i in zip(reranked, order):
        r.signals["cross"] = float(cross[i])
    return reranked + results[topk:]
