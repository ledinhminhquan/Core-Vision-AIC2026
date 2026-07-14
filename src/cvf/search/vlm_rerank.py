"""Listwise VLM re-ranking of the head of a ranking (UIT CVPRW'25: +10% H@1).

A vision-language model sees the query plus the top-N candidate keyframes and
scores each frame's relevance 0–10; the head of the ranking is re-ordered by
(vlm_score desc, original rank asc) — the tail is left untouched, so a VLM
hallucination can cost at most N positions, never the whole ranking.

Providers (settings.search.vlm_rerank_provider):
  * ``gemini``  — one multi-image call (fast, needs GEMINI_API_KEY),
  * ``vintern`` — local 5CD-AI/Vintern-1B-v3_5, one call per frame (offline),
  * ``none``    — passthrough.

Every failure path returns the input ranking unchanged.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import TYPE_CHECKING

from cvf.config import Settings

if TYPE_CHECKING:  # circular-import guard — engine imports this module
    from cvf.search.engine import SearchResult

log = logging.getLogger(__name__)

_GEMINI_PROMPT = """You are re-ranking video keyframes for a retrieval system.
Query (may be Vietnamese): {query}

You are given {n} candidate frames, numbered 1..{n} in order.
For EACH frame, rate how well it matches the query from 0 (unrelated) to 10
(exactly the described moment). Judge only what is visible.
Return STRICT JSON (no markdown): {{"scores": [s1, s2, ...]}} with exactly {n} numbers."""

_VINTERN_PROMPT = (
    "<image>\nCâu truy vấn: {query}\n"
    "Ảnh này khớp với câu truy vấn ở mức nào? Trả lời CHỈ MỘT SỐ từ 0 đến 10."
)


def _parse_scores(text: str, n: int) -> list[float] | None:
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        data = json.loads(text)
        scores = [float(x) for x in data.get("scores", [])]
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    if len(scores) != n:
        return None
    return [min(max(s, 0.0), 10.0) for s in scores]


def _gemini_scores(query: str, paths: list[str], settings: Settings) -> list[float] | None:
    from google import genai
    from PIL import Image

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set")
    client = genai.Client(api_key=api_key)
    parts: list = [_GEMINI_PROMPT.format(query=query, n=len(paths))]
    for p in paths:
        img = Image.open(p).convert("RGB")
        img.thumbnail((448, 448))
        parts.append(img)
    resp = client.models.generate_content(model=settings.vqa.gemini_model, contents=parts)
    return _parse_scores(resp.text or "", len(paths))


_LOCAL_VLM = None  # (model, tokenizer, device, dtype) — module-level lazy singleton


def _vintern_scores(query: str, paths: list[str], settings: Settings) -> list[float] | None:
    global _LOCAL_VLM
    import torch
    from transformers import AutoModel, AutoTokenizer

    from cvf.auxindex.vintern_preprocess import load_image_tiles

    if _LOCAL_VLM is None:
        model_id = settings.vqa.local_model
        log.info("Loading local rerank VLM %s", model_id)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        model = AutoModel.from_pretrained(
            model_id, torch_dtype=dtype, trust_remote_code=True
        ).to(device).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=False)
        _LOCAL_VLM = (model, tokenizer, device, dtype)
    model, tokenizer, device, dtype = _LOCAL_VLM

    scores: list[float] = []
    for p in paths:
        try:
            pixel_values = load_image_tiles(p, max_num=2).to(device=device, dtype=dtype)
            out = model.chat(
                tokenizer,
                pixel_values,
                _VINTERN_PROMPT.format(query=query),
                dict(max_new_tokens=8, do_sample=False),
            )
            m = re.search(r"\d+(?:\.\d+)?", str(out))
            scores.append(min(max(float(m.group()), 0.0), 10.0) if m else 0.0)
        except Exception:  # noqa: BLE001 — one bad frame must not sink the batch
            scores.append(0.0)
    return scores


def vlm_rerank(
    results: list["SearchResult"],
    query: str,
    settings: Settings,
    topk: int | None = None,
) -> list["SearchResult"]:
    """Re-order the top-``topk`` rows by VLM relevance; tail untouched."""
    provider = settings.search.vlm_rerank_provider
    topk = topk or settings.search.vlm_rerank_topk
    head = results[:topk]
    if provider in ("", "none") or len(head) < 2:
        return results
    paths = [r.ref.path for r in head]
    try:
        if provider == "gemini":
            scores = _gemini_scores(query, paths, settings)
        elif provider == "vintern":
            scores = _vintern_scores(query, paths, settings)
        else:
            log.warning("Unknown vlm_rerank_provider %r — skipping", provider)
            return results
    except Exception as e:  # noqa: BLE001 — reranking must never sink the query
        log.warning("VLM rerank failed (%s) — keeping original order", e)
        return results
    if not scores:
        log.warning("VLM rerank returned no usable scores — keeping original order")
        return results
    order = sorted(range(len(head)), key=lambda i: (-scores[i], i))
    reranked = [head[i] for i in order]
    for rank, (r, i) in enumerate(zip(reranked, order)):
        r.signals["vlm"] = scores[i]
    return reranked + results[topk:]
