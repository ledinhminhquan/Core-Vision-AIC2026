"""Vietnamese-aware text normalization for search (BM25 over OCR/ASR/captions).

Vietnamese OCR/ASR output is noisy; queries are typed fast during competition.
Matching therefore happens on two parallel views of every string:
  1. ``normalize_text`` — NFC, lowercase, punctuation stripped (keeps diacritics)
  2. ``fold_diacritics`` — additionally maps ``đ``→``d`` and strips accents,
     so "Hà Nội" matches "ha noi" and OCR that lost its accents.
"""

from __future__ import annotations

import re
import unicodedata

_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WS_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """NFC-normalize, lowercase, collapse whitespace, strip punctuation."""
    text = unicodedata.normalize("NFC", text or "")
    text = text.lower()
    text = _PUNCT_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


def fold_diacritics(text: str) -> str:
    """Remove Vietnamese diacritics: 'Hà Nội' -> 'ha noi', 'đường' -> 'duong'."""
    text = unicodedata.normalize("NFC", text or "")
    text = text.replace("đ", "d").replace("Đ", "D")
    decomposed = unicodedata.normalize("NFD", text)
    stripped = "".join(c for c in decomposed if unicodedata.category(c) != "Mn")
    return unicodedata.normalize("NFC", stripped)


def tokenize_vi(text: str, fold: bool = True) -> list[str]:
    """Whitespace tokens of the normalized (optionally diacritic-folded) string."""
    text = normalize_text(text)
    if fold:
        text = fold_diacritics(text)
    return text.split()
