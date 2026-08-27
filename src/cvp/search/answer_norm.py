"""QA answer canonicalization — equivalence classes for voting & diagnosis.

The organisers grade QA answers as (near-)exact text, so a CORRECT answer in
the WRONG surface form scores 0: "sáu" vs "6", "300kg" vs "300 kg",
'"Khung trời mơ ước"' vs "Khung trời mơ ước". This module folds such variants
into one canonical key so that:

* the self-consistency vote in :meth:`cvp.search.vqa.VqaAssistant.answer_group`
  can count "2" / "hai" / "02" as the SAME candidate (knob
  ``vqa.answer_canonicalize``) instead of splitting the majority;
* ``scripts/67_qa_diag.py`` can tell "answer đúng nội dung nhưng sai định
  dạng" (FORMAT_MISS) apart from a genuinely wrong answer (ANSWER_MISS).

Tiers (each includes the previous one):

    0   the official scorer's own :func:`cvp.eval.official.normalize_answer`
        (casefold + whitespace + TRAILING punctuation only) — the baseline.
    1   + strip decoration: preamble ("Đáp án:", "Trả lời là…"), markdown
        ``**…**``/backticks, wrapping quotes/brackets on BOTH ends.
    2   + Vietnamese number words → digits (hai mươi bảy → 27), decimal comma
        (3,5 → 3.5), thousands dots (1.000 → 1000), unit folding
        (300kg → "300 kg", "88 %" → "88%", "ki-lô-gam" → "kg"), internal
        punctuation → space.  ← the voting/matching tier.
    3   + Vietnamese diacritics stripped ("màu xanh" → "mau xanh").
        DIAGNOSTIC ONLY: the organisers preserve diacritics, so a tier-3-only
        match means the answer would still score 0 as submitted.

Known deliberate over-folds at tier 2 (symmetric on both sides, so they can
only ever MERGE equivalents, never split them): "năm" the year-word folds to
"5", "một chiếc" folds to "1 chiếc". Both sides of a comparison fold the same
way, and votes all answer the SAME question, so this is harmless in practice.

Pure stdlib; safe to import from scripts, tests and the engine path alike.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass

from cvp.eval.official import normalize_answer

__all__ = [
    "VoteResult",
    "canonical_key",
    "fold_numbers_vi",
    "fold_units",
    "looks_decorated",
    "majority_vote",
    "strip_decoration",
]

_WS_RE = re.compile(r"\s+")
_MD_RE = re.compile(r"[*`]+")
# "Đáp án: X" / "Trả lời là X" / "Câu trả lời: X" / "Answer - X" preambles the
# prompt forbids but VLMs still occasionally prepend.
_PREAMBLE_RE = re.compile(
    r"^(?:đáp án|dap an|trả lời|câu trả lời|answer)\b\s*(?:là\b|:|=|-|–|—|,)?\s*",
    re.IGNORECASE,
)
# Both-ends strip: quotes/brackets/sentence punctuation + whitespace.
# NOTE: unlike the official scorer this strips the LEADING side too — the
# scorer's rstrip-only rule is exactly the F3 trap ('"x"' → '"x').
# Round-73: a leading ASCII "-" directly before a digit is a SIGN, not
# decoration — "-5" must never merge with "5" ('tầng -1' ≠ 'tầng 1').
_EDGE_L_RE = re.compile(r"^[ \t\"'“”‘’«»‹›()\[\]{}.,;:!?…–—]+")
_EDGE_L_DASH_RE = re.compile(r"^-+(?!\d)")
_EDGE_R_RE = re.compile(r"[ \t\"'“”‘’«»‹›()\[\]{}.,;:!?…\-–—]+$")
_DECIMAL_RE = re.compile(r"(?<=\d),(?=\d)")
_THOUSANDS_RE = re.compile(r"(?<=\d)\.(?=\d{3}(?:\D|$))")
# Internal punctuation → space (keep word chars incl. Vietnamese, %, dots for
# the digit-internal case handled separately below; ASCII "-" handled by
# _DASH_TO_SPACE_RE so a numeric sign survives).
_INNER_PUNCT_RE = re.compile(r"[^\w\s.%-]")
# Any dash that is NOT a numeric sign (sign = preceded by non-word, digit next).
_DASH_TO_SPACE_RE = re.compile(r"-(?!\d)|(?<=\w)-")
_DOT_NOT_DECIMAL_RE = re.compile(r"(?<!\d)\.|\.(?!\d)")
# "06" ≡ "6" — leading zeros of standalone numbers (never inside words: there
# is no \b between two word chars, so "l01" stays intact).
_LEADING_ZERO_RE = re.compile(r"\b0+(?=\d)")

# ── Vietnamese number words ──────────────────────────────────────────────────
# Standalone-safe digit words. "không" is EXCLUDED on purpose (it would mangle
# "không rõ"); "mốt/tư/lăm/nhăm" are continuation-only (they mean 1/4/5 only
# after mươi/mười — standalone "tư" would mangle "tư vấn").
_NUM_SIMPLE = {
    "một": 1, "hai": 2, "ba": 3, "bốn": 4, "năm": 5,
    "sáu": 6, "bảy": 7, "bẩy": 7, "tám": 8, "chín": 9,
}
_NUM_AFTER_TENS = {**_NUM_SIMPLE, "mốt": 1, "tư": 4, "lăm": 5, "nhăm": 5}

# Unit synonyms folded to one canonical spelling — applied ONLY right after a
# number so ordinary prose ("mét vuông nhà") is never rewritten. Hyphenated
# spellings arrive space-separated (tier 2 turns "-" into a space first).
_UNIT_SYNONYMS: tuple[tuple[str, str], ...] = (
    (r"phần trăm", "%"),
    (r"ki ?lô ?gam|kilôgam|kilogam|kí ?lô|ký ?lô", "kg"),
    (r"ki ?lô ?mét|kilômét|kilomet|cây số", "km"),
    (r"xăng ?ti ?mét|xentimét|xentimet|centimét|centimet", "cm"),
    (r"mi ?li ?mét|milimét|milimet", "mm"),
    (r"mi ?li ?lít|mililít|mililit", "ml"),
    (r"mét", "m"),
    (r"lít", "l"),
    (r"gam|gram", "g"),
    (r"vnđ|vnd|đồng", "đ"),
)
_UNIT_SYNONYM_RES = [
    (re.compile(rf"(?<=\d)\s*(?:{pat})\b"), f" {canon}") for pat, canon in _UNIT_SYNONYMS
]
# Glue a missing space between a digit and a short symbol unit ("300kg").
_UNIT_GLUE_RE = re.compile(r"(?<=\d)(kg|km|cm|mm|ml|g|m|l|đ)\b")
_PERCENT_RE = re.compile(r"(?<=\d)\s*%")


def _strip_edges(s: str) -> str:
    """Both-ends decoration strip, preserving a leading numeric sign ("-5")."""
    while True:
        prev = s
        s = _EDGE_L_RE.sub("", s)
        s = _EDGE_L_DASH_RE.sub("", s)
        s = _EDGE_R_RE.sub("", s)
        if s == prev:
            return s


def strip_decoration(answer: str) -> str:
    """Remove preamble/markdown/wrapping quotes; collapse whitespace; NFC.

    Never returns "" for a non-empty input (falls back to the whitespace-
    collapsed original) — a decoration stripper must not eat the answer.

    Round-73: preamble and edge stripping alternate to a FIXPOINT, so quoted/
    bracketed preambles ('"Đáp án: 5"', '(Trả lời: X)') peel completely —
    the one-pass order previously left 'đáp án 5' ≠ '5' (non-idempotent).
    """
    raw = _WS_RE.sub(" ", unicodedata.normalize("NFC", str(answer))).strip()
    out = _MD_RE.sub("", raw)
    while True:
        prev = out
        out = _PREAMBLE_RE.sub("", out, count=1)
        out = _strip_edges(out.strip())
        if out == prev:
            break
    out = _WS_RE.sub(" ", out).strip()
    return out or raw


def looks_decorated(answer: str) -> bool:
    """True when the raw answer carries a preamble / markdown / wrapping quotes
    that :func:`strip_decoration` would remove — a cheap diagnostic flag."""
    raw = _WS_RE.sub(" ", unicodedata.normalize("NFC", str(answer))).strip()
    if not raw:
        return False
    return strip_decoration(raw) != raw


def _parse_number_phrase(tokens: list[str], i: int) -> tuple[int, int]:
    """(value, tokens consumed) of the Vietnamese number phrase at ``tokens[i]``.

    Grammar subset (0 never matches — see module notes): ``X trăm [linh|lẻ U |
    tens]``, ``X mươi [U]``, ``mười [U]``, standalone digit word. Returns
    ``(0, 0)`` when no phrase starts here.
    """
    n = len(tokens)
    val, j, matched = 0, i, False
    if j + 1 < n and tokens[j] in _NUM_SIMPLE and tokens[j + 1] == "trăm":
        val = _NUM_SIMPLE[tokens[j]] * 100
        j += 2
        matched = True
        if (j + 1 < n and tokens[j] in ("linh", "lẻ")
                and tokens[j + 1] in _NUM_AFTER_TENS):
            return val + _NUM_AFTER_TENS[tokens[j + 1]], j + 2 - i
    if j + 1 < n and tokens[j] in _NUM_SIMPLE and tokens[j + 1] == "mươi":
        val += _NUM_SIMPLE[tokens[j]] * 10
        j += 2
        matched = True
        if j < n and tokens[j] in _NUM_AFTER_TENS:
            val += _NUM_AFTER_TENS[tokens[j]]
            j += 1
    elif j < n and tokens[j] == "mười":
        val += 10
        j += 1
        matched = True
        if j < n and tokens[j] in _NUM_AFTER_TENS:
            val += _NUM_AFTER_TENS[tokens[j]]
            j += 1
    elif not matched and j < n and tokens[j] in _NUM_SIMPLE:
        val += _NUM_SIMPLE[tokens[j]]
        j += 1
        matched = True
    if not matched:
        return 0, 0
    return val, j - i


def fold_numbers_vi(text: str) -> str:
    """Vietnamese number words → digits over a casefolded string.

    "hai mươi bảy" → "27", "một trăm linh năm" → "105", "mười lăm" → "15",
    "sáu" → "6". Non-number tokens pass through untouched.
    """
    tokens = text.split()
    out: list[str] = []
    i = 0
    while i < len(tokens):
        val, consumed = _parse_number_phrase(tokens, i)
        if consumed:
            out.append(str(val))
            i += consumed
        else:
            out.append(tokens[i])
            i += 1
    return " ".join(out)


def fold_units(text: str) -> str:
    """Normalise number-unit spelling: synonyms → symbols, spacing → "N unit",
    percent glued ("88%"). Only touches text right after a digit."""
    s = text
    for rx, repl in _UNIT_SYNONYM_RES:
        s = rx.sub(repl, s)
    s = _UNIT_GLUE_RE.sub(r" \1", s)
    s = _PERCENT_RE.sub("%", s)
    return _WS_RE.sub(" ", s).strip()


def _strip_accents(text: str) -> str:
    s = text.replace("đ", "d").replace("Đ", "D")
    decomposed = unicodedata.normalize("NFD", s)
    return "".join(c for c in decomposed if unicodedata.category(c) != "Mn")


def canonical_key(answer: str, tier: int = 2) -> str:
    """Equivalence-class key of one answer at the given tier (module docstring).

    Deterministic and idempotent; never "" for a non-empty input.
    """
    if tier <= 0:
        return normalize_answer(answer)
    s = strip_decoration(answer).casefold()
    s = _WS_RE.sub(" ", _strip_edges(s)).strip()
    t1 = s or str(answer).casefold().strip()
    if tier == 1:
        return t1
    s = _DECIMAL_RE.sub(".", t1)
    while True:
        folded = _THOUSANDS_RE.sub("", s)
        if folded == s:
            break
        s = folded
    s = _INNER_PUNCT_RE.sub(" ", s)
    s = _DASH_TO_SPACE_RE.sub(" ", s)
    s = _DOT_NOT_DECIMAL_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    s = fold_numbers_vi(s)
    s = _LEADING_ZERO_RE.sub("", s)
    s = fold_units(s)
    if tier >= 3:
        s = _strip_accents(s)
    return s or t1


@dataclass(frozen=True)
class VoteResult:
    """Outcome of one majority vote over raw answer strings."""

    answer: str      # representative RAW string to submit
    votes_for: int   # ballots in the winning class
    total: int       # all non-empty ballots

    @property
    def agree(self) -> float:
        """Winning-class share of the ballots, in [0, 1] (0.0 for no ballots)."""
        return self.votes_for / self.total if self.total else 0.0


def majority_vote(votes: list[str], canonicalize: bool = False,
                  tier: int = 2) -> VoteResult:
    """Majority answer over raw ballots.

    ``canonicalize=False`` reproduces the legacy ``answer_group`` semantics
    EXACTLY (byte-identical winner): classes keyed by ``strip().casefold()``,
    ties broken by first-seen class, representative = FIRST raw ballot of the
    winning class. With ``canonicalize=True`` classes are keyed by
    :func:`canonical_key` (tier 2) and the representative is the MOST COMMON
    raw form inside the winning class (first-seen on ties) — so "2"/"hai"/"02"
    pool their ballots instead of splitting them.
    """
    ballots = [str(v) for v in votes if str(v).strip()]
    if not ballots:
        return VoteResult("", 0, 0)
    if not canonicalize:
        counts = Counter(b.strip().casefold() for b in ballots)
        best, n = counts.most_common(1)[0]
        rep = next(b for b in ballots if b.strip().casefold() == best)
        return VoteResult(rep, n, len(ballots))
    counts = Counter(canonical_key(b, tier) for b in ballots)
    best, n = counts.most_common(1)[0]
    raws = [b for b in ballots if canonical_key(b, tier) == best]
    rep = Counter(raws).most_common(1)[0][0]
    return VoteResult(rep, n, len(ballots))
