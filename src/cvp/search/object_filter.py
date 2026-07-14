"""Object-constraint boosts from the organiser's OpenImages detections.

Parses lightweight Vietnamese constraints out of a query — counts, classes and
coarse positions ("3 người bên trái", "một chiếc xe máy ở giữa") — and boosts
candidates whose detections satisfy them. Soft scores only: detections are
imperfect, so constraints never hard-filter the candidate list.

A query may carry several constraints for the *same* class: "2 người bên trái
và 1 người bên phải" yields two independent Person constraints. Counts are
read from digits or Vietnamese number words (một…mười) up to three tokens
before the noun; positions (trái/phải/giữa/trên/dưới phrases) from a short
window after the noun that stops at a "và" conjunction.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from cvp.config import Settings
from cvp.data.catalog import KeyframeRef
from cvp.data.metadata import Detection, ObjectStore
from cvp.utils.text import fold_diacritics, normalize_text

log = logging.getLogger(__name__)

_NUM_WORDS = {
    "một": 1, "mot": 1, "hai": 2, "ba": 3, "bốn": 4, "bon": 4, "tư": 4,
    "năm": 5, "nam": 5, "sáu": 6, "sau": 6, "bảy": 7, "bay": 7,
    "tám": 8, "tam": 8, "chín": 9, "chin": 9, "mười": 10, "muoi": 10,
}

# How many whitespace tokens before a noun may hold its count ("hai chiếc xe").
_COUNT_LOOKBACK = 3
# Digits above this are years / route numbers, never visible object counts.
_MAX_COUNT = 30

_POSITIONS = {
    "bên trái": "left", "phía trái": "left", "góc trái": "left",
    "bên phải": "right", "phía phải": "right", "góc phải": "right",
    "ở giữa": "center", "chính giữa": "center", "trung tâm": "center",
    "phía trên": "top", "bên trên": "top", "trên cùng": "top",
    "phía dưới": "bottom", "bên dưới": "bottom", "dưới cùng": "bottom",
}
# Folded phrases, longest first, so "phía bên trái" still hits "bên trái".
_POSITIONS_FOLDED: list[tuple[str, str]] = sorted(
    {(fold_diacritics(k), v) for k, v in _POSITIONS.items()},
    key=lambda e: (-len(e[0]), e[0]),
)


def _parse_count(prefix_folded: str) -> int | None:
    """Nearest digit or Vietnamese number word within the last few tokens.

    Scans backwards from the noun; the first parseable number wins. An
    out-of-range digit (e.g. the year in "năm 2024 người dân…") stops the
    scan so earlier tokens like "năm" cannot be misread as a count.
    """
    for tok in reversed(prefix_folded.split()[-_COUNT_LOOKBACK:]):
        if tok.isdigit():
            val = int(tok)
            return val if 1 <= val <= _MAX_COUNT else None
        if tok in _NUM_WORDS:
            return _NUM_WORDS[tok]
    return None


def _parse_position(suffix_folded: str, window: int = 24) -> str | None:
    """Nearest position phrase after the noun, stopping at a "và" conjunction.

    Trimming at "và" keeps "người và xe máy bên phải" from assigning the
    motorcycle's position to the person; nearest-match wins when the window
    still contains several phrases.
    """
    win = suffix_folded[:window]
    cut = re.search(r"\bva\b", win)
    if cut is not None:
        win = win[: cut.start()]
    best: tuple[int, str] | None = None
    for phrase, pos in _POSITIONS_FOLDED:
        i = win.find(phrase)
        if i >= 0 and (best is None or i < best[0]):
            best = (i, pos)
    return best[1] if best is not None else None


@dataclass
class ObjectConstraint:
    entity: str                # OpenImages class, e.g. "Person"
    count: int | None = None   # exact expected count (None = at least one)
    position: str | None = None  # left|right|center|top|bottom


def _load_vocab(path: Path) -> list[tuple[str, str, str]]:
    """[(norm_key, folded_key, entity)] sorted longest-first."""
    if not path.is_file():
        return []
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    entries = []
    for k, v in raw.items():
        norm_key = normalize_text(str(k))
        entries.append((norm_key, fold_diacritics(norm_key), str(v)))
    return sorted(entries, key=lambda e: len(e[1]), reverse=True)


class ObjectBooster:
    def __init__(self, settings: Settings, vocab_path: Path | None = None):
        self.settings = settings
        # Prefer the compact parquet index when built (one file read instead of
        # thousands of tiny JSON opens — decisive on Drive/network mounts);
        # fall back to the per-keyframe ObjectStore transparently.
        from cvp.data.objects_compact import CompactObjects

        compact = CompactObjects(settings)
        self.store = compact if compact.available() else ObjectStore(settings)
        vp = vocab_path or (Path(__file__).resolve().parents[3] / "configs" / "object_vocab_vi.yaml")
        self._vocab_entries = _load_vocab(vp)

    # ── parsing ──────────────────────────────────────────────────────────

    def parse(self, query_vi: str) -> list[ObjectConstraint]:
        """Extract every (entity, count, position) constraint from a query.

        Each occurrence of a vocabulary noun yields its own constraint, so
        "2 người bên trái và 1 người bên phải" produces two Person
        constraints; exact duplicates collapse to one.
        """
        norm = normalize_text(query_vi)
        folded = fold_diacritics(norm)  # same length as norm — spans align
        occupied = [False] * len(norm)
        constraints: list[ObjectConstraint] = []
        for norm_key, folded_key, entity in self._vocab_entries:
            # Exact (diacritic) match always allowed; diacritic-FOLDED matching
            # only for multi-word or long keys — folding short nouns creates
            # false hits on function words ('cờ'→'co' would match 'có').
            spans = [m.span() for m in re.finditer(rf"(?<!\w){re.escape(norm_key)}(?!\w)", norm)]
            if not spans and (" " in folded_key or len(folded_key) >= 5):
                spans = [m.span() for m in re.finditer(rf"(?<!\w){re.escape(folded_key)}(?!\w)", folded)]
            for a, b in spans:
                if any(occupied[a:b]):
                    continue  # inside a longer, already-claimed noun ("xe" in "xe cứu thương")
                for i in range(a, b):
                    occupied[i] = True
                c = ObjectConstraint(
                    entity=entity,
                    count=_parse_count(folded[:a]),
                    position=_parse_position(folded[b:]),
                )
                if c not in constraints:
                    constraints.append(c)
        return constraints

    # ── scoring ──────────────────────────────────────────────────────────

    @staticmethod
    def _in_position(det: Detection, position: str) -> bool:
        ymin, xmin, ymax, xmax = det.box
        cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
        return {
            "left": cx < 0.45,
            "right": cx > 0.55,
            "center": 0.30 < cx < 0.70,
            "top": cy < 0.45,
            "bottom": cy > 0.55,
        }.get(position, True)

    def score(self, constraints: list[ObjectConstraint], ref: KeyframeRef,
              min_score: float = 0.30) -> float:
        """0..1 satisfaction score for one keyframe (0 if no constraints)."""
        if not constraints:
            return 0.0
        detections = [d for d in self.store.get(ref.video_id, ref.n) if d.score >= min_score]
        if not detections:
            return 0.0
        total = 0.0
        for c in constraints:
            matching = [d for d in detections if d.entity == c.entity]
            if c.position:
                matching = [d for d in matching if self._in_position(d, c.position)]
            if not matching:
                continue
            if c.count is None:
                total += 1.0
            else:
                # full credit for exact count, partial for off-by-one
                diff = abs(len(matching) - c.count)
                total += 1.0 if diff == 0 else (0.5 if diff == 1 else 0.2)
        return total / len(constraints)
