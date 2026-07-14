"""Temporal-context boost for single-moment queries (Vortex AIC-2025 recipe).

Queries like "người đàn ông ngã xuống SAU KHI vụ nổ kết thúc" describe the
TARGET moment plus context that happens BEFORE (or after) it. Frame-level
retrieval scores only the target text; this module re-scores candidates by how
well their temporal NEIGHBOURS match the context part:

    S_final(f) = S(f) + w · max_{g ∈ neighbours_dir(f)} cos(ctx, emb(g))

(Vortex/FocusOnFun used exactly this before/now/after per-video max boost and
finished 79.6/88 at AIC 2025; the organisers call temporal logic the hardest
of their "Big Three" challenges.)

Deterministic + offline: the query split is a Vietnamese-marker regex (no LLM
call), and the boost reads neighbour vectors from the already-loaded primary
index. OFF by default (``search.temporal_boost``) — enable after measuring on
the dev pack.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Markers that split a query into (target moment, context). Each entry:
# (compiled regex, direction of the CONTEXT relative to the target).
#   "X sau khi Y"   → target X, context Y happens BEFORE the target.
#   "X trước khi Y" → target X, context Y happens AFTER the target.
_MARKERS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"(?i)\b(?:ngay\s+)?sau\s+khi\b"), "before"),
    (re.compile(r"(?i)\b(?:ngay\s+)?trư[ớo]c\s+khi\b"), "after"),
    (re.compile(r"(?i)\bafter\b"), "before"),
    (re.compile(r"(?i)\bbefore\b"), "after"),
]

# A context clause shorter than this is noise ("sau khi đó", "after that") —
# there is nothing to embed, keep the plain ranking.
_MIN_CLAUSE_CHARS = 12


@dataclass(frozen=True)
class TemporalParts:
    target: str    # what the submitted frame must show
    context: str   # what the neighbours in `direction` should show
    direction: str  # "before" | "after" — where the context lives in time


def split_temporal_query(query: str) -> TemporalParts | None:
    """Split at the FIRST temporal marker; None when the query has none.

    Both sides must be substantial clauses — markers inside short tails
    ("… ngay sau đó") do not trigger.
    """
    for rx, direction in _MARKERS:
        m = rx.search(query)
        if not m:
            continue
        head = query[: m.start()].strip(" .,;")
        tail = query[m.end():].strip(" .,;")
        if len(head) >= _MIN_CLAUSE_CHARS and len(tail) >= _MIN_CLAUSE_CHARS:
            return TemporalParts(target=head, context=tail, direction=direction)
    return None


def neighbor_rows(gid: int, span: tuple[int, int], direction: str,
                  window: int) -> list[int]:
    """Global-row ids of the temporal neighbours of ``gid`` on one side.

    ``span`` is the video's (first_row, last_row) inclusive — neighbours never
    cross video boundaries.
    """
    lo, hi = span
    if direction == "before":
        return list(range(max(lo, gid - window), gid))
    return list(range(gid + 1, min(hi, gid + window) + 1))


def apply_context_boost(fused: dict[int, float], ctx_scores: dict[int, float],
                        weight: float) -> dict[int, float]:
    """Blend min-max-normalised context scores into the fused map.

    Only candidates present in ``ctx_scores`` move; everything else keeps its
    fused score, so a failed/partial context pass can never hurt the tail.
    """
    if not ctx_scores or weight <= 0:
        return fused
    vals = list(ctx_scores.values())
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-12:
        return fused
    out = dict(fused)
    for gid, s in ctx_scores.items():
        if gid in out:
            out[gid] = out[gid] + weight * ((s - lo) / (hi - lo))
    return out
