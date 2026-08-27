"""Submission row-budget optimizer (Nhiệm vụ D đợt 2) — knob ``search.row_strategy``.

The official metric takes the MAX R-Score over the first k rows (k up to 100),
so every submitted row is a lottery ticket. The legacy KIS/QA submission just
dumps the ranking: by rank ~40 the fused scores are usually a flat, diluted
tail that almost never wins. ``diversify_tail`` spends that budget ON PURPOSE
instead (the TRAKE-jitter philosophy of round-72, applied to single-frame
tasks):

* the HEAD of the ranking (``search.row_strategy_head`` rows) is kept
  verbatim — R@1/R@5/R@20 must never be touched;
* the tail budget is then filled with NEIGHBOUR-FRAME variants around the
  best row of each distinct video seen in the head (round-robin across
  videos, best rank first): midpoints toward the previous/next keyframe
  first — the grid is sparse (~5-7 s at Batch-1 density) while GT windows
  are ±~5 s, so a frame BETWEEN two keyframes can hit windows that no grid
  frame hits — then the previous/next keyframes themselves;
* whatever budget remains keeps the original ranking tail, in order.

Variants take at most HALF the tail budget (``_VARIANT_TAIL_SHARE``): the
original mid-ranking still carries real signal and must not be evicted
wholesale by lottery tickets.

QA rows carry ``(video, frame, answer)`` — a variant row inherits its anchor
row's answer (same moment, same answer). Everything here is pure data → fully
offline-testable; the ONLY engine coupling is :func:`catalog_grid_fn`, which
adapts a ``KeyframeCatalog`` into the ``grid_fn`` callback and degrades to
"no grids" (→ no variants, tail unchanged) on stub engines without a catalog.

Default ``search.row_strategy: legacy`` bypasses this module entirely —
bit-identical rows.
"""

from __future__ import annotations

import logging
from bisect import bisect_left
from typing import Callable, Sequence

log = logging.getLogger(__name__)

# Variants may consume at most this share of the tail budget (see module doc).
_VARIANT_TAIL_SHARE = 0.5

GridFn = Callable[[str], Sequence[int] | None]


def variant_frames(frame: int, grid: Sequence[int] | None,
                   max_variants: int = 4) -> list[int]:
    """Neighbour-frame lottery tickets around one grid frame, best guess first.

    Order: early midpoint (toward the previous keyframe), late midpoint
    (toward the next), previous keyframe, next keyframe — midpoints first
    because they cover the inter-grid gaps no submitted keyframe can reach.
    Returns [] when the grid is unknown or ``frame`` is not ON the grid (an
    already-jittered row cannot be navigated further).
    """
    if not grid or max_variants <= 0:
        return []
    i = bisect_left(grid, frame)
    if i >= len(grid) or grid[i] != frame:
        return []
    prev = int(grid[i - 1]) if i > 0 else None
    nxt = int(grid[i + 1]) if i + 1 < len(grid) else None
    ordered = [
        (prev + frame) // 2 if prev is not None else None,   # early midpoint
        (frame + nxt) // 2 if nxt is not None else None,     # late midpoint
        prev,
        nxt,
    ]
    out: list[int] = []
    for v in ordered:
        if v is not None and v != frame and v >= 0 and v not in out:
            out.append(v)
    return out[: int(max_variants)]


def diversify_tail(rows: list[tuple], grid_fn: GridFn, head_keep: int = 30,
                   budget: int = 100, variants_per_anchor: int = 4) -> list[tuple]:
    """Head verbatim + targeted neighbour variants + original tail, ≤ ``budget``.

    Args:
        rows: ranked submission rows, best first — ``(video, frame)`` for KIS,
            ``(video, frame, answer)`` for QA (any extra payload rides along).
        grid_fn: ``video_id -> sorted frame_idx grid`` (None = unknown video).
        head_keep: rows kept verbatim at the top.
        budget: total output rows (the organiser cap).
        variants_per_anchor: neighbour variants generated per anchor video.

    Returns:
        The reshaped row list. When there is nothing to do (short list, no
        grids, no room) the ORIGINAL list object comes back unchanged.
    """
    head_keep = max(1, int(head_keep))
    if len(rows) <= head_keep or budget <= head_keep:
        return rows
    head = rows[:head_keep]
    # One anchor per distinct video, in rank order over the WHOLE head.
    anchors: list[tuple] = []
    anchor_videos: set[str] = set()
    for r in head:
        vid = str(r[0])
        if vid not in anchor_videos:
            anchor_videos.add(vid)
            anchors.append(r)

    seen = {(str(r[0]), int(r[1])) for r in rows}  # never duplicate a base row
    per_anchor: list[list[int]] = []
    for r in anchors:
        grid = grid_fn(str(r[0]))
        per_anchor.append(variant_frames(int(r[1]), grid, variants_per_anchor))

    max_variant_rows = int((budget - head_keep) * _VARIANT_TAIL_SHARE)
    variant_rows: list[tuple] = []
    for j in range(int(variants_per_anchor)):          # round-robin across anchors
        for a_idx, anchor in enumerate(anchors):
            if len(variant_rows) >= max_variant_rows:
                break
            vs = per_anchor[a_idx]
            if j >= len(vs):
                continue
            key = (str(anchor[0]), int(vs[j]))
            if key in seen:
                continue
            seen.add(key)
            variant_rows.append((anchor[0], vs[j], *anchor[2:]))
        if len(variant_rows) >= max_variant_rows:
            break

    if not variant_rows:
        log.info("row_strategy=diversify_tail: không dựng được variant nào "
                 "(thiếu lưới keyframe / frame ngoài lưới) — giữ nguyên ranking.")
        return rows
    out = list(head) + variant_rows + list(rows[head_keep:])
    log.info("row_strategy=diversify_tail: +%d variant quanh %d video top "
             "(đầu %d dòng giữ nguyên văn).", len(variant_rows), len(anchors), head_keep)
    return out[: int(budget)]


def catalog_grid_fn(catalog) -> GridFn:
    """Adapt a ``KeyframeCatalog`` (or None / stub) into a cached ``grid_fn``.

    Stub engines without a catalog (or a catalog whose load fails) yield a
    grid-less callback — ``diversify_tail`` then keeps the ranking unchanged
    and says so at INFO, never crashes the query.
    """
    if catalog is None or not hasattr(catalog, "load"):
        return lambda _vid: None
    cache: dict[str, list[int] | None] = {}

    def _grid(video_id: str) -> list[int] | None:
        if video_id not in cache:
            try:
                df = catalog.load()
                sub = df[df["video_id"] == video_id]
                cache[video_id] = (sorted(int(x) for x in sub["frame_idx"])
                                   if len(sub) else None)
            except Exception as e:  # noqa: BLE001 — variants are optional polish
                log.warning("row_strategy: không đọc được lưới keyframe của %s (%s)",
                            video_id, e)
                cache[video_id] = None
        return cache[video_id]

    return _grid
