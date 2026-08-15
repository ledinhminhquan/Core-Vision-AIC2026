"""Visual keyframe↔video alignment: the map-keyframes reconstruction core.

WHY THIS EXISTS: submissions must carry the ORIGINAL-video ``frame_idx``, and
the bridge is ``map-keyframes/{vid}.csv``. The 2026 prelim Batch 1 DOES ship
the official map csvs (verified 2026-08-15, 873/873 — always prefer those),
but future batches or video-only drops may not.
``scripts/05_rebuild_map_keyframes.py`` reconstructs them by perceptual-hash
matching; every pure-logic piece lives here so it is importable and testable
without video IO (script module names start with a digit and cannot be
imported).

Ported from Core-Vision_HCMC-AI ``data/keyframe_selection.py`` (the flagship
predecessor never inherited it). The reconstruction is APPROXIMATE — always
prefer the organiser's map-keyframes package once released.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np

from cvp.constants import MAP_KEYFRAMES_COLUMNS, parse_keyframe_ordinal
from cvp.utils.io import atomic_write_text


# ── Vectorised Hamming machinery ─────────────────────────────────────────────
def _popcount64(x: np.ndarray) -> np.ndarray:
    """Vectorised 64-bit popcount (SWAR) — works on numpy<2 where
    ``np.bitwise_count`` does not exist. Input/output are uint64/uint8 arrays."""
    m1 = np.uint64(0x5555555555555555)
    m2 = np.uint64(0x3333333333333333)
    m4 = np.uint64(0x0F0F0F0F0F0F0F0F)
    h01 = np.uint64(0x0101010101010101)
    x = x - ((x >> np.uint64(1)) & m1)
    x = (x & m2) + ((x >> np.uint64(2)) & m2)
    x = (x + (x >> np.uint64(4))) & m4
    return ((x * h01) >> np.uint64(56)).astype(np.uint8)


def hamming_matrix(a: Sequence[int], b: Sequence[int]) -> np.ndarray:
    """Pairwise Hamming distances between two lists of 64-bit hashes.

    Returns a (len(a), len(b)) uint8 matrix.
    """
    aa = np.asarray(list(a), dtype=np.uint64)[:, None]
    bb = np.asarray(list(b), dtype=np.uint64)[None, :]
    return _popcount64(aa ^ bb)


def match_hashes_monotonic(
    key_hashes: Sequence[int], stream_hashes: Sequence[int]
) -> list[int]:
    """Best NON-DECREASING assignment of ordered keyframes to stream positions.

    ``key_hashes`` are perceptual hashes of keyframes in temporal (rank) order;
    ``stream_hashes`` are hashes of strided video positions. Because keyframe
    ranks are temporally ordered, their matched positions must be monotonic —
    a DP over the full Hamming-cost matrix finds the assignment minimising the
    total cost subject to that constraint (a per-keyframe greedy argmin could
    jump backwards on repeated/similar shots). O(K·S) time, vectorised over S.

    Returns one stream index per key hash (non-decreasing); ``[]`` if either
    input is empty.
    """
    K, S = len(key_hashes), len(stream_hashes)
    if K == 0 or S == 0:
        return []
    cost = hamming_matrix(key_hashes, stream_hashes).astype(np.float32)
    # dp[s] = min cost of matching keys[:k+1] with the k-th key at position s.
    dp = cost[0]
    args: list[np.ndarray] = []  # args[k-1][s] = best prev position <= s for row k
    positions = np.arange(S)
    for k in range(1, K):
        run_min = np.minimum.accumulate(dp)
        arg = np.where(dp <= run_min, positions, 0)
        arg = np.maximum.accumulate(arg)  # forward-fill the prefix argmin
        args.append(arg)
        dp = cost[k] + run_min
    pos = [0] * K
    pos[K - 1] = int(np.argmin(dp))
    for k in range(K - 2, -1, -1):
        pos[k] = int(args[k][pos[k + 1]])
    return pos


def refinement_window(approx: int, prev: int, stride: int, last: int) -> list[int]:
    """Frame positions to search when refining a coarse strided match.

    ``last`` must be the highest frame index PROVEN decodable (the final
    position the strided pass actually yielded) — NOT the container header
    count, which can exceed the decodable range on truncated videos and would
    send the refinement reading past the end. ``prev`` is the previous
    keyframe's refined position; the window starts after it so refined
    ``frame_idx`` values stay strictly increasing. Never returns an empty
    window or an index outside ``[0, last]``.
    """
    lo = max(0, prev + 1, approx - stride)
    hi = min(last, approx + stride)
    if lo > hi:  # window collapsed by monotonicity — pin to the edge
        lo = hi = min(max(prev + 1, 0), last)
    return list(range(lo, hi + 1))


def decode_window(
    window: Sequence[int],
    get_batch: Callable[[Sequence[int]], Iterable[np.ndarray]],
    get_one: Callable[[int], np.ndarray],
) -> tuple[list[int], list[np.ndarray]]:
    """Decode ``window`` positions, salvaging what a failed batch decode can.

    Tries one batched decode first; if it raises (corrupt GOP, truncated tail)
    falls back to per-frame reads and drops only the positions that still fail.
    Returns ``(positions, frames)`` with positions a subset of ``window`` —
    possibly empty, in which case the caller should fall back to its coarse
    approximation instead of aborting the whole video.
    """
    window = list(window)
    try:
        return window, list(get_batch(window))
    except Exception:
        ok: list[int] = []
        frames: list[np.ndarray] = []
        for i in window:
            try:
                frames.append(get_one(i))
            except Exception:
                continue
            ok.append(i)
        return ok, frames


# ── Disk helpers shared with the rebuild script ──────────────────────────────
def keyframe_files(kf_dir: Path) -> list[tuple[int, Path]]:
    """(ordinal, path) for every keyframe image, sorted by 1-based rank."""
    out = [
        (o, f)
        for f in Path(kf_dir).iterdir()
        if (o := parse_keyframe_ordinal(f.name)) is not None
    ]
    return sorted(out)


def prune_stale_keyframes(out_dir: str | Path, keep_n: int = 0) -> int:
    """Delete every ordinal-named keyframe image with ordinal > ``keep_n``.

    Keeps the jpgs on disk and the map csv a bijection: a re-run that produces
    FEWER keyframes than the previous one must not leave stale higher-n files
    behind — the catalog would invent rows for them and corrupt submissions.
    Returns the number of files removed.
    """
    removed = 0
    for f in Path(out_dir).iterdir():
        o = parse_keyframe_ordinal(f.name)
        if o is not None and o > keep_n:
            f.unlink()
            removed += 1
    return removed


def write_map_keyframes_csv(path: str | Path, rows: Sequence[tuple]) -> Path:
    """Write a map-keyframes csv (n,pts_time,fps,frame_idx) atomically.

    Refuses rows with a negative ``frame_idx`` — those only arise from broken
    readers (e.g. a container header reporting 0 frames) and would silently
    corrupt every submission built on the csv.
    """
    bad = [r for r in rows if int(r[3]) < 0]
    if bad:
        raise ValueError(
            f"refusing to write {path}: {len(bad)} row(s) with negative "
            f"frame_idx (first: {bad[0]})"
        )
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(MAP_KEYFRAMES_COLUMNS)
    w.writerows(rows)
    atomic_write_text(path, buf.getvalue())
    return Path(path)
