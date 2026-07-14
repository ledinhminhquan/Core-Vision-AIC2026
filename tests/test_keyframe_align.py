"""map-keyframes reconstruction toolkit (port from Core-Vision_HCMC-AI).

Pure-logic tests: monotone hash matching, refinement windows, decode salvage,
csv guards, stale-keyframe pruning and the rebuild script's truncated-tail
survival. No video IO — synthetic arrays, hashes and fake readers only.

WHY this matters here: the user's AIC data drops ship keyframes WITHOUT
map-keyframes csvs, and submissions need the original-video ``frame_idx``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from cvp.data.keyframe_align import (
    decode_window,
    hamming_matrix,
    keyframe_files,
    match_hashes_monotonic,
    prune_stale_keyframes,
    refinement_window,
    write_map_keyframes_csv,
)
from cvp.utils.images import dhash, hamming, is_near_duplicate

REPO = Path(__file__).resolve().parents[1]


# ── hamming / popcount ───────────────────────────────────────────────────────
def test_hamming_matrix_matches_scalar_hamming():
    rng = np.random.default_rng(42)
    a = [int(x) for x in rng.integers(0, 2**63, size=6, dtype=np.int64)]
    b = [int(x) for x in rng.integers(0, 2**63, size=4, dtype=np.int64)]
    mat = hamming_matrix(a, b)
    assert mat.shape == (6, 4)
    for i, ha in enumerate(a):
        for j, hb in enumerate(b):
            assert int(mat[i, j]) == hamming(ha, hb)


def test_hamming_matrix_full_64bit_range():
    all_ones = (1 << 64) - 1
    assert hamming_matrix([0, all_ones], [all_ones]).tolist() == [[64], [0]]


def test_dhash_and_near_duplicate_roundtrip():
    from PIL import Image

    rng = np.random.default_rng(7)
    img = Image.fromarray(rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8))
    h = dhash(img)
    assert 0 <= h < 2**64
    assert is_near_duplicate(h, [h], threshold=0)
    assert not is_near_duplicate(h, [], threshold=64)


# ── monotone matching ────────────────────────────────────────────────────────
def _distinct_hashes(n: int, seed: int = 0) -> list[int]:
    rng = np.random.default_rng(seed)
    return [int(x) for x in rng.integers(0, 2**63, size=n, dtype=np.int64)]


def test_match_recovers_exact_positions():
    stream = _distinct_hashes(12)
    keys = [stream[2], stream[5], stream[9]]
    assert match_hashes_monotonic(keys, stream) == [2, 5, 9]


def test_match_tolerates_bit_noise():
    stream = _distinct_hashes(12, seed=1)
    keys = [stream[3] ^ 1, stream[7] ^ (1 << 20)]  # one flipped bit each
    assert match_hashes_monotonic(keys, stream) == [3, 7]


def test_match_enforces_monotonic_order():
    a, b = _distinct_hashes(2, seed=2)
    # Unconstrained argmins would be [1, 0] (reversed); the DP must return a
    # non-decreasing assignment with the minimal constrained cost hamming(a,b).
    pos = match_hashes_monotonic([b, a], [a, b])
    assert pos[0] <= pos[1]
    cost = hamming_matrix([b, a], [a, b])
    assert int(cost[0, pos[0]]) + int(cost[1, pos[1]]) == hamming(a, b)


def test_match_is_monotonic_on_random_input():
    keys = _distinct_hashes(20, seed=3)
    stream = _distinct_hashes(50, seed=4)
    pos = match_hashes_monotonic(keys, stream)
    assert len(pos) == 20
    assert all(p1 <= p2 for p1, p2 in zip(pos, pos[1:]))
    assert all(0 <= p < 50 for p in pos)


def test_match_allows_repeated_position_when_keys_outnumber_stream():
    stream = _distinct_hashes(2, seed=5)
    keys = [stream[0], stream[1], stream[1], stream[1]]
    assert match_hashes_monotonic(keys, stream) == [0, 1, 1, 1]


def test_match_empty_inputs():
    assert match_hashes_monotonic([], [1, 2, 3]) == []
    assert match_hashes_monotonic([1], []) == []


# ── refinement windows ───────────────────────────────────────────────────────
def test_refinement_window_basic():
    assert refinement_window(approx=10, prev=-1, stride=5, last=100) == list(range(5, 16))


def test_refinement_window_starts_after_prev():
    assert refinement_window(approx=10, prev=8, stride=5, last=100) == list(range(9, 16))


def test_refinement_window_bounded_by_last_decodable():
    win = refinement_window(approx=50, prev=-1, stride=5, last=52)
    assert win == list(range(45, 53)) and max(win) <= 52


def test_refinement_window_collapsed_pins_to_edge():
    assert refinement_window(approx=10, prev=50, stride=2, last=50) == [50]


def test_refinement_window_zero_frame_header_never_negative():
    assert refinement_window(approx=0, prev=-1, stride=5, last=0) == [0]


# ── decode-window salvage ────────────────────────────────────────────────────
def _flat(v: int) -> np.ndarray:
    return np.full((2, 2, 3), v, dtype=np.uint8)


def test_decode_window_batch_success_passthrough():
    frames = {i: _flat(i) for i in range(5)}
    pos, out = decode_window(
        [1, 2, 3], lambda idx: [frames[i] for i in idx], frames.__getitem__
    )
    assert pos == [1, 2, 3]
    assert [int(f[0, 0, 0]) for f in out] == [1, 2, 3]


def test_decode_window_salvages_per_frame_on_batch_failure():
    good = {1: _flat(1), 2: _flat(2)}  # frame 3 is past the decodable tail

    def get_batch(idx):
        raise RuntimeError("truncated tail")

    pos, out = decode_window([1, 2, 3], get_batch, lambda i: good[i])
    assert pos == [1, 2]


def test_decode_window_all_undecodable_returns_empty():
    def boom(*_a):
        raise RuntimeError("no")

    assert decode_window([7, 8], boom, boom) == ([], [])


# ── csv guards + pruning + listing ───────────────────────────────────────────
def test_write_map_keyframes_csv_refuses_negative_frame_idx(tmp_path):
    out = tmp_path / "L01_V001.csv"
    with pytest.raises(ValueError, match="negative"):
        write_map_keyframes_csv(out, [(1, 0.0, 25.0, 0), (2, -0.04, 25.0, -1)])
    assert not out.exists()


def test_write_map_keyframes_csv_writes_valid_rows(tmp_path):
    out = tmp_path / "L01_V001.csv"
    write_map_keyframes_csv(out, [(1, 0.0, 25.0, 0), (2, 0.2, 25.0, 5)])
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert lines[0] == "n,pts_time,fps,frame_idx" and len(lines) == 3


def test_prune_stale_keyframes_removes_only_higher_ordinals(tmp_path):
    for name in ("001.jpg", "002.jpg", "003.jpg", "004.jpg", "notes.txt"):
        (tmp_path / name).write_bytes(b"x")
    assert prune_stale_keyframes(tmp_path, keep_n=2) == 2
    assert sorted(f.name for f in tmp_path.iterdir()) == ["001.jpg", "002.jpg", "notes.txt"]


def test_keyframe_files_sorted_by_rank(tmp_path):
    for name in ("010.jpg", "002.jpg", "001.jpg", "junk.txt"):
        (tmp_path / name).write_bytes(b"x")
    assert [n for n, _ in keyframe_files(tmp_path)] == [1, 2, 10]


# ── scripts/05 rebuild_one (truncated tail, fake reader) ─────────────────────
def _load_rebuild_script():
    scripts = REPO / "scripts"
    if str(scripts) not in sys.path:  # the script imports _bootstrap
        sys.path.insert(0, str(scripts))
    spec = importlib.util.spec_from_file_location(
        "rebuild_map_keyframes_script", scripts / "05_rebuild_map_keyframes.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_rebuild_one_survives_truncated_tail(tmp_path, monkeypatch):
    """Container header claims 100 frames but only 0..9 decode: the refinement
    must stay within the proven range instead of failing the whole video."""
    from PIL import Image

    mod = _load_rebuild_script()
    rng = np.random.default_rng(2)
    decodable = {
        i: rng.integers(0, 256, size=(24, 24, 3), dtype=np.uint8) for i in range(10)
    }

    class _TruncatedReader:
        def __init__(self, path, size=None):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return None

        def __len__(self):
            return 100  # header overstates the decodable range

        @property
        def fps(self):
            return 25.0

        def iter_strided(self, stride):
            for i in range(0, 10, stride):
                yield i, decodable[i]

        def get(self, i):
            if i not in decodable:
                raise RuntimeError(f"cannot decode frame {i}")
            return decodable[i]

        def get_batch(self, idx):
            return [self.get(i) for i in idx]

    monkeypatch.setattr(mod, "VideoFrames", _TruncatedReader)

    kf_dir = tmp_path / "kf"
    kf_dir.mkdir()
    for n, src in ((1, 5), (2, 9)):
        Image.fromarray(decodable[src]).save(kf_dir / f"{n:03d}.jpg", quality=95)

    out_csv = tmp_path / "L01_V001.csv"
    assert mod.rebuild_one("L01_V001", tmp_path / "fake.mp4", kf_dir, out_csv, stride=5) == 2
    rows = out_csv.read_text(encoding="utf-8").strip().splitlines()[1:]
    frame_idx = [int(r.split(",")[3]) for r in rows]
    assert all(0 <= f <= 9 for f in frame_idx)  # never past the proven tail
    assert frame_idx == sorted(frame_idx)
