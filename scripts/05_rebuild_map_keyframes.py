"""Reconstruct missing map-keyframes csvs by visually matching keyframes to videos.

WHY: the AIC data drops on disk ship organiser keyframes (keyframes/{vid}/nnn.jpg,
names are 1-based temporal ranks) but often NO map-keyframes csvs — yet
submissions must contain the real ``frame_idx`` in the original video. For every
video that has keyframes AND a raw video but no map csv, this script rebuilds
``map-keyframes/{vid}.csv`` (n,pts_time,fps,frame_idx):

1. dhash every organiser keyframe jpg;
2. stream the video strided (default every 5th frame, decoded small) and dhash
   each sampled position;
3. align keyframes to positions with a monotone DP (ranks are temporally
   ordered, which constrains the match and disambiguates repeated shots);
4. refine each match exactly within +/- stride frames.

The result is APPROXIMATE — prefer the organiser map-keyframes when released.
Resumable: existing csvs are skipped unless --overwrite.

    python scripts/05_rebuild_map_keyframes.py [--videos-dir DIR]
        [--keyframes-dir DIR] [--out-dir DIR] [--stride 5] [--overwrite]
"""

import argparse
import logging
from pathlib import Path

from _bootstrap import init

import numpy as np
from PIL import Image

from cvp.constants import is_video_id
from cvp.data.keyframe_align import (
    decode_window,
    keyframe_files,
    match_hashes_monotonic,
    refinement_window,
    write_map_keyframes_csv,
)
from cvp.data.video_frames import VideoFrames
from cvp.utils.images import dhash, hamming, load_rgb

log = logging.getLogger("scripts.rebuild_map_keyframes")

VIDEO_EXTS = (".mp4", ".mkv", ".webm", ".avi", ".mov")
# Decode size for hashing: dhash reduces to 9x8 grayscale anyway, so a small
# decode is lossless for matching and ~10x faster than full resolution.
DECODE_SIZE = (160, 90)


def find_video(videos_dir: Path, video_id: str) -> Path | None:
    for ext in VIDEO_EXTS:
        p = videos_dir / f"{video_id}{ext}"
        if p.exists():
            return p
    return None


def rebuild_one(video_id: str, video_path: Path, kf_dir: Path, out_csv: Path,
                stride: int) -> int:
    """Reconstruct one map csv. Returns the number of rows written."""
    keyframes = keyframe_files(kf_dir)
    if not keyframes:
        raise RuntimeError(f"no keyframe jpgs in {kf_dir}")
    key_hashes = []
    for _n, p in keyframes:
        img = load_rgb(p)
        if img is None:
            raise RuntimeError(f"unreadable keyframe {p}")
        key_hashes.append(dhash(img))

    rows: list[tuple[int, float, float, int]] = []
    with VideoFrames(video_path, size=DECODE_SIZE) as vf:
        fps = vf.fps
        # Pass 1: strided dhash stream (small decode, bounded memory).
        stream_pos: list[int] = []
        stream_hashes: list[int] = []
        for frame_idx, arr in vf.iter_strided(stride):
            stream_pos.append(frame_idx)
            stream_hashes.append(dhash(Image.fromarray(arr)))
        if not stream_hashes:
            raise RuntimeError(f"decoded 0 frames from {video_path}")
        # Highest frame PROVEN decodable. The container header (len(vf)) can
        # overstate the range on truncated videos — or report 0 — and either
        # would corrupt or abort the refinement below.
        last = stream_pos[-1]

        # Pass 2: monotone alignment of keyframe ranks to stream positions.
        match = match_hashes_monotonic(key_hashes, stream_hashes)

        # Pass 3: exact refinement within +/- stride frames of each match
        # (inclusive, so a coarse match one slot off is still recoverable).
        prev = -1
        for (n, _path), kh, s in zip(keyframes, key_hashes, match):
            approx = stream_pos[s]
            window = refinement_window(approx, prev, stride, last)
            ok_pos, frames = decode_window(window, vf.get_batch, vf.get)
            if len(frames):
                dists = [hamming(kh, dhash(Image.fromarray(np.asarray(f)))) for f in frames]
                best = ok_pos[int(np.argmin(dists))]
            else:  # window undecodable — keep the coarse (proven) position
                best = approx
            best = min(max(best, prev + 1), last)  # keep frame_idx increasing
            prev = best
            rows.append((n, round(best / fps, 3), round(fps, 3), best))

    write_map_keyframes_csv(out_csv, rows)
    return len(rows)


def main():
    ap = argparse.ArgumentParser(
        description="Rebuild map-keyframes csvs by visual keyframe->video matching."
    )
    ap.add_argument("--settings", default=None)
    ap.add_argument("--videos-dir", default=None, help="raw videos (default: settings)")
    ap.add_argument("--keyframes-dir", default=None, help="organiser keyframes (default: settings)")
    ap.add_argument("--out-dir", default=None, help="map-keyframes output (default: settings)")
    ap.add_argument("--stride", type=int, default=5, help="coarse-scan stride in frames")
    ap.add_argument("--overwrite", action="store_true", help="rebuild existing csvs too")
    args = ap.parse_args()

    s = init(args.settings)
    videos_dir = Path(args.videos_dir) if args.videos_dir else s.paths.data(s.paths.videos_dir)
    kf_root = Path(args.keyframes_dir) if args.keyframes_dir else s.paths.data(s.paths.keyframes_dir)
    out_dir = Path(args.out_dir) if args.out_dir else s.paths.data(s.paths.map_keyframes_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.warning(
        "map-keyframes RECONSTRUCTION is approximate (perceptual-hash matching). "
        "Prefer the organiser's map-keyframes package when it is released."
    )
    if not kf_root.exists():
        raise SystemExit(f"keyframes dir not found: {kf_root}")

    video_ids = sorted(d.name for d in kf_root.iterdir() if d.is_dir() and is_video_id(d.name))
    done = skipped = missing = failed = 0
    for vid in video_ids:
        out_csv = out_dir / f"{vid}.csv"
        if out_csv.exists() and not args.overwrite:
            skipped += 1
            continue
        video_path = find_video(videos_dir, vid)
        if video_path is None:
            missing += 1
            continue
        try:
            n = rebuild_one(vid, video_path, kf_root / vid, out_csv, max(1, args.stride))
        except Exception as e:  # noqa: BLE001 — one broken video must not stop the sweep
            log.error("[%s] reconstruction failed: %s", vid, e)
            failed += 1
            continue
        done += 1
        print(f"  {vid}: {n} keyframes mapped -> {out_csv.name}")

    print(
        f"\nDone: {done} rebuilt, {skipped} already present (skipped), "
        f"{missing} without a raw video, {failed} failed."
    )


if __name__ == "__main__":
    main()
