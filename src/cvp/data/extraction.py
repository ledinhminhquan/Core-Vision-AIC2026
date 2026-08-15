"""Keyframe self-extraction for videos without organiser keyframes (K-batches).

Pipeline per video (the winning-team recipe):

  1. Shot boundaries — TransNetV2 (if installed) → PySceneDetect → fixed 2s
     windows, in that fallback order.
  2. 3 keyframes per shot at the 15% / 50% / 85% positions (MERVIN/LLandMark).
  3. Near-duplicate suppression by mean-absolute-difference on grey thumbnails.
  4. Writes ``keyframes/{vid}/{n:03d}.jpg`` and a CORRECT
     ``map-keyframes/{vid}.csv`` (n, pts_time, fps, frame_idx) — submissions
     depend on this file, so it is generated together with the frames.
"""

from __future__ import annotations

import csv
import logging
import os
from pathlib import Path

import numpy as np

from cvp.config import Settings
from cvp.data.catalog import KeyframeCatalog  # noqa: F401 — rebuilt by callers after extraction

log = logging.getLogger(__name__)

# Defaults — override per-corpus via settings.extraction (round-3 enhancement:
# UIT's ablation credits DENSER keyframes with +60–90% H@1 on hard queries, so
# K-batch density is a competition knob, e.g.
#   CVP_EXTRACTION__SHOT_POSITIONS='[0.1,0.3,0.5,0.7,0.9]'
SHOT_PERCENTILES = (0.15, 0.50, 0.85)
DEDUP_MAD_THRESHOLD = 6.0  # mean abs diff on 32x32 grey thumbs (0-255)


# ── shot detection backends ──────────────────────────────────────────────────


def _shots_transnet(video_path: Path) -> list[tuple[int, int]] | None:
    """TransNetV2 shot boundaries via the ``transnetv2-pytorch`` PyPI package.

    On Colab install it with ``--no-deps`` (torch/numpy/opencv are already
    there — Colab rule #1: never touch the preinstalled torch); notebook 01
    does this automatically when ``INSTALL_TRANSNETV2 = True``.
    """
    try:
        from transnetv2_pytorch import TransNetV2
    except ImportError:
        return None
    try:
        import torch

        model = TransNetV2()
        model.eval()
        if torch.cuda.is_available():
            model = model.cuda()
        with torch.no_grad():
            _, single_frame_pred, _ = model.predict_video(str(video_path))
        # predictions_to_scenes expects a 1-D numpy probability array; the
        # package may hand back a torch tensor (possibly (N, 1)-shaped).
        pred = single_frame_pred
        if hasattr(pred, "detach"):
            pred = pred.detach().cpu().numpy()
        scenes = model.predictions_to_scenes(np.asarray(pred).reshape(-1))
        return [(int(a), int(b)) for a, b in scenes]
    except Exception as e:  # noqa: BLE001
        log.warning("TransNetV2 failed on %s (%s) — falling back", video_path.name, e)
        return None


def _shots_scenedetect(video_path: Path) -> list[tuple[int, int]] | None:
    try:
        from scenedetect import ContentDetector, detect
    except ImportError:
        return None
    try:
        scenes = detect(str(video_path), ContentDetector(threshold=27.0))
        out = [(s.get_frames(), e.get_frames() - 1) for s, e in scenes if e.get_frames() > s.get_frames()]
        return out or None
    except Exception as e:  # noqa: BLE001
        log.warning("PySceneDetect failed on %s (%s) — falling back", video_path.name, e)
        return None


def _shots_fixed(total_frames: int, fps: float, window_s: float = 2.0) -> list[tuple[int, int]]:
    step = max(1, int(round(fps * window_s)))
    return [(a, min(a + step - 1, total_frames - 1)) for a in range(0, total_frames, step)]


def detect_shots(video_path: Path, total_frames: int, fps: float) -> list[tuple[int, int]]:
    for fn in (_shots_transnet, _shots_scenedetect):
        shots = fn(video_path)
        if shots:
            return shots
    return _shots_fixed(total_frames, fps)


# ── extraction ───────────────────────────────────────────────────────────────


def _pick_frames(shots: list[tuple[int, int]],
                 positions: tuple[float, ...] = SHOT_PERCENTILES) -> list[int]:
    picked: list[int] = []
    for a, b in shots:
        span = b - a
        for p in positions:
            picked.append(a + int(round(span * p)))
    return sorted(set(picked))


def extract_video(video_path: Path, keyframes_dir: Path, map_dir: Path,
                  overwrite: bool = False,
                  shot_positions: tuple[float, ...] | None = None,
                  dedup_mad: float | None = None) -> int:
    """Extract keyframes + map CSV for one video. Returns keyframe count."""
    import csv as _csv

    vid = video_path.stem
    out_dir = keyframes_dir / vid
    map_path = map_dir / f"{vid}.csv"
    # Written while THIS extractor is mid-run; removed after the map csv lands.
    # Its absence proves existing jpgs came from somewhere else (organiser zip).
    sentinel = out_dir / ".cvp-extracting"
    if out_dir.is_dir() and map_path.is_file() and not overwrite:
        existing = len([f for f in out_dir.iterdir() if f.suffix.lower() == ".jpg"])
        with open(map_path, "r", encoding="utf-8-sig", newline="") as f:
            csv_rows = sum(1 for _ in _csv.DictReader(f))
        if existing > 0 and existing == csv_rows:
            sentinel.unlink(missing_ok=True)
            return existing
        log.warning("%s: %d jpgs vs %d map rows", vid, existing, csv_rows)
    if out_dir.is_dir() and not overwrite and not sentinel.exists():
        existing = len([f for f in out_dir.iterdir() if f.suffix.lower() == ".jpg"])
        if existing > 0:
            # Organiser keyframes with a missing/mismatched map csv (e.g. the
            # map zip not unzipped yet, or a partial unzip). Deleting them and
            # substituting approximate self-extracted frames would desync the
            # official clip-features/objects packs — refuse instead. Return 0:
            # nothing was extracted and there is no usable map, so callers must
            # not count this as fresh work (extract_missing's tally, nb01's
            # forced catalog rebuild).
            log.error(
                "%s: %d existing jpgs but no matching map csv and they were NOT "
                "written by this extractor — REFUSING to replace what may be "
                "organiser keyframes. Unzip the official map-keyframes package "
                "(or finish the partial unzip), or pass overwrite=True.",
                vid, existing,
            )
            return 0
    # Written next to every csv THIS extractor produces — unlike the in-dir
    # sentinel it SURVIVES deleting the keyframes dir, so our own csvs stay
    # re-extractable (K-batch dense re-extraction, round-6) while organiser
    # csvs stay protected.
    selfmade = map_dir / f"{vid}.csv.selfmade"
    if (map_path.is_file() and not overwrite and not sentinel.exists()
            and not selfmade.exists()):
        # Mirror guard for the MAP side (round-5 HIGH): the organiser map csv
        # can land BEFORE the big Keyframes zips finish uploading. With no
        # keyframes dir the jpg guards above never fire, and self-extraction
        # would os.replace() the official csv with approximate shot-detector
        # rows — silently corrupting the n↔frame_idx bridge for submissions.
        log.error(
            "%s: an official-looking map csv already exists but the keyframes "
            "dir is missing/empty and the csv was NOT written by this extractor "
            "— REFUSING to overwrite it with approximate rows. Organiser packs: "
            "unzip the Keyframes package first. Self-extracted video: delete "
            "the map csv too, or rerun with overwrite (scripts/01 --overwrite).",
            vid,
        )
        return 0

    import cv2

    # Re-extraction must not leave stale high-n jpgs from a previous run.
    if out_dir.is_dir():
        for f in out_dir.iterdir():
            if f.suffix.lower() in (".jpg", ".tmp"):
                f.unlink(missing_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        cap.release()
        raise RuntimeError(f"Cannot read frame count: {video_path}")

    # `is not None` (not truthiness): a configured-but-empty list must fail in
    # ExtractionCfg's validator, never silently revert to the defaults here.
    positions = tuple(shot_positions) if shot_positions is not None else SHOT_PERCENTILES
    mad_threshold = dedup_mad if dedup_mad is not None else DEDUP_MAD_THRESHOLD
    shots = detect_shots(video_path, total, fps)
    frame_ids = [f for f in _pick_frames(shots, positions) if 0 <= f < total]
    log.info("%s: %d shots → %d candidate keyframes (fps=%.2f, frames=%d)",
             vid, len(shots), len(frame_ids), fps, total)

    out_dir.mkdir(parents=True, exist_ok=True)
    sentinel.touch()
    rows: list[tuple[int, float, float, int]] = []
    prev_thumb: np.ndarray | None = None
    n = 0
    for fid in frame_ids:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        thumb = cv2.cvtColor(cv2.resize(frame, (32, 32)), cv2.COLOR_BGR2GRAY).astype(np.float32)
        if prev_thumb is not None and float(np.abs(thumb - prev_thumb).mean()) < mad_threshold:
            continue  # near-duplicate of the previous pick
        prev_thumb = thumb
        n += 1
        # cv2.imwrite picks the codec from the EXTENSION (".tmp" would fail) —
        # encode to JPEG bytes, write tmp, then atomic-rename.
        ok_enc, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        if not ok_enc:
            n -= 1
            continue
        tmp = out_dir / f"{n:03d}.jpg.tmp"
        tmp.write_bytes(buf.tobytes())
        os.replace(tmp, out_dir / f"{n:03d}.jpg")
        rows.append((n, fid / fps, fps, fid))
    cap.release()

    map_dir.mkdir(parents=True, exist_ok=True)
    tmp_csv = map_path.with_suffix(".csv.tmp")
    with open(tmp_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n", "pts_time", "fps", "frame_idx"])
        for row in rows:
            w.writerow([row[0], f"{row[1]:.2f}", row[2], row[3]])
    os.replace(tmp_csv, map_path)
    selfmade.touch()
    sentinel.unlink(missing_ok=True)
    log.info("%s: wrote %d keyframes + map CSV", vid, n)
    return n


def extract_missing(settings: Settings, overwrite: bool = False) -> int:
    """Extract every video under data/videos that has no keyframes yet.

    ``overwrite=True`` (scripts/01 --overwrite) force-re-extracts everything —
    the documented path for re-running with denser ``shot_positions``.
    """
    video_root = settings.paths.data(settings.paths.videos_dir)
    keyframes_dir = settings.paths.data(settings.paths.keyframes_dir)
    map_dir = settings.paths.data(settings.paths.map_keyframes_dir)
    if not video_root.is_dir():
        log.info("No raw videos folder (%s) — nothing to extract", video_root)
        return 0
    count = 0
    # The organiser Videos zips wrap mp4s in a `video/` (singular) dir — accept
    # both the flat layout and an unflattened unzip; flat wins on collision.
    by_stem: dict[str, Path] = {}
    if (video_root / "video").is_dir():
        by_stem.update({vp.stem: vp for vp in (video_root / "video").glob("*.mp4")})
    by_stem.update({vp.stem: vp for vp in video_root.glob("*.mp4")})
    for vid in sorted(by_stem):
        vp = by_stem[vid]
        try:
            # extract_video itself decides whether existing output is complete
            # (jpg count must reconcile with the map CSV) — a bare folder check
            # here would accept partial extractions.
            before = (keyframes_dir / vid).is_dir() and (map_dir / f"{vid}.csv").is_file()
            ex_cfg = getattr(settings, "extraction", None)
            n = extract_video(
                vp, keyframes_dir, map_dir, overwrite=overwrite,
                shot_positions=tuple(ex_cfg.shot_positions) if ex_cfg else None,
                dedup_mad=ex_cfg.dedup_mad_threshold if ex_cfg else None,
            )
            if not before and n > 0:
                count += 1
        except Exception as e:  # noqa: BLE001 — a broken file must not stop the batch
            log.error("Extraction failed for %s: %s", vid, e)
    return count
