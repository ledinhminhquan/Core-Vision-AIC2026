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


def _map_csv_sha256(p: Path) -> str:
    import hashlib

    return hashlib.sha256(p.read_bytes()).hexdigest()


def _is_selfmade(marker: Path, map_path: Path) -> bool:
    """The .selfmade marker vouches ONLY for the exact csv it was written for.

    The marker stores the csv's sha256 (round-7): a bare-existence check let a
    STALE marker keep bypassing the organiser-protection guard after the
    official csv replaced ours.
    """
    if not (marker.is_file() and map_path.is_file()):
        return False
    try:
        return marker.read_text(encoding="utf-8").strip() == _map_csv_sha256(map_path)
    except OSError:
        return False


def extract_video(video_path: Path, keyframes_dir: Path, map_dir: Path,
                  overwrite: bool = False,
                  force_organiser: bool = False,
                  shot_positions: tuple[float, ...] | None = None,
                  dedup_mad: float | None = None) -> int:
    """Extract keyframes + map CSV for one video. Returns keyframe count.

    Organiser-made keyframes/map csvs (anything without OUR content-bound
    ``.selfmade`` marker) are never destroyed — not even with ``overwrite=True``
    — unless ``force_organiser=True`` is passed explicitly (round-7 HIGH: a
    bare global overwrite would have replaced all 873 official Batch-1 map
    csvs with approximate rows).
    """
    import csv as _csv

    vid = video_path.stem
    out_dir = keyframes_dir / vid
    map_path = map_dir / f"{vid}.csv"
    # Written while THIS extractor is mid-run; removed after the map csv lands.
    # Its absence proves existing jpgs came from somewhere else (organiser zip).
    sentinel = out_dir / ".cvp-extracting"
    # Written next to every csv THIS extractor produces (content = csv sha256)
    # — unlike the in-dir sentinel it SURVIVES deleting the keyframes dir, so
    # our own csvs stay re-extractable while organiser csvs stay protected.
    selfmade = map_dir / f"{vid}.csv.selfmade"
    if selfmade.is_file() and not _is_selfmade(selfmade, map_path):
        # The csv at this path is no longer the one we wrote (organiser csv
        # landed on top, or a crash) — the marker must not vouch for it.
        selfmade.unlink(missing_ok=True)

    ours = sentinel.exists() or _is_selfmade(selfmade, map_path)
    existing_jpgs = (len([f for f in out_dir.iterdir() if f.suffix.lower() == ".jpg"])
                     if out_dir.is_dir() else 0)

    if out_dir.is_dir() and map_path.is_file() and not overwrite:
        with open(map_path, "r", encoding="utf-8-sig", newline="") as f:
            csv_rows = sum(1 for _ in _csv.DictReader(f))
        if existing_jpgs > 0 and existing_jpgs == csv_rows:
            sentinel.unlink(missing_ok=True)
            return existing_jpgs
        log.warning("%s: %d jpgs vs %d map rows", vid, existing_jpgs, csv_rows)

    organiser_data = (map_path.is_file() or existing_jpgs > 0) and not ours
    if organiser_data and not (overwrite and force_organiser):
        # Organiser keyframes and/or an official map csv (e.g. the map zip
        # landed before the big Keyframes zips finished, or a partial unzip).
        # Deleting/replacing them with approximate shot-detector output would
        # desync the official clip-features/objects packs and corrupt the
        # n↔frame_idx bridge — refuse, EVEN under a global overwrite. Return 0
        # so callers never count a refusal as fresh work.
        log.error(
            "%s: existing keyframes/map csv were NOT written by this extractor "
            "— REFUSING to replace what may be the organiser's official data "
            "(jpgs=%d, map=%s). Organiser packs: unzip the missing package "
            "instead. To force-destroy organiser data anyway: scripts/01 "
            "--overwrite --force-organiser --video %s.",
            vid, existing_jpgs, map_path.is_file(), vid,
        )
        return 0

    import cv2

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
    # Clean stale jpgs only NOW — after the video opened and its frame count
    # read (round-7: a corrupt mp4 used to get its existing jpgs deleted
    # BEFORE VideoCapture failed, leaving nothing behind).
    for f in out_dir.iterdir():
        if f.suffix.lower() in (".jpg", ".tmp"):
            f.unlink(missing_ok=True)
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

    if not rows:
        # Every cap.read() failed after the container opened (bit-rotted mp4).
        # NEVER land a header-only csv over a previous good one — the jpgs are
        # already gone (logged), but the old map must survive so the corruption
        # is recoverable and visible (round-10 split finding).
        log.error(
            "%s: video opened but produced 0 frames — keeping the previous map "
            "csv untouched; existing jpgs were cleared. Fix/redownload the "
            "video and re-run.", vid,
        )
        return 0

    map_dir.mkdir(parents=True, exist_ok=True)
    tmp_csv = map_path.with_suffix(".csv.tmp")
    with open(tmp_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n", "pts_time", "fps", "frame_idx"])
        for row in rows:
            w.writerow([row[0], f"{row[1]:.2f}", row[2], row[3]])
    os.replace(tmp_csv, map_path)
    try:
        selfmade.write_text(_map_csv_sha256(map_path), encoding="utf-8")
    except OSError as e:  # marker is protection metadata, never a hard failure
        log.warning("%s: could not write .selfmade marker (%s)", vid, e)
    sentinel.unlink(missing_ok=True)
    log.info("%s: wrote %d keyframes + map CSV", vid, n)
    return n


def extract_missing(settings: Settings, overwrite: bool = False,
                    only: list[str] | None = None,
                    force_organiser: bool = False) -> int:
    """Extract every video under data/videos that has no keyframes yet.

    ``overwrite=True`` (scripts/01 --overwrite) force-re-extracts — scoped to
    ``only`` video ids when given (scripts/01 --video, repeatable). Organiser
    keyframes/map csvs stay protected even then unless ``force_organiser``.
    """
    video_root = settings.paths.data(settings.paths.videos_dir)
    keyframes_dir = settings.paths.data(settings.paths.keyframes_dir)
    map_dir = settings.paths.data(settings.paths.map_keyframes_dir)
    if not video_root.is_dir():
        log.info("No raw videos folder (%s) — nothing to extract", video_root)
        return 0
    count = 0
    reextracted = 0
    refused = 0
    # The organiser Videos zips wrap mp4s in a `video/` (singular) dir — accept
    # both the flat layout and an unflattened unzip; flat wins on collision.
    by_stem: dict[str, Path] = {}
    if (video_root / "video").is_dir():
        by_stem.update({vp.stem: vp for vp in (video_root / "video").glob("*.mp4")})
    by_stem.update({vp.stem: vp for vp in video_root.glob("*.mp4")})
    if only:
        wanted = set(only)
        missing = wanted - set(by_stem)
        if missing:
            log.error("--video ids with no matching mp4 under %s: %s",
                      video_root, sorted(missing))
        by_stem = {v: p for v, p in by_stem.items() if v in wanted}
    for vid in sorted(by_stem):
        vp = by_stem[vid]
        try:
            # extract_video itself decides whether existing output is complete
            # (jpg count must reconcile with the map CSV) — a bare folder check
            # here would accept partial extractions.
            map_p = map_dir / f"{vid}.csv"
            before = (keyframes_dir / vid).is_dir() and map_p.is_file()
            had_any = (keyframes_dir / vid).is_dir() or map_p.is_file()
            digest_before = _map_csv_sha256(map_p) if map_p.is_file() else None
            ex_cfg = getattr(settings, "extraction", None)
            n = extract_video(
                vp, keyframes_dir, map_dir, overwrite=overwrite,
                force_organiser=force_organiser,
                shot_positions=tuple(ex_cfg.shot_positions) if ex_cfg else None,
                dedup_mad=ex_cfg.dedup_mad_threshold if ex_cfg else None,
            )
            # Round-8/9: fresh extractions, RE-extractions (incl. the
            # no-overwrite mismatch repair — round-9) and guard REFUSALS must
            # all be told apart; callers key the catalog rebuild off the
            # return. A rewrite is detected by the map csv CONTENT changing
            # (hash, not mtime — NTFS timestamp updates can lag ~15ms): the
            # skip path returns n>0 without touching it, and a re-extraction
            # producing byte-identical rows changes nothing downstream.
            wrote_map = (map_p.is_file()
                         and _map_csv_sha256(map_p) != digest_before)
            if n > 0 and (not before or wrote_map):
                count += 1
                if before:
                    reextracted += 1
            elif n == 0 and had_any:
                refused += 1
        except Exception as e:  # noqa: BLE001 — a broken file must not stop the batch
            log.error("Extraction failed for %s: %s", vid, e)
    if reextracted or refused:
        log.warning(
            "extract_missing summary: %d extracted (%d of those RE-extracted), "
            "%d refused/unchanged by the organiser-protection guard%s",
            count, reextracted, refused,
            " — see the per-video REFUSING lines above" if refused else "",
        )
    return count
