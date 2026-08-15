"""DRES submission CSV writers — the exact organiser contract.

All three tasks share the rules: **no header**, **≤100 rows**, ranked best
first, de-duplicated. ``frame_idx`` is the frame number in the ORIGINAL video
(from map-keyframes), never the keyframe ordinal ``n``.

    KIS   : video_id,frame_idx
    QA    : video_id,frame_idx,"answer"
    TRAKE : video_id,frame_e1,frame_e2,...,frame_ek
"""

from __future__ import annotations

import logging
import os
import re
import unicodedata
from pathlib import Path
from typing import Iterable, Sequence

from cvp.constants import MAX_QA_ANSWER_CHARS, MAX_SUBMISSION_ROWS, VIDEO_ID_RE

log = logging.getLogger(__name__)

_ANSWER_BAD_CHARS = re.compile(r"[\r\n\t]")


def _validate_video_id(video_id: str) -> str:
    if not VIDEO_ID_RE.match(video_id):
        raise ValueError(f"Bad video_id for submission: {video_id!r}")
    return video_id


def _validate_frame_idx(frame_idx: int) -> int:
    fi = int(frame_idx)
    if fi < 0:
        raise ValueError(f"Negative frame_idx: {frame_idx}")
    return fi


def _finalize(path: str | os.PathLike, lines: list[str]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8", newline="\n")
    os.replace(tmp, path)
    return path


def sanitize_answer(answer: str) -> str:
    """QA answers ride in a CSV field: NFC-normalize, strip newlines/tabs, quote if needed.

    Unicode NFC first — VQA output may arrive decomposed (e.g. "a" + combining
    accent) and the organisers grade the composed Vietnamese text. Answers
    longer than the organiser cap (:data:`cvp.constants.MAX_QA_ANSWER_CHARS`)
    are truncated with a warning — a hand-typed over-long answer exported from
    the UI must never produce a CSV the submission server would reject.
    """
    a = unicodedata.normalize("NFC", str(answer))
    a = _ANSWER_BAD_CHARS.sub(" ", a).strip()
    a = re.sub(r"\s+", " ", a)
    if len(a) > MAX_QA_ANSWER_CHARS:
        log.warning(
            "QA answer truncated to %d chars (was %d): %r",
            MAX_QA_ANSWER_CHARS, len(a), a[:120],
        )
        a = a[:MAX_QA_ANSWER_CHARS].rstrip()
    if "," in a or '"' in a:
        a = '"' + a.replace('"', '""') + '"'
    return a


def _all_invalid(task: str, skipped: int, path: str | os.PathLike) -> ValueError:
    return ValueError(
        f"All {skipped} {task} candidate rows were invalid — refusing to write {path}"
    )


def write_kis(path: str | os.PathLike, ranked: Iterable[tuple[str, int]]) -> Path:
    """ranked: iterable of (video_id, frame_idx), best first.

    Invalid candidates (bad video id, negative frame) are SKIPPED with a
    warning — one bad row must not abort the whole 100-row export
    mid-competition. An entirely-invalid non-empty input still raises.
    """
    seen: set[tuple[str, int]] = set()
    lines: list[str] = []
    skipped = 0
    for video_id, frame_idx in ranked:
        try:
            key = (_validate_video_id(video_id), _validate_frame_idx(frame_idx))
        except ValueError as e:
            log.warning("KIS row skipped: %s", e)
            skipped += 1
            continue
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"{key[0]},{key[1]}")
        if len(lines) >= MAX_SUBMISSION_ROWS:
            break
    if skipped and not lines:
        raise _all_invalid("KIS", skipped, path)
    return _finalize(path, lines)


def write_qa(path: str | os.PathLike, ranked: Iterable[tuple[str, int, str]]) -> Path:
    """ranked: iterable of (video_id, frame_idx, answer), best first.

    Invalid candidates (bad video id, negative frame) are SKIPPED with a
    warning — one bad row must not abort the whole 100-row export
    mid-competition. An entirely-invalid non-empty input still raises.
    """
    # Dedup on the FULL emitted row (video, frame, sanitized answer): under the
    # official max-over-rows formula, the same frame with DIFFERENT candidate
    # answers is a legitimate score-raising strategy — only true duplicates
    # (same locus AND same answer after sanitization) may collapse.
    seen: set[tuple[str, int, str]] = set()
    lines: list[str] = []
    skipped = 0
    for video_id, frame_idx, answer in ranked:
        try:
            vid, fi = _validate_video_id(video_id), _validate_frame_idx(frame_idx)
        except ValueError as e:
            log.warning("QA row skipped: %s", e)
            skipped += 1
            continue
        ans = sanitize_answer(answer)
        key = (vid, fi, ans.casefold())
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"{vid},{fi},{ans}")
        if len(lines) >= MAX_SUBMISSION_ROWS:
            break
    if skipped and not lines:
        raise _all_invalid("QA", skipped, path)
    return _finalize(path, lines)


def write_trake(path: str | os.PathLike, ranked: Iterable[tuple[str, Sequence[int]]]) -> Path:
    """ranked: iterable of (video_id, [frame_e1..frame_ek]) sequences, best first.

    Frames within a sequence must be strictly increasing (events in order);
    invalid candidates (bad video id, bad frames, empty or non-increasing
    sequences) are SKIPPED with a warning — one bad candidate must not abort
    the whole 100-row export mid-competition. An entirely-invalid non-empty
    input still raises.
    """
    seen: set[tuple] = set()
    lines: list[str] = []
    skipped = 0
    for video_id, frames in ranked:
        try:
            vid = _validate_video_id(video_id)
            fs = [_validate_frame_idx(f) for f in frames]
        except ValueError as e:
            log.warning("TRAKE row skipped: %s", e)
            skipped += 1
            continue
        if len(fs) < 1:
            log.warning("TRAKE row skipped (no frames) for %s", vid)
            skipped += 1
            continue
        if any(b <= a for a, b in zip(fs, fs[1:])):
            log.warning("TRAKE row skipped (frames not strictly increasing) for %s: %s", vid, fs)
            skipped += 1
            continue
        key = (vid, tuple(fs))
        if key in seen:
            continue
        seen.add(key)
        lines.append(",".join([vid, *[str(f) for f in fs]]))
        if len(lines) >= MAX_SUBMISSION_ROWS:
            break
    if skipped and not lines:
        raise _all_invalid("TRAKE", skipped, path)
    return _finalize(path, lines)
