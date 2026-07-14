"""Video frame reading. Prefers decord (fast), falls back to OpenCV.

(Ported from Core-Vision_HCMC-AI ``data/frame_extraction.py`` — self-contained.)

Used by the self-extraction pipeline (shot detection + keyframe selection), the
map-keyframes reconstruction script and the UI to grab the exact original frame
for a retrieved keyframe.

WHY the rework: the original helpers opened a fresh reader per call, so reading
K frames of one video cost K container opens + seeks, and whole-video reads
materialised every frame in RAM. ``VideoFrames`` holds ONE persistent reader per
video and streams bounded batches, so each frame is decoded at most once and
peak memory stays at (batch x frame) size. The old per-call helpers remain as
thin wrappers for one-shot use (e.g. the UI fetching a single frame).
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

# Frames per decord get_batch chunk when streaming — bounds decode memory.
_STREAM_BATCH = 256
# Max frames the cv2 fallback grabs forward instead of an explicit seek.
# (POS_FRAMES seeks land on the nearest GOP keyframe for some codecs, so short
# forward grabs are both faster AND more exact than a seek.)
_CV2_SEEK_AHEAD = 64


def _backend() -> str:
    try:
        import decord  # noqa: F401
        return "decord"
    except Exception:
        try:
            import cv2  # noqa: F401
            return "cv2"
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "Install `decord` (recommended) or `opencv-python` to read videos."
            ) from e


def _resize_rgb(arr: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize an RGB uint8 array to (W, H). cv2 when present (fast), else PIL."""
    try:
        import cv2
        return cv2.resize(arr, size)
    except Exception:  # pragma: no cover - PIL is a hard dependency
        from PIL import Image
        return np.asarray(Image.fromarray(arr).resize(size), dtype=np.uint8)


class VideoFrames:
    """Context manager around ONE persistent video reader.

    Frames come back as RGB uint8 (H, W, 3). Pass ``size=(w, h)`` to decode at a
    reduced resolution: decord accepts width/height and resizes inside the
    decoder (cheap); the cv2 fallback resizes each decoded frame.

    Usage::

        with VideoFrames(path, size=(48, 27)) as vf:
            for frame_idx, frame in vf.iter_strided(stride=5):
                ...
    """

    def __init__(self, path: str | Path, size: tuple[int, int] | None = None) -> None:
        self.path = Path(path)
        self.size = size
        self.backend = _backend()
        self._vr = None   # decord.VideoReader
        self._cap = None  # cv2.VideoCapture
        self._cv2_pos = 0  # next frame index cv2 will decode (avoids re-seeks)
        if self.backend == "decord":
            import decord
            decord.bridge.set_bridge("native")
            if size is not None:
                try:
                    self._vr = decord.VideoReader(str(self.path), width=size[0], height=size[1])
                except Exception:
                    # Some decord builds lack decoder-side resize; fall back to
                    # full-size decode + per-frame resize (see _finalize).
                    self._vr = decord.VideoReader(str(self.path))
            else:
                self._vr = decord.VideoReader(str(self.path))
        else:
            import cv2
            self._cap = cv2.VideoCapture(str(self.path))
            if not self._cap.isOpened():
                raise RuntimeError(f"Could not open video: {self.path}")

    # ------------------------------------------------------------ lifecycle
    def __enter__(self) -> VideoFrames:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._vr = None

    # ------------------------------------------------------------ metadata
    def __len__(self) -> int:
        if self.backend == "decord":
            return len(self._vr)
        import cv2
        return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def fps(self) -> float:
        if self.backend == "decord":
            return float(self._vr.get_avg_fps()) or 25.0
        import cv2
        return float(self._cap.get(cv2.CAP_PROP_FPS)) or 25.0

    # ------------------------------------------------------------- reading
    def _finalize(self, frame: np.ndarray, size: tuple[int, int] | None) -> np.ndarray:
        """Apply the requested output size (no-op when the decoder already did)."""
        size = size or self.size
        if size is not None and (frame.shape[1], frame.shape[0]) != size:
            frame = _resize_rgb(frame, size)
        return frame

    def _clamp(self, frame_idx: int) -> int:
        n = len(self)
        return max(0, min(int(frame_idx), n - 1)) if n > 0 else int(frame_idx)

    def _cv2_read_at(self, frame_idx: int) -> np.ndarray:
        """Position-tracked read: grab forward when close, seek only when far."""
        import cv2
        if frame_idx < self._cv2_pos or frame_idx - self._cv2_pos > _CV2_SEEK_AHEAD:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            self._cv2_pos = frame_idx
        while self._cv2_pos < frame_idx:
            if not self._cap.grab():  # decode-free skip
                break
            self._cv2_pos += 1
        ok, frame = self._cap.read()
        if not ok:
            raise RuntimeError(f"Could not read frame {frame_idx} from {self.path}")
        self._cv2_pos = frame_idx + 1
        if self.size is not None:
            frame = cv2.resize(frame, self.size)
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def get(self, frame_idx: int) -> np.ndarray:
        """Return a single RGB frame (H, W, 3) uint8 (index clamped to bounds)."""
        frame_idx = self._clamp(frame_idx)
        if self.backend == "decord":
            return self._finalize(self._vr[frame_idx].asnumpy(), None)
        return self._cv2_read_at(frame_idx)

    def get_batch(self, indices: Sequence[int]) -> np.ndarray:
        """Return frames at ``indices`` as (len(indices), H, W, 3) uint8."""
        idx = [self._clamp(i) for i in indices]
        if not idx:
            return np.zeros((0, 0, 0, 3), dtype=np.uint8)
        if self.backend == "decord":
            batch = self._vr.get_batch(idx).asnumpy().astype(np.uint8, copy=False)
            if self.size is not None and (batch.shape[2], batch.shape[1]) != self.size:
                batch = np.stack([_resize_rgb(f, self.size) for f in batch])
            return batch
        return np.stack([self._cv2_read_at(i) for i in idx])

    def iter_strided(
        self, stride: int = 1, size: tuple[int, int] | None = None
    ) -> Iterator[tuple[int, np.ndarray]]:
        """Yield ``(frame_idx, RGB frame)`` every ``stride`` frames.

        Streams in bounded chunks — never materialises the whole video. ``size``
        overrides the reader-level output size for this iteration only.
        """
        stride = max(1, int(stride))
        if self.backend == "decord":
            n = len(self)
            for start in range(0, n, _STREAM_BATCH * stride):
                idx = list(range(start, min(start + _STREAM_BATCH * stride, n), stride))
                batch = self._vr.get_batch(idx).asnumpy()
                for i, frame in zip(idx, batch):
                    yield i, self._finalize(frame.astype(np.uint8, copy=False), size)
            return
        # cv2: one sequential pass; skipped frames are grab()bed, never decoded.
        import cv2
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self._cv2_pos = 0
        out_size = size or self.size
        i = 0
        while True:
            if i % stride == 0:
                ok, frame = self._cap.read()
                if not ok:
                    break
                if out_size is not None:
                    frame = cv2.resize(frame, out_size)
                yield i, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            elif not self._cap.grab():
                break
            i += 1
            self._cv2_pos = i


# ----------------------------------------------------------------------------
# One-shot convenience wrappers (kept for the UI and existing callers). For any
# multi-frame access open a VideoFrames once instead.
# ----------------------------------------------------------------------------
def video_fps(path: str | Path) -> float:
    with VideoFrames(path) as vf:
        return vf.fps


def num_frames(path: str | Path) -> int:
    with VideoFrames(path) as vf:
        return len(vf)


def read_frame(path: str | Path, frame_idx: int) -> np.ndarray:
    """Return a single RGB frame (H, W, 3) uint8."""
    with VideoFrames(path) as vf:
        return vf.get(frame_idx)


def read_all_resized(path: str | Path, size: tuple[int, int] = (48, 27)) -> np.ndarray:
    """Read the whole video resized to (W, H) for TransNetV2 -> (N, H, W, 3) uint8.

    Streams via ``VideoFrames.iter_strided`` so peak memory is the RESIZED
    output array plus one small decode chunk — never the full-resolution video.
    """
    with VideoFrames(path, size=size) as vf:
        frames = [f for _, f in vf.iter_strided(1)]
    if not frames:
        return np.zeros((0, size[1], size[0], 3), dtype=np.uint8)
    return np.stack(frames).astype(np.uint8, copy=False)
