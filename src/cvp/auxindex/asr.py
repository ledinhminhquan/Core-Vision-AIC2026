"""Vietnamese ASR over raw videos → artifacts/asr/{video_id}.json.

News anchors narrate exactly what KIS queries describe, so speech is a strong
recall channel. Uses PhoWhisper (whisper fine-tuned on 844h of Vietnamese) via
either faster-whisper (CTranslate2, ~4× faster, used when installed) or the
plain transformers pipeline. Output keeps segment timestamps so TextSignals
can map speech to the keyframes it overlaps::

    {"segments": [{"start": 12.3, "end": 17.8, "text": "..."}]}

Resumable per video.
"""

from __future__ import annotations

import logging
from pathlib import Path

from cvp.config import Settings
from cvp.data.catalog import KeyframeCatalog
from cvp.utils.io import atomic_write_json

log = logging.getLogger(__name__)


class _FasterWhisperBackend:
    """Only used when the configured model id IS a CTranslate2 conversion
    (contains "ct2" / "faster") — PhoWhisper HF checkpoints are transformers
    format and go through the pipeline backend instead."""

    def __init__(self, settings: Settings):
        from faster_whisper import WhisperModel

        model_id = settings.asr.model
        device, compute = self._device()
        log.info("Loading faster-whisper %s (%s/%s)", model_id, device, compute)
        self.model = WhisperModel(model_id, device=device, compute_type=compute)

    @staticmethod
    def _device() -> tuple[str, str]:
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda", "float16"
        except ImportError:
            pass
        return "cpu", "int8"

    def transcribe(self, media_path: str) -> list[dict]:
        segments, _info = self.model.transcribe(media_path, language="vi", vad_filter=True)
        return [{"start": float(s.start), "end": float(s.end), "text": s.text.strip()} for s in segments]


def _extract_wav(media_path: str) -> str | None:
    """Decode the audio track to a local 16 kHz mono wav and return its path.

    The transformers ASR pipeline loads files by piping their BYTES through
    ffmpeg's stdin (``ffmpeg_read``) — an mp4 whose moov atom sits at the END
    of the file (the organiser encodes) cannot be decoded from a pipe, which
    made every Batch-1 video fail with "Soundfile is malformed" (live run 6).
    Decoding from the seekable path ourselves sidesteps that entirely.

    Returns None when the video simply has NO audio stream (a silent video is
    a valid input → empty transcript, not a failure).
    """
    import subprocess
    import tempfile

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    try:
        # timeout: a Drive-FUSE stall must become a per-video failure that
        # feeds the systemic guard, not an invisible all-night hang (verify-R15)
        r = subprocess.run(
            ["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", media_path,
             "-vn", "-ac", "1", "-ar", "16000", "-f", "wav", tmp.name],
            capture_output=True, text=True, timeout=600)
    except FileNotFoundError as e:
        Path(tmp.name).unlink(missing_ok=True)
        raise RuntimeError("ffmpeg binary not found — ASR needs ffmpeg installed") from e
    except subprocess.TimeoutExpired as e:
        Path(tmp.name).unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg timed out after 600s on {media_path}") from e
    try:
        if r.returncode == 0 and Path(tmp.name).stat().st_size > 44:  # > wav header
            return tmp.name
    except OSError:
        pass
    Path(tmp.name).unlink(missing_ok=True)
    try:
        p = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a",
             "-show_entries", "stream=codec_type", "-of", "csv=p=0", media_path],
            capture_output=True, text=True, timeout=60)
    except FileNotFoundError as e:
        raise RuntimeError("ffprobe binary not found — ASR needs ffmpeg installed") from e
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"ffprobe timed out on {media_path}") from e
    if p.returncode == 0 and not p.stdout.strip():
        return None                      # no audio stream — silent video
    raise RuntimeError(f"ffmpeg audio extraction failed: {(r.stderr or '').strip()[:300]}")


class _TransformersBackend:
    def __init__(self, settings: Settings):
        import torch
        from transformers import pipeline

        device = 0 if torch.cuda.is_available() else -1
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        log.info("Loading ASR pipeline %s", settings.asr.model)
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=settings.asr.model,
            device=device,
            torch_dtype=dtype,
            chunk_length_s=settings.asr.chunk_length_s,
            batch_size=settings.asr.batch_size,
            return_timestamps=True,
        )

    def transcribe(self, media_path: str) -> list[dict] | None:
        """Segments; ``None`` means the video has NO audio track (silent)."""
        wav = _extract_wav(media_path)
        if wav is None:
            return None
        try:
            out = self.pipe(wav)
        finally:
            Path(wav).unlink(missing_ok=True)
        segments = []
        for chunk in out.get("chunks", []):
            ts = chunk.get("timestamp") or (None, None)
            if ts[0] is None:
                continue
            segments.append({
                "start": float(ts[0]),
                "end": float(ts[1] if ts[1] is not None else ts[0] + 5.0),
                "text": str(chunk.get("text", "")).strip(),
            })
        if not segments:
            # Whisper fine-tunes (PhoWhisper) can emit text WITHOUT usable
            # timestamps — salvage it as one whole-video segment so the speech
            # recall channel survives, instead of silently dropping everything
            # (verify-R15). 86400s covers any video length.
            text = str(out.get("text", "")).strip()
            if text:
                segments = [{"start": 0.0, "end": 86400.0, "text": text}]
        return segments


def _build_backend(settings: Settings):
    model_id = settings.asr.model.lower()
    if "ct2" in model_id or "faster" in model_id:
        try:
            return _FasterWhisperBackend(settings)
        except ImportError:
            log.info("faster-whisper not installed — using transformers pipeline")
        except Exception as e:  # noqa: BLE001 — bad/missing CT2 conversion
            log.warning("faster-whisper backend failed (%s) — using transformers pipeline", e)
    return _TransformersBackend(settings)


def asr_all_videos(settings: Settings, catalog: KeyframeCatalog,
                   videos: list[str] | None = None, overwrite: bool = False) -> int:
    """Transcribe raw videos that exist under data/videos; skip finished ones."""
    out_dir = settings.paths.art("asr")
    video_root = settings.paths.data(settings.paths.videos_dir)
    todo = videos or catalog.videos()
    backend = None
    done = 0
    attempted = 0
    files_found = 0
    consecutive_failures = 0
    consecutive_empty_audio = 0
    for vid in todo:
        out_path = out_dir / f"{vid}.json"
        if out_path.exists() and not overwrite:
            continue
        attempted += 1
        src = _find_video_file(video_root, vid)
        if src is None:
            continue
        files_found += 1
        if backend is None:
            backend = _build_backend(settings)
        try:
            segments = backend.transcribe(str(src))
        except Exception as e:  # noqa: BLE001 — a corrupt file must not stop the batch
            consecutive_failures += 1
            log.warning("ASR failed for %s: %s", vid, e)
            if consecutive_failures >= 5:
                # Round-13: 5 straight failures is a systemic breakage (model/
                # decode API drift), not 5 coincidentally corrupt videos.
                raise RuntimeError(
                    "ASR failed on 5 consecutive videos — systemic failure, "
                    "aborting the sweep"
                ) from e
            continue
        consecutive_failures = 0
        no_audio = segments is None
        if no_audio:
            segments = []
        elif not segments:
            # Audio track EXISTS but the decoder produced nothing. On a news
            # corpus that cannot be true 5 videos in a row — and writing the
            # empty artifact would lock the corruption in behind the resume
            # skip forever (verify-R15). Leave it unwritten (retried next run)
            # and abort loudly if it looks systemic.
            consecutive_empty_audio += 1
            log.warning("ASR %s: audio present but 0 segments — artifact NOT "
                        "written (sẽ thử lại lần chạy sau)", vid)
            if consecutive_empty_audio >= 5:
                raise RuntimeError(
                    "ASR produced 0 segments for 5 consecutive videos that DO "
                    "have an audio track — decoder/output drift (systemic "
                    "failure), aborting instead of writing empty transcripts")
            continue
        else:
            consecutive_empty_audio = 0
        payload: dict = {"segments": segments}
        if no_audio:
            payload["no_audio"] = True    # phân biệt video câm với decode hỏng
        atomic_write_json(out_path, payload)
        done += 1
        log.info("ASR %s: %d segments (%d done)", vid, len(segments), done)
    if attempted and files_found == 0:
        raise RuntimeError(
            f"ASR: KHÔNG tìm thấy file video nào cho {attempted} video cần xử lý "
            f"dưới {video_root} — thư mục videos trống/symlink chết? Kiểm tra "
            "Videos_*.zip đã bung trên Drive và mount còn sống, rồi chạy lại.")
    return done


def _find_video_file(video_root: Path, video_id: str) -> Path | None:
    # Accept both the flat layout and the organiser zips' `video/` wrapper dir
    # (every 2026 Batch-1 Videos zip nests its mp4s under video/).
    for base in (video_root, video_root / "video"):
        for ext in (".mp4", ".mkv", ".webm", ".avi"):
            p = base / f"{video_id}{ext}"
            if p.is_file():
                return p
    return None
