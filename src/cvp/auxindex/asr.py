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

    def transcribe(self, media_path: str) -> list[dict]:
        out = self.pipe(media_path)
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
    for vid in todo:
        out_path = out_dir / f"{vid}.json"
        if out_path.exists() and not overwrite:
            continue
        src = _find_video_file(video_root, vid)
        if src is None:
            continue
        if backend is None:
            backend = _build_backend(settings)
        try:
            segments = backend.transcribe(str(src))
        except Exception as e:  # noqa: BLE001 — a corrupt file must not stop the batch
            log.warning("ASR failed for %s: %s", vid, e)
            continue
        atomic_write_json(out_path, {"segments": segments})
        done += 1
        log.info("ASR %s: %d segments (%d done)", vid, len(segments), done)
    return done


def _find_video_file(video_root: Path, video_id: str) -> Path | None:
    for ext in (".mp4", ".mkv", ".webm", ".avi"):
        p = video_root / f"{video_id}{ext}"
        if p.is_file():
            return p
    return None
