"""Offline OCR over keyframes → artifacts/ocr/{video_id}.json.

TV-news frames carry dense burned-in text (headlines, tickers, location
chyrons) — often the single most discriminative KIS signal. Output contract::

    {"n_to_text": {"1": "BẢN TIN 60 GIÂY ...", "5": "..."}}

Resumable per video. Engines: easyocr (default — good vi accuracy / zero
config), paddle (PP-OCRv4+), none.
"""

from __future__ import annotations

import logging


from cvp.config import Settings
from cvp.data.catalog import KeyframeCatalog
from cvp.utils.io import atomic_write_json, read_json

log = logging.getLogger(__name__)


class _EasyOcrEngine:
    def __init__(self, settings: Settings):
        import easyocr

        self.reader = easyocr.Reader(settings.ocr.langs, gpu=self._gpu())
        self.min_conf = settings.ocr.min_confidence

    @staticmethod
    def _gpu() -> bool:
        try:
            import torch

            return torch.cuda.is_available()
        except ImportError:
            return False

    def read(self, image_path: str) -> str:
        results = self.reader.readtext(image_path, detail=1, paragraph=False)
        texts = [t for (_, t, conf) in results if conf >= self.min_conf]
        return " ".join(texts).strip()


class _PaddleEngine:
    """Best-effort PaddleOCR wrapper (2.x and 3.x expose different APIs)."""

    def __init__(self, settings: Settings):
        from paddleocr import PaddleOCR

        # Round-67 (live 07a): paddleocr 3.x rejects unknown kwargs with
        # ValueError ("Unknown argument: show_log"), NOT TypeError — the old
        # 2.x-first chain never reached its 3.x branch and every init died.
        # Try the 3.x signature FIRST (the 07 notebooks pin >=3.0,<4) and
        # catch BOTH exception types at every step.
        try:  # 3.x — doc-preprocessing off (round-60): useless for TV
            # keyframes and it multiplies per-frame latency across ~177k frames
            self.ocr = PaddleOCR(
                lang="vi",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            )
        except (TypeError, ValueError):  # 2.x signature
            try:
                self.ocr = PaddleOCR(use_angle_cls=True, lang="vi", show_log=False)
            except (TypeError, ValueError):  # any other drift — bare defaults
                self.ocr = PaddleOCR(lang="vi")
        self.min_conf = settings.ocr.min_confidence

    def read(self, image_path: str) -> str:
        try:
            result = self.ocr.ocr(image_path) or []
        except TypeError:
            result = self.ocr.predict(image_path) or []
        texts: list[str] = []
        for page in result:
            # 3.x returns dict-like results with rec_texts/rec_scores
            if isinstance(page, dict) or hasattr(page, "get"):
                for text, conf in zip(page.get("rec_texts", []), page.get("rec_scores", [])):
                    if float(conf) >= self.min_conf:
                        texts.append(str(text))
                continue
            for line in page or []:  # 2.x nested-list format
                try:
                    text, conf = line[1][0], float(line[1][1])
                except (IndexError, TypeError, ValueError):
                    continue
                if conf >= self.min_conf:
                    texts.append(text)
        return " ".join(texts).strip()


def _build_engine(settings: Settings):
    engine = settings.ocr.engine
    if engine == "easyocr":
        return _EasyOcrEngine(settings)
    if engine == "paddle":
        return _PaddleEngine(settings)
    raise ValueError(f"Unsupported OCR engine: {engine!r}")


def ocr_all_keyframes(settings: Settings, catalog: KeyframeCatalog,
                      videos: list[str] | None = None, overwrite: bool = False) -> int:
    """Run OCR for all (or selected) videos; skip finished ones."""
    if settings.ocr.engine == "none":
        log.info("OCR disabled by config")
        return 0
    out_dir = settings.paths.art("ocr")
    df = catalog.load()
    todo_videos = videos or [str(v) for v in df["video_id"].unique()]
    engine = None
    done = 0
    consecutive_failures = 0
    for vid in todo_videos:
        out_path = out_dir / f"{vid}.json"
        grp = df[df["video_id"] == vid].sort_values("n")
        if out_path.exists() and not overwrite:
            existing = read_json(out_path, default={}) or {}
            # done == every frame was PROCESSED (zero text found is a valid outcome)
            done_count = existing.get("processed_count", len(existing.get("n_to_text", {})) or -1)
            if done_count >= len(grp):
                continue
        if engine is None:
            engine = _build_engine(settings)
        n_to_text: dict[str, str] = {}
        processed = 0
        last_err: Exception | None = None
        for _, row in grp.iterrows():
            try:
                text = engine.read(catalog.resolve_path(str(row["path"])))
            except Exception as e:  # noqa: BLE001 — one bad frame must not kill the run
                last_err = e
                log.warning("OCR failed on %s n=%s: %s", vid, row["n"], e)
                continue
            processed += 1
            if text:
                n_to_text[str(int(row["n"]))] = text
        if vid == todo_videos[0] and len(grp) > 0 and processed == 0 and last_err is not None:
            # Round-13: EVERY frame of the very first video failing is a
            # SYSTEMIC breakage (API drift, broken weights) — fail loud in a
            # minute instead of spending hours writing empty artifacts that
            # report as success.
            # Round-61 (audit): anchor on the FIRST video of the todo LIST, not
            # "first processed this run" — otherwise one video with locally
            # corrupt keyframes (skipped by the round-60 guard, so retried
            # every run after everything else resume-skips) masquerades as a
            # systemic failure forever and wedges the whole shard.
            raise RuntimeError(
                f"OCR failed on every frame of the first video ({vid}) — "
                "systemic failure, aborting the sweep"
            ) from last_err
        if len(grp) > 0 and processed == 0 and last_err is not None:
            # Round-60 (audit): a mid-run sticky engine death (CUDA error/OOM)
            # fails EVERY later video in minutes. Writing those as empty jsons
            # would poison the shared store — the shard gates count by file
            # NAME, so finalize would swap in a store with no text for that
            # slice. Skip the write (resume retries the video next run) and
            # abort early after 3 consecutive all-fail videos.
            consecutive_failures += 1
            log.warning("OCR %s: engine failed on every frame — not writing, "
                        "will retry next run", vid)
            if consecutive_failures >= 3:
                raise RuntimeError(
                    "OCR engine failed on every frame of 3 consecutive videos — "
                    "aborting early to keep the store clean (rerun resumes)"
                ) from last_err
            continue
        consecutive_failures = 0
        atomic_write_json(out_path, {"n_to_text": n_to_text, "processed_count": processed})
        done += 1
        log.info("OCR %s: %d/%d frames had text (%d/%d videos)",
                 vid, len(n_to_text), len(grp), done, len(todo_videos))
    return done
