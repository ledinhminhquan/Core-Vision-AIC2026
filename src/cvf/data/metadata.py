"""Loaders for the organiser's sidecar metadata: media-info JSON + object detections."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from functools import lru_cache

from cvf.config import Settings

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Detection:
    entity: str      # OpenImages class name, e.g. "Person"
    score: float
    box: tuple[float, float, float, float]  # ymin, xmin, ymax, xmax in [0,1]


class MediaInfoStore:
    """media-info/{video_id}.json — YouTube title / description / keywords."""

    def __init__(self, settings: Settings):
        self.root = settings.paths.data(settings.paths.media_info_dir)

    @lru_cache(maxsize=4096)
    def get(self, video_id: str) -> dict:
        p = self.root / f"{video_id}.json"
        if not p.is_file():
            return {}
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            log.warning("Unreadable media-info for %s", video_id)
            return {}

    def text_blob(self, video_id: str) -> str:
        """Searchable text: title + description + keywords."""
        info = self.get(video_id)
        parts = [str(info.get("title", "")), str(info.get("description", ""))]
        kw = info.get("keywords") or []
        if isinstance(kw, list):
            parts.append(" ".join(str(k) for k in kw))
        return "\n".join(p for p in parts if p)

    def watch_url(self, video_id: str, at_seconds: float | None = None) -> str | None:
        url = self.get(video_id).get("watch_url")
        if url and at_seconds is not None:
            sep = "&" if "?" in url else "?"
            return f"{url}{sep}t={int(at_seconds)}"
        return url


class ObjectStore:
    """objects/{video_id}/{nnn}.json — OpenImages detections per keyframe."""

    def __init__(self, settings: Settings):
        self.root = settings.paths.data(settings.paths.objects_dir)

    # Cached per (video_id, n): a single query can touch ~500 frames and the
    # JSONs are immutable artifacts (often on slow Drive mounts) — no
    # invalidation needed.
    @lru_cache(maxsize=4096)
    def get(self, video_id: str, n: int) -> list[Detection]:
        # Organiser packs use 3-digit names; self-generated may use wider pads.
        vdir = self.root / video_id
        for width in (3, 4, 5):
            p = vdir / f"{n:0{width}d}.json"
            if p.is_file():
                break
        else:
            return []
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        entities = raw.get("detection_class_entities") or raw.get("detection_class_names") or []
        scores = raw.get("detection_scores") or []
        boxes = raw.get("detection_boxes") or []
        out: list[Detection] = []
        for i, ent in enumerate(entities):
            try:
                score = float(scores[i]) if i < len(scores) else 0.0
                box = tuple(float(x) for x in boxes[i]) if i < len(boxes) else (0.0, 0.0, 1.0, 1.0)
                out.append(Detection(entity=str(ent), score=score, box=box))  # type: ignore[arg-type]
            except (TypeError, ValueError):
                continue
        return out

    def has_video(self, video_id: str) -> bool:
        return (self.root / video_id).is_dir()
