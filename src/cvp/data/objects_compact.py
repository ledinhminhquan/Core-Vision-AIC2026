"""Compact object index: ~178k per-keyframe OpenImages JSONs → one parquet.

The organiser ships one detection json per keyframe (up to 100 boxes each).
Opening thousands of tiny files at query time is slow — catastrophically so on
Google Drive / network filesystems — so this module folds them into a single
``artifacts/objects_index/objects.parquet`` with one row per ``(video_id, n)``,
keeping only detections with ``score >= 0.3`` (``ObjectBooster`` filters at
exactly that floor at query time, so nothing usable is lost). The build streams
one video at a time through ``pyarrow.parquet.ParquetWriter`` and never holds
more than a few MB in RAM; it writes to a ``.tmp`` sibling then ``os.replace``s
(crash-safe, same discipline as ``utils.io``).

``CompactObjects`` is the read side: a drop-in replacement for
``data.metadata.ObjectStore`` (same ``get(video_id, n) -> list[Detection]``
surface) that loads the whole index with a single file read. ``ObjectBooster``
auto-prefers it whenever the parquet exists.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from cvp.config import Settings
from cvp.data.catalog import KeyframeCatalog
from cvp.data.metadata import Detection, ObjectStore

log = logging.getLogger(__name__)

PARQUET_NAME = "objects.parquet"
# Below this OpenImages boxes are noise; ObjectBooster.score filters at 0.30
# anyway, so the build floor is lossless for retrieval.
MIN_SCORE = 0.3


def _schema():
    import pyarrow as pa

    return pa.schema([
        ("video_id", pa.string()),
        ("n", pa.int32()),
        ("entities", pa.list_(pa.string())),
        ("scores", pa.list_(pa.float32())),
        ("boxes", pa.list_(pa.list_(pa.float32()))),  # ymin, xmin, ymax, xmax
    ])


def index_path(settings: Settings) -> Path:
    return settings.paths.art("objects_index") / PARQUET_NAME


def build_objects_index(settings: Settings,
                        catalog: KeyframeCatalog | None = None,
                        overwrite: bool = False,
                        min_score: float = MIN_SCORE) -> Path:
    """Fold every per-keyframe objects json into one streamed parquet file.

    One row per catalog ``(video_id, n)`` — keyframes with no json (or no
    detection above ``min_score``) get empty lists so lookups stay total.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    if catalog is None:
        catalog = KeyframeCatalog(settings)
        catalog.load()
    out = index_path(settings)
    if out.exists() and not overwrite:
        log.info("Objects index already built at %s (overwrite=True to rebuild)", out)
        return out

    obj_root = settings.paths.data(settings.paths.objects_dir)
    if not obj_root.exists():
        raise FileNotFoundError(
            f"objects dir not found: {obj_root}. Download the organiser's objects "
            f"package (or skip this stage — the per-file ObjectStore still works)."
        )

    store = ObjectStore(settings)          # reuse the battle-tested JSON parsing
    out.parent.mkdir(parents=True, exist_ok=True)
    schema = _schema()
    tmp = out.parent / (out.name + ".tmp")
    n_rows = n_dets = 0
    try:
        writer = pq.ParquetWriter(tmp, schema, compression="zstd")
        try:
            for vid in catalog.videos():
                cols: dict[str, list] = {
                    "video_id": [], "n": [], "entities": [], "scores": [], "boxes": [],
                }
                for n in catalog.video_rows(vid)["n"].tolist():
                    dets = [d for d in store.get(vid, int(n)) if d.score >= min_score]
                    cols["video_id"].append(vid)
                    cols["n"].append(int(n))
                    cols["entities"].append([d.entity for d in dets])
                    cols["scores"].append([float(d.score) for d in dets])
                    cols["boxes"].append([[float(x) for x in d.box] for d in dets])
                    n_dets += len(dets)
                writer.write_table(pa.Table.from_pydict(cols, schema=schema))
                n_rows += len(cols["n"])
                # Per-video JSONs are never re-read — clear the class-level
                # lru_cache so a 178k-keyframe build cannot balloon RAM.
                ObjectStore.get.cache_clear()
        finally:
            writer.close()
        os.replace(tmp, out)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    log.info("Objects index: %d keyframes, %d detections (score>=%.2f) -> %s",
             n_rows, n_dets, min_score, out)
    return out


class CompactObjects:
    """Query-time reader over ``objects.parquet`` — ObjectStore-compatible.

    Loads the whole table once (arrow buffers, tens of MB) on first ``get``
    and answers from an in-memory ``(video_id, n) → row`` map; no per-lookup
    file IO at all.
    """

    def __init__(self, settings: Settings) -> None:
        self.path = index_path(settings)
        self._table = None                     # pyarrow.Table, loaded lazily once
        self._row_of: dict[tuple[str, int], int] = {}
        self._videos: set[str] = set()

    def available(self) -> bool:
        """True when the parquet exists (i.e. build_objects_index has run)."""
        return self.path.exists()

    def _load(self):
        if self._table is None and self.available():
            import pyarrow.parquet as pq

            self._table = pq.read_table(self.path)
            vids = self._table.column("video_id").to_pylist()
            ns = self._table.column("n").to_pylist()
            self._row_of = {(v, int(n)): i for i, (v, n) in enumerate(zip(vids, ns))}
            self._videos = set(vids)
            log.info("Compact objects index loaded: %d keyframes from %s",
                     len(self._row_of), self.path)
        return self._table

    def get(self, video_id: str, n: int) -> list[Detection]:
        tbl = self._load()
        if tbl is None:
            return []
        row = self._row_of.get((str(video_id), int(n)))
        if row is None:
            return []
        ents = tbl.column("entities")[row].as_py() or []
        scores = tbl.column("scores")[row].as_py() or []
        boxes = tbl.column("boxes")[row].as_py() or []
        out: list[Detection] = []
        for e, s, b in zip(ents, scores, boxes):
            if b is None or len(b) != 4:
                continue
            out.append(Detection(entity=str(e), score=float(s),
                                 box=(float(b[0]), float(b[1]), float(b[2]), float(b[3]))))
        return out

    def has_video(self, video_id: str) -> bool:
        self._load()
        return video_id in self._videos
