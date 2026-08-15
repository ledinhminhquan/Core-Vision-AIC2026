"""Keyframe catalog — the single source of truth for corpus identity.

The whole system hangs off one invariant:

    ``global_id`` == row in ``manifest.parquet`` == row in every FAISS index.

``global_id`` is assigned sequentially over keyframes sorted by
``(video_id, n)``, so rebuilding the catalog over the same corpus always
produces the same ids. Every index stores the catalog *signature* (a hash of
the (video_id, keyframe-count) set) and refuses to serve queries when stale —
a wrong ``frame_idx`` scores zero in DRES, so we fail loud rather than drift.
"""

from __future__ import annotations

import csv
import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from cvp.config import Settings
from cvp.constants import KEYFRAME_NAME_RE, VIDEO_ID_RE
from cvp.utils.io import atomic_write_json, read_json

log = logging.getLogger(__name__)

MANIFEST_COLUMNS = ["global_id", "video_id", "n", "frame_idx", "pts_time", "fps", "path", "has_map"]


@dataclass(frozen=True)
class KeyframeRef:
    """One keyframe row, resolved for display / submission."""

    global_id: int
    video_id: str
    n: int              # 1-indexed keyframe ordinal
    frame_idx: int      # frame number in the ORIGINAL video (what DRES wants)
    pts_time: float     # seconds into the video
    fps: float
    path: str           # absolute path to the .jpg
    has_map: bool       # False → frame_idx is a fallback estimate


def _read_map_csv(path: Path) -> dict[int, tuple[float, float, int]]:
    """map-keyframes/{vid}.csv → {n: (pts_time, fps, frame_idx)}."""
    out: dict[int, tuple[float, float, int]] = {}
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            try:
                n = int(float(row["n"]))
                out[n] = (float(row["pts_time"]), float(row["fps"]), int(float(row["frame_idx"])))
            except (KeyError, TypeError, ValueError):
                continue
    return out


class KeyframeCatalog:
    """Builds and serves the keyframe manifest."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.manifest_path = settings.paths.art("catalog", "manifest.parquet")
        self.signature_path = settings.paths.art("catalog", "signature.json")
        self._df: pd.DataFrame | None = None
        self._video_index: dict[str, tuple[int, int]] | None = None  # vid -> (start_gid, count)

    # ── build ────────────────────────────────────────────────────────────

    def build(self, force: bool = False) -> pd.DataFrame:
        """Scan keyframes/ + map-keyframes/ and (re)write the manifest."""
        if self.manifest_path.exists() and not force:
            df = self.load()
            if self._current_signature() == self._stored_signature():
                log.info("Catalog up-to-date: %d keyframes / %d videos", len(df), df["video_id"].nunique())
                return df
            log.info("Corpus changed on disk — rebuilding catalog.")

        kf_root = self.settings.paths.data(self.settings.paths.keyframes_dir)
        map_root = self.settings.paths.data(self.settings.paths.map_keyframes_dir)
        if not kf_root.is_dir():
            raise FileNotFoundError(
                f"Keyframes folder not found: {kf_root}. Set CVP_PATHS__DATA_ROOT to your dataset root."
            )

        rows: list[dict] = []
        video_dirs = sorted(d for d in kf_root.iterdir() if d.is_dir())
        for vdir in video_dirs:
            vid = vdir.name
            if not VIDEO_ID_RE.match(vid):
                log.warning("Skipping non-video folder: %s", vid)
                continue
            frames: list[tuple[int, Path]] = []
            for f in vdir.iterdir():
                m = KEYFRAME_NAME_RE.match(f.name)
                if m:
                    frames.append((int(m.group(1)), f))
            frames.sort(key=lambda t: t[0])
            if not frames:
                log.warning("Video %s has no keyframes — skipped", vid)
                continue

            mp = map_root / f"{vid}.csv"
            mapping = _read_map_csv(mp) if mp.is_file() else {}
            if not mapping:
                log.warning("No map-keyframes for %s — frame_idx falls back to estimate", vid)

            for n, fpath in frames:
                if n in mapping:
                    pts, fps, fidx = mapping[n]
                    has_map = True
                else:
                    # Fallback: assume ~1 keyframe per shot at 25fps is unknowable;
                    # estimate frame_idx from neighbours if partially mapped, else n-1.
                    pts, fps, fidx = float(n - 1), 25.0, n - 1
                    has_map = False
                # Store the path RELATIVE to data_root: the manifest stays valid
                # when artifacts move between Colab and the laptop.
                try:
                    stored = str(
                        fpath.resolve().relative_to(self.settings.paths.data_root.resolve())
                    ).replace("\\", "/")
                except ValueError:  # keyframes outside data_root — keep absolute
                    stored = str(fpath.resolve())
                rows.append(
                    dict(video_id=vid, n=n, frame_idx=fidx, pts_time=pts, fps=fps,
                         path=stored, has_map=has_map)
                )

        if not rows:
            raise RuntimeError(f"No keyframes found under {kf_root}")

        df = pd.DataFrame(rows).sort_values(["video_id", "n"], kind="mergesort").reset_index(drop=True)
        df.insert(0, "global_id", range(len(df)))
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.manifest_path.with_suffix(".parquet.tmp")
        df.to_parquet(tmp, index=False)
        tmp.replace(self.manifest_path)
        atomic_write_json(self.signature_path, {"signature": self._signature_of(df)})
        self._df = df
        self._video_index = None
        log.info("Catalog built: %d keyframes / %d videos (%d with map-keyframes)",
                 len(df), df["video_id"].nunique(), int(df["has_map"].sum()))
        return df

    # ── load / access ────────────────────────────────────────────────────

    def load(self) -> pd.DataFrame:
        if self._df is None:
            if not self.manifest_path.exists():
                raise FileNotFoundError(
                    f"Catalog missing: {self.manifest_path}. Run scripts/00_build_catalog.py first."
                )
            self._df = pd.read_parquet(self.manifest_path)
        return self._df

    def __len__(self) -> int:
        return len(self.load())

    def videos(self) -> list[str]:
        return self.load()["video_id"].unique().tolist()

    def video_rows(self, video_id: str) -> pd.DataFrame:
        df = self.load()
        return df[df["video_id"] == video_id]

    def resolve_path(self, stored: str) -> str:
        """Manifest paths are data_root-relative (absolute tolerated for
        manifests built by older versions)."""
        p = Path(stored)
        if p.is_absolute():
            return str(p)
        return str(self.settings.paths.data_root / p)

    def ref(self, global_id: int) -> KeyframeRef:
        row = self.load().iloc[int(global_id)]
        return KeyframeRef(
            global_id=int(row["global_id"]), video_id=str(row["video_id"]), n=int(row["n"]),
            frame_idx=int(row["frame_idx"]), pts_time=float(row["pts_time"]), fps=float(row["fps"]),
            path=self.resolve_path(str(row["path"])), has_map=bool(row["has_map"]),
        )

    def refs(self, global_ids: list[int]) -> list[KeyframeRef]:
        # One positional gather + itertuples: this runs with up to topk ids on
        # EVERY query, and per-id ``.iloc`` singles measured ~50ms/500 ids at
        # the 177k-row Batch-1 shape (round-10 scale rehearsal).
        if not global_ids:
            return []
        rows = self.load().iloc[[int(g) for g in global_ids]]
        return [
            KeyframeRef(
                global_id=int(r.global_id), video_id=str(r.video_id), n=int(r.n),
                frame_idx=int(r.frame_idx), pts_time=float(r.pts_time), fps=float(r.fps),
                path=self.resolve_path(str(r.path)), has_map=bool(r.has_map),
            )
            for r in rows.itertuples(index=False)
        ]

    def video_ids(self, global_ids: list[int]) -> list[str]:
        """Just the video id per global id — a single column gather, for hot
        paths that need row→video routing without full KeyframeRefs."""
        if not global_ids:
            return []
        col = self.load()["video_id"]
        return [str(v) for v in col.iloc[[int(g) for g in global_ids]]]

    def video_span(self, video_id: str) -> tuple[int, int]:
        """(first_global_id, count) for a video — contiguous by construction."""
        if self._video_index is None:
            df = self.load()
            grp = df.groupby("video_id", sort=False)["global_id"]
            self._video_index = {v: (int(g.min()), int(g.count())) for v, g in grp}
        return self._video_index[video_id]

    # ── signature (staleness detection) ──────────────────────────────────

    @staticmethod
    def _signature_of(df: pd.DataFrame) -> str:
        counts = df.groupby("video_id", sort=True)["n"].count()
        payload = ";".join(f"{v}:{c}" for v, c in counts.items())
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def signature(self) -> str:
        return self._signature_of(self.load())

    def _stored_signature(self) -> str | None:
        return (read_json(self.signature_path, default={}) or {}).get("signature")

    def _current_signature(self) -> str | None:
        """Signature of what is on disk right now (cheap scan of counts)."""
        kf_root = self.settings.paths.data(self.settings.paths.keyframes_dir)
        if not kf_root.is_dir():
            return None
        counts: dict[str, int] = {}
        for vdir in kf_root.iterdir():
            if vdir.is_dir() and VIDEO_ID_RE.match(vdir.name):
                c = sum(1 for f in vdir.iterdir() if KEYFRAME_NAME_RE.match(f.name))
                if c:
                    counts[vdir.name] = c
        payload = ";".join(f"{v}:{c}" for v, c in sorted(counts.items()))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
