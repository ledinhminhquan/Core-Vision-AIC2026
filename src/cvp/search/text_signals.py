"""BM25 text signals over OCR / ASR / captions / video metadata.

Candidate-restricted by design: FAISS proposes the top-K dense candidates and
BM25 only scores those — O(K) per query, never O(corpus). Documents are indexed
in a diacritic-folded view so accent-less typing and accent-mangled OCR still
match (see cvp.utils.text).

Two scoring paths, transparent to callers:

1. **Persisted** (preferred): ``scripts/03_build_aux_indexes.py`` tokenizes
   every field once and stores BM25 sufficient statistics under
   ``artifacts/text_index/`` (:mod:`cvp.index.text_store`). When the stored
   catalog signature matches the live catalog, candidates are scored per-id
   with closed-form BM25+ math — the corpus is never rescanned at query time.
2. **In-memory fallback**: the original rank-bm25 path that builds a BM25Plus
   corpus from artifact JSONs on first use. Kept so artifact sets produced by
   older notebooks (no ``text_index/``) keep working unchanged.

Artifact contract (written by the aux pipelines):

    artifacts/ocr/{video_id}.json       {"n_to_text": {"1": "..."}}
    artifacts/asr/{video_id}.json       {"segments": [{"start": s, "end": e, "text": ...}]}
    artifacts/captions/{video_id}.json  {"n_to_caption": {"1": "..."}}
"""

from __future__ import annotations

import logging
from pathlib import Path

from cvp.config import Settings
from cvp.data.catalog import KeyframeCatalog, KeyframeRef
from cvp.data.metadata import MediaInfoStore
from cvp.index.text_store import TextIndexField, TextIndexStore
from cvp.utils.io import read_json
from cvp.utils.text import tokenize_vi

log = logging.getLogger(__name__)

FRAME_FIELDS = ("ocr", "asr", "caption")
ALL_FIELDS = FRAME_FIELDS + ("metadata",)

# ASR segments cover keyframes whose pts_time falls within ±ASR_PAD_S of them.
ASR_PAD_S = 2.0


# ── document collection (single source for BOTH scoring paths) ───────────────


def collect_field_documents(
    settings: Settings, catalog: KeyframeCatalog, field: str
) -> tuple[list, list[str]] | None:
    """Raw ``(keys, texts)`` for one BM25 field, or None when artifacts are absent.

    The single source of document construction: the in-memory fallback and the
    persisted text_index build (scripts/03) both consume this, so the two paths
    always score identical corpora. Frame fields key on ``global_id`` (int);
    metadata keys on ``video_id`` (str, broadcast to frames at scoring time).
    Texts are RAW — consumers tokenize (and drop token-less docs) themselves.
    """
    if field == "ocr":
        return _collect_frame_docs(catalog, settings.paths.art("ocr"), "n_to_text")
    if field == "caption":
        return _collect_frame_docs(catalog, settings.paths.art("captions"), "n_to_caption")
    if field == "asr":
        return _collect_asr_docs(settings, catalog)
    if field == "metadata":
        return _collect_metadata_docs(settings, catalog)
    return None


def _collect_frame_docs(
    catalog: KeyframeCatalog, art_dir: Path, json_key: str
) -> tuple[list, list[str]] | None:
    if not art_dir.is_dir():
        return None
    df = catalog.load()
    keys: list = []
    texts: list[str] = []
    by_video = {vid: grp for vid, grp in df.groupby("video_id", sort=False)}
    for f in sorted(art_dir.glob("*.json")):
        vid = f.stem
        grp = by_video.get(vid)
        if grp is None:
            continue
        data = read_json(f, default={}) or {}
        n_to_text = data.get(json_key) or {}
        if not n_to_text:
            continue
        n_to_gid = dict(zip(grp["n"].astype(int), grp["global_id"].astype(int)))
        for n_str, text in n_to_text.items():
            try:
                gid = n_to_gid[int(n_str)]
            except (KeyError, ValueError):
                continue
            keys.append(int(gid))
            texts.append(str(text))
    return keys, texts


def _collect_asr_docs(settings: Settings, catalog: KeyframeCatalog) -> tuple[list, list[str]] | None:
    """ASR segments → nearest keyframes by pts_time (a segment covers frames)."""
    art_dir = settings.paths.art("asr")
    if not art_dir.is_dir():
        return None
    df = catalog.load()
    keys: list = []
    texts: list[str] = []
    by_video = {vid: grp.sort_values("pts_time") for vid, grp in df.groupby("video_id", sort=False)}
    for f in sorted(art_dir.glob("*.json")):
        vid = f.stem
        grp = by_video.get(vid)
        if grp is None:
            continue
        segments = (read_json(f, default={}) or {}).get("segments") or []
        if not segments:
            continue
        pts = grp["pts_time"].to_numpy()
        gids = grp["global_id"].to_numpy()
        for seg in segments:
            try:
                start, end = float(seg["start"]), float(seg["end"])
                text = str(seg.get("text", ""))
            except (KeyError, TypeError, ValueError):
                continue
            if not text.strip():
                continue
            # keyframes whose pts_time falls inside [start-2s, end+2s]
            lo, hi = start - ASR_PAD_S, end + ASR_PAD_S
            for gid, t in zip(gids, pts):
                if lo <= t <= hi:
                    keys.append(int(gid))
                    texts.append(text)
    return keys, texts


def _collect_metadata_docs(settings: Settings, catalog: KeyframeCatalog) -> tuple[list, list[str]]:
    media = MediaInfoStore(settings)
    keys: list = []
    texts: list[str] = []
    for vid in catalog.videos():
        blob = media.text_blob(vid)
        if blob.strip():
            keys.append(str(vid))
            texts.append(blob)
    return keys, texts


# ── in-memory fallback field ─────────────────────────────────────────────────


class _Bm25Field:
    """One BM25 corpus, keyed by global_id (frame fields) or video_id (metadata)."""

    def __init__(self, keys: list, docs: list[list[str]]):
        from rank_bm25 import BM25Plus  # lazy: fallback-only dependency

        self.keys = keys
        # A key may own SEVERAL documents (ASR: one per covering segment) —
        # keep every position and score with the best-matching one.
        self.key_to_pos: dict = {}
        for i, k in enumerate(keys):
            self.key_to_pos.setdefault(k, []).append(i)
        # BM25Plus: IDF is never negative (BM25Okapi goes negative on small
        # corpora / very common terms, ranking matching docs BELOW absent ones).
        self.bm25 = BM25Plus(docs) if docs else None

    def scores_for(self, query_tokens: list[str], keys: list) -> dict:
        """Sparse positive scores for the requested keys (absent == no signal).

        BM25Plus scores a zero-term-overlap document Σ_q idf(q)·δ (its +δ
        floor), not 0 — that analytic baseline is subtracted so only true
        term matches contribute signal.
        """
        if self.bm25 is None or not query_tokens:
            return {}
        positions = [(k, self.key_to_pos[k]) for k in set(keys) if k in self.key_to_pos]
        if not positions:
            return {}
        all_scores = self.bm25.get_scores(query_tokens)
        baseline = sum(
            float(self.bm25.idf.get(tok, 0.0)) * float(self.bm25.delta) for tok in query_tokens
        )
        out = {}
        for k, plist in positions:
            s = max(float(all_scores[p]) for p in plist) - baseline
            if s > 1e-9:
                out[k] = s
        return out


class TextSignals:
    """Scores candidates with BM25 over aux text fields (persisted or in-memory)."""

    def __init__(self, settings: Settings, catalog: KeyframeCatalog):
        self.settings = settings
        self.catalog = catalog
        self.media = MediaInfoStore(settings)
        self._fields: dict[str, _Bm25Field | None] = {}
        self._persisted: dict[str, TextIndexField | None] = {}
        self._persisted_ok: bool | None = None

    # ── persisted path ───────────────────────────────────────────────────

    def _persisted_enabled(self) -> bool:
        """True when artifacts/text_index exists AND was built for this corpus."""
        if self._persisted_ok is None:
            try:
                self._persisted_ok = TextIndexStore.signature_matches(
                    self.settings.paths.artifacts_root, self.catalog.signature()
                )
            except Exception as e:  # noqa: BLE001 — meta problems must not sink the query
                log.warning("text_index meta check failed (%s) — using in-memory BM25", e)
                self._persisted_ok = False
            if self._persisted_ok:
                log.info("TextSignals: persisted text_index matches catalog — candidate-restricted scoring")
            else:
                # Silent degradation here cost seconds/query + GBs of RAM at
                # the 177k-frame Batch-1 shape — the operator must know.
                log.warning(
                    "TextSignals: persisted text_index is ABSENT or STALE for this "
                    "catalog — falling back to slow in-memory BM25 (full-corpus "
                    "scan per query). Rerun scripts/03_build_aux_indexes.py "
                    "--text-index after ingest.")
        return self._persisted_ok

    # ── in-memory fallback construction ──────────────────────────────────

    def _build_memory_field(self, name: str) -> _Bm25Field | None:
        collected = collect_field_documents(self.settings, self.catalog, name)
        if collected is None:
            return None
        keys, texts = collected
        f_keys: list = []
        f_docs: list[list[str]] = []
        for k, text in zip(keys, texts):
            toks = tokenize_vi(text)
            if toks:
                f_keys.append(k)
                f_docs.append(toks)
        if not f_docs:
            return None
        log.info("BM25 field %s (in-memory): %d docs", name, len(f_docs))
        return _Bm25Field(f_keys, f_docs)

    def _get_field(self, name: str) -> _Bm25Field | None:
        if name not in self._fields:
            self._fields[name] = self._build_memory_field(name) if name in ALL_FIELDS else None
        return self._fields[name]

    def _get_scorer(self, name: str) -> TextIndexField | _Bm25Field | None:
        """Persisted field when available and fresh; else in-memory fallback.

        A field persisted with zero documents means "built, nothing to match"
        — skipped WITHOUT rescanning artifacts. A field missing from a fresh
        persisted build (its artifacts appeared after scripts/03 ran) falls
        back to the in-memory path for that field only.
        """
        if self._persisted_enabled():
            if name not in self._persisted:
                self._persisted[name] = TextIndexStore.load(name, self.settings.paths.artifacts_root)
            f = self._persisted[name]
            if f is not None:
                return f if f.n_docs > 0 else None
        return self._get_field(name)

    def persisted_field_ready(self, name: str) -> bool:
        """True when ``score_field(name, …)`` would hit the PERSISTED index.

        Round-72 (Cursor-lab audit): the TRAKE caption-step scorer calls
        ``score_field`` k×~30 times per query — on the in-memory fallback each
        call is a full-corpus rank_bm25 scan (minutes per query on the dense
        caption store). Callers that would multiply the fallback cost this way
        must check here and disable themselves loudly instead.
        """
        if not self._persisted_enabled():
            return False
        if name not in self._persisted:
            self._persisted[name] = TextIndexStore.load(name, self.settings.paths.artifacts_root)
        f = self._persisted[name]
        return f is not None and f.n_docs > 0

    # ── scoring ──────────────────────────────────────────────────────────

    def score_field(self, field: str, query_text: str,
                    candidates: list[KeyframeRef]) -> dict[int, float]:
        """Raw BM25 scores for ONE frame field only ({} when absent/unknown).

        The TRAKE caption-step signal scores K event texts per pooled video —
        going through :meth:`score_candidates` would tokenize and score all
        four fields K times per video for nothing.
        """
        if field not in FRAME_FIELDS:
            return {}
        f = self._get_scorer(field)
        if not f:
            return {}
        toks = tokenize_vi(query_text)
        if not toks:
            return {}
        return f.scores_for(toks, [c.global_id for c in candidates])

    def score_candidates(self, query_text: str, candidates: list[KeyframeRef]) -> dict[str, dict[int, float]]:
        """Per-field raw BM25 scores for candidate frames.

        Returns {field: {global_id: score}} — missing keys mean zero. Frame
        fields key on global_id; metadata scores a video and is broadcast to
        its candidate frames.
        """
        toks = tokenize_vi(query_text)
        out: dict[str, dict[int, float]] = {}
        gids = [c.global_id for c in candidates]

        for field in FRAME_FIELDS:
            f = self._get_scorer(field)
            if f:
                out[field] = f.scores_for(toks, gids)

        meta = self._get_scorer("metadata")
        if meta:
            vids = [c.video_id for c in candidates]
            vid_scores = meta.scores_for(toks, vids)
            out["metadata"] = {
                c.global_id: vid_scores.get(c.video_id, 0.0) for c in candidates if c.video_id in vid_scores
            }
        return out
