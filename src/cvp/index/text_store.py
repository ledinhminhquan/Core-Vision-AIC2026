"""Persistent BM25 text index — tokenize once at build time, score O(K·|q|) at query time.

``scripts/03_build_aux_indexes.py`` tokenizes every text field (ocr / asr /
caption / metadata) ONCE with the same diacritic-folded view used for queries
(:func:`cvp.utils.text.tokenize_vi`) and persists the BM25 sufficient
statistics:

    artifacts/text_index/{field}.json.gz   per-key term counts + doc lengths,
                                           document frequencies, N, avgdl
    artifacts/text_index/meta.json         catalog signature + build params

At query time :class:`cvp.search.text_signals.TextSignals` loads a field once
and scores ONLY the candidate keys with closed-form BM25+ math — numerically
identical to ``rank_bm25.BM25Plus`` with its zero-overlap ``+delta`` baseline
subtracted, but without ever scanning the corpus.
"""

from __future__ import annotations

import gzip
import json
import logging
import math
import os
from dataclasses import dataclass, field
from pathlib import Path

from cvp.utils.io import atomic_write_bytes, atomic_write_json, read_json
from cvp.utils.text import tokenize_vi

log = logging.getLogger(__name__)

FORMAT_VERSION = 1
DIR_NAME = "text_index"

# rank_bm25.BM25Plus defaults — the persisted math must reproduce the
# in-memory fallback path exactly (same k1/b/delta ⇒ same scores to float eps).
DEFAULT_PARAMS: dict[str, float] = {"k1": 1.5, "b": 0.75, "delta": 1.0}


@dataclass
class TextIndexField:
    """One loaded field: enough statistics to score any candidate per-id.

    ``docs`` maps a key (frame ``global_id`` int, or ``video_id`` str for the
    metadata field) to a list of ``[doc_len, {token: tf}]`` entries — a key may
    own several documents (ASR: one per covering segment) and scores with its
    best one.
    """

    name: str
    n_docs: int
    avgdl: float
    df: dict[str, int]
    docs: dict[int | str, list] = field(default_factory=dict)
    k1: float = 1.5
    b: float = 0.75
    delta: float = 1.0

    def idf(self, token: str) -> float:
        """BM25Plus idf: ln((N+1)/df); 0 for out-of-corpus tokens."""
        d = self.df.get(token)
        if not d:
            return 0.0
        return math.log((self.n_docs + 1) / d)

    def scores_for(self, query_tokens: list[str], keys: list) -> dict:
        """Sparse positive BM25+ scores for the requested keys (absent == 0).

        Mirrors ``cvp.search.text_signals._Bm25Field.scores_for``: the BM25Plus
        ``+delta`` zero-overlap baseline is already subtracted, so only true
        term matches contribute — and work is O(len(keys) · len(query_tokens));
        the corpus is never scanned.
        """
        if self.n_docs <= 0 or not query_tokens or self.avgdl <= 0:
            return {}
        # Keep duplicates: rank_bm25 sums per query occurrence, not per type.
        weighted = [(tok, self.idf(tok)) for tok in query_tokens]
        weighted = [(tok, w) for tok, w in weighted if w > 0.0]
        if not weighted:
            return {}
        k1, b, avgdl = self.k1, self.b, self.avgdl
        out: dict = {}
        for k in set(keys):
            doc_list = self.docs.get(k)
            if not doc_list:
                continue
            best = 0.0
            for dl, tf in doc_list:
                norm = k1 * (1.0 - b + b * dl / avgdl)
                s = 0.0
                for tok, w in weighted:
                    f = tf.get(tok)
                    if f:
                        s += w * (f * (k1 + 1.0)) / (norm + f)
                if s > best:
                    best = s
            if best > 1e-9:
                out[k] = best
        return out


class TextIndexStore:
    """Builds / loads the persisted per-field BM25 index under ``artifacts/text_index``."""

    # ── paths / meta ─────────────────────────────────────────────────────

    @staticmethod
    def root(artifacts_root: str | os.PathLike) -> Path:
        return Path(artifacts_root) / DIR_NAME

    @classmethod
    def field_path(cls, artifacts_root: str | os.PathLike, field_name: str) -> Path:
        return cls.root(artifacts_root) / f"{field_name}.json.gz"

    @classmethod
    def meta_path(cls, artifacts_root: str | os.PathLike) -> Path:
        return cls.root(artifacts_root) / "meta.json"

    @classmethod
    def read_meta(cls, artifacts_root: str | os.PathLike) -> dict:
        return read_json(cls.meta_path(artifacts_root), default={}) or {}

    @classmethod
    def signature_matches(cls, artifacts_root: str | os.PathLike, signature: str) -> bool:
        """True when a persisted index exists and was built for this exact corpus."""
        meta = cls.read_meta(artifacts_root)
        return bool(signature) and meta.get("version") == FORMAT_VERSION and meta.get("signature") == signature

    # ── build ────────────────────────────────────────────────────────────

    @classmethod
    def build(
        cls,
        field_name: str,
        keys: list,
        raw_texts: list[str],
        artifacts_root: str | os.PathLike,
        signature: str,
        params: dict[str, float] | None = None,
    ) -> TextIndexField:
        """Tokenize one field's documents once and persist its BM25 statistics.

        Args:
            field_name: ocr | asr | caption | metadata.
            keys: parallel to ``raw_texts`` — global_id (int) for frame fields,
                video_id (str) for metadata. Duplicate keys are legal (a key
                owning several documents scores with its best one).
            raw_texts: raw document strings; token-less ones are dropped, so
                N/avgdl match the in-memory rank-bm25 fallback exactly.
            artifacts_root: the artifacts folder (``settings.paths.artifacts_root``).
            signature: catalog signature the corpus keys refer to.
            params: BM25+ parameters; defaults to rank_bm25's k1/b/delta.

        Returns:
            The freshly persisted field, reloaded from disk (roundtrip-checked).
        """
        p = dict(DEFAULT_PARAMS if params is None else params)

        kept_keys: list = []
        kept_docs: list[list[str]] = []
        for k, text in zip(keys, raw_texts):
            toks = tokenize_vi(str(text))
            if toks:
                kept_keys.append(k)
                kept_docs.append(toks)

        n_docs = len(kept_docs)
        total_len = sum(len(t) for t in kept_docs)
        avgdl = (total_len / n_docs) if n_docs else 0.0
        key_type = "str" if any(isinstance(k, str) for k in kept_keys) else "int"

        df: dict[str, int] = {}
        docs_by_key: dict[str, list] = {}
        for k, toks in zip(kept_keys, kept_docs):
            tf: dict[str, int] = {}
            for tok in toks:
                tf[tok] = tf.get(tok, 0) + 1
            for tok in tf:
                df[tok] = df.get(tok, 0) + 1
            ks = str(k) if key_type == "str" else str(int(k))
            docs_by_key.setdefault(ks, []).append([len(toks), tf])

        payload = {
            "version": FORMAT_VERSION,
            "field": field_name,
            "key_type": key_type,
            "params": p,
            "n_docs": n_docs,
            "avgdl": avgdl,
            "df": df,
            "docs": docs_by_key,
        }
        blob = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        atomic_write_bytes(cls.field_path(artifacts_root, field_name), gzip.compress(blob))

        meta = cls.read_meta(artifacts_root)
        if (
            meta.get("version") != FORMAT_VERSION
            or meta.get("signature") != signature
            or meta.get("params") != p
        ):
            meta = {"version": FORMAT_VERSION, "signature": signature, "params": p, "fields": {}}
        meta.setdefault("fields", {})[field_name] = {
            "n_docs": n_docs,
            "n_keys": len(docs_by_key),
            "avgdl": avgdl,
        }
        atomic_write_json(cls.meta_path(artifacts_root), meta)
        log.info("text_index[%s]: %d docs / %d keys persisted", field_name, n_docs, len(docs_by_key))

        loaded = cls.load(field_name, artifacts_root)
        if loaded is None:  # pragma: no cover — write just succeeded
            raise RuntimeError(f"text_index[{field_name}] failed to reload after build")
        return loaded

    # ── load ─────────────────────────────────────────────────────────────

    @classmethod
    def load(cls, field_name: str, artifacts_root: str | os.PathLike) -> TextIndexField | None:
        """Load one persisted field, or None when absent/unreadable/foreign-format."""
        path = cls.field_path(artifacts_root, field_name)
        if not path.is_file():
            return None
        try:
            payload = json.loads(gzip.decompress(path.read_bytes()).decode("utf-8"))
        except (OSError, EOFError, ValueError) as e:  # BadGzipFile/JSONDecodeError ⊂ these
            log.warning("Unreadable text index %s: %s", path, e)
            return None
        if payload.get("version") != FORMAT_VERSION:
            log.warning("text_index[%s]: unknown format version %r", field_name, payload.get("version"))
            return None
        params = payload.get("params") or DEFAULT_PARAMS
        conv = int if payload.get("key_type") == "int" else str
        docs: dict = {}
        for ks, doc_list in (payload.get("docs") or {}).items():
            try:
                docs[conv(ks)] = [[int(dl), tf] for dl, tf in doc_list]
            except (TypeError, ValueError):
                continue
        return TextIndexField(
            name=str(payload.get("field", field_name)),
            n_docs=int(payload.get("n_docs", 0)),
            avgdl=float(payload.get("avgdl") or 0.0),
            df=payload.get("df") or {},
            docs=docs,
            k1=float(params.get("k1", 1.5)),
            b=float(params.get("b", 0.75)),
            delta=float(params.get("delta", 1.0)),
        )
