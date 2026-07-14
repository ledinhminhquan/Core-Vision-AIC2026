"""SearchEngine — the online orchestrator behind the UI and batch runners.

Per query (<1s target, excluding optional LLM query processing):

    VI query ─► QueryProcessor (translate/enhance/expand — cached, optional)
             ─► encode all query variants, per ensemble member
             ─► FAISS top-K per member  ─► max over variants ─► fuse members
             ─► + BM25 (OCR/ASR/caption/metadata) + object boosts   (top-K only)
             ─► + temporal neighbour smoothing
             ─► ranked SearchResults (each carries video_id / frame_idx / signals)

Everything degrades gracefully: no OCR artifacts → that signal contributes 0;
no Gemini key → raw-query search still works (SigLIP-2 is multilingual).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from cvf.config import Settings
from cvf.data.catalog import KeyframeCatalog, KeyframeRef
from cvf.models.query_processor import ProcessedQuery, QueryProcessor
from cvf.models.registry import build_model, index_key_for
from cvf.index.store import IndexStore
from cvf.search import fusion
from cvf.search.avs import avs_diversify
from cvf.search.feedback import rocchio
from cvf.search.object_filter import ObjectBooster
from cvf.search.superglobal import superglobal_rerank
from cvf.search.temporal import TrakeCandidate, trake_search
from cvf.search.text_signals import TextSignals
from cvf.search.vlm_rerank import vlm_rerank

log = logging.getLogger(__name__)


@dataclass
class SearchResult:
    ref: KeyframeRef
    score: float
    signals: dict[str, float] = field(default_factory=dict)

    # convenience passthroughs for the UI / writers
    @property
    def video_id(self) -> str:
        return self.ref.video_id

    @property
    def frame_idx(self) -> int:
        return self.ref.frame_idx

    @property
    def global_id(self) -> int:
        return self.ref.global_id


class SearchEngine:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.catalog = KeyframeCatalog(settings)
        self.catalog.load()

        # Ensemble members: 1 model when embedding.model != "ensemble".
        names = (
            settings.embedding.ensemble_members
            if settings.embedding.model == "ensemble"
            else [settings.embedding.model]
        )
        weights = (
            list(settings.embedding.ensemble_weights)
            if settings.embedding.model == "ensemble"
            else [1.0]
        )
        self.members = []  # list[(model, IndexStore)]
        self.member_weights: list[float] = []
        self.member_names: list[str] = []
        errors: list[str] = []
        for name, weight in zip(names, weights):
            try:
                model = build_model(settings, name)
                store = IndexStore(settings, index_key_for(name))
                store.load(self.catalog)  # fail loud on stale index per member
            except Exception as e:  # noqa: BLE001 — degrade to the lanes that work
                errors.append(f"{name}: {e}")
                log.error(
                    "Ensemble member %r failed to load (%s) — CONTINUING WITHOUT IT. "
                    "Search quality is degraded; fix before the competition round.",
                    name, e,
                )
                continue
            # Checkpoint-drift guard: the 'openclip' lane resolves to one of
            # several checkpoints; searching a PE-Core index with DFN5B query
            # vectors would silently return garbage.
            built_tag = store.meta().get("model_tag")
            live_tag = getattr(model, "model_tag", None)
            if built_tag and live_tag and built_tag != live_tag:
                log.warning(
                    "[%s] index was built by %r but this machine loaded %r — "
                    "re-embed + rebuild the index before trusting results",
                    name, built_tag, live_tag,
                )
            self.members.append((model, store))
            self.member_weights.append(weight)
            self.member_names.append(name)
        if not self.members:
            raise RuntimeError(
                "No ensemble member could be loaded: " + " | ".join(errors)
            )

        self.primary_model, self.primary_store = self.members[0]
        self.query_processor = QueryProcessor(settings)
        self.text_signals = TextSignals(settings, self.catalog)
        self.object_booster = ObjectBooster(settings)
        log.info(
            "SearchEngine ready: %d keyframes, members=%s",
            len(self.catalog), [m.key for m, _ in self.members],
        )

    # ── internals ────────────────────────────────────────────────────────

    def _vectors_for(self, store: IndexStore, gids: list[int]) -> np.ndarray:
        """Stored embedding rows for global ids (grouped per video, mmap reads).

        Embedding .npy rows are POSITIONAL over each video's catalog rows
        (sorted by n) — ordinal gaps are legal, so the row is
        ``global_id - video_start``, never ``n - 1``.
        """
        refs = self.catalog.refs(gids)
        by_video: dict[str, list[tuple[int, int]]] = {}
        for pos, ref in enumerate(refs):
            start, _count = self.catalog.video_span(ref.video_id)
            by_video.setdefault(ref.video_id, []).append((pos, ref.global_id - start))
        dim = store.dim() or self.primary_model.dim
        out = np.zeros((len(gids), dim), dtype=np.float32)
        for vid, items in by_video.items():
            try:
                vecs = np.load(store.embedding_path(vid), mmap_mode="r")
            except (OSError, ValueError) as e:
                log.warning("Missing embeddings for %s: %s", vid, e)
                continue
            for pos, row in items:
                if 0 <= row < len(vecs):
                    out[pos] = vecs[row]
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms

    def _dense_scores(self, processed: ProcessedQuery, topk: int) -> dict[int, float]:
        """Ensemble dense retrieval: variants→max, SuperGlobal refine, members→weighted sum."""
        member_maps: list[dict[int, float]] = []
        for model, store in self.members:
            texts = processed.texts_for_search(model.multilingual)
            vecs = model.encode_text(texts)
            scores, gids = store.search(vecs, topk)
            per_variant = []
            for qi in range(len(texts)):
                m = {
                    int(g): float(s)
                    for s, g in zip(scores[qi], gids[qi])
                    if g >= 0
                }
                per_variant.append(m)
            member_map = fusion.aggregate_queries(per_variant, self.settings.query.multi_query_agg)
            if self.settings.search.rerank and member_map:
                # All query variants participate — max-fused inside the reranker,
                # matching the dense stage, so expansion-found hits can't be demoted.
                member_map = self._superglobal(store, vecs, member_map)
            member_maps.append(member_map)
        if len(member_maps) == 1:
            return member_maps[0]
        return fusion.weighted_sum(member_maps, self.member_weights)

    def _superglobal(self, store: IndexStore, query_vecs: np.ndarray,
                     dense: dict[int, float]) -> dict[int, float]:
        """SuperGlobal refinement of one member's candidate scores (best-effort).

        Candidates whose stored vector is unreadable (zero row — e.g. a
        partially-synced embeddings folder) keep their ORIGINAL dense score:
        FAISS retrieved them for a reason, and replacing that with the
        reranker's 0 would silently drop every frame of the affected video.
        """
        try:
            gids = list(dense.keys())
            cand = self._vectors_for(store, gids)
            refined = superglobal_rerank(query_vecs, cand)
            if len(refined) == len(gids):
                valid = np.linalg.norm(cand, axis=1) > 1e-6
                n_bad = int((~valid).sum())
                if n_bad:
                    log.warning(
                        "SuperGlobal: %d/%d candidates have no stored vector — "
                        "keeping their dense scores (check embeddings sync)",
                        n_bad, len(gids),
                    )
                return {
                    g: (float(s) if ok else dense[g])
                    for g, s, ok in zip(gids, refined, valid)
                }
        except Exception as e:  # noqa: BLE001 — reranking must never sink the query
            log.warning("SuperGlobal rerank failed: %s", e)
        return dense

    def _spans_for(self, gids: list[int]) -> dict[int, tuple[int, int]]:
        out: dict[int, tuple[int, int]] = {}
        for gid in gids:
            ref = self.catalog.ref(gid)
            out[gid] = self.catalog.video_span(ref.video_id)
        return out

    def _finalize(
        self,
        dense: dict[int, float],
        query_text_for_bm25: str,
        display_k: int,
        signal_dump: dict[str, dict[int, float]] | None = None,
    ) -> list[SearchResult]:
        """Fuse dense + text + object signals over the candidate pool.

        ``signal_dump`` (optional dict) receives the RAW per-signal score maps —
        used by scripts/23_dump_signals.py to feed the weight-tuning harness.
        """
        w = self.settings.search.weights
        candidates = self.catalog.refs(list(dense.keys()))

        text_scores = {}
        try:
            text_scores = self.text_signals.score_candidates(query_text_for_bm25, candidates)
        except Exception as e:  # noqa: BLE001 — aux search must never sink the query
            log.warning("Text-signal scoring failed: %s", e)

        constraints = self.object_booster.parse(query_text_for_bm25)
        object_scores: dict[int, float] = {}
        if constraints:
            for ref in candidates:
                s = self.object_booster.score(constraints, ref)
                if s > 0:
                    object_scores[ref.global_id] = s

        maps = [dense]
        weights = [w.visual]
        for name, weight in (("ocr", w.ocr), ("asr", w.asr), ("caption", w.caption), ("metadata", w.metadata)):
            m = text_scores.get(name)
            if m:
                maps.append(m)
                weights.append(weight)
        if object_scores:
            maps.append(object_scores)
            weights.append(w.object)

        if signal_dump is not None:
            signal_dump["visual"] = dict(dense)
            for name in ("ocr", "asr", "caption", "metadata"):
                if text_scores.get(name):
                    signal_dump[name] = dict(text_scores[name])
            if object_scores:
                signal_dump["object"] = dict(object_scores)

        if self.settings.search.fusion_method == "rrf":
            fused = fusion.rrf(maps, weights, k=self.settings.search.rrf_k)
        else:
            fused = fusion.weighted_sum(maps, weights)
        fused = fusion.neighbor_boost(
            fused,
            self._spans_for(list(fused.keys())),
            boost=self.settings.search.neighbor_boost,
            window=self.settings.search.neighbor_window,
        )

        ranked = sorted(fused.items(), key=lambda kv: -kv[1])[:display_k]
        results = []
        for gid, score in ranked:
            signals = {"visual": dense.get(gid, 0.0)}
            for name in ("ocr", "asr", "caption", "metadata"):
                v = (text_scores.get(name) or {}).get(gid)
                if v:
                    signals[name] = v
            if gid in object_scores:
                signals["object"] = object_scores[gid]
            results.append(SearchResult(ref=self.catalog.ref(gid), score=float(score), signals=signals))

        if self.settings.search.vlm_rerank and results:
            results = vlm_rerank(results, query_text_for_bm25, self.settings)
        return results

    # ── public API ───────────────────────────────────────────────────────

    def search_text(self, query_vi: str, topk: int | None = None, display_k: int | None = None) -> list[SearchResult]:
        if not query_vi or not query_vi.strip():
            return []
        topk = topk or self.settings.search.topk
        display_k = display_k or self.settings.search.display_k
        processed = self.query_processor.process(query_vi)
        dense = self._dense_scores(processed, topk)
        # BM25 fields hold Vietnamese text (OCR/ASR/captions) — score with the
        # original query; diacritic folding handles sloppy typing.
        return self._finalize(dense, query_vi, display_k)

    def search_text_debug(
        self, query_vi: str, topk: int | None = None, display_k: int | None = None
    ) -> tuple[list[SearchResult], dict[str, dict[int, float]]]:
        """search_text + the raw per-signal score maps (for the tuning harness)."""
        if not query_vi or not query_vi.strip():
            return [], {}
        topk = topk or self.settings.search.topk
        display_k = display_k or self.settings.search.display_k
        processed = self.query_processor.process(query_vi)
        dense = self._dense_scores(processed, topk)
        dump: dict[str, dict[int, float]] = {}
        results = self._finalize(dense, query_vi, display_k, signal_dump=dump)
        return results, dump

    def search_image(self, image: Image.Image, display_k: int | None = None) -> list[SearchResult]:
        """Query-by-example with the primary model (click a frame → similar)."""
        display_k = display_k or self.settings.search.display_k
        vec = self.primary_model.encode_image([image])
        scores, gids = self.primary_store.search(vec, display_k)
        return [
            SearchResult(ref=self.catalog.ref(int(g)), score=float(s), signals={"visual": float(s)})
            for s, g in zip(scores[0], gids[0])
            if g >= 0
        ]

    def nearest(self, global_id: int, k: int = 60) -> list[SearchResult]:
        """Visual neighbourhood of an indexed frame (uses its stored vector)."""
        ref = self.catalog.ref(global_id)
        vecs = np.load(self.primary_store.embedding_path(ref.video_id))
        start, _count = self.catalog.video_span(ref.video_id)
        row = vecs[ref.global_id - start]  # positional row, robust to ordinal gaps
        row = row / (np.linalg.norm(row) or 1.0)
        scores, gids = self.primary_store.search(row[None, :].astype(np.float32), k + 1)
        return [
            SearchResult(ref=self.catalog.ref(int(g)), score=float(s), signals={"visual": float(s)})
            for s, g in zip(scores[0], gids[0])
            if g >= 0 and int(g) != int(global_id)
        ][:k]

    def temporal_neighbors(self, global_id: int, window: int = 8) -> list[KeyframeRef]:
        """Frames around a hit inside the same video — for eyeballing context."""
        ref = self.catalog.ref(global_id)
        start, count = self.catalog.video_span(ref.video_id)
        lo = max(start, global_id - window)
        hi = min(start + count - 1, global_id + window)
        return [self.catalog.ref(g) for g in range(lo, hi + 1)]

    def search_avs(self, query_vi: str, per_video_cap: int | None = None,
                   min_gap_s: float | None = None, limit: int = 100) -> list[SearchResult]:
        """AVS: cover as many *distinct* relevant moments as possible in ≤100 rows.

        MMR over primary-lane embeddings de-duplicates near-identical scenes
        across different videos (news reuse the same b-roll footage).
        """
        s = self.settings.search
        per_video_cap = per_video_cap if per_video_cap is not None else s.avs_per_video_cap
        min_gap_s = min_gap_s if min_gap_s is not None else s.avs_min_gap_s
        results = self.search_text(query_vi, display_k=max(limit * 3, 300))
        cand_vecs = None
        if results and 0.0 <= s.avs_mmr_lambda < 1.0:
            try:
                cand_vecs = self._vectors_for(self.primary_store, [r.global_id for r in results])
            except Exception as e:  # noqa: BLE001 — MMR is optional polish
                log.warning("AVS MMR vector fetch failed: %s", e)
        return avs_diversify(
            results,
            per_video_cap=per_video_cap,
            min_gap_s=min_gap_s,
            limit=limit,
            cand_vecs=cand_vecs,
            mmr_lambda=s.avs_mmr_lambda,
        )

    def search_with_feedback(self, query_vi: str, positive_gids: list[int],
                             negative_gids: list[int] | None = None,
                             display_k: int | None = None) -> list[SearchResult]:
        """Rocchio: nudge the query toward marked-good tiles and re-search.

        Every ensemble member applies the feedback in its own embedding space;
        member score maps are then fused with the configured weights, so the
        second lane benefits from the marks too.
        """
        if not query_vi or not query_vi.strip():
            return []
        display_k = display_k or self.settings.search.display_k
        processed = self.query_processor.process(query_vi)
        member_maps: list[dict[int, float]] = []
        map_weights: list[float] = []
        for (model, store), weight in zip(self.members, self.member_weights):
            try:
                texts = processed.texts_for_search(model.multilingual)
                qvec = model.encode_text(texts[:1])[0]
                pos = self._vectors_for(store, positive_gids) if positive_gids else None
                neg = self._vectors_for(store, negative_gids) if negative_gids else None
                q_new = rocchio(qvec, pos, neg)
                scores, gids = store.search(q_new[None, :], self.settings.search.topk)
                member_maps.append(
                    {int(g): float(s) for s, g in zip(scores[0], gids[0]) if g >= 0}
                )
                map_weights.append(weight)  # paired with ITS map, not positionally
            except Exception as e:  # noqa: BLE001 — one lane failing must not kill feedback
                log.warning("Feedback failed for member %s: %s", model.key, e)
        if not member_maps:
            return []
        dense = (
            member_maps[0]
            if len(member_maps) == 1
            else fusion.weighted_sum(member_maps, map_weights)
        )
        return self._finalize(dense, query_vi, display_k)

    def search_trake(self, event_queries: list[str], max_results: int = 100) -> list[TrakeCandidate]:
        """TRAKE: ordered events → per-video keyframe sequences.

        With ``temporal.use_ensemble`` every member scores every event; per-video
        similarity matrices are the weighted sum of per-member cosine sims
        (min-max normalized per event), so an event only one lane understands
        still anchors the chain.
        """
        processed = [self.query_processor.process(q) for q in event_queries]

        def _texts_for(model) -> list[str]:
            return [
                p.original if model.multilingual else (p.enhanced or p.translation or p.original)
                for p in processed
            ]

        event_vecs = self.primary_model.encode_text(_texts_for(self.primary_model))
        ensemble_kwargs: dict = {}
        if self.settings.temporal.use_ensemble and len(self.members) > 1:
            event_vecs_by_member: dict[str, np.ndarray] = {}
            stores_by_member: dict[str, IndexStore] = {}
            for name, (model, store) in zip(self.member_names, self.members):
                try:
                    event_vecs_by_member[name] = (
                        event_vecs if model is self.primary_model
                        else model.encode_text(_texts_for(model))
                    )
                    stores_by_member[name] = store
                except Exception as e:  # noqa: BLE001 — a lane failing must not kill TRAKE
                    log.warning("TRAKE ensemble: member %s skipped (%s)", name, e)
            if len(event_vecs_by_member) > 1:
                ensemble_kwargs = dict(
                    event_vecs_by_member=event_vecs_by_member,
                    member_weights=dict(zip(self.member_names, self.member_weights)),
                    stores_by_member=stores_by_member,
                )
        return trake_search(
            event_vecs=event_vecs,
            index_search=self.primary_store.search,
            embedding_path=self.primary_store.embedding_path,
            catalog=self.catalog,
            settings=self.settings,
            max_results=max_results,
            **ensemble_kwargs,
        )
