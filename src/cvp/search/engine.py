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

from cvp.config import Settings
from cvp.data.catalog import KeyframeCatalog, KeyframeRef
from cvp.models.query_processor import ProcessedQuery, QueryProcessor
from cvp.models.registry import build_model, index_key_for
from cvp.index.store import IndexStore
from cvp.search import fusion
from cvp.search.avs import avs_diversify
from cvp.search.feedback import rocchio
from cvp.search.object_filter import ObjectBooster
from cvp.search.superglobal import superglobal_rerank
from cvp.search.temporal import TrakeCandidate, caption_scorer_from_signals, trake_search
from cvp.search.text_signals import TextSignals
from cvp.search.cross_rerank import cross_rerank
from cvp.search.vlm_rerank import vlm_rerank

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
                # verify-R23 (HIGH): a torn Drive-cache read can make the hub
                # loader silently fall back to a DIFFERENT checkpoint; querying
                # this index with those vectors returns garbage (or a dim
                # mismatch that fails EVERY query quietly). Disable the lane
                # LOUDLY at startup instead.
                errors.append(f"{name}: model_tag mismatch "
                              f"(index {built_tag!r} vs loaded {live_tag!r})")
                log.error(
                    "[%s] index was built by %r but this machine loaded %r — "
                    "lane DISABLED (kết quả sẽ là rác nếu cứ dùng). Xóa cache "
                    "của checkpoint này trong artifacts/hf_cache rồi chạy lại, "
                    "hoặc re-embed với FORCE_EMBED=True.",
                    name, built_tag, live_tag,
                )
                continue
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
        # Column gather + span dict — building full KeyframeRefs here cost
        # tens of ms per query at Batch-1 scale (round-10): only the video id
        # is needed to locate each row.
        vids = self.catalog.video_ids(gids)
        by_video: dict[str, list[tuple[int, int]]] = {}
        for pos, (gid, vid) in enumerate(zip(gids, vids)):
            start, _count = self.catalog.video_span(vid)
            by_video.setdefault(vid, []).append((pos, int(gid) - start))
        dim = store.dim() or self.primary_model.dim
        out = np.zeros((len(gids), dim), dtype=np.float32)
        for vid, items in by_video.items():
            try:
                # Cached mmap handle (round-10): np.load per video per query
                # was the single largest cost of the whole search path.
                vecs = store.vectors_mmap(vid)
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
        """Ensemble dense retrieval: variants→max, SuperGlobal refine, members→weighted sum.

        Per-member guard (review C16): a runtime failure in ONE lane (encode
        OOM, index error) degrades to the surviving lanes — exactly like
        ``search_image`` and load-time degradation — instead of killing every
        text search of the session.
        """
        member_maps: list[dict[int, float]] = []
        weights: list[float] = []
        for (model, store), w in zip(self.members, self.member_weights):
            if (not model.multilingual and not processed.has_english()
                    and len(self.members) > 1):
                # No translation available (Gemini + Google Translate both
                # down): an English-only tower would rank near-randomly on raw
                # Vietnamese and fuse that noise at full weight. Degrade to the
                # multilingual lanes — same policy as a runtime lane failure.
                log.warning(
                    "Lane %r is English-only but the query has no English "
                    "variant — SKIPPING this lane (translation unavailable)",
                    getattr(model, "key", "?"))
                continue
            try:
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
            except Exception as e:  # noqa: BLE001 — degrade to the lanes that work
                log.error("Dense lane %r failed at query time (%s) — CONTINUING "
                          "without it; search quality is degraded",
                          getattr(model, "key", "?"), e)
                continue
            member_maps.append(member_map)
            weights.append(w)
        if not member_maps:
            log.error("EVERY dense lane failed for this query — empty ranking")
            return {}
        if len(member_maps) == 1:
            return member_maps[0]
        return fusion.weighted_sum(member_maps, weights)

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
        # Batched column gather (round-10): per-gid catalog.ref() built a full
        # KeyframeRef just to read .video_id — ~45ms/500 ids at Batch-1 scale.
        vids = self.catalog.video_ids(gids)
        return {gid: self.catalog.video_span(vid) for gid, vid in zip(gids, vids)}

    def _finalize(
        self,
        dense: dict[int, float],
        query_text_for_bm25: str,
        display_k: int,
        signal_dump: dict[str, dict[int, float]] | None = None,
        skip_rerank: bool = False,
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
        fused = self._maybe_temporal_boost(query_text_for_bm25, fused)

        ranked = sorted(fused.items(), key=lambda kv: -kv[1])[:display_k]
        ranked_refs = self.catalog.refs([gid for gid, _ in ranked])  # one gather (round-10)
        results = []
        for (gid, score), ref in zip(ranked, ranked_refs):
            signals = {"visual": dense.get(gid, 0.0)}
            for name in ("ocr", "asr", "caption", "metadata"):
                v = (text_scores.get(name) or {}).get(gid)
                if v:
                    signals[name] = v
            if gid in object_scores:
                signals["object"] = object_scores[gid]
            results.append(SearchResult(ref=ref, score=float(score), signals=signals))

        if skip_rerank:
            # Low-confidence retry alts feed an RRF rank-merge — running the
            # full cross/VLM stack per alt quadrupled reranker latency and
            # Gemini quota for zero head-order benefit (round-10).
            return results
        if self.settings.search.reranker != "none" and results:
            results = cross_rerank(results, query_text_for_bm25, self.settings)
        if self.settings.search.vlm_rerank and results:
            results = vlm_rerank(results, query_text_for_bm25, self.settings)
        return results

    def _maybe_temporal_boost(self, query_vi: str, fused: dict[int, float]) -> dict[int, float]:
        """Vortex-style before/now/after context boost (off by default).

        When the query carries a temporal marker ("… sau khi …"), re-score the
        head of the fused ranking by how well each candidate's one-sided
        temporal neighbours match the CONTEXT clause in the primary lane.
        Every failure path returns ``fused`` unchanged.
        """
        cfg = self.settings.search
        if not cfg.temporal_boost or not fused:
            return fused
        from cvp.search.temporal_boost import (
            apply_context_boost,
            neighbor_rows_from_video_span,
            split_temporal_query,
        )

        parts = split_temporal_query(query_vi)
        if parts is None:
            return fused
        try:
            model, store = self.members[0]
            texts = [parts.context]
            if not getattr(model, "multilingual", False):
                # English-only primary lane: the raw VI clause would embed
                # poorly — skip rather than boost with garbage.
                return fused
            ctx_vec = model.encode_text(texts)[0]
            head = sorted(fused.items(), key=lambda kv: -kv[1])[: cfg.temporal_boost_topk]
            spans = self._spans_for([gid for gid, _ in head])
            # ONE batched vector fetch for every candidate's neighbours
            # (round-10): a _vectors_for call per head candidate cost +1.0s
            # per boosted query at the 177k-row Batch-1 scale.
            rows_by_gid: dict[int, list[int]] = {}
            for gid, _score in head:
                # _spans_for returns catalog.video_span = (first_row, COUNT).
                rows = neighbor_rows_from_video_span(gid, spans[gid], parts.direction,
                                                     cfg.temporal_boost_window)
                if rows:
                    rows_by_gid[gid] = rows
            if not rows_by_gid:
                return fused
            uniq_rows = sorted({r for rows in rows_by_gid.values() for r in rows})
            vecs = self._vectors_for(store, uniq_rows)
            if vecs.size == 0:
                return fused
            sims = vecs @ ctx_vec
            pos = {r: i for i, r in enumerate(uniq_rows)}
            ctx_scores = {
                gid: float(max(sims[pos[r]] for r in rows))
                for gid, rows in rows_by_gid.items()
            }
            return apply_context_boost(fused, ctx_scores, cfg.temporal_boost_weight)
        except Exception as e:  # noqa: BLE001 — an optional boost must never sink a query
            log.warning("Temporal-context boost failed (%s) — keeping plain ranking", e)
            return fused

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

    def search_prepared(self, text: str, topk: int | None = None,
                        display_k: int | None = None,
                        skip_rerank: bool = False) -> list[SearchResult]:
        """Search an ALREADY-prepared text VERBATIM — the query processor is
        bypassed entirely (no Gemini call, no enhancement-of-enhancement).

        Used by the low-confidence retry: its inputs are the processor's own
        cached enhanced/expansion strings, so re-processing them would both
        cost fresh API round-trips and search a re-description of a
        re-description (review finding C2). The retry passes
        ``skip_rerank=True`` — its alts feed an RRF rank-merge, and running
        the cross/VLM stack once per alt multiplied reranker latency and
        Gemini quota 4× (round-10).
        """
        if not text or not text.strip():
            return []
        topk = topk or self.settings.search.topk
        display_k = display_k or self.settings.search.display_k
        processed = ProcessedQuery(original=text, translation=text,
                                   provider_used="prepared")
        dense = self._dense_scores(processed, topk)
        return self._finalize(dense, text, display_k, skip_rerank=skip_rerank)

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
        """Query-by-example fused over ALL ensemble members.

        The finals KIS-V path (clip may only be WATCHED → the team re-creates
        an image and feeds it here) deserves the same lane diversity as text
        queries: each member encodes the image into ITS OWN space and searches
        its own index; maps are min-max normalised and weight-summed exactly
        like the text path. A member that fails (missing lane on this machine,
        OOM) is skipped with a warning — one lane is enough to answer.
        """
        display_k = display_k or self.settings.search.display_k
        maps: list[dict[int, float]] = []
        weights: list[float] = []
        for (model, store), w in zip(self.members, self.member_weights):
            try:
                vec = model.encode_image([image])
                # Wider per-lane pool so fusion has candidates to agree on.
                scores, gids = store.search(vec, min(3 * display_k, 1000))
                m = {int(g): float(s) for s, g in zip(scores[0], gids[0]) if g >= 0}
            except Exception as e:  # noqa: BLE001 — degrade to the lanes that work
                log.warning("search_image: member %r failed (%s) — skipping lane",
                            getattr(model, "key", "?"), e)
                continue
            if m:
                maps.append(m)
                weights.append(w)
        if not maps:
            return []
        fused = fusion.weighted_sum(maps, weights)
        ranked = sorted(fused.items(), key=lambda kv: -kv[1])[:display_k]
        return [
            SearchResult(ref=self.catalog.ref(int(g)), score=float(s),
                         signals={"visual": float(s)})
            for g, s in ranked
        ]

    def nearest(self, global_id: int, k: int = 60) -> list[SearchResult]:
        """Visual neighbourhood of an indexed frame (uses its stored vector)."""
        ref = self.catalog.ref(global_id)
        try:
            vecs = np.load(self.primary_store.embedding_path(ref.video_id))
        except OSError as e:
            # Machine synced artifacts/indexes but not artifacts/embeddings —
            # a valid lightweight deployment; the similar-frames button must
            # degrade, not crash the UI/service (review C18).
            log.warning("nearest(): embeddings for %s unavailable (%s) — "
                        "sync artifacts/embeddings to enable similar-search",
                        ref.video_id, e)
            return []
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
            if (not model.multilingual and not processed.has_english()
                    and len(self.members) > 1):
                log.warning("Feedback: skipping English-only lane %r — query has "
                            "no English variant", getattr(model, "key", "?"))
                continue
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

    def search_trake(self, event_queries: list[str], max_results: int = 100,
                     *, context: str | None = None) -> list[TrakeCandidate]:
        """TRAKE: ordered events → per-video keyframe sequences.

        With ``temporal.use_ensemble`` every member scores every event; per-video
        similarity matrices are the weighted sum of per-member cosine sims
        (min-max normalized per event), so an event only one lane understands
        still anchors the chain.

        Optional upgrades, each behind a ``temporal.*`` knob (defaults = old
        behaviour — see TemporalCfg):

        * ``event_query_variants: all`` — encode every cached query-processor
          variant per event and max-fuse per event (KIS parity).
        * ``pool_context: prepend`` + ``context=<query header>`` — the video
          POOLING stage searches "<header>. <event>" texts on multilingual
          lanes; the DP keeps scoring the bare events.
        * ``caption_signal_weight > 0`` — blend per-event caption-BM25 into
          the DP similarity matrix (dense Vintern captions as step signal).
        * ``submit_strategy: jitter`` — densify the row budget around the top
          chains with frame variants (handled inside ``trake_search``).
        """
        # Blank/whitespace events (e.g. a query file with bare "E1:" markers)
        # would reach np.concatenate([]) inside encode_text and crash the call.
        event_queries = [q for q in event_queries if q and q.strip()]
        if not event_queries:
            log.warning("search_trake called with no usable events — returning [].")
            return []
        processed = [self.query_processor.process(q) for q in event_queries]
        tcfg = self.settings.temporal
        use_variants = getattr(tcfg, "event_query_variants", "original") == "all"

        def _texts_for(model) -> list[str]:
            # English fallback order mirrors has_english(): expansions ARE an
            # English variant (round-6 — _lane_usable counted them, so the
            # text picker must consume them too, never raw Vietnamese).
            return [
                p.original if model.multilingual
                else (p.enhanced or p.translation
                      or (p.expansions[0] if p.expansions else p.original))
                for p in processed
            ]

        def _variant_texts_for(model) -> tuple[list[str], list[int]]:
            """(texts, row→event map). Default mode: the classic one text per
            event; ``event_query_variants: all`` adds every cached processor
            variant (texts_for_search — the exact KIS list) per event."""
            if not use_variants:
                return _texts_for(model), list(range(len(processed)))
            texts: list[str] = []
            vmap: list[int] = []
            for j, p in enumerate(processed):
                cand = p.texts_for_search(model.multilingual) or [p.original]
                texts.extend(cand)
                vmap.extend([j] * len(cand))
            return texts, vmap

        def _lane_usable(model) -> bool:
            # An English-only tower fed raw Vietnamese for ANY event would fold
            # near-random similarities into that event's ensemble row — skip
            # the lane unless every event has an English variant. The primary
            # (multilingual by default) is exempt: some ranking beats none.
            return model.multilingual or all(p.has_english() for p in processed)

        # Pooling-context texts (multilingual lanes only): video-level context
        # helps SELECT videos, but would dilute the per-event DP alignment.
        pool_texts: list[str] | None = None
        if getattr(tcfg, "pool_context", "none") == "prepend" and context and context.strip():
            ctx = context.strip().rstrip(" :.")
            pool_texts = [f"{ctx}. {q}" for q in event_queries]

        def _pool_vecs_for(model) -> np.ndarray | None:
            if pool_texts is None or not getattr(model, "multilingual", False):
                return None
            try:
                return model.encode_text(pool_texts)
            except Exception as e:  # noqa: BLE001 — pooling falls back to event texts
                log.warning("TRAKE: pool-context encode failed for %r (%s) — "
                            "pooling on event texts", getattr(model, "key", "?"), e)
                return None

        primary_texts, primary_vmap = _variant_texts_for(self.primary_model)
        event_vecs = self.primary_model.encode_text(primary_texts)
        primary_pool_vecs = _pool_vecs_for(self.primary_model)
        if not _lane_usable(self.primary_model):
            log.warning("TRAKE: primary lane %r is English-only but some events have "
                        "no English variant — rankings may be degraded",
                        getattr(self.primary_model, "key", "?"))

        caption_scorer = None
        if float(getattr(tcfg, "caption_signal_weight", 0.0) or 0.0) > 0:
            # Round-72 (Cursor-lab audit): scorer này gọi score_field k×~30
            # lần/query — rơi vào fallback BM25 in-memory là MỖI lần một cú
            # quét toàn kho (nhiều phút/query). Chỉ bật trên index persisted;
            # thiếu thì tắt TO TIẾNG thay vì âm thầm chậm gấp trăm lần.
            if self.text_signals.persisted_field_ready("caption"):
                caption_scorer = caption_scorer_from_signals(
                    self.text_signals, self.catalog, [p.original for p in processed])
            else:
                log.warning(
                    "TRAKE caption_signal_weight bật nhưng text_index[caption] "
                    "persisted chưa sẵn sàng — TẮT tín hiệu caption (chạy "
                    "scripts/03 --text-index trước; fallback in-memory chậm "
                    "hàng trăm lần nên không được phép).")

        ensemble_kwargs: dict = {}
        if tcfg.use_ensemble and len(self.members) > 1:
            event_vecs_by_member: dict[str, np.ndarray] = {}
            stores_by_member: dict[str, IndexStore] = {}
            variant_maps_by_member: dict[str, list[int]] = {}
            pool_vecs_by_member: dict[str, np.ndarray] = {}
            for name, (model, store) in zip(self.member_names, self.members):
                if model is not self.primary_model and not _lane_usable(model):
                    log.warning("TRAKE ensemble: member %s skipped — English-only "
                                "lane with no English variant for some events", name)
                    continue
                try:
                    if model is self.primary_model:
                        vecs_m, vmap_m, pool_m = event_vecs, primary_vmap, primary_pool_vecs
                    else:
                        texts_m, vmap_m = _variant_texts_for(model)
                        vecs_m = model.encode_text(texts_m)
                        pool_m = _pool_vecs_for(model)
                    event_vecs_by_member[name] = vecs_m
                    variant_maps_by_member[name] = vmap_m
                    if pool_m is not None:
                        pool_vecs_by_member[name] = pool_m
                    stores_by_member[name] = store
                except Exception as e:  # noqa: BLE001 — a lane failing must not kill TRAKE
                    log.warning("TRAKE ensemble: member %s skipped (%s)", name, e)
            if len(event_vecs_by_member) > 1:
                ensemble_kwargs = dict(
                    event_vecs_by_member=event_vecs_by_member,
                    member_weights=dict(zip(self.member_names, self.member_weights)),
                    stores_by_member=stores_by_member,
                    variant_maps_by_member=variant_maps_by_member,
                    pool_vecs_by_member=pool_vecs_by_member,
                )
        return trake_search(
            event_vecs=event_vecs,
            index_search=self.primary_store.search,
            embedding_path=self.primary_store.embedding_path,
            catalog=self.catalog,
            settings=self.settings,
            max_results=max_results,
            event_variant_map=primary_vmap,
            pool_event_vecs=primary_pool_vecs,
            caption_scorer=caption_scorer,
            **ensemble_kwargs,
        )
