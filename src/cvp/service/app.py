"""FastAPI app exposing the retrieval engine over HTTP/JSON.

    uvicorn cvp.service.app:app --host 0.0.0.0 --port 8000

The engine (models + FAISS) loads once at startup via the lifespan handler.
``create_app(engine=...)`` accepts a pre-built (or stub) engine — that is how
the CPU test-suite exercises every endpoint without models or indexes, and how
the auto-agent can embed the service in-process.

Endpoints (one per competition task family + plumbing):

    GET  /health              liveness + corpus size + lanes
    POST /search/text         KIS / KIS-V ranking
    POST /search/qa           ranking + per-group VQA answers (QA task rows)
    POST /search/trake        ordered event sequences (TRAKE)
    POST /search/avs          diversified coverage ranking (AVS, behind flag)
    GET  /nearest/{gid}       visual similar (query-by-example)
    GET  /keyframe/{gid}      the JPEG itself

NOTE: deliberately NO ``from __future__ import annotations`` here — FastAPI
must see real (non-string) parameter annotations to resolve the request-body
models that are imported inside ``create_app``.
"""

import logging
from contextlib import asynccontextmanager

from cvp.config import Settings, load_settings

log = logging.getLogger(__name__)


def _to_item(engine, r, ItemCls, answer: str | None = None):
    ref = r.ref
    watch = None
    media = getattr(getattr(engine, "text_signals", None), "media", None)
    if media is not None:
        try:
            watch = media.watch_url(ref.video_id, at_seconds=ref.pts_time)
        except Exception:  # noqa: BLE001 — metadata must never sink a response
            watch = None
    return ItemCls(
        global_id=ref.global_id, video_id=ref.video_id, n=ref.n,
        frame_idx=ref.frame_idx, pts_time=round(float(ref.pts_time), 3),
        score=round(float(r.score), 6), signals=dict(r.signals),
        answer=answer, image_url=f"/keyframe/{ref.global_id}", watch_url=watch,
    )


def create_app(engine=None, settings: Settings | None = None):
    """Build the FastAPI app. Pass ``engine`` to skip lazy startup loading."""
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.responses import FileResponse

    from cvp.service.schemas import (
        AvsQuery,
        ImageQuery,
        QaQuery,
        ResultItem,
        SearchResponse,
        TextQuery,
        TrakeItem,
        TrakeQuery,
        TrakeResponse,
    )

    state: dict = {"engine": engine, "settings": settings}

    @asynccontextmanager
    async def lifespan(app):
        if state["engine"] is None:
            log.info("Loading SearchEngine …")
            from cvp.search.engine import SearchEngine

            state["settings"] = state["settings"] or load_settings()
            state["engine"] = SearchEngine(state["settings"])
            log.info("SearchEngine loaded.")
        yield

    app = FastAPI(title="Core-Vision Perfect V1 Retrieval API",
                  version="1.0.0", lifespan=lifespan)

    def _engine():
        e = state["engine"]
        if e is None:
            raise HTTPException(503, "Engine not ready")
        return e

    @app.get("/health")
    def health():
        e = state["engine"]
        if e is None:
            return {"status": "loading", "keyframes": 0, "members": []}
        return {
            "status": "ok",
            "keyframes": len(e.catalog),
            "members": list(getattr(e, "member_names", [])),
        }

    @app.post("/search/text", response_model=SearchResponse)
    def search_text(q: TextQuery):
        e = _engine()
        results = e.search_text(q.query, topk=q.topk, display_k=q.display_k)
        return SearchResponse(query=q.query, count=len(results),
                              results=[_to_item(e, r, ResultItem) for r in results])

    @app.post("/search/qa", response_model=SearchResponse)
    def search_qa(q: QaQuery):
        e = _engine()
        results = e.search_text(q.query, display_k=q.display_k)
        answers: list[str | None] = [None] * len(results)
        if q.answers and results:
            try:
                from cvp.pipeline.run_queries import compute_qa_answers
                from cvp.search.vqa import VqaAssistant

                s = state["settings"] or getattr(e, "settings", None) or load_settings()
                vqa = VqaAssistant(s) if s.vqa.provider not in ("", "none") else None
                answers = list(compute_qa_answers(results, q.question or q.query, vqa, s))
            except Exception as exc:  # noqa: BLE001 — answers are best-effort extras
                log.warning("QA answering failed (%s) — returning ranking only", exc)
                answers = [None] * len(results)
        return SearchResponse(
            query=q.query, count=len(results),
            results=[_to_item(e, r, ResultItem, answer=a)
                     for r, a in zip(results, answers)])

    @app.post("/search/image", response_model=SearchResponse)
    def search_image(q: ImageQuery):
        """KIS-V path: the shown clip may only be WATCHED — feed a re-created
        image (sketch→generated, screenshot of your own drawing, …) instead."""
        import base64
        import binascii
        import io

        from PIL import Image, UnidentifiedImageError

        e = _engine()
        payload = q.image_b64.split(",", 1)[-1]  # tolerate data: URLs
        try:
            img = Image.open(io.BytesIO(base64.b64decode(payload))).convert("RGB")
        except (binascii.Error, UnidentifiedImageError, OSError, ValueError):
            raise HTTPException(422, "image_b64 is not a decodable image") from None
        results = e.search_image(img, display_k=q.display_k)
        return SearchResponse(query="(image)", count=len(results),
                              results=[_to_item(e, r, ResultItem) for r in results])

    @app.post("/search/trake", response_model=TrakeResponse)
    def search_trake(q: TrakeQuery):
        e = _engine()
        cands = e.search_trake(q.events, max_results=q.max_results)
        return TrakeResponse(events=q.events, count=len(cands), results=[
            TrakeItem(video_id=c.video_id, frame_sequence=list(c.frame_idxs),
                      pts_times=[round(float(t), 3) for t in c.pts_times],
                      score=round(float(c.score), 6))
            for c in cands
        ])

    @app.post("/search/avs", response_model=SearchResponse)
    def search_avs(q: AvsQuery):
        e = _engine()
        results = e.search_avs(q.query, limit=q.limit)
        return SearchResponse(query=q.query, count=len(results),
                              results=[_to_item(e, r, ResultItem) for r in results])

    @app.get("/nearest/{global_id}", response_model=SearchResponse)
    def nearest(global_id: int, k: int = Query(default=60, ge=1, le=500)):
        e = _engine()
        try:
            results = e.nearest(global_id, k=k)
        except (KeyError, IndexError):
            raise HTTPException(404, f"unknown global_id {global_id}") from None
        return SearchResponse(query=f"nearest:{global_id}", count=len(results),
                              results=[_to_item(e, r, ResultItem) for r in results])

    @app.get("/keyframe/{global_id}")
    def keyframe(global_id: int):
        e = _engine()
        try:
            path = e.catalog.ref(int(global_id)).path
        except Exception:  # noqa: BLE001 — any lookup failure is a plain 404
            raise HTTPException(404, "keyframe not found") from None
        return FileResponse(str(path), media_type="image/jpeg")

    return app


def get_app():
    """Uvicorn factory: ``uvicorn cvp.service.app:get_app --factory``."""
    return create_app()


# Module-level ASGI app for the plain `uvicorn cvp.service.app:app` form.
# Built lazily on first attribute access so importing this module stays free
# of FastAPI/engine costs (and works in environments without fastapi).
def __getattr__(name: str):
    if name == "app":
        return create_app()
    raise AttributeError(name)
