"""Pydantic request/response schemas for the retrieval service."""

from __future__ import annotations

from pydantic import BaseModel, Field


class TextQuery(BaseModel):
    query: str = Field(..., description="Vietnamese (or English) text query")
    # Bounded (review finding C22): an unbounded topk allocates
    # n_variants × topk result arrays per FAISS lane — a fat-fingered client
    # posting 10**9 would OOM the service mid-competition.
    topk: int | None = Field(default=None, ge=1, le=5000)
    display_k: int | None = Field(default=None, ge=1, le=1000)


class QaQuery(BaseModel):
    query: str = Field(..., description="Scene description used for retrieval")
    question: str | None = Field(
        default=None,
        description="The actual question; defaults to `query` when omitted")
    display_k: int | None = Field(default=None, ge=1, le=1000)
    answers: bool = Field(
        default=True, description="Run VQA per candidate group (needs a provider)")


class ImageQuery(BaseModel):
    """Query-by-image — the finals KIS-V path (clip may only be WATCHED, so the
    team re-describes/sketches/generates an image and feeds it here)."""

    image_b64: str = Field(..., description="Base64 image bytes (raw or data: URL)")
    display_k: int | None = Field(default=None, ge=1, le=1000)


class TrakeQuery(BaseModel):
    events: list[str] = Field(..., min_length=1,
                              description="Ordered event descriptions E1..Ek")
    max_results: int = 100


class AvsQuery(BaseModel):
    query: str
    limit: int = Field(default=100, ge=1, le=100,
                       description="max rows (submission cap is 100)")


class ResultItem(BaseModel):
    global_id: int
    video_id: str
    n: int
    frame_idx: int
    pts_time: float
    score: float
    signals: dict[str, float] = Field(default_factory=dict)
    answer: str | None = None
    image_url: str
    watch_url: str | None = None


class SearchResponse(BaseModel):
    query: str
    count: int
    results: list[ResultItem]


class TrakeItem(BaseModel):
    video_id: str
    frame_sequence: list[int]
    pts_times: list[float]
    score: float


class TrakeResponse(BaseModel):
    events: list[str]
    count: int
    results: list[TrakeItem]
