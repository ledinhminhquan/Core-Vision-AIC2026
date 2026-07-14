"""Typed settings for the whole system, loaded from ``configs/settings.yaml``.

Any key can be overridden per-environment (laptop / Colab / competition
machine) without editing the file, via environment variables of the form::

    CVP_<SECTION>__<KEY>=value        e.g.  CVP_PATHS__DATA_ROOT=/content/data
    CVP_<SECTION>__<SUB>__<KEY>=value e.g.  CVP_SEARCH__WEIGHTS__OCR=0.5

Values are parsed as YAML (so ``true``, ``0.5``, ``[a,b]`` all work).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator

ENV_PREFIX = "CVP_"


# ── Sections ─────────────────────────────────────────────────────────────────


class PathsCfg(BaseModel):
    data_root: Path = Path("./data")
    artifacts_root: Path = Path("./artifacts")
    # Sub-folders of data_root (organiser package names).
    keyframes_dir: str = "keyframes"
    clip_features_dir: str = "clip-features-32"
    map_keyframes_dir: str = "map-keyframes"
    media_info_dir: str = "media-info"
    objects_dir: str = "objects"
    videos_dir: str = "videos"

    def data(self, sub: str) -> Path:
        return self.data_root / sub

    def art(self, *parts: str) -> Path:
        return self.artifacts_root.joinpath(*parts)


class EmbeddingCfg(BaseModel):
    """Primary dense encoder + optional ensemble members."""

    # siglip2 | finetuned | openclip | qwen_embed | mclip | provided_clip32 | ensemble
    model: str = "siglip2"
    siglip2_id: str = "google/siglip2-so400m-patch16-384"
    # English lane. The loader tries `openclip_hub_ids` (open_clip hf-hub checkpoints,
    # strongest first — Meta Perception Encoder per MERVIN/AIC-2025) and falls back to
    # openclip_arch/openclip_pretrained (Apple DFN5B) if a hub download fails.
    openclip_hub_ids: list[str] = Field(default_factory=lambda: [
        "hf-hub:timm/PE-Core-bigG-14-448",
        "hf-hub:timm/PE-Core-L-14-336",
    ])
    openclip_arch: str = "ViT-H-14-378-quickgelu"
    openclip_pretrained: str = "dfn5b"
    # Optional native-Vietnamese MLLM embedding lane (heavy at index time).
    qwen_embed_id: str = "Qwen/Qwen3-VL-Embedding-2B"
    qwen_embed_dim: int = 1024          # MRL truncation (model supports 64–2048)
    qwen_embed_instruction: str = "Represent this news-video keyframe / query for retrieval."
    mclip_id: str = "M-CLIP/XLM-Roberta-Large-Vit-L-14"
    mclip_image_arch: str = "ViT-L-14"
    mclip_image_pretrained: str = "openai"
    # Optional native-multilingual diversity lanes (2026 additions).
    jina_id: str = "jinaai/jina-clip-v2"
    jina_dim: int = 1024               # Matryoshka truncation, valid 64–1024
    metaclip2_id: str = "facebook/metaclip-2-worldwide-huge-quickgelu"
    # Ensemble = late score fusion across per-model FAISS indexes.
    ensemble_members: list[str] = Field(default_factory=lambda: ["siglip2", "openclip"])
    ensemble_weights: list[float] = Field(default_factory=lambda: [0.55, 0.45])
    text_max_length: int = 64
    device: str = "auto"  # auto | cuda | cpu
    dtype: str = "auto"   # auto | bf16 | fp16 | fp32
    batch_size: int = 64


class FinetunedCfg(BaseModel):
    base_id: str = "google/siglip2-so400m-patch16-384"
    checkpoint: Path = Path("./artifacts/checkpoints/vi_siglip2_best")


class IndexCfg(BaseModel):
    type: Literal["flatip", "ivf", "hnsw"] = "flatip"
    ivf_nlist: int = 4096
    ivf_nprobe: int = 32
    hnsw_m: int = 32
    hnsw_ef_search: int = 128
    use_gpu: bool = False


class FusionWeights(BaseModel):
    visual: float = 1.0
    ocr: float = 0.35
    asr: float = 0.30
    caption: float = 0.25
    metadata: float = 0.15
    object: float = 0.25


class SearchCfg(BaseModel):
    topk: int = 500          # dense candidates fetched before fusion
    display_k: int = 120     # results surfaced to the UI / submission
    rerank: bool = True      # SuperGlobal re-ranking of dense candidates
    # Cross-signal fusion: tuned weighted_sum (default, per Bruch et al. TOIS'23)
    # or rrf when score distributions are incomparable / weights untuned.
    fusion_method: Literal["weighted_sum", "rrf"] = "weighted_sum"
    rrf_k: int = 60
    weights: FusionWeights = Field(default_factory=FusionWeights)
    # Temporal context boost: neighbours of strong frames get a small lift.
    neighbor_boost: float = 0.10
    neighbor_window: int = 2
    # Vortex-style before/now/after context boost for "… sau khi …" queries
    # (off by default — measure on the dev pack before enabling).
    temporal_boost: bool = False
    temporal_boost_weight: float = 0.25
    temporal_boost_window: int = 12    # neighbour rows scanned on the context side
    temporal_boost_topk: int = 200     # candidates re-scored (head of the fused map)
    # Batch/auto-track: when the ranking looks flat (low confidence), re-search
    # the cached enhanced/expansion texts VERBATIM (engine.search_prepared —
    # bypasses the query processor, so no extra API calls) and RRF-merge.
    # Cost when triggered: up to 3 extra dense searches on flagged queries.
    low_confidence_retry: bool = False
    low_confidence_threshold: float = 0.25
    # Optional PAIRWISE cross-encoder rerank of the fused head (Unified-IMMR
    # 76.4/88 AIC-2025 recipe). Runs BEFORE the listwise VLM rerank; both are
    # off by default (latency). qwen_reranker = Qwen3-VL-Reranker (Jan 2026).
    reranker: Literal["none", "blip2_itm", "qwen_reranker"] = "none"
    rerank_topk: int = 100
    rerank_weight: float = 0.5           # blend: (1-w)·fused + w·cross (both min-max)
    rerank_batch_size: int = 8
    blip2_itm_id: str = "Salesforce/blip2-itm-vit-g"
    qwen_reranker_id: str = "Qwen/Qwen3-VL-Reranker-2B"
    # Optional listwise VLM re-rank of the head of the ranking (UIT CVPRW'25: +10% H@1).
    vlm_rerank: bool = False
    vlm_rerank_topk: int = 24
    vlm_rerank_provider: str = "gemini"   # gemini | vintern | none
    # AVS diversification: MMR trade-off between relevance and novelty.
    avs_mmr_lambda: float = 0.7
    avs_per_video_cap: int = 3
    avs_min_gap_s: float = 10.0


class QueryCfg(BaseModel):
    """Vietnamese query understanding: translate + visually re-describe + expand."""

    provider: str = "gemini"          # none | google | gemini
    # Stable mid-2026 default; fallbacks cover preview retirement / regional
    # gaps so a stale id degrades to the next Gemini model, not to Translate.
    gemini_model: str = "gemini-3.5-flash"
    gemini_model_fallbacks: list[str] = Field(default_factory=lambda: [
        "gemini-3-flash-preview", "gemini-2.5-flash",
    ])
    enhance: bool = True              # rewrite as concrete visual description
    enhance_english: bool = True      # also enhance pure-English queries
    expansions: int = 2              # extra paraphrase queries for multi-query fusion
    multi_query_agg: str = "max"     # max | mean over expanded queries
    cache: bool = True
    timeout_s: float = 8.0


class TemporalCfg(BaseModel):
    """TRAKE: ordered multi-event search within one video (DP over keyframes)."""

    algo: Literal["dante", "beam"] = "dante"  # dante = O(N·T) running-max DP (SOICT'25)
    use_ensemble: bool = True    # score events with the full ensemble, not just the primary
    per_event_topk: int = 100
    max_gap_s: float = 150.0     # max seconds between consecutive events
    min_gap_s: float = 0.0
    sim_floor: float = 0.10
    beam_size: int = 8
    max_videos: int = 30         # videos considered (pooled from per-event hits)
    # DANTE-style soft gap penalty: transitions lose gap_penalty_per_s * Δt,
    # so tighter event chains win ties (λ≈0.001–0.01 per keyframe in the paper).
    gap_penalty_per_s: float = 0.0005
    # Organiser TRAKE files carry a context/header line before the "E1:…"
    # events; "prepend" merges that header into every event text (A/B knob —
    # extra video-level context vs diluted event specificity). Default: drop.
    event_context: Literal["none", "prepend"] = "none"


class ExtractionCfg(BaseModel):
    """K-batch keyframe self-extraction (round-3 enhancement: density knob).

    UIT's CVPRW'25 ablation credits DENSER keyframes with +60–90% H@1 on hard
    queries — add positions (e.g. ``[0.1, 0.3, 0.5, 0.7, 0.9]``) to trade disk
    and embed time for recall on the self-extracted batches.
    """

    shot_positions: list[float] = Field(default_factory=lambda: [0.15, 0.50, 0.85])
    dedup_mad_threshold: float = Field(default=6.0, ge=0.0)

    @field_validator("shot_positions")
    @classmethod
    def _positions_valid(cls, v: list[float]) -> list[float]:
        # A misconfigured knob must fail LOUDLY, not silently revert to the
        # defaults (round-4 fix): empty list / out-of-range values are typos.
        if not v:
            raise ValueError("extraction.shot_positions must not be empty")
        if any(not (0.0 < p < 1.0) for p in v):
            raise ValueError(f"extraction.shot_positions must be within (0, 1): {v}")
        return v


class OcrCfg(BaseModel):
    engine: str = "easyocr"  # easyocr | paddle | none
    langs: list[str] = Field(default_factory=lambda: ["vi", "en"])
    min_confidence: float = 0.30


class AsrCfg(BaseModel):
    model: str = "vinai/PhoWhisper-medium"
    chunk_length_s: int = 30
    batch_size: int = 8


class CaptionCfg(BaseModel):
    model: str = "5CD-AI/Vintern-1B-v3_5"
    max_new_tokens: int = 96
    max_tiles: int = 6


class VqaCfg(BaseModel):
    provider: str = "gemini"          # gemini | vintern | none
    gemini_model: str = "gemini-3.5-flash"
    local_model: str = "5CD-AI/Vintern-1B-v3_5"
    top_frames: int = 5               # frames sent to the VQA model per answer group
    # Frames per answer_group strip (ONE Gemini call sees the whole strip).
    # 1 = old single-frame behaviour; 3 covers text that spans several frames.
    frames_per_answer: int = 3
    # Batch/auto mode: answer the top-N distinct candidate groups instead of writing
    # one answer on every row (VQA R-Score needs the *right* row to carry the right answer).
    answers_per_query: int = 5
    max_calls_per_query: int = 5


class SubmissionCfg(BaseModel):
    """Packaging + (finals) DRES-style submission endpoint."""

    package_name: str = "submission"     # folder name inside the Codabench zip
    # DRES endpoint for the on-site finals; credentials via env DRES_USER / DRES_PASSWORD.
    dres_base_url: str = ""              # e.g. https://dres.example.org/api/v2
    dres_timeout_s: float = 6.0
    auto_submit: bool = False            # automatic track: submit without confirmation


class LoggingCfg(BaseModel):
    level: str = "INFO"


class Settings(BaseModel):
    project_name: str = "core-vision-perfect-v1"
    paths: PathsCfg = Field(default_factory=PathsCfg)
    embedding: EmbeddingCfg = Field(default_factory=EmbeddingCfg)
    finetuned: FinetunedCfg = Field(default_factory=FinetunedCfg)
    index: IndexCfg = Field(default_factory=IndexCfg)
    search: SearchCfg = Field(default_factory=SearchCfg)
    query: QueryCfg = Field(default_factory=QueryCfg)
    temporal: TemporalCfg = Field(default_factory=TemporalCfg)
    extraction: ExtractionCfg = Field(default_factory=ExtractionCfg)
    ocr: OcrCfg = Field(default_factory=OcrCfg)
    asr: AsrCfg = Field(default_factory=AsrCfg)
    caption: CaptionCfg = Field(default_factory=CaptionCfg)
    vqa: VqaCfg = Field(default_factory=VqaCfg)
    submission: SubmissionCfg = Field(default_factory=SubmissionCfg)
    logging: LoggingCfg = Field(default_factory=LoggingCfg)


# ── Loader ───────────────────────────────────────────────────────────────────


def _deep_set(d: dict, keys: list[str], value: Any) -> None:
    cur = d
    for k in keys[:-1]:
        nxt = cur.get(k)
        if not isinstance(nxt, dict):  # missing or scalar in the way — replace IN PARENT
            nxt = {}
            cur[k] = nxt
        cur = nxt
    cur[keys[-1]] = value


def _apply_env_overrides(raw: dict) -> dict:
    for env_key, env_val in os.environ.items():
        if not env_key.startswith(ENV_PREFIX):
            continue
        path = env_key[len(ENV_PREFIX):].lower().split("__")
        if len(path) < 2:
            continue
        try:
            value = yaml.safe_load(env_val)
        except yaml.YAMLError:
            value = env_val
        _deep_set(raw, path, value)
    return raw


def find_settings_file(explicit: str | os.PathLike | None = None) -> Path | None:
    """Locate settings.yaml: explicit arg > $CVP_SETTINGS > repo configs/ > cwd."""
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    if os.environ.get("CVP_SETTINGS"):
        candidates.append(Path(os.environ["CVP_SETTINGS"]))
    here = Path(__file__).resolve()
    for parent in [here.parents[2], here.parents[3] if len(here.parents) > 3 else here.parents[2]]:
        candidates.append(parent / "configs" / "settings.yaml")
    candidates.append(Path.cwd() / "configs" / "settings.yaml")
    for c in candidates:
        if c and c.is_file():
            return c
    return None


def load_settings(path: str | os.PathLike | None = None) -> Settings:
    """Load YAML settings, apply CVP_* env overrides, return typed Settings."""
    f = find_settings_file(path)
    raw: dict = {}
    if f is not None:
        raw = yaml.safe_load(f.read_text(encoding="utf-8")) or {}
    raw = _apply_env_overrides(raw)
    return Settings.model_validate(raw)
