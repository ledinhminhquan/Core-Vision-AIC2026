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
from pydantic import BaseModel, Field, field_validator, model_validator

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

    # siglip2 | finetuned | openclip | qwen_embed | mclip | jina | metaclip2 | provided_clip32 | ensemble
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
    # Round-80: mặc định 8B (nghiên cứu 29/08 — MMEB-V2 77.8 vs 73.2 của 2B);
    # lane là optional nên đổi default không đụng trận 2-lane hiện tại.
    qwen_embed_id: str = "Qwen/Qwen3-VL-Embedding-8B"
    qwen_embed_dim: int = 1536          # MRL truncation (8B supports 64–4096)
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

    @field_validator("ensemble_weights")
    @classmethod
    def _weights_match_members(cls, v: list[float], info) -> list[float]:
        # Fail LOUD on a config typo: zip() in the engine would silently drop
        # the unmatched tail member otherwise (review C17).
        members = info.data.get("ensemble_members")
        if members is not None and len(v) != len(members):
            raise ValueError(
                f"embedding.ensemble_weights has {len(v)} entries but "
                f"ensemble_members has {len(members)} — they must match 1:1"
            )
        return v

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

    @model_validator(mode="after")
    def _sane_weights(self) -> "FusionWeights":
        # All-zero weights silently zeroed EVERY ranking (round-10 config
        # matrix): weighted_sum skips zero-weight maps, so nothing survived
        # and no CSV was written — a config typo must fail loud instead.
        vals = [self.visual, self.ocr, self.asr, self.caption, self.metadata, self.object]
        if any(v < 0 for v in vals):
            raise ValueError(f"fusion weights must be >= 0, got {vals}")
        if all(v <= 0 for v in vals):
            raise ValueError("ALL fusion weights are 0 — every query would return nothing")
        return self


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
    # Second-pass consistency boost (round-73, Nhiệm vụ C đợt 2): candidates
    # whose NEIGHBOUR frames also score high get a lift — true peaks are
    # plateaus, noise is a lone spike. Runs on the RAW fused map BEFORE
    # neighbor_boost (which would fake plateaus by spreading spikes). KIS/QA/AVS
    # ranking only (search_trake has its own path). 0.0 = off, bit-identical.
    neighbor_consistency_boost: float = 0.0
    neighbor_consistency_window: int = 2
    neighbor_consistency_decay: float = Field(0.5, gt=0.0, le=1.0)
    # Submission row budget (round-74, Nhiệm vụ D đợt 2): legacy = dump the
    # ranking (old behaviour); diversify_tail = keep the head verbatim, then
    # spend the tail budget on neighbour-frame variants around the head's
    # distinct videos (TRAKE-jitter philosophy — the metric maxes over rows,
    # so diluted rank-40+ rows are better spent as targeted lottery tickets).
    # KIS + QA only (AVS has its own MMR diversity; TRAKE has jitter).
    row_strategy: Literal["legacy", "diversify_tail"] = "legacy"
    row_strategy_head: int = Field(30, ge=1)       # rows kept verbatim
    row_strategy_variants: int = Field(4, ge=1, le=8)  # variants per anchor video
    # Vortex-style before/now/after context boost for "… sau khi …" queries
    # (off by default — measure on the dev pack before enabling).
    temporal_boost: bool = False
    temporal_boost_weight: float = 0.25
    temporal_boost_window: int = 12    # neighbour rows scanned on the context side
    temporal_boost_topk: int = 200     # candidates re-scored (head of the fused map)
    # Batch/auto-track: when the ranking looks flat (low confidence), re-search
    # the cached enhanced/expansion texts VERBATIM (engine.search_prepared —
    # bypasses the query processor, so no extra API calls) and RRF-merge.
    # Cost when triggered: up to 3 extra dense searches on flagged queries
    # (the alts skip the optional cross/VLM rerank stack — round-10).
    low_confidence_retry: bool = False
    low_confidence_threshold: float = 0.25
    # Optional PAIRWISE cross-encoder rerank of the fused head (Unified-IMMR
    # 76.4/88 AIC-2025 recipe). Runs BEFORE the listwise VLM rerank; both are
    # off by default (latency). qwen_reranker = Qwen3-VL-Reranker (Jan 2026).
    reranker: Literal["none", "blip2_itm", "qwen_reranker"] = "none"
    rerank_topk: int = 100
    rerank_weight: float = Field(0.5, ge=0.0, le=1.0)  # blend: (1-w)·fused + w·cross
    rerank_batch_size: int = 8
    blip2_itm_id: str = "Salesforce/blip2-itm-vit-g"
    # Round-41: the 8B sibling scores 80.7 vs the 2B's 73.8 on MMEB-v2 image
    # retrieval (same Apache-2.0 stack, ~18GB BF16 — comfortable on A100 80GB).
    qwen_reranker_id: str = "Qwen/Qwen3-VL-Reranker-8B"
    # Optional listwise VLM re-rank of the head of the ranking (UIT CVPRW'25: +10% H@1).
    vlm_rerank: bool = False
    vlm_rerank_topk: int = 24
    vlm_rerank_provider: str = "gemini"   # gemini | vintern | none
    # Round-45: the listwise scorer gets its own CHEAP model ($0.30/$2.50,
    # thinking defaults to minimal) — this call family was ~80% of the live
    # bill on gemini-3.5-flash defaults. Empty = follow vqa.gemini_model.
    vlm_rerank_model: str = "gemini-3.5-flash-lite"
    # Round-40 test-time compute: call the VLM N times and AVERAGE the score
    # vectors. Evidence: two same-config live runs scored 9.4 vs 9.0 purely on
    # single-call sampling noise — averaging trades API calls for stability.
    vlm_rerank_votes: int = Field(1, ge=1, le=5)
    # AVS diversification: MMR trade-off between relevance and novelty.
    avs_mmr_lambda: float = 0.7
    avs_per_video_cap: int = 3
    avs_min_gap_s: float = 10.0


class QueryCfg(BaseModel):
    """Vietnamese query understanding: translate + visually re-describe + expand."""

    provider: str = "gemini"          # none | google | gemini
    # Round-41 (researched 22/08/2026): gemini-3.7-flash is the CURRENT Flash
    # line — newer than 3.5-flash and half its price through 2026 ($0.75/$3.75
    # promo vs $1.50/$9). Battle-proven 3.5-flash stays first fallback; the
    # "-latest" tail can't retire (pinned ids 404 for new users — the old 2.5
    # pin died that way live 20/08).
    # Round-45: translation/enhancement is an easy text task — the Lite tier
    # ($0.30/$2.50, thinking already minimal) does it at ~1/10 the old cost.
    gemini_model: str = "gemini-3.5-flash-lite"
    gemini_model_fallbacks: list[str] = Field(default_factory=lambda: [
        "gemini-3.7-flash", "gemini-3.5-flash", "gemini-flash-latest",
    ])
    enhance: bool = True              # rewrite as concrete visual description
    enhance_english: bool = True      # also enhance pure-English queries
    expansions: int = 2              # extra paraphrase queries for multi-query fusion
    multi_query_agg: Literal["max", "mean"] = "max"  # over expanded queries (typo = loud)
    cache: bool = True
    # The Gemini API rejects client deadlines under 10s (400 INVALID_ARGUMENT,
    # live nb03 gemini A/B) — keep this ≥10 or every call fails before running.
    timeout_s: float = 15.0


class TemporalCfg(BaseModel):
    """TRAKE: ordered multi-event search within one video (DP over keyframes)."""

    algo: Literal["dante", "beam"] = "dante"  # dante = O(N·T) running-max DP (SOICT'25)
    use_ensemble: bool = True    # score events with the full ensemble, not just the primary
    per_event_topk: int = 100
    max_gap_s: float = 150.0     # max seconds between consecutive events
    min_gap_s: float = 0.0

    @model_validator(mode="after")
    def _sane_gaps(self) -> "TemporalCfg":
        # min > max makes the DP window empty → EVERY TRAKE query returns 0
        # candidates with an error that reads like a data problem (round-10).
        if self.min_gap_s < 0:
            raise ValueError(f"temporal.min_gap_s must be >= 0, got {self.min_gap_s}")
        if self.min_gap_s > self.max_gap_s:
            raise ValueError(
                f"temporal.min_gap_s ({self.min_gap_s}) > max_gap_s ({self.max_gap_s}) "
                "— the DP gap window would be empty and every TRAKE query would die"
            )
        return self
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
    # ── TRAKE-upgrade knobs (session 2026-08-27, see report.md) ─────────────
    # Every default reproduces the OLD behaviour exactly — flip one at a time
    # on the Colab bench (bench-before-adopt).
    #
    # "all": encode EVERY cached query-processor variant per event (original +
    # enhanced + translation + expansions — KIS parity) and max-fuse the
    # per-variant similarities per event; "original" = the single raw text.
    event_query_variants: Literal["original", "all"] = "original"
    # "prepend": the video-pooling stage searches "<header>. <event>" texts
    # (multilingual lanes only) while the per-video DP keeps scoring the bare
    # event texts — video-level context helps pick videos without diluting
    # the event alignment. Needs the runner to pass the query header through.
    # Do NOT combine with event_context: prepend (events would already carry
    # the header → pooling text doubles it; harmless but diluted).
    pool_context: Literal["none", "prepend"] = "none"
    # > 0: blend per-event caption-BM25 scores (dense Vintern captions) into
    # the DP similarity matrix: sim += w · minmax-per-event(caption BM25 over
    # the pooled videos). 0 = off. Degrades to no-op when caption artifacts
    # are absent.
    caption_signal_weight: float = Field(0.0, ge=0.0)
    # "jitter": keep the top rows as-is, then densify the 100-row budget with
    # frame variants around the best per-video chains (neighbouring keyframes
    # + midpoints BETWEEN keyframes, early-biased — TRAKE asks for the FIRST
    # moment). The official metric takes the best row per cutoff, so extra
    # rows around strong candidates are free lottery tickets.
    submit_strategy: Literal["legacy", "jitter"] = "legacy"
    jitter_videos: int = Field(4, ge=1)  # distinct top videos expanded by jitter


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
    gemini_model: str = "gemini-3.7-flash"
    # Round-41 "nghiền ngẫm": the QA-track answer path (strip-VQA) runs on the
    # strongest callable Pro (gemini-3.5-pro is still a closed Vertex preview
    # as of Aug 2026). Pro thinks longer → its own wall timeout below. Empty
    # string = use gemini_model (old behaviour). The chain still degrades
    # Pro → 3.7-flash → 3.5-flash → flash-latest on failure.
    answer_model: str = "gemini-3.1-pro-preview"
    answer_timeout_s: float = 90.0
    local_model: str = "5CD-AI/Vintern-1B-v3_5"
    # Round-79: chọn backend fallback local. "vintern" = đường cũ y nguyên;
    # "hf_auto" = VLM chat-template bất kỳ (Qwen3.5-class) — id khai ở dưới,
    # nạp lười CHỈ khi Gemini sập. Đổi backend không đụng đường Gemini.
    local_backend: Literal["vintern", "hf_auto"] = "vintern"
    local_hf_id: str = ""            # vd "Qwen/Qwen3.5-9B-Instruct" sau khi kiểm chứng id
    top_frames: int = 5               # frames sent to the VQA model per answer group
    # Frames per answer_group strip (ONE Gemini call sees the whole strip).
    # 1 = old single-frame behaviour; 3 covers text that spans several frames.
    frames_per_answer: int = 3
    # Round-40: ask the strip-VQA N times, keep the MAJORITY answer. Evidence:
    # live q3 flip-flopped '300 kg' ↔ '30 kg' between two same-config runs —
    # one sample is a coin toss, three votes pick the stable reading.
    self_consistency: int = Field(1, ge=1, le=5)
    # Batch/auto mode: answer the top-N distinct candidate groups instead of writing
    # one answer on every row (VQA R-Score needs the *right* row to carry the right answer).
    answers_per_query: int = 5
    max_calls_per_query: int = 5
    # ── QA overhaul đợt 2 (round-73) — all three default OFF = old behaviour ──
    # Vote by EQUIVALENCE CLASS instead of raw casefolded string: "2"/"hai"/"02"
    # pool their ballots ("300 kg"/"300kg" likewise) — see cvp.search.answer_norm.
    answer_canonicalize: bool = False
    # >0: per answered group, ask up to N EXTRA neighbour strips around the
    # moment and vote across ALL ballots (a slightly-off frame still converges
    # on the right answer). Extra strips consume max_calls_per_query — raise it
    # (e.g. 15) or the knob has no budget and WARNS loudly instead of silently
    # doing nothing.
    answer_neighbor_frames: int = Field(0, ge=0, le=4)
    # Promote the row-block of a candidate group whose ballots are UNANIMOUS
    # when the top group's ballots disagree (an unstable top answer usually
    # means the wrong moment or an unreadable frame).
    consistency_rerank: bool = False
    # Round-77: nộp thêm dòng song-định-dạng số↔chữ trên cùng (video, frame)
    # cho các answer có số ("sáu"↔"6") — thay các dòng ĐUÔI cùng số lượng,
    # đầu bảng không đổi. BTC chấm exact text nên đây là bảo hiểm định dạng.
    answer_variant_rows: bool = False
    # Round-77 (bài học q19 sơ tuyển 2: đáp án lệch GT đúng MỘT chữ): khi bật,
    # prompt QA yêu cầu CHÉP NGUYÊN VĂN chữ hiển thị thay vì diễn đạt lại.
    exact_transcription: bool = False
    # Round-82 (đêm 28/08: QA ăn ~2/3 thời gian pack vì 5 nhóm × 10 strip gọi
    # Gemini TUẦN TỰ): số nhóm ứng viên được hỏi SONG SONG. Kết quả áp theo
    # đúng thứ tự nhóm nên bit-identical với 1; chỉ đổi thời gian + áp lực RPM.
    parallel_calls: int = Field(1, ge=1, le=8)


class SubmissionCfg(BaseModel):
    """Packaging + (finals) DRES-style submission endpoint."""

    package_name: str = "submission"     # folder name inside the Codabench zip
    # DRES endpoint for the on-site finals; credentials via env DRES_USER / DRES_PASSWORD.
    dres_base_url: str = ""              # e.g. https://dres.example.org
    # DRES v2 submits to /api/v2/submit/{evaluationId} — the id is announced at
    # the finals; without it the client posts to the id-less legacy path (C19).
    dres_evaluation_id: str = ""
    dres_timeout_s: float = 6.0
    auto_submit: bool = False            # automatic track: submit without confirmation
    # Round-77 (bài học đêm 28/08: bão 504 kéo pack từ 60' lên 142' và đề đóng
    # cửa khi còn 3 câu): quá mốc này (phút, 0 = tắt) run_auto BẬT chế độ nước
    # rút cho các câu còn lại — QA votes 1, tắt neighbor strips, tắt VLM rerank
    # (giữ retrieval + cross-rerank local) — và la lớn trong log.
    pack_deadline_min: float = Field(0.0, ge=0.0)


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
