> **Historical working notes, superseded by the submitted papers.** Figures here may be
> mid-development (the test suite has since grown to 970 `test_` functions in 104 files) and
> any reading of other teams' results is unsourced and NOT asserted by the papers.
>
> *Ghi chú làm việc cũ, đã bị thay thế bởi các bài báo đã nộp. Số liệu có thể là giữa chặng, và mọi
> nhận định về kết quả của đội khác đều chưa truy được nguồn và KHÔNG được khẳng định trong bài.*

# SOTA methods & models — research report (2026-07-06)

*Evidence base for the architecture. Exact model IDs verified on HF as of mid-2026.*

## The six papers from the organisers' training deck
1. **GRAB** (CVPRw'25, arXiv 2504.09298) — BEiT-3; **SuperGlobal reranking** (GeM; S=(S1+S2)/2); ABTS bidirectional temporal search; TransNetV2 4kf/shot; pHash dedup.
2. **Multi-Granularity + Temporal Reranking** (CVPRw'25, 2504.08384) — OpenCLIP (coarse) + BEiT-3 (fine), top-50/model, score/max normalize, weighted sum; neighbor-score temporal reranking; cosine dedup >0.9.
3. **LLandMark** (AAAIw'26, 2603.02888) — 4-stage multi-agent; ConvNeXt-XXLarge (laion2B); landmark→visual-description rewriting + web-image image-to-image; WhisperX, PaddleOCR+Gemini; weakest-link min-aggregation for multi-step. **77.40/88**.
4. **Unified-IMMR** (AAAIw'26, 2512.12935) — BEiT-3+SigLIP in Qdrant, SRRF fusion; BLIP-2 ITM rerank top-100; TRAKE beam (width 8) with e^(−0.01Δt) decay; GPT-4o 4 query variants. **76.4/88**.
5. **ISTA/DANTE** (SOICT'25, 2512.13169) — BEiT-3+Milvus, Gemini OCR→Elasticsearch(vi plugin); QUEST rewriting + web-image lane; **DANTE DP: DP[i,t]=S[i,t]+max_τ(DP[i−1,τ]−λ(t−τ))**, λ=0.001–0.01, O(N·T).
6. **MADTempo** (SOICT'25, 2512.12929) — CLIP-LAION, Milvus; TRAKE = events + context, FinalScore=α·EventScore+(1−α)·LLM ContextScore; Vintern-1B OCR, PhoWhisper ASR; Google-Image query augmentation.

Also: **MERVIN** (2605.16120): **PE-Core-bigG-14-448 beat CLIP ViT-H/14** (COCO T2I 58.1 vs ~49.5); TransNetV2 3kf/shot @0.15/0.50/0.85; runs on RTX 3060; **79/88**. **Vortex** (2606.19682): CLIP+SigLIP2 RRF, AutoShot, Qwen2.5-VL captions, **Rocchio relevance feedback**.

## Embedding models (exact IDs)
| Model | Dim | COCO T2I R@1 | vi? | Notes |
|---|---|---|---|---|
| google/siglip2-so400m-patch16-384 | 1152 | ~54-55 | partial (multilingual WebLI) | best quality/VRAM; our default |
| facebook/PE-Core-G14-448 (timm/vit_pe_core_gigantic_patch14_448.fb) | 1280 | **58.1** | no | open SOTA; MERVIN's pick |
| apple/DFN5B-CLIP-ViT-H-14-378 (open_clip ViT-H-14-378-quickgelu/dfn5b) | 1024 | ~54 | no | our English lane |
| laion/CLIP-ViT-H-14-laion2B | 1024 | 49.5 | no | legacy strong |
| jinaai/jina-clip-v2 | 1024 | mid | **yes (89 langs)** | CC-BY-NC — research only |
| BEiT-3 (microsoft/unilm, COCO-ft) | 1024 | 61–67 (ft) | no | the AIC fine-grained favorite |

**Key decision:** ALL winning systems run English encoders + LLM translate/rewrite (Gemini/GPT) — multilingual towers lose 15–25 R@1 on Vietnamese (ViCLIP-OT baselines: SigLIP 34.75 vs ViSigLIP-OT 39.19 on UIT-OpenViIC). ⇒ dual-lane: translated→English-lane + native-vi lane (SigLIP2 / LiT-tuned), fused.

## Vietnamese components
- Translate: Gemini 2.5 Flash (free ~15RPM/1.5k RPD — verify), VietAI/envit5-translation (open), NLLB-200.
- OCR: PaddleOCR-det + VietOCR-rec; PP-OCRv5 multilingual; VLM-OCR via 5CD-AI/Vintern-1B-v3_5 (0.94B MIT) or Gemini (what winners used).
- ASR: vinai/PhoWhisper-large/-medium (BSD-3, SOTA vi WER); faster-whisper CT2 conversions ~4×.
- VLM: Vintern-1B-v3_5, Qwen/Qwen2.5-VL-7B-Instruct, OpenGVLab/InternVL3-8B, Gemini Flash.

## Techniques adopted
- SuperGlobal rerank (ICCV'23, 2308.06954): +3.7–7.1 pts on ROxford at ~0 cost.
- RRF across heterogeneous lanes; tuned weighted fusion within homogeneous lanes.
- LLM query expansion (~4 variants) + web-image image-to-image fallback for OOD entities.
- Rocchio feedback (α=1, β=0.75, γ=0.15).
- FAISS: 3M×1152 = 13.8 GB fp32 — flat exact on GPU/H100; IVF16384 or OPQ+IVFPQ+rerank for laptop.
- LiT (CVPR'22) + LoRA-on-text-tower for vi tuning; keep sigmoid-loss learnable scale/bias from pretrained; video-level splits. Precedents: PhoCLIP, ViCLIP-OT (2602.22678).
