# AIC (HCMC AI Challenge) — Competition research report (2026-07-06)

*Compiled via deep web research; drives the design decisions in PROJECT_CONTEXT.md.*

## Format & tasks
- Organized by HCMC DOST + Thành Đoàn (TST); modeled on VBS + LSC (Cathal Gurrin judged the 2024 final). 2026 theme: intelligent assistant for deep multimedia retrieval; traditional + experimental automated (assistant-vs-assistant) formats under discussion.
- Tasks seen 2025 (confirmed in team papers + Codabench): Textual KIS (final: progressive — 5 hints at 1-min intervals, 5 min/query), Video KIS (≤20s clip shown, 4 min, no screen recording), Video QA (answer ≤100 chars; 2025 finals had reasoning questions e.g. solve the math problem shown), TRAKE (frame per event, in order).
- 2026 training session (Buổi 1, 2026-07-05, Nguyễn Hải Đăng, SELab HCMUS) presented four families: KIS, **AVS (ad-hoc: as many matching segments as possible)**, Video QA, **KIS-C conversational** ("điểm nhấn công nghệ 2026", may or may not appear). Direction: agent-driven systems (LLM does query expansion, clarification, tool fusion, reranking, auto-submit).

## Scoring
- **Qualifier (Codabench):** ≤100 ranked rows/query; score = (1/5)·Σ_{k∈{1,5,20,50,100}} [top-k contains correct]. 2025: 3 rounds (23+30+35=88 pts); MERVIN 79/88, LLandMark 77.40/88; cutoff ≈ top-56 of 680+ teams. Max 3 submissions/round, last counts.
- **Final (DRES):** earlier correct = more points; wrong submissions penalized. Reference model (VBS): score = max(0, 100 − 50·t/T − 10·w) — exact AIC constants unpublished.
- Ground truth = **contiguous frame segment**; any frame inside counts. Submit **original-video frame_idx** (from map-keyframes), not keyframe ordinal.

## Dataset (AIC 2025)
- 1,478 videos / 324 h / ~250 GB, 9 Vietnamese TV programs (HTV 60 Giây 665×20min; Món ngon mỗi ngày 498×5min; Bí quyết ôn thi THPT 88×25min; Lan tỏa năng lượng tích cực 96; Lion Dance Cup 43; HTV Cup cycling 25; Tản mạn/Đôi mắt Mê Kông 47; Việt Nam đi là ghiền 16).
- Batch 1 = L21–L30 (keyframes + clip-features-32 ViT-B/32 + map-keyframes + media-info + objects Faster R-CNN Inception-ResNet-V2 / Open Images V4); batch 2 = K01–K20 (videos only).
- 2026 dataset unreleased; trainer speculated up to 1,000+ h; session leaned on lifelog/egocentric examples (possible signal).

## Winners
- 2023 Chicken (PTNK) · 2024 TycheVid (UIT) · 2025 OpenCubee-1 (UIT), 2nd OpenCubee-2, 3rd LunchRetrieval (UIT). 797 teams in 2025.
- Published stacks converge on: TransNetV2/AutoShot keyframes, big CLIP-family encoder(s) + rank fusion, OCR+ASR text channels (Elasticsearch), temporal search UI, LLM query processing.

## AIC 2026 timeline
Launch ~May 2026 · registration to Jun 15 · training Jun–Jul (Buổi 1 = Jul 5; submission spec in Buổi 2+) · qualifiers ~Aug (Codabench, Sundays) · final Sep 12–26 · awards Oct (WISE week).

## Key sources
aichallenge.hochiminhcity.gov.vn · codabench.org/competitions/10187 · arXiv: 2605.23274 (U-CESE), 2606.19682 (Vortex), 2605.16120 (MERVIN), 2603.02888 (LLandMark) · github.com/dres-dev/DRES · full Buổi-1 transcript: aic2026_taphuan_buoi1_transcript.txt
