# 📝 PAPER_NOTES — Định vị bài báo (SOICT 2026 / Multimedia Tools & Applications)

> BTC AIC 2026 hỗ trợ công bố tại **SOICT 2026** và **Multimedia Tools and Applications**
> cho các đội top. UI/hệ thống có thể công bố bất kể thứ hạng (BTC xác nhận tại tập huấn 1).
> File này gom sẵn: đóng góp có thể nhấn, ablation cần chạy, bảng số liệu cần thu, related work.

## 1. Tựa đề gợi ý (chọn 1, chỉnh theo kết quả)

- *Core-Vision: Ensemble Keyframe Retrieval with Candidate-Restricted Multimodal Fusion
  and Exact Temporal Alignment for Vietnamese News Video*
- *Winning-Recipe Engineering for Interactive Video Retrieval: A Reproducible System for
  the HCMC AI Challenge 2026*

## 2. Đóng góp có thể claim (theo thứ tự mạnh → yếu)

1. **Corpus-specific LoRA-LiT + WiSE-FT cho tiếng Việt, tự giám sát bằng VLM captions**
   — không cần nhãn người: Vintern-1B caption chính corpus thi → LiT (image tower đóng băng
   nên FAISS index bất biến) + anchor tiếng Anh 25% + distill teacher + WiSE-FT α-sweep.
   *Điểm mới:* pipeline tự cải thiện trên đúng phân phối dữ liệu thi; ablation vs zero-shot.
2. **SuperGlobal đa biến thể truy vấn (max-fused)** — chứng minh mismatch giữa rerank
   1-biến-thể và dense max-fusion nhiều-biến-thể làm tụt hit của biến thể mở rộng;
   fix gần như miễn phí. *Đo:* R@1/R@5 có/không multi-variant trên bộ đề practice.
3. **BM25 candidate-restricted persisted index** cho fusion đa tín hiệu tương tác
   (O(K·|q|) thay vì O(corpus)) — hệ tương tác <1s/query kể cả truy vấn đầu tiên.
4. **DANTE-DP ensemble cho TRAKE** — sim ma trận = tổng có trọng số nhiều encoder
   (min-max theo sự kiện, degrade khi thiếu lane); so beam-DP vs DANTE vs MERVIN-heuristic.
5. **MMR mức embedding cho AVS xuyên video** — tin tức tái sử dụng b-roll; đo unique-segment
   coverage trên bộ đề AVS.
6. **Thể thức tự động 2026**: kiến trúc auto-agent (task-route → retrieve → group-VQA →
   validate → submit) — mô tả hệ thống, đóng góp kỹ nghệ.

**🆕 Đóng góp bổ sung Perfect-V1 (07/2026):**

7. **Tầng rerank cross-encoder pairwise** (`search.reranker`): BLIP-2 ITM
   (blueprint Unified-IMMR 76.4/88) HOẶC **Qwen3-VL-Reranker-2B** (model 01/2026)
   — theo khảo sát của ta là **lần dùng đầu tiên trong thi đấu AIC/VBS**; blend
   `(1−w)·fused + w·cross` sau min-max, đo bằng A9.
8. **Multi-frame VQA strips** (`vqa.frames_per_answer`): mỗi nhóm ứng viên gửi
   DẢI frame (top ± láng giềng) trong 1 call — fix lớp câu QA "diễn tiến/giải
   toán trong video" mà single-frame VQA 2025 trả lời sai.
9. **Temporal-context boost deterministic** (`search.temporal_boost`): tách
   target/context bằng regex marker tiếng Việt ("sau khi"/"trước khi", không gọi
   LLM), cộng max-cos láng giềng đúng hướng thời gian (công thức Vortex) — đo A10.
10. **Confidence-gated reformulation retry** (`search.low_confidence_retry`):
    track tự động tự phát hiện ranking phẳng (độ tách top-1 vs median dưới
    threshold) → re-search các expansion đã cache + RRF-merge; mọi đường lỗi
    trả về ranking gốc (fail-open).
11. **Kiến trúc hai-track dùng chung engine**: một `SearchEngine` phục vụ cả app
    tương tác lẫn HTTP service (`cvp serve`: /health, /search/{text,image,qa,trake,avs},
    /nearest, /keyframe) — nền tảng máy-gọi-được cho thể thức assistant-vs-assistant.

## 3. Ablation bắt buộc phải chạy (khi có data + GT)

| # | Ablation | Metric | Script |
|---|---|---|---|
| A1 | siglip2 đơn vs +PE-Core ensemble vs +finetuned | Mean R@k Final | `20_run_queries` + `40_eval_official` |
| A2 | SuperGlobal off / 1-variant / multi-variant | Final, R@1 | env `CVP_SEARCH__RERANK` |
| A3 | fusion: visual-only / +BM25 / +object / tuned weights / RRF | Final | `21_tune_weights` báo cáo sẵn deltas |
| A4 | TRAKE: beam vs dante vs dante-ensemble | TRAKE R-Score | `CVP_TEMPORAL__ALGO` |
| A5 | zero-shot vs LoRA-LiT vs +WiSE-FT (+anchor/hardneg on/off) | val R@1/5/10 + Final | nb02 tự in bảng |
| A6 | VLM rerank on/off (Gemini vs Vintern) | R@1, latency | `CVP_SEARCH__VLM_RERANK` |
| A7 | AVS greedy vs MMR (λ sweep) | #video phủ đúng | `search_avs` |
| A8 | QA: 1 answer chung vs per-group answers | QA R-Score | so 2 chế độ run_queries |
| A9 | cross-rerank: none vs blip2_itm vs qwen_reranker | Final, R@1, latency | `CVP_SEARCH__RERANKER` |
| A10 | temporal_boost off/on + low_confidence_retry on | Final (nhóm câu "sau khi/trước khi") | `CVP_SEARCH__TEMPORAL_BOOST`, `CVP_SEARCH__LOW_CONFIDENCE_RETRY` |

Mỗi ablation chỉ là biến env/config — không sửa code. Ghi kết quả vào bảng này luôn.

🆕 **Một lệnh chạy cả battery** (trừ A5 — bảng riêng trong nb02; A3 full sweep —
`scripts/21`): `python scripts/26_run_ablations.py --query-dir queries/dev
--gt queries/dev/gt.json [--only A9 A10]` — mỗi variant chạy pack qua engine,
chấm bằng scorer chính thức, in + lưu `artifacts/ablations/ablation_results.json`.

## 4. Số liệu hệ thống cần thu để viết phần Experiments

- Kích thước corpus 2026 (videos/giờ/keyframes), thời gian build từng stage trên Colab
  (embed/lane, OCR, ASR, captions, text-index) + chi phí compute units.
- Latency/query (p50/p95) laptop CPU + Colab GPU, có/không rerank + VLM.
- Điểm practice queries (89 câu 2025 làm dev-set) trước/sau từng nâng cấp.
- Điểm thật các đợt sơ tuyển (3 đợt) — leaderboard là bằng chứng mạnh nhất.

## 5. Related work chính (đã có sẵn citation trong docs/research của repo cũ)

- **AIC 2025**: MERVIN (arXiv:2605.16120), Vortex (2606.19682), LLandMark (2603.02888),
  Unified-IMMR cascaded (2512.12935), MADTempo (2512.12929), DANTE (2512.13169),
  EEIoT (2512.06334); tổng quan AIC 2024 (Springer 978-981-96-4291-5_1).
- **VBS/LSC**: VISIONE 5.0, vibro, PraK V4 (MMM'26), NII-UIT (MMM'25), kết quả VBS 2024/2025
  (arXiv:2502.15683, 2509.12000). UIT CVPRW'25 (LLM-assist, ablation H@1).
  🆕 VBS 2026: **SnapMind** (MMM'26 — agent LLM-planner điều phối truy xuất, đối
  chứng trực tiếp cho thể thức tự động của ta) · **NII-UIT VBS2026** (MMM'26 —
  VQA bằng Answer Span Prediction + Candidate Answer Suggestion, đối chứng cho
  per-group VQA + multi-frame strips).
- **Kỹ thuật**: SuperGlobal (ICCV'23), SigLIP/SigLIP2, Perception Encoder (2504.13181),
  **Qwen3-VL-Embedding/Reranker (2601.04720)** — lane embed + tầng cross-rerank của ta,
  **MetaCLIP 2 (2507.22062)** — lane worldwide-huge tùy chọn, jina-clip-v2 (lane 89 ngôn
  ngữ tùy chọn), LiT (CVPR'22), WiSE-FT (2109.01903), mCLIP distill
  (2004.09813), ViCLIP-OT (2602.22678 — SOTA vi retrieval, đối chứng), Bruch et al. fusion
  (TOIS'23), TransNetV2, PhoWhisper, Vintern (2408.12480).

## 6. Khung bài (8 trang SOICT)

1. Intro — bài toán AIC/VBS, gap: hệ tiếng Việt tương tác + tự động.
2. Related work (½ trang, bảng so sánh đội 2025).
3. System — hình pipeline (mục 2 PROJECT_CONTEXT), 6+5 đóng góp đánh số.
4. Vietnamese adaptation — LiT+WiSE-FT recipe + data tự sinh.
5. Experiments — A1–A10 (`scripts/26_run_ablations.py`) + leaderboard + latency
   (`scripts/50_bench_latency.py`).
6. Lessons & failure cases (progressive-KIS timing, OCR khó, gatekeeping questions).
7. Conclusion + reproducibility statement (repo + notebooks công khai sau thi).
