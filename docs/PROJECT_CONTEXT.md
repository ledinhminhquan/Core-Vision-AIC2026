# 📖 PROJECT_CONTEXT — Toàn bộ ngữ cảnh dự án Core-Vision_Perfect_V1

> **Đây là tài liệu gốc (source of truth) của dự án.** Đọc file này là nắm được toàn bộ:
> bài toán, ý tưởng cốt lõi, kiến trúc, thuật toán, pipeline, cách chạy, và cơ sở
> bằng chứng đằng sau từng quyết định thiết kế.
>
> **Phả hệ:** Core Vision (3 tuần) → Core-Vision_Ultimate → Core-Vision_Ultimate_Final
> (5 vòng review đối kháng, 316 test, 0 finding bất ổn) → **Core-Vision_Perfect_V1**
> (bản này, 14/07/2026): kế thừa nguyên vẹn phần lõi đã kiểm chứng (package `cvf`→`cvp`,
> env `CVF_`→`CVP_`), **đóng nốt cả 8 tinh chỉnh vòng 5 còn mở**, và bổ sung các nâng
> cấp 2026 — xem mục 0 ngay dưới.

## 0. Perfect V1 — có gì mới so với Ultimate_Final (14/07/2026)

**Vá (8/8 finding vòng 5 đã đóng):** câu-hỏi-mệnh-lệnh của đề QA thật (`Hảy cho biết…`
không có `?` — typo nguyên bản của BTC) giờ được tách đúng (M-R5-1 + L-R5-2/3, kèm
tách theo ranh giới câu cho QA không marker); lệnh cài TransNetV2 local đủ deps
(L-R5-1); test găm block bash trong DRIVE_SETUP (C-R5-1); comment builder hết nói quá
(C-R5-2); `.gitattributes` chuẩn hoá EOL toàn repo (C-R5-4).

**Kế thừa từ các repo tiền nhiệm (các delta chưa từng vào flagship):**
1. **Cross-encoder rerank theo cặp** (`search/cross_rerank.py`) — công thức vô địch
   Unified-IMMR 76.4/88: `search.reranker: blip2_itm` (BLIP-2 ITM) **hoặc
   `qwen_reranker`** = Qwen3-VL-Reranker-2B (01/2026 — cross-encoder đa phương thức
   MỞ đầu tiên). Chạy TRƯỚC VLM listwise rerank; chỉ xáo đầu bảng; mọi lỗi → giữ nguyên.
2. **Objects parquet** (`data/objects_compact.py` + `scripts/03 --objects-index`) —
   gộp ~178k JSON per-keyframe thành MỘT file zstd; `ObjectBooster` tự ưu tiên khi có
   (IO trên Drive nhanh hơn hàng trăm lần).
3. **Tái dựng map-keyframes** (`scripts/05_rebuild_map_keyframes.py` + `data/keyframe_align.py`)
   — dhash + DP đơn điệu + tinh chỉnh cửa sổ. **BẢO HIỂM SỐNG CÒN:** bộ data đang tải
   về KHÔNG có map-keyframes; thiếu nó là không nộp được `frame_idx` thật.
4. **HTTP service** (`cvp serve`, `service/app.py`): /health · /search/{text,qa,trake,avs}
   · /nearest · /keyframe — bộ mặt máy-gọi-máy cho **thể thức tự động 2026**; test bằng
   stub engine thuần CPU. + CLI `cvp` (serve/search/version).
5. **2 lane đa ngữ tùy chọn mới:** `jina` (jina-clip-v2, 89 ngôn ngữ, Matryoshka 64–1024)
   và `metaclip2` (MetaCLIP 2 worldwide — SOTA đa ngữ giữa 2026, XM3600 64.3%).

**Nâng cấp 2026 mới toanh:**
6. **VQA đa khung hình** (`vqa.frames_per_answer=3`): MỖI nhóm ứng viên gửi MỘT dải
   frame liên tiếp trong MỘT call Gemini — fix đúng dạng đề "giải toán trong video"
   2025 mà single-frame chịu chết.
7. **Temporal-context boost** (`search.temporal_boost`, tắt mặc định) — công thức
   before/now/after của Vortex cho truy vấn "… sau khi …": láng giềng một phía của
   ứng viên được chấm với vế ngữ cảnh, max cộng vào điểm. Tách vế bằng regex tiếng Việt
   (không tốn call LLM), thuần offline.
8. **Retry khi ranking "phẳng"** (`search.low_confidence_retry`, tắt mặc định) — đường
   batch/auto: điểm tin cậy = (top1 − median)/|top1|; thấp hơn ngưỡng thì re-search các
   expansion ĐÃ CACHE (không tốn call API) và RRF-merge.
9. **Chuỗi fallback Gemini** `gemini-3.5-flash → gemini-3-flash-preview → 2.5-flash`
   (model id chết giữa mùa → tự rơi xuống model kế, không rơi thẳng về Google Translate).
10. **Ops:** `scripts/26_run_ablations.py` (trận A1–A10 một lệnh, chấm bằng scorer
    chính thức) · `scripts/50_bench_latency.py` (gate p50≤200ms/p95≤500ms) · toggle
    **📺 group-by-video** kiểu VISIONE trong app (BTC dạy đúng pattern này ở buổi 2).

**Facts 2026 đã xác minh (tập huấn buổi 1–3 + FAQ + portal, 14/07/2026):**
- **2 hình thức thi:** interactive (dùng trợ lý ảo trong tool là **TÙY CHỌN**) +
  **automatic** (thử nghiệm, "thi giữa các Trợ lý ảo" — CHƯA có spec/protocol/API;
  giảng viên khẳng định chắc chắn KHÔNG phải "nộp file tự động rồi chấm").
- **Bán kết: nộp ĐÚNG 1 FILE** trong khung thời gian cho trước.
- **Dataset + đề bài + baseline + metric: phát hành CHẬM NHẤT 25/07/2026** (FAQ);
  sơ tuyển cần nộp kèm **BÁO CÁO giải pháp** (→ `report/` LaTeX kit).
- Vòng loại: truy vấn **CHỈ text**; chung kết thêm clip **chỉ được XEM** (cấm quay
  chụp/screen-capture — được phép mô tả lại/vẽ/sinh ảnh để nhập vào hệ); **audio có
  thể bị TẮT**; 2025: KIS 5 phút với 5 hint nhỏ giọt mỗi 1 phút, VKIS 4 phút/clip 20s.
- **AVS KHÔNG chắc thi năm nay** (giảng viên buổi 2 nhớ là không có; thể lệ đổi hằng
  năm → giữ nguyên module sau cờ). KIS-C là "phong cách hệ thống" chưa phải dạng đề.
- FAQ row 9: **không giới hạn model/thuật toán/công cụ**; cả tự động lẫn tương tác đều
  hợp lệ. Theo dõi spec qua: Q&A sheet của BTC, facebook.com/AICHCMC, Codabench
  (organizer `vnaic`, trang 2025 = id 10187; trang 2026 dự kiến xuất hiện đầu-giữa T8).
- Zip Codabench **PHẢI chứa folder tên `submission`** (đã là mặc định của packager).

---

## 1. Bài toán: HCMC AI Challenge 2026 (AIC)

**Hội thi Thử thách Trí tuệ Nhân tạo TP.HCM** — truy xuất khoảnh khắc trong hàng trăm
đến ~1000+ giờ video (tin tức HTV + có thể lifelog/egocentric theo định hướng 2026)
bằng truy vấn tiếng Việt. Mô hình theo VBS (Video Browser Showdown) và LSC.

**Timeline 2026 (đã xác minh từ cổng chính thức):** đăng ký đến 15/6 · tập huấn 6–7/2026
· sơ tuyển 8/2026 (kết quả 30/8) · **chung kết 12–26/9/2026** · trao giải 10/2026.
**Mới 2026:** ngoài thể thức tương tác (người + trợ lý) còn có **thể thức tự động**
(trợ lý đấu trợ lý, không người can thiệp) → module `pipeline/auto_agent.py`.

| Dạng | Mô tả | Nộp gì |
|---|---|---|
| **Textual KIS** | Tìm ĐÚNG MỘT khoảnh khắc từ mô tả văn bản. Chung kết: mô tả tiết lộ dần | `video_id,frame_idx` |
| **Video KIS (KIS-V)** | Xem clip ≤20s chiếu trên màn hình (cấm quay) → tự mô tả lại và tìm | `video_id,frame_idx` |
| **Q&A (Video QA)** | Như KIS + trả lời câu hỏi về khoảnh khắc | `video_id,frame_idx,answer` |
| **TRAKE** | Chuỗi sự kiện có thứ tự trong MỘT video → frame cho TỪNG sự kiện | `video_id,f1,f2,...,fk` |
| **AVS** | Tìm CÀNG NHIỀU đoạn khớp mô tả càng tốt | như KIS, nhiều dòng đa dạng |
| **KIS-C** *(điểm nhấn 2026, chưa chắc thi)* | Hội thoại: gợi ý tối thiểu ban đầu, tiết lộ thêm sau 60s theo câu hỏi của đội | như KIS |

**Chấm điểm vòng loại (Codabench)** — công thức CHÍNH THỨC (đã cài trong `cvp/eval/official.py`):
- KIS: `R-Score = 1` khi và chỉ khi đúng video VÀ `frame_idx ∈ [s,e]`; sai một trong hai → 0.
- QA: thêm điều kiện answer khớp (so sánh casefold, GIỮ NGUYÊN dấu tiếng Việt).
- TRAKE: sai video → 0; đúng video → `(1/N)·Σ 1(frame_j ∈ [s_j,e_j])` (ví dụ BTC: 3/4 = 0.75).
- Mỗi truy vấn: `R@k = max R-Score trong k dòng đầu`, k ∈ {1,5,20,50,100};
  **Final = trung bình 5 giá trị R@k**. Hit càng sớm điểm càng cao.
- **Đáp án là MỘT ĐOẠN frame liên tục** — nộp bất kỳ frame nào trong đoạn là ăn trọn điểm
  (BTC Q&A 2026: các keyframe thỏa mãn liên tiếp tạo thành đoạn; frame chuyển cảnh không tính).
- **`frame_idx` là số frame trong video GỐC** (từ `map-keyframes`), *không phải* thứ tự keyframe!
- Sơ tuyển 2026: tối đa **20 lượt nộp, 5 lượt/ngày** (theo thể lệ công bố). Chung kết dùng
  server BTC kiểu DRES: nộp sớm điểm cao, nộp sai bị trừ — client trong `submission/dres_client.py`.

## 2. Ý tưởng cốt lõi

Bài toán đưa về **truy xuất ảnh mức keyframe** (BTC đảm bảo mọi đoạn đáp án chứa ≥1 keyframe):

```
OFFLINE (build 1 lần, resumable)                 ONLINE (mỗi truy vấn <1s)
────────────────────────────────                 ─────────────────────────
videos ──► TransNetV2/SceneDetect ──► keyframes   query VI ──► Gemini: dịch + mô tả
keyframes ──► SigLIP-2 so400m (đa ngữ) ──┐                     thị giác + mở rộng (cả EN)
keyframes ──► PE-Core-bigG (EN, #1 2026)─┼──► FAISS   FAISS top-K/lane ──► SuperGlobal
keyframes ──► [Qwen3-VL-Embed-2B (vi)]  ─┤            (max mọi biến thể) ──► fusion lane
keyframes ──► OCR (EasyOCR vi+en)       ─┤──► BM25    ──► + BM25 O(K) (OCR/ASR/caption/meta)
videos    ──► ASR (PhoWhisper)          ─┤   PERSISTED──► + object boost (228 danh từ VI)
keyframes ──► Captions (Vintern-1B)     ─┤            ──► + neighbor boost ──► [VLM rerank]
media-info + objects (Open Images)      ─┘            ──► kết quả / CSV / DRES
```

**Các nâng cấp "Final" so với Ultimate (mỗi mục là một điểm yếu đã sửa):**

1. **BM25 bền vững + giới hạn ứng viên** — chỉ số text được tokenize MỘT LẦN lúc build
   (`artifacts/text_index/*.json.gz`, chữ ký catalog), truy vấn chấm điểm CHỈ trên ứng viên
   → O(K·|q|) thật sự, hết trễ ở truy vấn đầu tiên. Fallback in-memory khi thiếu index cũ.
2. **SuperGlobal trên MỌI biến thể truy vấn** (max-fuse) — khớp với cách dense fuse,
   biến thể mở rộng tìm ra hit sẽ không bị rerank dìm nữa.
3. **TRAKE/feedback dùng cả ensemble** — sự kiện chỉ lane tiếng Anh hiểu vẫn neo được chuỗi;
   Rocchio áp trong không gian của TỪNG lane rồi fuse.
4. **QA trả lời THEO TỪNG dòng** — ứng viên gộp nhóm theo (video, ngắt cảnh >10s/250 frame),
   VQA từng nhóm (≤`vqa.max_calls_per_query`), mỗi dòng mang answer đúng nhóm của nó
   (điểm QA yêu cầu ĐÚNG dòng mang ĐÚNG answer).
5. **Đóng gói + nộp tự động** — `packager.py` validate mọi CSV theo thể lệ rồi zip Codabench
   kèm MANIFEST sha256; `dres_client.py` (urllib thuần, không bao giờ tự retry lệnh bị từ chối);
   `auto_agent.py` chạy trọn gói cho **thể thức tự động 2026**.
6. **provided_clip32 tự bật** trong notebook 01 khi có features của BTC.
7. **Resume chính xác giữa epoch** (`plan_epoch_steps`) — không train lặp batch sau khi đứt.
8. **Hết lệch config** — pydantic defaults == settings.yaml (patch16, weights 0.55/0.45).
9. **Từ vựng object 96 → 228 danh từ VI** (đã xác minh với 600 lớp Open Images V4);
   parser đa ràng buộc ("2 người bên trái và 1 người bên phải"), số chữ (một→mười),
   chặn nhầm năm ("năm 2024" không phải 5 vật).
10. **RRF có thể bật** (`search.fusion_method: rrf`) khi chưa tune trọng số;
    mặc định weighted-sum (Bruch et al. TOIS'23: tuned weighted-sum > RRF)
    + **harness tune trọng số** trên bộ đề dev có đáp án (`scripts/21` + `scripts/23`).
11. **AVS dùng MMR mức embedding** — khử near-duplicate XUYÊN video (tin tức hay dùng lại
    b-roll), vẫn giữ cap/video + khoảng cách thời gian.
12. **Frame hỏng (vector 0) bị loại khỏi pool SuperGlobal** — không pha loãng refinement.
13. **Test tầng tích hợp**: SearchEngine end-to-end với fake model (6 tests), official
    scorer 60+ tests, packager 34 tests... tổng cộng **412 tests** (Perfect V1: 316 kế thừa + 96 mới cho cross-rerank/objects-parquet/keyframe-align/service/VQA-đa-khung/temporal-boost/confidence-retry) + GitHub Actions CI.
14. **KIS-C nhận tóm tắt kết quả hiện tại** → câu hỏi làm rõ có tính phân biệt;
    thêm **đồng hồ 5 phút** progressive-KIS trong app.
15. **Truy vấn tiếng Anh cũng được enhance** (trước đây bị bỏ qua hoàn toàn).

## 3. Model stack (đã xác minh HF id, 07/2026)

| Vai trò | Model | Ghi chú |
|---|---|---|
| Lane đa ngữ (mặc định, train được) | `google/siglip2-so400m-patch16-384` | đọc tiếng Việt trực tiếp |
| Lane tiếng Anh (thi đấu) | **`timm/PE-Core-bigG-14-448`** (open_clip), fallback `PE-Core-L-14-336` → DFN5B ViT-H/14-378 | Meta Perception Encoder = dual-encoder mạnh nhất 2026 (COCO T2I 58.1); MERVIN (79/88 AIC-2025) dùng chính nó |
| Lane vi bản địa (tùy chọn, nặng) | `Qwen/Qwen3-VL-Embedding-2B` (MRL cắt 1024-d) | R@1 tiếng Việt zero-shot cao nhất trong model mở |
| Lane features BTC | OpenAI CLIP ViT-B/32 512-d | tức thì, không cần GPU |
| Fine-tuned | LoRA-LiT text tower trên siglip2 (+ WiSE-FT) | dùng chung FAISS index với siglip2 |
| Rerank chéo | BLIP-2 ITM / **VLM listwise** (Gemini hoặc Vintern, `search.vlm_rerank`) | UIT CVPRW'25: +10% H@1 |
| VQA | Gemini 3.5 Flash (fallback 3-flash-preview → 2.5-flash) → Vintern-1B-v3.5 (offline) | trả lời theo NHÓM ứng viên, mỗi nhóm MỘT dải nhiều frame (`vqa.frames_per_answer`) |
| Caption/OCR/ASR | Vintern-1B-v3.5 · EasyOCR vi+en · PhoWhisper | các kênh BM25 |

**Chống lệch checkpoint:** lane `openclip` có thể resolve ra checkpoint khác nhau giữa các máy
(PE-Core-bigG 1280-d vs DFN5B 1024-d). `model_tag` được ghi vào meta của index; engine cảnh báo
khi checkpoint đang load ≠ checkpoint đã build; `missing_videos` bắt cả lệch dim.

## 4. Thuật toán chi tiết

### 4.1 Catalog — bất biến trung tâm
`global_id == số dòng manifest.parquet == số dòng FAISS index`. Index lưu **signature** corpus;
lệch → từ chối phục vụ (sai hàng = sai `frame_idx` = 0 điểm). Hàng embedding là VỊ TRÍ
(`global_id − video_start`), sống sót khi tên keyframe có khoảng trống.

### 4.2 Một truy vấn KIS/QA
1. `QueryProcessor`: 1 call Gemini → {dịch, mô tả thị giác, N biến thể} (cache đĩa;
   degrade Gemini → Google Translate → raw — không bao giờ chết vì mạng).
2. Mỗi lane encode các biến thể phù hợp → FAISS top-K (500) → **max theo biến thể**.
3. **SuperGlobal** (mọi biến thể, max-fuse; zero-row bị mask): S1 = q·(vectors refine bằng
   trung bình láng giềng trong pool), S2 = (q + mean top-10)·vectors; final = (S1+S2)/2.
4. Fusion lane: weighted sum sau min-max (0.55 siglip2 / 0.45 openclip).
5. Trên ứng viên: BM25+ persisted (OCR/ASR/caption/metadata, bỏ dấu) + object boost
   → `weighted_sum` (hoặc `rrf`) theo `search.weights` → **neighbor boost** (±2 frame).
6. Tùy chọn `search.vlm_rerank`: VLM chấm 0–10 cho top-24, chỉ xáo đầu bảng, lỗi → giữ nguyên.

### 4.3 TRAKE — DANTE DP (mặc định) hoặc beam
1. Encode từng sự kiện (mọi lane khi `temporal.use_ensemble`) → FAISS top-100/sự kiện
   → pool ≤30 video; sim ma trận (frames × events) = tổng có trọng số các lane
   (min-max theo sự kiện; lane thiếu embedding tự rút, renormalize).
2. **DANTE O(N·T)**: `DP[j,t] = sim[t,j] + max_{τ<t}(DP[j−1,τ] − λ·(time_t − time_τ))`
   với running-max trượt; ràng buộc gap ∈ [min_gap_s, max_gap_s], sim_floor, λ=0.0005/s.
   (SOICT'25 — "Outstanding TRAKE" AIC-2025.) `temporal.algo: beam` giữ beam-DP đa dạng cũ.
3. Trả top-100 chuỗi strictly-increasing đúng format nộp.

### 4.4 AVS
Search display_k lớn → **MMR**: `λ·điểm − (1−λ)·max cos(ứng viên, đã chọn)` (λ=0.7)
+ ≤3 dòng/video + cách ≥10s; phần dư lấp bằng điểm cao còn lại.

### 4.5 Huấn luyện encoder tiếng Việt (LoRA-LiT + WiSE-FT) — notebook 02
- **Dữ liệu**: Vintern-1B caption tiếng Việt mọi keyframe của corpus (in-domain, giá trị nhất)
  + parquet public (KTVIC 21.6k caption, UIT-ViIC 19.3k — `scripts/12`); lọc 15–400 ký tự,
  khử near-dup (cos>0.97), **split theo VIDEO** (val ~3%) chống rò rỉ.
- **LiT**: đóng băng image tower → FAISS index giữ nguyên; LoRA r=32 trên text tower
  (q/k/v/out/fc1/fc2 + head), **giữ logit_scale/bias pretrained**.
- **Loss**: SigLIP sigmoid (mặc định; `loss: infonce` để A/B — nghiên cứu cho thấy InfoNCE
  hay thắng khi fine-tune nhỏ) + distillation anchor 1−cos(student, teacher) β: 0.3→0
  + **trộn 25% cặp anchor tiếng Anh** (`anchor_mix_ratio`) chống quên đa ngữ
  + tùy chọn **hard negative cùng video** (`hard_negative_per_sample`).
- **WiSE-FT**: sau train, quét α ∈ {0.4,0.5,0.6,1.0} nội suy base↔tuned (α=1.0 = tower thô nên winner không bao giờ tệ hơn nó), đánh giá val R@5,
  xuất winner → `checkpoints/vi_siglip2_best/wiseft_best` (thường robust hơn checkpoint thô).
- **Autopilot H100**: micro-batch dò OOM thực nghiệm giữ effective batch ≈2048, bf16, TF32,
  checkpoint atomic lên Drive, **resume chính xác giữa epoch**, run-pointer
  `active_train_run.json` gắn sha256 config, early-stop val R@5, EMA.
- Dùng: `CVP_EMBEDDING__MODEL=finetuned` (hoặc ensemble [finetuned, openclip]).
- **Lưu ý bằng chứng**: mọi đội top AIC/VBS đều thắng bằng ZERO-SHOT ensemble + dịch câu;
  fine-tune chỉ đáng giá như **kênh bổ sung** cho tên riêng/địa danh tiếng Việt —
  đo bằng `scripts/eval_model.py` trước khi bật trong ensemble thi đấu.

### 4.6 Tune trọng số tín hiệu trên bộ đề dev
```
python scripts/23_dump_signals.py --query-dir queries/dev          # engine chạy 1 lần/câu
python scripts/21_tune_weights.py --signals-dir artifacts/signal_dumps/dev \
       --gt queries/dev/gt.json --trials 60                        # tối ưu offline, in env CVP_...
```
Harness thay trọng số trên score map đã cache (không chạy lại engine), chấm bằng scorer
chính thức → in đúng dòng `CVP_SEARCH__WEIGHTS__*` để dán vào máy thi.

## 5. Cấu trúc repo

```
Core-Vision_Perfect_V1/
├── configs/settings.yaml         ← MỌI cấu hình (override: CVP_SECTION__KEY=...)
├── configs/object_vocab_vi.yaml  ← 228 danh từ VI → lớp Open Images V4
├── src/cvp/
│   ├── config.py constants.py
│   ├── utils/ data/ (catalog bất biến · extraction TransNetV2 · metadata)
│   ├── models/  siglip2 (+finetuned/wiseft graft) · openclip (PE-Core→DFN5B)
│   │            · qwen_embed · mclip · jina · metaclip2 · query_processor (Gemini)
│   │            · agent (KIS-C)
│   ├── index/   embedder (resumable) · store (FAISS + signature + model_tag)
│   │            · text_store (BM25 persisted)
│   ├── search/  engine · fusion (weighted_sum/rrf) · superglobal (đa biến thể)
│   │            · temporal (DANTE/beam, ensemble) · temporal_boost (Vortex)
│   │            · avs (MMR) · feedback (ensemble) · text_signals (O(K))
│   │            · object_filter · cross_rerank (BLIP-2 ITM/Qwen) · vqa · vlm_rerank
│   ├── submission/ writer · packager (Codabench zip) · dres_client
│   ├── training/ build_dataset · datamodule (anchor+hardneg) · losses (sigmoid/infonce)
│   │            · lit_trainer (exact resume + WiSE-FT) · public_datasets
│   ├── eval/    metrics · official (công thức BTC, pure stdlib)
│   ├── service/  app (FastAPI: search/qa/trake/avs/nearest/keyframe) · schemas
│   ├── cli.py   console `cvp` (serve / search / version)
│   └── pipeline/ ingest · run_queries (QA theo nhóm + confidence-retry) · auto_agent
├── scripts/  00 catalog · 01 extract · 02 embed+index · 03 aux(+text/objects-index)
│             · 05 rebuild-map-keyframes(!) · 11 train-data · 12 public-datasets
│             · 20 run-queries(--zip --gt) · 21 tune-weights · 23 dump-signals
│             · 25 auto-agent · 26 ablations A1-A10 · 30 ingest · 40 eval-official
│             · 50 bench-latency · doctor · eval_model
├── app/streamlit_app.py          ← 5 tab + đồng hồ 5' + basket + export + feedback
├── notebooks/_build_notebooks.py ← nguồn sinh 3 notebook (đừng sửa tay .ipynb)
├── docs/    file này · ARCHITECTURE · EVALUATION · DATASET_INGESTION · COLAB_GUIDE
│            · PROJECT_PLAN · DATA_FORMAT · DRIVE_SETUP · TRAINING · PLAYBOOK · PAPER_NOTES
├── report/  LaTeX kit báo cáo giải pháp (BẮT BUỘC nộp kèm vòng sơ tuyển)
├── HUONG_DAN.md  hướng dẫn A-Z tiếng Việt (Drive → Colab → thi đấu)
├── tests/   412 tests CPU thuần (không cần GPU/mạng/data thật)
└── .github/workflows/ci.yml
```

## 6. Quy trình A→Z

1. **Local**: `pip install -e ".[search,dev]"` → `pytest` (bộ đầy đủ **412 pass** khi có torch-cpu như CI; không torch thì các test training/model bị skip — đo lại con số torch-less sau khi cài).
2. **Drive**: upload theo `docs/DRIVE_SETUP.md`.
3. **Colab nb 01** (GPU bất kỳ): catalog → extract K-batch → embed các lane → FAISS
   → OCR/ASR/captions → **text index**. Mọi bước resumable, FORCE_* để làm lại.
4. **Colab nb 02** (ưu tiên H100): public parquets → train set → LoRA-LiT → WiSE-FT
   → so baseline. Run all là tự resume.
5. **Colab nb 03** / laptop: smoke test, latency, CSV mẫu, **validate + zip Codabench**,
   chấm thử với GT, dry-run auto-agent.
6. **Vòng loại**: `python scripts/20_run_queries.py --query-dir <đề> --zip` → nộp Codabench
   (nhớ: 5 lượt/ngày, 20 lượt tổng).
7. **Chung kết**: laptop chạy app + artifacts sync về; đấu pháp trong PLAYBOOK;
   DRES client cấu hình `submission.dres_base_url` khi BTC công bố endpoint.

## 7. Cơ sở bằng chứng (đối chiếu các đội 2025)

- MERVIN (PE-Core-bigG-448, TransNetV2 3kf/shot, Milvus, Gemini-cleaned ASR): **79/88**.
- Vortex (CLIP+SigLIP2 RRF + Rocchio): 79.6/88 — ta có cả hai cơ chế.
- Unified-IMMR (BEiT-3+SigLIP → BLIP-2 ITM rerank top-100): **76.4/88** — đúng blueprint rerank.
- DANTE (DP gap-penalty): "Outstanding TRAKE" — thuật toán TRAKE mặc định của ta.
- UIT (vô địch 4 năm liên tiếp + VBS 2025): LLM decompose (+103% H@1), VLM rerank (+10% H@1),
  dense keyframes (+60–90%) — tất cả đều đã tích hợp.
- NIST TRECVID 2025: "fast embedding recall + VLM verification" là pattern thắng — chính là
  dense → SuperGlobal → VLM rerank của ta.

**Định vị bài báo (SOICT 2026 / MTA):** xem `docs/PAPER_NOTES.md`.

## 8. Trạng thái & việc còn lại

- ✅ Source + 412 tests + CI + 3 notebooks + docs (11 file) + app + service + report/ LaTeX kit.
- ⏳ Cần dữ liệu 2026 (BTC chưa phát hành — "sẽ gửi cho các đội khi sẵn sàng"):
  build artifacts (nb01), train (nb02), đo chất lượng thật (nb03 + tune weights).
- 📌 Theo dõi buổi tập huấn kế (spec nộp bài + công cụ BTC 2026), endpoint DRES chung kết,
  và khả năng xuất hiện dạng đề mới (KIS-C / lifelog metadata).

## 9. 🆕 Bản vá 2026-07-08 — kết quả review đối kháng vòng 2 (đối chiếu đề THẬT 2025)

> Đợt review đối kháng toàn diện (chi tiết trong `Report_CoreVision.md` ngoài repo) đã
> đối chiếu code với **gói đề chung kết AIC 2025 nguyên bản** và vá/nâng cấp các điểm sau.
> Không có thay đổi phá vỡ tương thích; mọi hành vi cũ được giữ nguyên khi input ở
> định dạng cũ (test regression đầy đủ).

1. **[M1 — quan trọng] Parser đề TRAKE/QA theo định dạng đề thật của BTC**
   (`pipeline/run_queries.py`): file TRAKE thật có dòng ngữ cảnh + sự kiện tiền tố
   `E1:`…`Ek:` — trước đây dòng ngữ cảnh thành "sự kiện ma" làm lệch TOÀN BỘ chuỗi frame
   khi chạy batch/auto (TRAKE ≈ 0 điểm). Mới: `parse_trake_events()` (bỏ header, bóc
   tiền tố; định dạng mỗi-dòng-một-sự-kiện giữ nguyên hành vi cũ) + `split_qa_line()`
   (đề QA một-dòng "… Hỏi …?" tách mô tả/câu hỏi — dense search chỉ nhận phần mô tả).
   Knob A/B mới `temporal.event_context: none|prepend` (prepend = ghép ngữ cảnh header
   vào từng sự kiện). `scripts/23_dump_signals.py` dùng cùng parser để dump khớp engine.
   Kèm 2 file ví dụ đúng định dạng BTC trong `queries/example/` + 10 test mới.
2. **[E5] Scorer offline hỗ trợ GT nhiều cửa sổ** (`eval/official.py` +
   `scripts/21_tune_weights.py`): thêm cách viết `"ranges": [[s1,e1],[s2,e2],…]` —
   một dòng KIS/QA ăn điểm khi frame rơi vào BẤT KỲ cửa sổ nào (phục vụ bộ đề dev kiểu
   AVS và đáp án lặp lại trong video). GT một cửa sổ giữ nguyên hành vi cũ.
3. **[L5] Alias task mới**: `kis-v` / `kisv` / `vkis` / `video-kis` → chấm theo công thức
   KIS (trước đây rơi vào `unscored`).
4. **[L3] `sanitize_answer` tự cắt answer > 100 ký tự** (kèm cảnh báo) + hằng số chung
   `MAX_QA_ANSWER_CHARS` dùng thống nhất ở writer / packager / VQA / run_queries —
   answer gõ tay quá dài từ UI không còn tạo được CSV bị server từ chối.
5. **[L4] Cache key của QueryProcessor** thêm `gemini_model` + `enhance_english` —
   đổi model Gemini hay bật/tắt enhance tiếng Anh giữa trận sẽ tạo enhancement mới,
   không dùng lại cache cũ.
6. **[L2] Lane `qwen_embed`**: floor `transformers>=4.57` (bản đầu tiên có lớp Qwen3-VL,
   ghi rõ trong pyproject + requirements-colab); instruction chỉ áp cho PHÍA QUERY
   (ảnh/document encode không instruction — đúng quy ước Qwen3VLEmbedder chính thức).
7. **[E2] TransNetV2 tự cài trong notebook 01** (`INSTALL_TRANSNETV2 = True`, pip
   `--no-deps` để không bao giờ đụng torch của Colab; lỗi cài → tự fallback
   PySceneDetect như cũ). `data/extraction.py` chuyển prediction tensor → numpy 1-D
   trước khi `predictions_to_scenes` cho khớp package `transnetv2-pytorch` trên PyPI.
8. **[E1] Đồng hồ 5 phút KIS-C chạy realtime** trong app (`st.fragment(run_every="1s")`)
   — trước đây chỉ cập nhật khi có tương tác.
9. **[C6] `.gitignore`**: `/data/` → `/data/*` để `!/data/.gitkeep` có hiệu lực
   (git không thể re-include file khi thư mục cha bị exclude) — clone mới giờ có sẵn
   skeleton `data/`. **[C7]** docstring `siglip2.py` sửa patch14 → patch16 (đúng config).
   **[C8]** số test trong docs cập nhật: **288 tests**.


## 10. Bản vá 2026-07-12 — vòng review đối kháng thứ 3 (18 finding + 2 enhancement)

Vòng 3 kiểm định độc lập trong clone nguyên sơ đã XÁC THỰC toàn bộ claim vòng 2, đồng thời
tìm ra 18 vấn đề mới trên các lăng kính chưa quét (đề THẬT nhiều dòng × batch path, app sâu,
execution smoke, Colab reality). Tất cả đã vá trong commit này (suite 288 → **311 tests**; vòng 4 kiểm chứng thêm 5 test + 6 fix nhỏ → **316**, xem mục 11):

1. **[H-R3-1] QA ≥3 dòng** — câu hỏi thật (dòng cuối) từng bị vứt, VQA nhận nhầm câu mô tả
   (đề thật `query-p2-3-qa.txt`). Fix: `parse_query_lines()` — question = dòng cuối có tính
   nghi vấn, mô tả = ghép các dòng còn lại; fallback tách marker trên toàn văn.
2. **[H-R3-2] KIS/AVS nhiều đoạn văn** — chỉ search đoạn 1 (≥5 file KIS chung kết 2025 bị cắt
   chi tiết phân biệt). Fix: ghép TẤT CẢ đoạn cho retrieval (BM25/OCR/metadata nhận đủ toàn văn;
   đường Gemini-enhance tự cô đọng); `scripts/23_dump_signals.py` dùng CHUNG helper → trọng số
   tune trên đúng query mà engine sẽ thấy. +2 fixture đề thật (`query-0-6/0-7`).
3. **[M-R3-1] TransNetV2 chết lâm sàng trên Colab** — `--no-deps` bỏ mất `ffmpeg-python`
   (dependency BẮT BUỘC của `predict_video`) → mọi video âm thầm rơi về PySceneDetect.
   Fix: cài kèm `ffmpeg-python` (thuần Python, an toàn với `--no-deps`).
4. **[M-R3-2] 8/14 scripts crash UnicodeEncodeError** khi stdout bị pipe/redirect trên Windows
   (ký tự `→` vs cp1252) — kể cả run thật `--zip > log` chết TRƯỚC khối zip. Fix: `_bootstrap.py`
   reconfigure stdout/stderr sang UTF-8 (một điểm vá cho mọi script).
5. **[M-R3-3] App: Export tab này từng ghi nhầm ranking tab khác** (buffer kết quả dùng chung).
   Fix: provenance tag `results_task` ghi tại MỌI điểm search; `_export` chỉ auto-append khi
   ranking thuộc đúng task (KIS-C export như KIS), khác task → cảnh báo + chỉ xuất basket.
6. **[L-R3-1..3] App**: gợi ý VQA 💡 chỉ render ở tab QA + tự xóa khi thay ranking; marks
   feedback hết hạn khi KIS-C đổi `last_query`; tên file export có NGÀY + chống trùng
   (`kis-YYYYMMDD-HHMMSS[-n].csv`) — hết đè file cùng giây/cùng giờ khác ngày.
7. **[L-R3-4] `scripts/eval_model.py`** chuyển heavy import vào `main()` — `--help` chạy được
   trên bản cài `[search,dev]` không torch (đúng quy ước lazy-import của repo).
8. **[L-R3-5..7] Docs đồng bộ số liệu thật**: bước Local ghi rõ 292 pass + 1 skip khi không
   torch (311 khi có); DRIVE_SETUP đổi block `set VAR=… # comment` (hỏng trên cả cmd/PS/bash)
   sang `$env:`; WiSE-FT α thống nhất `[0.4,0.5,0.6,1.0]` ở cả TrainConfig, notebook 02 lẫn docs
   (α=1.0 = tower thô → winner không bao giờ tệ hơn).
9. **[L-R3-8] Regex micro-fix `pip check`** giờ khớp CẢ HAI format thật của pip
   (`requires …, which is not installed` VÀ `has requirement …, but you have …`) — nhánh
   tự-vá conflict không còn là dead code.
10. **[C-R3-1..3]** Report vòng 2 đính chính 12 (không phải 10) test mới; sidebar basket render
    SAU tab handlers (số liệu tươi trong cùng rerun); notebooks có cell `id` deterministic
    (sha1(index:source)[:12]) — đạt chuẩn nbformat 4.5 strict.
11. **[F0]** 3 file `.ipynb` bị một writer ngoài (kiểu nbformat) ghi lại sau commit `2539a7a`
    (nội dung không đổi, chỉ EOL + thứ tự key) → đã `git restore` về đúng output builder.
12. **[E-A] Knob mật độ keyframe K-batch** — section config mới `extraction.shot_positions`
    (mặc định [0.15,0.5,0.85]) + `dedup_mad_threshold`: UIT đo dense keyframes +60–90% H@1
    trên bộ khó; giờ tăng mật độ chỉ là 1 dòng env
    (`CVP_EXTRACTION__SHOT_POSITIONS='[0.1,0.3,0.5,0.7,0.9]'`), không sửa code.
13. **[E-B] `scripts/20_run_queries.py --gt`** — lệnh MỘT PHÁT cho vòng loại: chạy đề → validate
    → zip → chấm offline đúng công thức BTC (in bảng per-query + per-task + MEAN FINAL) —
    tiết kiệm lượt nộp (5/ngày, 20 tổng).


## 11. Kiểm chứng vòng 4 (2026-07-12, commit `14ceb86`)

Vòng 4 chạy 3 agents kiểm chứng ĐỐI KHÁNG chính các fix vòng 3 trong clone nguyên sơ và bắt được
6 điểm cần tinh chỉnh — tất cả đã vá trong cùng ngày:
1. Fix DRIVE_SETUP (L-R3-6) hóa ra bị NO-OP im lặng ở vòng 3 (lệnh replace không khớp) — giờ đã
   land THẬT: 2 block env riêng cho PowerShell và bash, có test găm (`$env:` phải tồn tại,
   form `set VAR=` hỏng phải biến mất).
2. `ffmpeg-python` hard-import `past.builtins` từ gói `future` — cài `--no-deps` thiếu nó thì
   TransNetV2 vẫn chết. Lệnh cài thêm `future` + notebook chỉ báo READY sau khi IMPORT thật
   thành công (verify-then-report, không tin pip rc nữa).
3. `split_qa_line`: marker giờ CHỈ khớp dạng viết hoa (`Hỏi`/`Câu hỏi`) — động từ thường
   "hỏi đường"/"học hỏi" trong mô tả không marker không còn cắt cụt retrieval text.
4. `scripts/23` mirror cả knob `temporal.event_context=prepend` khi dump TRAKE.
5. `ExtractionCfg` validator: `shot_positions` rỗng/ngoài (0,1) fail to tiếng thay vì âm thầm
   quay về mặc định.
6. Số test torch-less trong docs đồng bộ theo phép đếm thật: **292 pass + 1 skip** (không torch),
   **316 pass** đủ bộ (CI cài torch-cpu). Battery vòng 4 trong clone: pytest ×2 sạch, tree sạch
   sau chạy, ruff sạch, 0/30 flake atomic-write, notebooks byte-identical với builder, venv
   `[search,dev]` import sweep 55/58 module (3 module training đúng thiết kế cần torch),
   15/15 script `--help` rc=0 qua pipe.
