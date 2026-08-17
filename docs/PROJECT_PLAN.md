# 🗺️ PROJECT_PLAN — Kế hoạch tác chiến AIC 2026 (Core-Vision Perfect V1)

> Viết ngày **14/07/2026**, thay thế mọi plan cũ. Lịch dưới đây là lịch **THẬT** đã xác minh
> từ cổng chính thức + buổi tập huấn. Cấu trúc: workstream có tiêu chí nghiệm thu đo được
> → sổ rủi ro → code freeze → định nghĩa hoàn thành. Mọi con số chất lượng đều chấm bằng
> **scorer chính thức** (`cvp/eval/official.py`), không bao giờ bằng cảm giác.

---

## 0. Lịch 2026 đã xác minh & nguyên tắc lập kế hoạch

| Mốc | Ngày | Ý nghĩa cho ta |
|---|---|---|
| BTC phát hành **dataset + đề bài + baseline + metrics** | **không muộn hơn 25/07/2026** | WS-2 kích hoạt ngay ngày phát hành |
| Sơ tuyển trên **Codabench** | **08/2026** (trang dự kiến mở đầu–giữa T8, organizer `vnaic`; trang 2025 id `10187` là mẫu tham chiếu) | WS-3; **kèm BÁO CÁO giải pháp** |
| Công bố kết quả sơ tuyển | **30/08/2026** | biết suất chung kết |
| **Chung kết on-site** | **12–26/09/2026** | WS-5; **code freeze 05/09** |
| Trao giải | 10/2026 | — |

**Thể thức 2026** (đã xác minh): MỘT bài toán "trợ lý ảo thông minh trên dữ liệu đa phương
tiện lớn", HAI hình thức: (1) **tương tác** — truyền thống, dùng agent trong tool là *tùy chọn*;
(2) **tự động** — pilot, trợ lý đấu trợ lý, **CHƯA có spec** (chắc chắn không phải "nộp file
tự động"). Dạng đề: KIS / KIS-V / QA chắc chắn; TRAKE có tiền lệ 2025; **AVS không chắc**
(giữ sau cờ config); KIS-C là *phong cách hệ thống* BTC muốn thấy, chưa phải task công bố.
FAQ: **không giới hạn model/thuật toán/công cụ**; dữ liệu 2026 có thể gồm video
sousveillance/egocentric và có thể 1000+ giờ.

**Nguyên tắc:**
1. **Không đốt lượt nộp** — 20 lượt tổng / ≤5 mỗi ngày ở sơ tuyển: mọi lượt nộp phải qua
   `scripts/20_run_queries.py --zip --gt` (validate + chấm offline) trước.
2. **Đo trước khi tin** — mọi thay đổi config phải thắng trên bộ đề dev 89 câu rồi mới
   được vào profile thi đấu (`scripts/26_run_ablations.py`).
3. **Mọi thứ resumable** — đứt Colab/mất mạng không được làm mất tiến độ (đã là bất biến
   của repo; giữ nguyên khi thêm bất kỳ bước mới nào).

## 1. Mục tiêu & phạm vi

**Mục tiêu:** vào chung kết qua sơ tuyển Codabench với ngân sách 20 lượt nộp, đồng thời sẵn
sàng cả hai hình thức (tương tác + tự động) cho chung kết 12–26/09. **Phạm vi:** đúng hệ đã
build — ensemble `[siglip2, openclip]` 0.55/0.45, SuperGlobal, BM25 persisted, object boost,
DANTE TRAKE, MMR AVS (sau cờ), cross-encoder rerank (`blip2_itm`/`qwen_reranker`), VQA dải
frame, temporal boost, retry khi tự tin thấp, FastAPI service + `cvp` CLI, 3 notebook Colab.
**Không thêm năng lực mới sau 05/09** (code freeze).

## 2. Bản đồ workstream ↔ thời gian

| WS | Cửa sổ | Nội dung một dòng |
|---|---|---|
| **WS-1** | 14/07 → 25/07 | Tổng duyệt toàn pipeline trên dữ liệu local (L28 + K08), dựng dev pack 89 câu, đăng ký câu hỏi với BTC |
| **WS-2** | 25/07 → ~08/08 | Sprint dữ liệu: ingest ngày-1, artifacts trong 48–72h, tune trọng số trên dev pack |
| **WS-3** | 08/2026 | Sơ tuyển: kỷ luật nộp 20/5-ngày qua ~3 đợt đề + viết báo cáo giải pháp (`report/`) |
| **WS-4** | T8 → T9 (song song) | Thể thức tự động: gắn protocol adapter khi có spec; diễn tập `scripts/25` với DRES 2.0.4 tự host |
| **WS-5** | 01/09 → 26/09 | Chung kết: drill nhịp hint 5', quy trình KIS-V không-quay-chụp, fallback tắt tiếng, cổng latency; **freeze 05/09** |

## 3. WS-1 · Tổng duyệt trước giờ G (14/07 → 25/07)

Dữ liệu local hiện có (cập nhật 15/08/2026): **Batch 1 chính thức — 32 zip / 873 video
L21–L30** trong `AIC2026-Info/Datasets1/` (keyframes + videos + clip-features-32 +
media-info + objects + **map-keyframes ĐỦ 873/873** — kiểm kê DATASET_INGESTION §1b);
gói **89 câu chung kết 2025** (73 KIS / 9 QA / 7 TRAKE). Đường tái tạo map chỉ còn là
bảo hiểm batch sau. Drive chuẩn: `MyDrive/AIC2025/{data,artifacts}` (xem [DRIVE_SETUP](DRIVE_SETUP.md)).

### 3.1 Dựng bộ đề dev chuẩn (`queries/dev-2025-finals/`)
Chuyển 89 câu 2025 về định dạng batch runner (parser đã đọc được đề nguyên bản BTC: TRAKE
header + `E1:`, QA một-dòng "Hỏi …?") + soạn `queries/dev-2025-finals/gt.json` (hỗ trợ nhiều cửa sổ
`"ranges": [[s1,e1],…]`). Đây là **thước đo duy nhất** của mọi quyết định config từ nay đến hết giải.

### 3.2 Rehearsal end-to-end trên L28 (đường keyframes BTC) + K08 (đường tự cắt)
```powershell
python scripts/00_build_catalog.py
python scripts/05_rebuild_map_keyframes.py --stride 5        # dhash + monotone DP — CHỈ là bảo hiểm; Batch 1 ĐÃ có đủ 873/873 map csv (§1b), bỏ qua được
python scripts/02_embed_and_index.py --model provided_clip32 # search được NGAY (features BTC)
python scripts/02_embed_and_index.py --all-members           # siglip2 + openclip (GPU/Colab)
python scripts/03_build_aux_indexes.py --ocr --asr --captions --text-index --objects-index
python scripts/20_run_queries.py --query-dir queries/dev-2025-finals --zip --gt queries/dev-2025-finals/gt.json
python scripts/25_auto_agent.py --query-dir queries/dev-2025-finals      # dry-run thể thức tự động
python scripts/50_bench_latency.py --n 200 --query-dir queries/dev-2025-finals
cvp serve --port 8000    # smoke: /health /search/text /search/image /search/qa /search/trake /search/avs /nearest /keyframe
```
K08: giải nén video K08 → `scripts/01_extract_keyframes.py` (TransNetV2 → fallback
PySceneDetect) — đường này tự sinh map-keyframes CHÍNH XÁC, dùng đối chứng chất lượng
tái tạo của scripts/05 trên L28.

### 3.3 Đăng ký câu hỏi vào Q&A sheet của BTC (deadline: trước buổi tập huấn kế)
1. **Tính hợp lệ của LLM API ngoài** (Gemini) tại chung kết on-site — mạng venue, quota, có bị coi là "ngoài hệ thống"?
2. **AVS có thi năm 2026 không?** (giảng viên buổi 2 nhớ là không — cần xác nhận chính thức).
3. **Audio ở chung kết**: clip KIS-V có bị tắt tiếng không? ASR có còn là tín hiệu hợp lệ với đề thi?
4. Spec + giao thức **thể thức tự động** (endpoint? DRES? định dạng đề máy-đọc-được?).
5. ~~Gói dữ liệu chính thức **có kèm map-keyframes không**~~ → **ĐÃ TRẢ LỜI 15/08/2026: CÓ, đủ 873/873** (Batch 1).
6. Chế độ **khung giờ nộp** Codabench 2026 (2025 từng giới hạn 9:00–11:59 sáng).

### 3.4 Khung báo cáo sơ tuyển
Dựng `report/` (LaTeX kit theo template AIO): khung sẵn phần kiến trúc + bảng ablation
(A1–A10 từ `scripts/26_run_ablations.py` xuất bảng cho paper) — sơ tuyển **bắt buộc nộp báo
cáo giải pháp** (FAQ), không để dồn sang tuần nộp bài.

**Nghiệm thu WS-1 (chốt 25/07):**
- [ ] `pytest` **575 tests** xanh trên máy dev (CI cài torch-cpu để đủ bộ).
- [ ] L28: `scripts/05` tái tạo map csv cho **100% video** có keyframes + video gốc; `frame_idx` tăng nghiêm ngặt; đối chiếu K08 (map tự cắt exact) sai lệch ≤ stride (5 frame).
- [ ] `scripts/20 --zip` tạo zip Codabench hợp lệ (folder trong zip tên `submission`, MANIFEST sha256) — 0 lỗi validate.
- [ ] Đo và GHI **B0** = MEAN FINAL trên dev-89 với config mặc định (đây là baseline mọi WS sau so vào).
- [ ] Latency ghi nhận trên laptop thi: mục tiêu **p50 ≤ 200 ms, p95 ≤ 500 ms** (`scripts/50` tự cảnh báo khi vượt).
- [ ] `cvp serve`: 8 endpoint trả 200 trên artifacts L28; `scripts/25` chạy trọn 89 câu ra zip hợp lệ không cần người can thiệp.
- [ ] 6 câu hỏi đã nằm trong Q&A sheet của BTC; `report/` compile ra PDF khung.

## 4. WS-2 · Sprint dữ liệu (25/07 → ~08/08)

**Ngày-1 (trong 24h kể từ khi BTC phát hành):**
1. Ném nguyên zip vào `MyDrive/AIC2025/data/` (notebook 01 tự nhận dạng + giải nén — "mẹo lười" trong DRIVE_SETUP).
2. **Soi schema ngay**: đề bài, metrics, baseline BTC, cấu trúc media-info/objects — đặc biệt
   nếu có lifelog/egocentric (schema metadata có thể khác HTV; xem rủi ro R2).
3. Kiểm kê thiếu gì: có map-keyframes không? clip-features không? → chọn nhánh build.
4. Notebook 01 **Run all** (mọi bước resumable; đứt phiên → Run all lại là tiếp tục).

**Ngân sách wall-clock artifacts (mục tiêu, tính từ lúc có dữ liệu):**

| Bước | Mục tiêu | Ghi chú |
|---|---|---|
| Catalog + (nếu thiếu) map-keyframes tái tạo | ≤ 12h | `scripts/05` chạy song song nhiều phiên Colab được (resumable, skip csv đã có) |
| Embed 2 lane + FAISS | ≤ 48h | A100/H100; `provided_clip32` cho search-ngay trong lúc chờ |
| OCR + ASR + captions + text-index + objects-index | ≤ 72h (streaming) | search dùng được TRƯỚC khi aux xong — độ chính xác tăng dần |

**Tuần 1 tháng 8 — tune trên dev-89:**
```powershell
python scripts/23_dump_signals.py --query-dir queries/dev-2025-finals
python scripts/21_tune_weights.py --signals-dir artifacts/signal_dumps/dev-2025-finals --gt queries/dev-2025-finals/gt.json --trials 60
# dán các dòng CVP_SEARCH__WEIGHTS__* nó in ra vào profile thi đấu
python scripts/26_run_ablations.py --query-dir queries/dev-2025-finals --gt queries/dev-2025-finals/gt.json   # A1–A10, chọn config thi
```
Ablation trọng tâm 2026: **A9** (cross-rerank `none|blip2_itm|qwen_reranker` —
`CVP_SEARCH__RERANKER=qwen_reranker`) và **A10** (`CVP_SEARCH__TEMPORAL_BOOST=true`,
`CVP_SEARCH__LOW_CONFIDENCE_RETRY=true`). Train nb02 (H100) là **kênh phụ**: chỉ vào ensemble
thi nếu `scripts/eval_model.py` chứng minh thắng zero-shot.

**Nghiệm thu WS-2:**
- [ ] Artifacts đủ cho 100% video của drop trong ≤72h (bảng trên); catalog in N keyframes / M videos khớp kiểm kê.
- [ ] Nếu drop thiếu map-keyframes: tái tạo phủ 100% video có raw video; số còn thiếu (không có video) = 0 hoặc có phương án (tự cắt `scripts/01`).
- [ ] Dev-89 sau tune: **MEAN FINAL ≥ B0 + 10% tương đối** (mục tiêu tạm — hiệu chỉnh sau lần đo B0; mọi mục tiêu chất lượng trong plan này đều theo quy tắc "đo B0 trước, chốt đích sau").
- [ ] Bảng A1–A10 đầy đủ trong `artifacts/ablations/` — vừa chọn config thi, vừa là ruột báo cáo sơ tuyển.
- [ ] Latency vẫn đạt cổng p50/p95 trên artifacts THẬT (corpus to hơn → nếu >1M keyframes cân nhắc `CVP_INDEX__TYPE=ivf`).

## 5. WS-3 · Sơ tuyển Codabench (08/2026)

**Hợp đồng nộp bài (spec 2025, cổng 2026 ghi giữ nguyên định dạng):** CSV UTF-8, **không
header**, ≤100 dòng, phân cách phẩy. KIS: `video_name, frame_id`; QA: `video_name, frame_id,
answer` (answer ≤100 ký tự, VN/EN, có phẩy thì bọc nháy); TRAKE: `video_name, f1..fN`. Zip
nộp **phải chứa folder tên `submission`** (packager của ta đã đúng chuẩn này). Điểm mỗi câu =
trung bình của max R-Score trong top-k, k ∈ {1,5,20,50,100}. `frame_idx` là **số frame video
GỐC** (từ map-keyframes) — sai bản đồ là 0 điểm dù tìm đúng cảnh.

**Kỷ luật trước MỌI lượt nộp** (2025 còn giới hạn khung 9:00–11:59 sáng — chuẩn bị bài từ
tối hôm trước, nộp đầu giờ sáng):
1. Config mới phải thắng config đã nộp gần nhất trên dev-89 (`scripts/26` hoặc `scripts/20 --gt`).
2. Chạy đề thật: `python scripts/20_run_queries.py --query-dir queries/p<N> --zip` — zip lỗi là KHÔNG tạo.
3. Ghi log lượt nộp (ngày, config env, điểm Codabench trả về) vào `report/submission_log.md`.

**Ngân sách 20 lượt / ≤5 mỗi ngày, kịch bản 3 đợt đề:**

| Đợt | Lượt | Chiến thuật |
|---|---|---|
| Đợt 1 | ≤6 | 1 baseline sớm (bắt lỗi format/pipeline với server thật) + cải tiến theo dev-89 |
| Đợt 2 | ≤6 | chỉ nộp khi dev-89 tăng; A/B tối đa 2 biến thể config/ngày |
| Đợt 3 | ≤6 | config tốt nhất + biến thể rerank; không thử nghiệm mới sát hạn |
| Dự phòng | 2 | sự cố server/định dạng/đề bổ sung |

**Báo cáo giải pháp** (điều kiện bắt buộc của sơ tuyển): viết trong `report/` song song đợt 1–2
(kiến trúc từ [PROJECT_CONTEXT](PROJECT_CONTEXT.md) §2–4, bảng ablation từ WS-2, số liệu
Codabench thật từ log) — **nộp trước hạn, không viết đêm chót**.

**Nghiệm thu WS-3:** dùng ≤20 lượt đúng kịch bản; 0 lượt bị từ chối vì format; báo cáo PDF
nộp đúng hạn; điểm Codabench của lượt cuối ≥ điểm lượt đầu (chứng minh vòng lặp offline-first
hoạt động); kết quả 30/08 — có tên trong danh sách chung kết.

## 6. WS-4 · Thể thức tự động (song song T8 → T9)

Nền đã có: `pipeline/auto_agent.py` (đề → search/TRAKE/AVS → QA theo nhóm → validate → zip
→ tùy chọn DRES) + **FastAPI service** làm bề mặt máy-gọi-được:
`cvp serve` → `GET /health`, `POST /search/text|image|qa|trake|avs`, `GET /nearest/{global_id}`,
`GET /keyframe/{global_id}` (cài `pip install -e ".[service]"`).

1. **Khi BTC công bố spec** (theo dõi buổi tập huấn + FAQ): viết adapter mỏng dịch giao thức
   BTC ↔ service/auto_agent hiện có. KHÔNG đoán trước spec — chỉ giữ bề mặt trừu tượng sẵn.
2. **Diễn tập DRES tự host**: dựng **DRES 2.0.4** (bản hiện hành 06/2026) local, cấu hình
   `CVP_SUBMISSION__DRES_BASE_URL` + env `DRES_USER`/`DRES_PASSWORD`, bật
   `submission.auto_submit: true` rồi chạy:
   `python scripts/25_auto_agent.py --query-dir queries/dev-2025-finals --submit`.
   Kỷ luật sắt: client **không bao giờ tự retry lệnh bị từ chối** (nộp sai bị trừ điểm).
3. **Profile auto-track** (env, không sửa code): `CVP_SEARCH__LOW_CONFIDENCE_RETRY=true`
   (reformulate + RRF-merge khi ranking phẳng), `CVP_SEARCH__TEMPORAL_BOOST=true`
   (đề "… sau khi …"), VQA dải frame `CVP_VQA__FRAMES_PER_ANSWER=3` (sửa lỗi "giải toán
   trong video" 2025) + Gemini fallback chain `gemini-3.5-flash → gemini-3-flash-preview →
   gemini-2.5-flash` (đã là default config, degrade tiếp → Google Translate → raw).

**Nghiệm thu WS-4:** 89 câu dev chạy tự động 0 can thiệp, 0 CSV bị packager chối; vòng DRES
local: submit top-1 đúng thứ tự, không retry sau reject; MEAN FINAL chế độ auto ≥ 90% chế độ
có người (đo cùng dev-89); adapter spec BTC (nếu đã công bố) có test giả lập server.

## 7. WS-5 · Chung kết (01/09 → 26/09) — và code freeze

**Bối cảnh đã xác minh:** đề chung kết là văn bản HOẶC clip ngắn **chỉ được XEM** (cấm quay/
chụp/capture màn hình; được phép mô tả lại/vẽ/sinh ảnh để đưa vào hệ của mình); audio có thể
bị **tắt tiếng**. Mẫu 2025: KIS 5 phút với 5 hint nhỏ giọt mỗi 1 phút; VKIS 4 phút/clip 20s;
server kiểu DRES — nộp sớm điểm cao, nộp sai bị trừ.

**Drill hàng tuần từ 01/09 (đấu pháp chi tiết: [COMPETITION_PLAYBOOK](COMPETITION_PLAYBOOK.md)):**
- **KIS nhịp hint**: dùng đồng hồ 5' trong app (tab 💬, realtime) + 1 người đóng vai BTC nhả
  hint mỗi 60s từ đề dev; Add hint để agent GỘP mọi hint (không chỉ search hint mới nhất).
  Mục tiêu drill: trung vị thời-gian-tới-đáp-án-đúng < 3 phút trên đề KIS dev.
- **KIS-V không-quay-chụp**: cả đội xem 1 lần → Spotter đọc to chi tiết phân biệt → Driver gõ
  2–3 biến thể; vẽ/sinh ảnh chỉ dùng làm ghi nhớ nội bộ đội (hệ nhận text — không dựng flow
  ảnh mới sau freeze); 🔍 similar + ♻ Refine để leo từ frame gần đúng.
- **Fallback tắt tiếng**: chạy lại bộ VKIS drill với `CVP_SEARCH__WEIGHTS__ASR=0` — nếu điểm
  tụt mạnh thì đấu pháp mặc định cho VKIS là không dựa tín hiệu lời thoại.
- **Cổng latency mỗi ngày thi**: `python scripts/50_bench_latency.py --n 200` trên đúng laptop
  thi — p50 ≤ 200 ms / p95 ≤ 500 ms, vượt là xử lý theo checklist script in ra (text-index?
  cache? display_k?). Kiểm tra cảnh báo **checkpoint drift** (index build bằng PE-Core-bigG
  nhưng máy load DFN5B → đổi máy hoặc rebuild).
- **DRES thật**: cấu hình `CVP_SUBMISSION__DRES_BASE_URL` ngay khi BTC công bố endpoint; chạy
  lại vòng diễn tập WS-4 với endpoint thật (nếu BTC mở practice round).

**CODE FREEZE 05/09/2026 (1 tuần trước chung kết):**
- Sau mốc này: **chỉ bugfix có test kèm**, không feature, không đổi default config đã đo.
- Đóng băng profile thi = 1 file env cho mỗi chế độ (interactive / auto) trong `report/profiles/`.
- Artifacts + checkpoints sync về 2 laptop; diễn tập rút-mạng (Gemini degrade) đạt.

**Nghiệm thu WS-5:** 3 buổi drill trọn vẹn (KIS-hint, VKIS-muted, TRAKE/QA); mọi chỉ số cổng
xanh vào 11/09; checklist PLAYBOOK §0 tick đủ trên cả 2 máy.

## 8. Chỉ tiêu tổng hợp (đo bằng lệnh, không bằng cảm giác)

| Chỉ tiêu | Đích | Lệnh đo |
|---|---|---|
| Test suite | 575 pass (CI, torch-cpu) | `pytest` |
| Dev-89 MEAN FINAL | ghi **B0** ở WS-1 → WS-2 ≥ B0+10% tương đối → trước freeze ≥ B0+20% (hiệu chỉnh sau khi có B0) | `scripts/20 --gt` / `scripts/26` |
| Latency search_text | p50 ≤ 200 ms, p95 ≤ 500 ms (laptop thi) | `scripts/50_bench_latency.py` |
| Artifacts từ drop mới | dùng được ≤48h, đủ aux ≤72h | notebook 01 + bảng WS-2 |
| Map-keyframes khi thiếu | 100% video có raw video được tái tạo, sai lệch ≤ stride so với self-extract đối chứng | `scripts/05` vs `scripts/01` |
| Auto-track | = quy trình người trên dev-89 ≥ 90%, 0 can thiệp | `scripts/25` |
| Lượt nộp sơ tuyển | ≤20 tổng, 0 lượt hỏng format | packager + `report/submission_log.md` |

## 9. Sổ rủi ro

| # | Rủi ro | Phát hiện | Đối sách |
|---|---|---|---|
| R1 | **Drop không có map-keyframes** (~~đã hóa giải cho Batch 1: BTC phát đủ 873/873~~ — còn áp dụng cho batch sau) | kiểm kê ngày-1 mỗi batch | `scripts/05_rebuild_map_keyframes.py` (dhash + monotone DP, resumable, đã tổng duyệt ở WS-1); video nào thiếu cả raw video → `scripts/01` tự cắt |
| R2 | **Lifelog/egocentric metadata schema lạ** (chủ đề 2026, khác media-info HTV) | soi schema ngày-1 | kênh metadata là BM25 tổng quát — nạp field text bất kỳ; nếu có timestamp/GPS đặc thù: chỉ thêm parser mỏng ở `data/metadata.py` TRƯỚC freeze, có test |
| R3 | **Chế độ khung giờ Codabench** (2025: 9:00–11:59 sáng, 2026 chưa rõ) | đọc rules khi trang mở (đầu–giữa T8) | quy trình "chuẩn bị tối hôm trước, nộp đầu giờ sáng" là mặc định của WS-3 — đúng cho CẢ hai chế độ |
| R4 | **Không thuê được H100** cho train | lịch Colab tuần cuối T7 | train là kênh phụ: zero-shot ensemble là config thi mặc định; notebook 02 tự hạ batch cho A100/L4/T4, resume chính xác giữa epoch |
| R5 | **Gemini quota/ban/mạng tại venue** | drill rút-mạng WS-5 | fallback chain 3 model Gemini → Google Translate → raw query (SigLIP-2 đọc tiếng Việt trực tiếp); `vlm_rerank`/`reranker` API tắt được bằng 1 env; đã hỏi BTC tính hợp lệ LLM API (WS-1.3) |
| R6 | **AVS bất ngờ xuất hiện** (tin buổi-2: không thi, nhưng luật đổi từng năm) | đề đợt 1 sơ tuyển / thông báo BTC | đường AVS + MMR vẫn sống sau cờ config, có endpoint `/search/avs`, có ablation A7 — chỉ cần 1 buổi drill kích hoạt lại |
| R7 | **Spec thể thức tự động khác dự đoán** | buổi tập huấn kế / FAQ | không đoán trước: bề mặt service + auto_agent trừu tượng, adapter viết sau khi có spec (WS-4.1), thời gian đệm đã chừa T8–T9 |
| R8 | **1000+ giờ dữ liệu** (corpus >1M keyframes) | catalog ngày-1 | `CVP_INDEX__TYPE=ivf` (hoặc `hnsw`) + đo lại latency; embed chia nhiều phiên Colab (resumable); hạ `extraction.shot_positions` nếu tự cắt |
| R9 | **Checkpoint drift giữa máy build và máy thi** (PE-Core-bigG vs DFN5B) | engine tự cảnh báo khi load | không thi trên máy cảnh báo — rebuild hoặc đổi máy (PLAYBOOK §9) |

## 10. Định nghĩa hoàn thành

Một đồng đội, trên laptop thi, có thể: gõ truy vấn KIS/QA tiếng Việt → thấy keyframe đúng
trong top đầu dưới 1 giây → xác nhận qua context/deep-link → xuất CSV/zip hợp lệ chuẩn
Codabench; chạy TRAKE ra chuỗi frame tăng dần; bật chế độ tự động chạy trọn bộ đề không cần
người; và toàn bộ các con số ở §8 xanh — trước 05/09/2026, trên artifacts build từ dữ liệu
THẬT của BTC 2026.
