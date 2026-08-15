# 🇻🇳 HƯỚNG DẪN A-Z — Core Vision Perfect V1

Tài liệu này dắt tay bạn **từng bước, từ đầu đến cuối**: kéo code từ GitHub,
tổ chức Google Drive, chạy 3 notebook trên Colab, kéo artifacts về laptop chạy UI,
nộp bài vòng sơ tuyển, và chuẩn bị cho thể thức tự động 2026. Bạn chỉ cần làm
đúng theo thứ tự — mọi thứ còn lại hệ thống tự lo.

> Đọc sâu hơn: `docs/PROJECT_CONTEXT.md` (toàn bộ ngữ cảnh + thuật toán),
> `docs/DRIVE_SETUP.md` (Drive chi tiết), `docs/COMPETITION_PLAYBOOK.md` (đấu pháp ngày thi).

---

## Bước 0 — Tổng quan 1 trang

**Hệ này làm gì?** Nhận truy vấn tiếng Việt (hoặc mô tả lại một clip) → tìm ĐÚNG
khoảnh khắc trong hàng trăm đến ~1000+ giờ video → xuất `video_name,frame_id`
đúng format nộp bài. Phủ đủ các dạng: **KIS · KIS-V · QA · TRAKE · AVS (để sau cờ
bật/tắt) · KIS-C (phong cách hội thoại)**.

**AIC 2026 có MỘT bài toán, HAI thể thức:**

| Thể thức | Là gì | Hệ này đáp ứng bằng |
|---|---|---|
| **Tương tác** (truyền thống) | Người + trợ lý cùng tìm; dùng agent trong tool là **TÙY CHỌN** | UI Streamlit 5 tab + đồng hồ 5 phút + basket + export CSV |
| **Tự động** (pilot 2026) | Trợ lý đấu trợ lý, không người can thiệp; **BTC CHƯA công bố spec** (chắc chắn KHÔNG phải "tự nộp file") | `scripts/25_auto_agent.py` + dịch vụ HTTP `cvp serve` (nền tảng máy-gọi-máy, đợi spec là lắp adapter) |

**Timeline 2026 (đã xác minh):** dataset + đề bài + baseline + metrics phát hành
**không muộn hơn 25/07/2026** → sơ tuyển **tháng 8/2026** trên Codabench (trang 2026
dự kiến xuất hiện đầu–giữa tháng 8 dưới organizer **vnaic**; trang 2025 id 10187 là
mẫu tham khảo) → kết quả sơ tuyển **30/8** → chung kết on-site **12–26/9/2026** →
trao giải **10/2026**. Sơ tuyển **BẮT BUỘC kèm báo cáo giải pháp** (bài viết mô tả
hệ thống) — đừng để sát ngày mới viết.

**Trình tự việc bạn sẽ làm:**

| # | Việc | Ở đâu | Mất bao lâu |
|---|---|---|---|
| 1 | `git pull` repo (đã có sẵn trên GitHub private) + tạo PAT cho Colab | máy của bạn | 5–10 phút |
| 2 | Ném các file `.zip` của BTC vào Drive (KHÔNG cần giải nén) | trình duyệt | tùy mạng |
| 3 | Colab nb01: build catalog → embed → FAISS → OCR/ASR/caption → BM25 | Colab (GPU nào cũng được) | vài giờ, tự resume |
| 4 | (Tùy chọn — kênh phụ) Colab nb02: train text tower tiếng Việt | Colab H100 | 30–60 phút H100 |
| 5 | Colab nb03: smoke test + latency + CSV mẫu + zip Codabench | Colab / laptop | 15 phút |
| 6 | Kéo `artifacts/` về laptop → chạy UI | máy của bạn | 15 phút |
| 7 | Sơ tuyển: chạy đề → validate → zip → chấm offline → nộp | máy của bạn | theo lịch BTC |

**Điểm quan trọng nhất:** Colab **rớt kết nối giữa chừng** → mở lại →
`Runtime ▸ Run all` → mọi stage **tự chạy tiếp từ chỗ dừng** (embedding, OCR/ASR/
caption, training đều resumable; checkpoint nằm trên Drive).

---

## Bước 1 — GitHub (✅ repo đã push sẵn, chỉ cần pull)

Repo **private** đã có tại:

> **https://github.com/ledinhminhquan/Core-Vision_Perfect_V1**

Trên máy bạn (đã có sẵn thư mục này) chỉ cần cập nhật:

```powershell
cd "E:\Vision AI HCMC\Core-Vision_Perfect_V1"
git pull
```

**Để Colab clone được repo private** — tạo Personal Access Token loại
**fine-grained** (an toàn hơn token cổ điển):

1. GitHub → avatar → **Settings** → **Developer settings** → **Personal access
   tokens ▸ Fine-grained tokens** → *Generate new token*.
2. **Repository access**: *Only select repositories* → chọn
   `Core-Vision_Perfect_V1`.
3. **Permissions**: Repository permissions → **Contents: Read-only** (đủ để clone).
4. Copy token → trong Colab bấm biểu tượng 🔑 (Secrets, thanh trái) → thêm secret
   tên **`GITHUB_TOKEN`**, dán token, bật *Notebook access*.

Cell 3 của cả 3 notebook tự đọc secret này và gắn vào URL clone — bạn không phải
sửa gì. Sau này sửa code trên máy: `git add -A && git commit -m "..." && git push`,
rồi trên Colab chỉ việc Run all lại (cell repo tự `fetch` + `checkout`).

---

## Bước 2 — Tổ chức Google Drive: ném NGUYÊN CÁC ZIP vào

Tạo thư mục `AIC2025` trong **My Drive** (tên này là mặc định của notebook — đổi
được bằng param `DRIVE_PROJECT_DIR`), rồi **ném nguyên các file zip của BTC vào
`data/` — KHÔNG cần giải nén**. Cell 5 của notebook 01 tự nhận diện theo tên file
và bung đúng chỗ (có marker chống bung lại lần sau):

```
MyDrive/AIC2025/
├── data/                  ← ⭐ NÉM NGUYÊN 32 ZIP BATCH-1 VÀO ĐÂY, giữ nguyên tên gốc
│   ├── Keyframes_L21.zip … Keyframes_L30.zip      (+ L26_a…_e — 14 zip keyframes)
│   ├── Videos_L21_a.zip … Videos_L30_a.zip        (+ L26_a…_e — 14 zip video; cần cho ASR)
│   ├── map-keyframes-aic25-b1.zip                 (✅ BTC ĐÃ PHÁT trong Batch 1 — QUAN TRỌNG NHẤT)
│   ├── clip-features-32-aic25-b1.zip
│   ├── media-info-aic25-b1.zip
│   └── objects-aic25-b1.zip
└── artifacts/             ← notebook TỰ TẠO — đừng đụng vào
```

Quy tắc nhận diện tên zip: bắt đầu bằng `Keyframes*` → `data/keyframes/`,
`Videos*` → `data/videos/`, chứa `clip-features` / `media-info` / `objects` /
`map-keyframes` → thư mục tương ứng. Đã giải nén sẵn thứ gì (ví dụ Keyframes_L28
bạn bung sẵn trên máy) thì upload thẳng thư mục con vào `data/keyframes/` cũng được —
hai cách sống chung hòa bình.

**⚠️ Zip nằm trong "Shared with me" KHÔNG dùng được trực tiếp.** *Add shortcut to
Drive* cũng không đủ để notebook ghi/merge. Hãy **copy về My Drive của bạn**
(mở từng file → Make a copy → move vào `AIC2025/data/`) hoặc tải về máy rồi upload lại.

### ✅ map-keyframes: Batch 1 (15/08/2026) ĐÃ PHÁT ĐỦ — cứ upload zip là xong

`map-keyframes/{video}.csv` là **cây cầu nộp bài**: nó đổi số thứ tự keyframe
(`001.jpg`) thành **`frame_idx` thật trong video gốc** — thứ BẮT BUỘC phải nộp.
**Batch 1 chính thức CÓ `map-keyframes-aic25-b1.zip` với ĐỦ 873/873 csv** (đã
kiểm kê từng file — `docs/DATASET_INGESTION.md` §1b): upload zip đó cùng các zip
khác là hết chuyện, KHÔNG cần làm gì thêm.

**Bảo hiểm cho batch sau** (nếu BTC phát gói chỉ-có-video hoặc thiếu csv):
`python scripts/05_rebuild_map_keyframes.py` tái dựng XẤP XỈ từ video bằng
dhash + DP đơn điệu (cần `Videos_*.zip` tương ứng; resumable; khi có bản chính
thức thì ưu tiên bản chính thức). Video tự cắt keyframe (kiểu K-batch) thì map
được sinh kèm CHÍNH XÁC, không cần script này.

**Cách tự kiểm:** ô Catalog của notebook 01 in dòng
`catalog: … keyframes / … videos (N frames with map-keyframes)` — N phải bằng
tổng số keyframe. Nếu N nhỏ hơn hẳn → thiếu map (batch mới chưa unzip đủ?) →
xử lý xong mới được nộp bài.

---

## Bước 3 — Secrets trên Colab (🔑 thanh trái)

| Secret | Bắt buộc? | Dùng cho |
|---|---|---|
| `GITHUB_TOKEN` | **Có** (repo private) | cell 3 clone repo (fine-grained PAT, Bước 1) |
| `GEMINI_API_KEY` | Không — **nên có** | dịch + mô tả lại truy vấn, VQA, rerank VLM, trợ lý KIS-C. Không có thì hệ tự degrade Google Translate → raw query, không chết |
| `HF_TOKEN` | Không | chỉ khi `PUSH_TO_HF_HUB=True` trong nb02 (đẩy model lên HF private) |

Model Gemini mặc định `gemini-3.5-flash`, tự fallback theo chuỗi
`gemini-3-flash-preview` → `gemini-2.5-flash` khi model bị nghỉ hưu/limit
(cấu hình `query.gemini_model` + `query.gemini_model_fallbacks` trong
`configs/settings.yaml`).

---

## Bước 4 — Chạy 3 notebook trên Colab

Mở notebook từ GitHub: Colab → `File ▸ Open notebook ▸ GitHub` → dán
`ledinhminhquan/Core-Vision_Perfect_V1` (cần đăng nhập GitHub trong Colab vì repo
private; hoặc upload file .ipynb từ máy). **Cả 3 notebook: đứt phiên = mở lại =
`Runtime ▸ Run all` — tự resume, không mất gì.**

### 4a. `notebooks/01_build_artifacts_colab.ipynb` — build toàn bộ artifacts

- GPU: **loại nào cũng được** (T4 chạy được; A100/H100 nhanh hơn nhiều cho
  SigLIP-2 + captions).
- Ô PARAMS (ô duy nhất có thể cần sửa): `DRIVE_PROJECT_DIR = "AIC2025"`,
  `EMBED_MODELS = ["siglip2", "openclip"]` (thêm `"qwen_embed"` nếu muốn lane
  tiếng Việt bản địa — nặng; lane `jina` / `metaclip2` cũng đăng ký sẵn trong
  registry cho ai muốn thử), các nút `FORCE_*` mặc định False = resume.
- Trình tự stage: bung zip → copy keyframes về SSD cục bộ (nhanh gấp ~50 lần
  Drive FUSE) → catalog (+ tự cắt keyframes K-batch bằng TransNetV2, fallback
  PySceneDetect) → embed từng lane → FAISS index → OCR/ASR/captions →
  **BM25 text index** → doctor tổng kiểm.
- `provided_clip32` (features ViT-B/32 của BTC) **tự được thêm** vào EMBED_MODELS
  khi thấy `data/clip-features-32/` phủ đủ video.
- **Nhìn dòng in của ô Catalog**: `(N frames with map-keyframes)` — N = 0 nghĩa là
  bạn đang ở tình huống 🔴 của Bước 2 → chạy `scripts/05_rebuild_map_keyframes.py`
  hoặc chờ BTC, đừng bỏ qua.
- Sau này BTC phát batch mới / map-keyframes: ném zip vào `data/` → Run all lại —
  mọi stage incremental.

### 4b. `notebooks/02_train_vi_encoder_H100.ipynb` — train (⚠️ KÊNH PHỤ)

**Đọc kỹ trước khi tốn giờ H100:** giảng viên buổi tập huấn BTC khuyên
**pretrained-only** — mọi đội top AIC/VBS đều thắng bằng zero-shot ensemble +
dịch câu. Train LoRA-LiT ở đây là **kênh bổ sung** cho tên riêng/địa danh tiếng
Việt; **chỉ bật `finetuned` vào ensemble thi đấu khi `scripts/eval_model.py`
đo thấy lãi thật** (Bước 8). FAQ BTC (dòng 9) xác nhận **không giới hạn
model/thuật toán/công cụ** — nghĩa là dùng hay không dùng model train là quyền của bạn.

- GPU: ưu tiên **H100 (~30–60 phút)**; A100 ~2×; L4/T4 chậm hơn nhưng chạy được
  (notebook tự dò micro-batch bằng OOM probe, effective batch giữ nguyên 2048 qua
  gradient accumulation → chất lượng không đổi).
- LiT đóng băng image tower → **FAISS index dùng chung, KHÔNG phải embed lại**.
- Sau train, WiSE-FT quét α ∈ {0.4, 0.5, 0.6, 1.0} → xuất winner vào
  `artifacts/checkpoints/vi_siglip2_best/wiseft_best`.
- Đứt phiên giữa epoch? Run all — resume **chính xác từng bước** (checkpoint
  atomic trên Drive, run-pointer gắn hash config).

### 4c. `notebooks/03_test_system.ipynb` — kiểm tra + đóng gói

Chạy truy vấn tiếng Việt thật (KIS/TRAKE/AVS) và hiện lưới ảnh kết quả, đo
latency, ghi CSV mẫu, **validate + zip đúng chuẩn Codabench**, chấm thử với GT
local (công thức chính thức), dry-run auto-agent (`RUN_AUTO_AGENT=True`), và có
thể phục vụ UI Streamlit qua proxy Colab (`LAUNCH_UI=True`). Ô 5 có biến đổi model
(`CVP_EMBEDDING__MODEL`) để test `siglip2` / `finetuned` / `ensemble` trong cùng phiên.

---

## Bước 5 — Kéo artifacts về laptop + chạy UI

1. Đồng bộ `MyDrive/AIC2025/artifacts/` về máy (Google Drive for Desktop hoặc
   `rclone`). Cần tối thiểu: `catalog/`, `indexes/`, `text_index/`, `embeddings/`
   (cho SuperGlobal/MMR), `ocr/asr/captions` nếu đã build; `data/` cần
   `keyframes/` (hiển thị ảnh) + `map-keyframes/` + `media-info/`.
2. Cài đặt:

   ```bash
   git clone https://github.com/ledinhminhquan/Core-Vision_Perfect_V1.git
   cd Core-Vision_Perfect_V1
   pip install -e ".[search,app,llm,ml]"    # ml: query text encoder chạy torch trên CPU
   ```

3. Trỏ đường dẫn + chạy UI (PowerShell trên Windows):

   ```powershell
   $env:CVP_PATHS__DATA_ROOT      = "D:\AIC\data"
   $env:CVP_PATHS__ARTIFACTS_ROOT = "D:\AIC\artifacts"
   $env:GEMINI_API_KEY            = "..."          # tùy chọn, nên có
   streamlit run app/streamlit_app.py
   ```

   ```bash
   # bash / macOS / Linux
   export CVP_PATHS__DATA_ROOT="/d/AIC/data"
   export CVP_PATHS__ARTIFACTS_ROOT="/d/AIC/artifacts"
   export GEMINI_API_KEY="..."
   streamlit run app/streamlit_app.py
   ```

4. Laptop yếu? `$env:CVP_EMBEDDING__MODEL = "provided_clip32"` — dùng features
   BTC, không cần GPU. `siglip2` trên CPU cũng chạy (chậm ~1–2 s/truy vấn).
5. Kiểm tra sức khỏe corpus bất cứ lúc nào: `python scripts/doctor.py`;
   đo latency: `python scripts/50_bench_latency.py` (mục tiêu **p50 ≤ 200 ms,
   p95 ≤ 500 ms** trên laptop).

Các "đồ chơi" mới của Perfect V1 — bật bằng env khi cần (mặc định tắt để giữ
baseline đã chứng minh):

```powershell
$env:CVP_SEARCH__RERANKER            = "qwen_reranker"   # hoặc "blip2_itm" — rerank chéo cặp (cần GPU/chậm trên CPU)
$env:CVP_SEARCH__TEMPORAL_BOOST      = "true"            # truy vấn "… sau khi …" (Vortex before/now/after)
$env:CVP_SEARCH__LOW_CONFIDENCE_RETRY= "true"            # track tự động: tự viết lại truy vấn khi ranking "phẳng"
$env:CVP_VQA__FRAMES_PER_ANSWER      = "3"               # VQA nhận DẢI frame (mặc định 3) — trị câu "giải toán trong video"
```

---

## Bước 6 — Vòng sơ tuyển (Codabench, tháng 8/2026)

**Một lệnh lo trọn gói** — chạy đề → validate → zip → chấm offline:

```bash
python scripts/20_run_queries.py --query-dir <thư-mục-đề> --zip --gt <gt.json>
# chưa có GT thì bỏ --gt; chưa muốn zip thì bỏ --zip
```

- Zip tạo ra **chứa folder tên `submission`** — đúng yêu cầu BTC (packager tự lo,
  kèm MANIFEST sha256). CSV được validate đủ thể lệ TRƯỚC khi zip: UTF-8,
  **không header, ≤100 dòng**, `video_name,frame_id` (KIS), thêm `answer` ≤100 ký
  tự cho QA (tự quote khi answer chứa dấu phẩy), `video_name,f1,…,fN` (TRAKE).
- Điểm mỗi truy vấn = **trung bình của max R-Score trong top-k, k ∈ {1,5,20,50,100}**
  → hit càng sớm điểm càng cao; luôn điền đủ 100 dòng.
- **KỶ LUẬT NỘP: tối đa 20 lượt tổng, ≤5 lượt/ngày.** Năm 2025 còn giới hạn khung
  giờ nộp buổi sáng 9:00–11:59 — chuẩn bị tinh thần cho CẢ HAI chế độ. Vì vậy:
  **LUÔN chấm offline trước** (`--gt`, hoặc `scripts/40_eval_official.py`) —
  một dòng sai format là mất một lượt.
- Có bộ đề dev kèm đáp án? Tune trọng số tín hiệu trước khi nộp:

  ```bash
  python scripts/23_dump_signals.py --query-dir queries/dev
  python scripts/21_tune_weights.py --signals-dir artifacts/signal_dumps/dev --gt queries/dev/gt.json
  # muốn biết từng nâng cấp đóng góp bao nhiêu:
  python scripts/26_run_ablations.py --query-dir queries/dev --gt queries/dev/gt.json
  ```

- **Đừng quên báo cáo giải pháp** — sơ tuyển yêu cầu nộp kèm bài viết mô tả hệ
  thống. Viết sớm từ nội dung `docs/PROJECT_CONTEXT.md` + số liệu ablation.
- Bạn có sẵn **gói đề chung kết 2025 (89 câu: 73 KIS / 9 QA / 7 TRAKE)** — đó là
  bộ dev tốt nhất hiện có để tập trận với các lệnh trên trong lúc chờ đề 2026.

---

## Bước 7 — Thể thức TỰ ĐỘNG 2026 (chuẩn bị sẵn, chờ spec)

BTC **chưa công bố** giao thức thi tự động (chỉ biết: trợ lý đấu trợ lý, không
người can thiệp). Repo đã chuẩn bị 2 lớp để "spec ra là lắp":

**1) Auto-agent trọn gói** — nhận thư mục đề → tự suy ra task theo tên file →
search (kể cả QA tự trả lời) → CSV → validate → zip:

```bash
python scripts/25_auto_agent.py --query-dir <thư-mục-đề>     # dry-run: KHÔNG nộp gì
# chỉ khi thi chung kết có endpoint DRES + đã bật submission.auto_submit: true:
python scripts/25_auto_agent.py --query-dir <đề> --submit    # đẩy top-1 lên DRES
```

**2) Dịch vụ HTTP máy-gọi-máy** — nếu BTC cho hệ khác gọi vào trợ lý của bạn:

```bash
pip install -e ".[service]"
cvp serve                          # mặc định 127.0.0.1:8000; --host 0.0.0.0 --port 8000
```

Endpoint: `GET /health` · `POST /search/text` · `POST /search/image` (KIS-V, base64) ·
`POST /search/qa` · `POST /search/trake` · `POST /search/avs` ·
`GET /nearest/{global_id}` · `GET /keyframe/{global_id}`. Thử nhanh không cần service: `cvp search "câu truy vấn" --k 10`.

Nên bật cho chế độ tự động: `CVP_SEARCH__LOW_CONFIDENCE_RETRY=true` (tự viết lại
truy vấn khi kết quả "phẳng") và cân nhắc `CVP_SEARCH__RERANKER`.

---

## Bước 8 — Test model sau khi train (3 cách)

**Cách 1 — số liệu chuẩn (quyết định bật/tắt):**

```bash
python scripts/eval_model.py                          # so siglip2 (baseline) vs finetuned
python scripts/eval_model.py --candidate finetuned --baseline siglip2   # tường minh
```

In bảng R@1/R@5/R@10, MRR, MedR kèm **delta** trên tập val (split theo video,
không rò rỉ). **Chỉ khi delta dương rõ ở R@1/R@5** mới đáng đưa vào ensemble thi đấu.

**Cách 2 — trong notebook:** ô Eval của nb02 tự so `siglip2` / `finetuned` /
`wiseft` ngay sau train; nb03 ô 5 đổi `CVP_EMBEDDING__MODEL` để chạy truy vấn
thật và NHÌN kết quả bằng mắt trên cùng câu hỏi.

**Cách 3 — trong UI trên laptop:**

```powershell
$env:CVP_EMBEDDING__MODEL      = "finetuned"
$env:CVP_FINETUNED__CHECKPOINT = "D:\AIC\artifacts\checkpoints\vi_siglip2_best\wiseft_best"
streamlit run app/streamlit_app.py
# đo thấy lãi rồi mới nâng cấp thành ensemble thi đấu:
$env:CVP_EMBEDDING__MODEL           = "ensemble"
$env:CVP_EMBEDDING__ENSEMBLE_MEMBERS= '["finetuned", "openclip"]'
```

Không cần embed lại ảnh trong mọi trường hợp — LiT giữ nguyên image tower nên
`finetuned` dùng chung FAISS index với `siglip2`.

---

## Bước 9 — Câu hỏi thường gặp (FAQ)

| Câu hỏi | Trả lời |
|---|---|
| Colab rớt mạng / hết phiên giữa chừng? | Mở lại notebook → `Runtime ▸ Run all`. Mọi stage tự bỏ qua phần đã xong; train resume chính xác giữa epoch từ checkpoint trên Drive. Mất vài phút, không mất giờ. |
| Không xin được H100? | nb01 chạy GPU nào cũng được. nb02: chọn A100/L4/T4 — notebook tự dò micro-batch (OOM probe), effective batch không đổi; có thể train dở trên L4 rồi resume trên H100. |
| Drive báo hết dung lượng? | Ô 2 của mọi notebook có preflight test ghi/đọc — hỏng là dừng NGAY thay vì chết sau 2 giờ. Dọn `artifacts/hf_cache/`, `artifacts/pip_cache/` (tự tạo lại), thùng rác Drive; cần ≥20 GB trống. |
| Zip nằm ở "Shared with me" dùng được không? | KHÔNG. Phải copy về My Drive của bạn (Make a copy) hoặc tải về rồi upload lại — shortcut không đủ quyền ghi/merge. |
| `index is STALE` khi search? | Bạn thêm data sau khi build index. Chạy lại nb01 (Run all) hoặc `python scripts/30_ingest.py`. Đây là guard cố ý: catalog lệch hàng = `frame_idx` sai = 0 điểm. |
| `No map-keyframes for …` hoặc catalog in `0 frames with map-keyframes`? | Tình huống 🔴 Bước 2: chờ BTC phát zip map-keyframes (ném vào `data/` rồi Run all), hoặc tự dựng `python scripts/05_rebuild_map_keyframes.py` từ videos. KHÔNG nộp bài khi chưa xử lý. |
| Cần API key gì không? | Toàn bộ build/train chạy KHÔNG cần key. `GEMINI_API_KEY` chỉ để dịch/enhance/VQA lúc tìm kiếm (nên có); Gemini lỗi/hết quota → tự fallback Google Translate → raw query, không bao giờ chặn thi đấu. |
| Nộp Codabench bị từ chối format? | Đừng zip tay! `scripts/20 --zip` validate mọi CSV theo thể lệ (không header, ≤100 dòng, answer ≤100 ký tự, TRAKE tăng dần) và tạo zip có folder `submission` chuẩn. Lỗi validate = zip bị chặn, không tốn lượt nộp. |
| Embedding chậm khủng khiếp trên Colab? | Đang đọc JPG qua Drive FUSE. Giữ `COPY_KEYFRAMES_LOCAL=True` (mặc định) — copy về SSD cục bộ trước khi embed. |
| Laptop không có GPU chạy nổi không? | Nổi. `CVP_EMBEDDING__MODEL=provided_clip32` (features BTC, tức thì) hoặc `siglip2` trên CPU (~1–2 s/truy vấn). Cross-encoder rerank (`blip2_itm`/`qwen_reranker`) thì nên để GPU hoặc tắt. |
| AVS năm nay có thi không? | CHƯA chắc (giảng viên buổi 2 nhớ là không có, nhưng thể lệ đổi từng năm). Hệ vẫn hỗ trợ đầy đủ (MMR đa dạng hóa) — cứ để đó, có đề là dùng được ngay. |
| KIS-C là task riêng à? | Chưa — 2026 mô tả nó như "phong cách trợ lý hội thoại" đáng có. App đã có tab hội thoại + đồng hồ 5 phút + episodic log, sẵn sàng nếu BTC biến nó thành task thật. |
| Chung kết được xem clip mấy lần, có tiếng không? | Clip chỉ được XEM trên màn hình (cấm quay/chụp/capture — nhưng được mô tả lại/vẽ/sinh ảnh để đưa vào hệ); âm thanh CÓ THỂ bị tắt. 2025: KIS 5 phút với 5 hint nhỏ giọt mỗi phút; V-KIS 4 phút/clip 20s; server kiểu DRES — nộp sớm điểm cao, nộp sai bị trừ (xem `docs/COMPETITION_PLAYBOOK.md`). |
| Làm sao biết train có "lãi" để bật vào ensemble? | `python scripts/eval_model.py` (Bước 8). Delta R@1/R@5 dương rõ → bật; lằng nhằng → thi bằng zero-shot ensemble mặc định `[siglip2, openclip]` 0.55/0.45 (công thức các đội top). |
| Objects boost đọc chậm trên Drive? | Chạy `python scripts/03_build_aux_indexes.py --objects-index` — gộp hàng trăm nghìn JSON thành 1 parquet, ObjectBooster tự ưu tiên dùng. |
| Kiểm tra tổng thể hệ đang thiếu gì? | `python scripts/doctor.py` — in coverage từng artifact + cảnh báo index stale. Test suite: `pytest` (514 tests, thuần CPU, không cần data thật). |

---

## Phụ lục — Việc nào THỦ CÔNG, việc nào TỰ ĐỘNG?

| Việc | Thủ công (bạn làm) | Tự động (hệ lo) |
|---|---|---|
| Code lên GitHub | `git pull` / `git push` | — |
| PAT + Colab secrets | tạo 1 lần (GITHUB_TOKEN, GEMINI_API_KEY) | notebook tự đọc secrets |
| Upload data | ném zip vào `MyDrive/AIC2025/data/` | nb01 tự nhận diện + giải nén + chống bung lại |
| map-keyframes | quyết định: chờ BTC hay chạy `scripts/05` | script tự khớp dhash + DP, resumable |
| Build artifacts | mở nb01 → Run all | catalog/embed/FAISS/OCR/ASR/caption/BM25 — tất cả resumable |
| Train (kênh phụ) | mở nb02 → Run all; quyết định bật/tắt sau khi đo | OOM probe, resume giữa epoch, WiSE-FT sweep, export best |
| Đánh giá model | đọc bảng delta của `scripts/eval_model.py` | tính R@k/MRR/MedR, chống rò rỉ split |
| Chạy đề sơ tuyển | gõ 1 lệnh `scripts/20 --zip --gt` | parse mọi layout đề BTC, search, VQA, validate, zip, chấm offline |
| Nộp Codabench | upload zip + canh 5 lượt/ngày, 20 tổng | packager đảm bảo format không bao giờ bị từ chối |
| Báo cáo giải pháp sơ tuyển | bạn viết (dựa PROJECT_CONTEXT + ablations) | `scripts/26_run_ablations.py` sinh số liệu |
| Thi tương tác (chung kết) | thao tác UI theo `COMPETITION_PLAYBOOK.md` | search <1s, basket, export, đồng hồ 5', DRES client |
| Thi tự động (chung kết) | chờ spec BTC → lắp adapter | `scripts/25` end-to-end + `cvp serve` HTTP sẵn sàng |
| Theo dõi BTC | ngó portal/Codabench (organizer `vnaic`, đầu–giữa T8) + buổi tập huấn | — |

Chúc đội mình thi đấu thật "perfect"! 🏆
