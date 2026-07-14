# ☁️ DRIVE_SETUP — Tổ chức Google Drive & chạy Colab (từng bước)

Làm đúng file này là chỉ việc mở notebook và **Run all**.

## 1. Cấu trúc thư mục trên Drive

Tạo trong **MyDrive** (tên `AIC2025` là mặc định của notebook — đổi được bằng
param `DRIVE_PROJECT_DIR`):

```
MyDrive/AIC2025/
├── data/                       ← DỮ LIỆU BAN TỔ CHỨC (bạn upload)
│   ├── keyframes/              ← gộp mọi Keyframes_*.zip đã giải nén
│   │   ├── L21_V001/001.jpg …
│   │   └── L30_V0xx/…
│   ├── map-keyframes/          ← L21_V001.csv … (n,pts_time,fps,frame_idx)
│   ├── media-info/             ← L21_V001.json …
│   ├── clip-features-32/       ← L21_V001.npy … (tùy chọn, cho provided_clip32)
│   ├── objects/                ← L21_V001/001.json … (tùy chọn)
│   └── videos/                 ← L21_V001.mp4 … K01_V001.mp4 …
│                                 (K-batch BẮT BUỘC có videos để tự cắt keyframes)
└── artifacts/                  ← NOTEBOOK TỰ TẠO — đừng đụng vào
    ├── catalog/  embeddings/  indexes/  ocr/  asr/  captions/  text_index/
    ├── train_data/  runs/  checkpoints/  submissions/  signal_dumps/
    ├── deliverables/  hf_cache/  pip_cache/  logs/
```

**Mẹo lười (khuyên dùng):** không cần giải nén! Cứ ném thẳng các file zip của ban tổ chức
(`Keyframes_L25.zip`, `clip-features-32-aic25-b1.zip`, `map-keyframes-aic25-b1.zip`,
`media-info-aic25-b1.zip`, `objects-aic25-b1.zip`, `Videos_L21_a.zip`, …) vào
`MyDrive/AIC2025/data/` — cell 5 của notebook 01 tự nhận dạng theo tên và giải nén
đúng chỗ (có marker chống giải nén lại).

Với gói "Shared with me" của ban tổ chức: chuột phải → **Add shortcut to Drive** không đủ
để notebook ghi/merge — hãy **copy** các zip cần dùng vào thư mục của bạn
(mở từng file → Make a copy → move vào `AIC2025/data/`), hoặc tải về rồi upload lại.

## 2. Lưu ý L-batch vs K-batch

| | L21–L30 (batch 1) | K01–K20 (batch 2) |
|---|---|---|
| Keyframes + features + objects + media-info | ✅ có sẵn | ❌ không có |
| Cần upload videos? | Tùy chọn (cho ASR) | **BẮT BUỘC** |
| Xử lý | dùng trực tiếp | notebook tự cắt keyframes (TransNetV2/SceneDetect) + tự tạo map-keyframes |

## 3. Secrets trên Colab (🔑 biểu tượng chìa khóa, thanh trái)

| Tên | Bắt buộc? | Dùng cho |
|---|---|---|
| `GEMINI_API_KEY` | Không (nên có) | dịch + mô tả lại truy vấn, VQA, trợ lý KIS-C |
| `HF_TOKEN` | Không | push model đã train lên HF Hub |

Trong notebook, secrets được đọc bằng `google.colab.userdata` — cell nào cần sẽ tự hỏi quyền.

## 4. Thứ tự chạy

1. `notebooks/01_build_artifacts_colab.ipynb` — GPU nào cũng được (A100/H100 nhanh hơn nhiều).
   Chạy nhiều phiên nếu hết giờ: **mọi bước resumable**, Run all lại là tiếp tục.
2. `notebooks/02_train_vi_encoder_H100.ipynb` — ưu tiên H100; không có thì A100/L4/T4
   (tự thu nhỏ batch). Đứt phiên → Run all lại là **tự resume**.
3. `notebooks/03_test_system.ipynb` — kiểm tra chất lượng + latency + CSV mẫu.

🆕 2026-07-08 — **TransNetV2 cho K-batch**: notebook 01 mặc định tự cài
`transnetv2-pytorch` (param `INSTALL_TRANSNETV2 = True`, cài `--no-deps` nên
KHÔNG BAO GIỜ đụng torch của Colab) để cắt keyframe bằng shot detector mà các
đội top dùng; cài lỗi/thiếu mạng → tự fallback PySceneDetect như cũ. Máy local
muốn dùng: `pip install --no-deps transnetv2-pytorch` (torch đã có từ `[ml]`).

## 5. Chạy UI thi đấu trên laptop

```bash
git clone https://github.com/ledinhminhquan/Core-Vision_Ultimate_Final.git
cd Core-Vision_Ultimate_Final
pip install -e ".[search,app,llm,ml]"        # ml chỉ cần nếu encode local
```

Đồng bộ artifacts từ Drive về (Drive for Desktop hoặc rclone), rồi đặt env theo
đúng shell của bạn *(sửa 12/07: form cũ `set VAR=value  # comment` hỏng trên CẢ
cmd lẫn PowerShell/bash — cmd nhét nguyên comment vào giá trị, PowerShell/bash
thì không set env)*:

```powershell
# PowerShell (khuyên dùng trên Windows)
$env:CVF_PATHS__DATA_ROOT = "D:\AIC\data"
$env:CVF_PATHS__ARTIFACTS_ROOT = "D:\AIC\artifacts"
$env:GEMINI_API_KEY = "..."        # tùy chọn
streamlit run app/streamlit_app.py
```

```bash
# bash / Git Bash / macOS-Linux
export CVF_PATHS__DATA_ROOT="/d/AIC/data"
export CVF_PATHS__ARTIFACTS_ROOT="/d/AIC/artifacts"
export GEMINI_API_KEY="..."        # tùy chọn
streamlit run app/streamlit_app.py
```

Laptop yếu? Đặt `CVF_EMBEDDING__MODEL=provided_clip32` (dùng features ViT-B/32 có sẵn —
không phải embed lại ảnh; vẫn cần `[ml]` extra vì query text encoder chạy torch trên CPU).
Notebook 01 **tự thêm** `provided_clip32` vào EMBED_MODELS khi thấy `data/clip-features-32/`.
Còn `siglip2` chạy CPU cũng được (chậm hơn ~1–2s/query).

## 5b. Nộp bài vòng loại + tune trọng số

```bash
# chạy cả gói đề → CSV + validate + zip Codabench (MANIFEST kèm sha256):
python scripts/20_run_queries.py --query-dir <gói-đề> --zip
# thể thức TỰ ĐỘNG 2026 (trọn gói, kể cả QA tự trả lời):
python scripts/25_auto_agent.py --query-dir <gói-đề>
# có bộ đề dev kèm đáp án? Tune trọng số tín hiệu:
python scripts/23_dump_signals.py --query-dir queries/dev
python scripts/21_tune_weights.py --signals-dir artifacts/signal_dumps/dev --gt queries/dev/gt.json
# chấm điểm offline đúng công thức BTC:
python scripts/40_eval_official.py --submission-dir artifacts/submissions --gt queries/dev/gt.json
```

Nhớ thể lệ sơ tuyển 2026: tối đa **5 lượt nộp/ngày, 20 lượt tổng** — chấm offline trước khi nộp!

## 6. Lỗi thường gặp

| Triệu chứng | Nguyên nhân → cách sửa |
|---|---|
| `Keyframes folder not found` | sai `DRIVE_PROJECT_DIR` hoặc chưa giải nén — kiểm tra `data/keyframes/` |
| `index is STALE` | thêm data sau khi build index → chạy lại nb 01 (hoặc `scripts/30_ingest.py`) |
| `No map-keyframes for …` (L-batch) | thiếu map-keyframes zip → frame_idx sẽ SAI khi nộp. Upload đủ! |
| Embedding chậm khủng khiếp | đọc JPG qua Drive FUSE → bật `COPY_KEYFRAMES_LOCAL=True` (mặc định) |
| OOM khi train | notebook tự chọn micro-batch; nếu vẫn OOM giảm `MICRO_BATCH` một nấc |
| Gemini lỗi/limit | hệ thống tự fallback Google Translate → raw query; không chặn thi đấu |
