# ☁️ COLAB_GUIDE — Toàn bộ pipeline GPU, từng bước một

Mọi thứ cần GPU đều chạy trên Colab qua **3 notebook, theo thứ tự**. File này nói rõ:
upload gì, bấm gì, mỗi stage mất bao lâu trên GPU nào, tốn khoảng bao nhiêu compute
unit (theo mức đo THẬT trên tài khoản Pro+ đang dùng), và chuyện gì xảy ra khi Colab
rớt kết nối (spoiler: không mất gì cả).

Điều kiện tiên quyết: layout Drive theo **[DRIVE_SETUP.md](DRIVE_SETUP.md)** — một thư mục
`MyDrive/AIC2025/{data, artifacts}` (`DRIVE_PROJECT_DIR` đổi được trong cell PARAMS),
zip dữ liệu BTC ném thẳng vào `data/` là notebook 01 tự giải nén.

| # | Notebook | Vai trò | GPU khuyên dùng |
|---|---|---|---|
| 01 | `01_build_artifacts_colab.ipynb` | catalog → tự cắt keyframe K-batch → embed + FAISS mọi lane → OCR/ASR/captions → BM25 text index | A100 (L4 được, chậm) |
| 02 | `02_train_vi_encoder_H100.ipynb` | LoRA-LiT fine-tune text tower SigLIP-2 (autopilot H100) | **H100** |
| 03 | `03_test_system.ipynb` | smoke test end-to-end: search, CSV, validate + zip Codabench, chấm GT offline, dry-run automatic track | GPU nào cũng được |

> ⚠ Các file `.ipynb` là **sản phẩm build** — đừng sửa tay. Nguồn duy nhất là
> `notebooks/_build_notebooks.py` (sửa xong chạy `python notebooks/_build_notebooks.py`).

---

## 0. Bộ khung chung (cả 3 notebook)

Mỗi notebook mở đầu bằng đúng 4 cell, thứ tự cố định:

1. **PARAMS** — cell DUY NHẤT có thể cần sửa (`DRIVE_PROJECT_DIR`, `EMBED_MODELS`,
   `FORCE_*`…). Đặt env chống phân mảnh VRAM **trước** khi import torch.
2. **Mount Drive + preflight write test** — Drive không ghi được (hết quota/mất quyền)
   thì fail NGAY tại đây, không phải sau 2 giờ chạy. HF cache + pip cache đặt trên Drive
   → model/wheel chỉ tải một lần cho mọi session.
3. **Repo + dependencies (kỷ luật v12)** — clone GitHub (repo private: thêm secret
   `GITHUB_TOKEN`; clone fail → tự fallback bản copy repo trên Drive). Check version qua
   `importlib.metadata` (KHÔNG import trước khi nâng cấp), **không bao giờ đụng torch
   của Colab**, chỉ cài đúng gói thiếu, `pip check` + micro-fix tối đa 2 vòng, purge
   `sys.modules` trước khi `import cvp`. Chạy lại là no-op.
4. **Env + GPU** — trỏ `CVP_PATHS__DATA_ROOT` / `CVP_PATHS__ARTIFACTS_ROOT` vào Drive,
   bật TF32/SDPA, nạp secrets → env.

**Dò năng lực, không dò tên GPU:** notebook đọc `VRAM_GB` + `torch.cuda.is_bf16_supported()`
rồi tự chọn batch (notebook 02 còn probe OOM thực nghiệm ở cell 6). Nghĩa là khi không
lấy được H100 thì A100/L4/T4 **cứ thế chạy** — không có nhánh code nào so sánh chuỗi
"H100"; GPU xịn hơn chỉ mua thêm tốc độ.

**Quy tắc phục hồi vạn năng** cho MỌI sự cố (rớt kết nối, OOM, Drive giật):
**reconnect → Runtime → Run all**. Mọi cell đều idempotent + có toggle `FORCE_*`;
việc đã xong được skip trong vài giây (chi tiết granularity ở mục 5).

### 0b. Secrets trên Colab (🔑 thanh trái → Secrets)

| Secret | Bắt buộc? | Dùng cho |
|---|---|---|
| `GEMINI_API_KEY` | Nên có | dịch + mô tả lại truy vấn, VQA, VLM rerank, trợ lý KIS-C. Model theo `configs/settings.yaml`: `gemini-3.7-flash`, tự fallback `gemini-3.5-flash` → `gemini-flash-latest`; hết cả chuỗi → Google Translate → raw query (không bao giờ chặn thi đấu) |
| `HF_TOKEN` | Không | `PUSH_TO_HF_HUB=True` ở notebook 02 (đẩy checkpoint lên repo HF private); kéo dataset public khi HF yêu cầu đăng nhập |
| `GITHUB_TOKEN` | Khi repo private | fine-grained PAT để cell 3 clone `Core-Vision_Perfect_V1` |

Cell 4 tự nạp `GEMINI_API_KEY`/`HF_TOKEN` vào env (mỗi secret sẽ hỏi quyền lần đầu).

---

## 1. Notebook 01 — `01_build_artifacts_colab.ipynb` (build toàn bộ artifacts)

**Làm gì:** giải nén zip BTC còn nằm trong `data/` (marker chống giải nén lại) → copy
keyframes về đĩa local (Drive FUSE chậm ~50× khi đọc trăm nghìn JPG nhỏ) → catalog +
**tự cắt keyframe cho video K-batch** (TransNetV2, fallback PySceneDetect) + sync
keyframes/map-keyframes mới về Drive → embed + FAISS **từng lane** trong `EMBED_MODELS`
→ OCR/ASR/captions (resumable từng video) → **BM25 text index persisted** → `doctor()`
báo cáo coverage. Timing từng stage ghi vào `artifacts/logs/nb01.log`.

**Knobs (cell PARAMS):**

| Param | Mặc định | Ý nghĩa |
|---|---|---|
| `DRIVE_PROJECT_DIR` | `"AIC2025"` | `MyDrive/<đây>/{data, artifacts}` |
| `EMBED_MODELS` | `["siglip2", "openclip"]` | các lane dense (thứ tự = thứ tự ensemble). Options: `siglip2` (đa ngữ, cần cho training), `openclip` (lane EN: PE-Core-bigG → fallback DFN5B), `qwen_embed` (nặng, tùy chọn), `provided_clip32` (features BTC — **tự thêm** khi `clip-features-32/` phủ đủ 100% video). Lane `jina` / `metaclip2` cũng hợp lệ (registry hỗ trợ) — xem mục 6 trước khi bật |
| `COPY_KEYFRAMES_LOCAL` | `True` | copy keyframes Drive → `/content/data` trước khi embed (tmp + rename, an toàn khi đứt giữa chừng) |
| `RUN_OCR / RUN_ASR / RUN_CAPTIONS` | `True, True, True` | các kênh BM25; captions chậm nhất và **cần cho notebook 02** |
| `CAPTION_STRIDE` | `2` | caption mỗi keyframe thứ 2 (nhanh gấp đôi) |
| `INSTALL_TRANSNETV2` | `True` | cài `transnetv2-pytorch --no-deps` (+ `ffmpeg-python` + `future`) — verify bằng import thật rồi mới báo READY; fail → PySceneDetect |
| `FORCE_CATALOG / FORCE_EMBED / FORCE_INDEX / FORCE_AUX / FORCE_TEXT_INDEX` | `False` | mặc định resume/skip; bật để làm lại từ đầu đúng stage đó |

**Chọn GPU:** embed là bài toán throughput — **A100 là điểm ngọt**; L4 đủ dùng nếu chạy
qua đêm; H100 chỉ đáng khi cần gấp (unit/h cao, xem mục 4). Lane `openclip`
(PE-Core-bigG-14-448) nặng hơn hẳn `siglip2` — T4 không khuyến khích cho lane này.

**Thời gian ước tính** (mỗi ~100k keyframes ≈ 1 L-batch lớn; corpus L21–L30 873 video
sẽ là vài lần con số này):

| Stage | H100 | A100 | L4 | T4 |
|---|---|---|---|---|
| giải nén + copy local (I/O, không phụ thuộc GPU) | 10–25 min | 10–25 min | 10–25 min | 10–25 min |
| SigLIP-2 embed + FAISS | ~15–25 min | ~25–40 min | ~1.5–2.5 h | ~3–4 h |
| openclip PE-Core-bigG (lane 2) | +~30–45 min | +~1–1.5 h | +~4–6 h | không khuyến khích |
| Vintern captions (stride 2) | ~1.5–3 h | ~3–5 h | qua đêm | qua đêm+ |
| OCR / ASR (mỗi cái) | ~1–2 h | ~2–3 h | chậm — qua đêm hoặc chạy sau | chậm |
| BM25 text index + doctor | phút | phút | phút | phút |

### 1b. ✅ map-keyframes — Batch 1 (15/08/2026) ĐÃ PHÁT ĐỦ

Batch 1 chính thức gồm 32 zip cho 873 video L21–L30, **TRONG ĐÓ CÓ
`map-keyframes-aic25-b1.zip` với đủ 873/873 csv** (kiểm kê từng file:
`DATASET_INGESTION.md` §1b). Upload zip đó cùng các zip khác vào `data/` —
notebook 01 tự giải nén, `frame_idx` nộp bài là số CHÍNH XÁC của BTC.

Bảo hiểm cho batch tương lai (chỉ khi thiếu csv), cả hai đã có sẵn trong repo:

- **Video tự cắt (kiểu K-batch):** `extract_missing` của notebook 01 tự sinh
  map-keyframes CHÍNH XÁC khi cắt — không phải làm gì thêm.
- **Keyframes BTC nhưng thiếu csv:** rebuild XẤP XỈ bằng dhash + monotone DP
  (cần có video gốc):

  ```bash
  python scripts/05_rebuild_map_keyframes.py \
      --videos-dir <data>/videos --keyframes-dir <data>/keyframes \
      --out-dir <data>/map-keyframes [--stride 5] [--overwrite]
  ```

  Chạy được cả trong một cell Colab sau khi stage videos, hoặc trên laptop.
  Khi BTC phát bản chính thức thì ưu tiên bản chính thức, xóa csv rebuild đi.

### 1c. Tùy chọn sau nb01: objects parquet nén

Objects JSON per-keyframe đọc qua Drive rất chậm. Gộp một lần thành parquet
(ObjectBooster tự ưu tiên dùng khi thấy):

```bash
python scripts/03_build_aux_indexes.py --objects-index
```

---

## 2. Notebook 02 — `02_train_vi_encoder_H100.ipynb` (train rồi đi ngủ)

**Làm gì:** LoRA-LiT fine-tune **text tower** SigLIP-2 trên caption tiếng Việt của chính
corpus (+ KTVIC/UIT-ViIC nếu `USE_PUBLIC_DATA=True`) — image tower đóng băng nên FAISS
index của notebook 01 dùng lại nguyên vẹn. Đây là autopilot đúng nghĩa:

- **OOM probe thực nghiệm** (cell 6, `PROBE_BATCH=True`): bảng VRAM chỉ là đoán ban đầu
  (H100/A100-80G→512, A100-40G→256, L4→128, T4→64); probe chạy forward+backward THẬT
  trên đúng text tower, OOM thì chia đôi (floor 8). `GRAD_ACCUM` được tính lại để
  **effective batch = `TARGET_EFF_BATCH` (2048) không bao giờ đổi** — probe chỉ tinh
  chỉnh throughput, không đổi ngữ nghĩa training.
- **Run pointer gắn sha256 config** (`artifacts/runs/vi_siglip2/active_train_run.json`,
  ghi atomic tmp+rename): Run all lại là **tự resume** chừng nào config chưa đổi;
  dashboard read-only in trạng thái phiên trước (status, checkpoint cuối, best R@5).
  `micro_batch`/`grad_accum`/`num_workers` là **resume-neutral** — probe retune giữa các
  session không phá identity của run. Config đổi thật → cảnh báo CONFIG MISMATCH to rõ.
- **Resume chính xác giữa epoch** (`plan_epoch_steps` + RNG state trong checkpoint) —
  không train lặp batch sau khi đứt; checkpoint atomic lên Drive mỗi
  `EVAL_EVERY_STEPS=200` bước.
- **WiSE-FT sweep α ∈ [0.4, 0.5, 0.6, 1.0]** cuối training — α=1.0 = tower thô nên
  winner không bao giờ tệ hơn checkpoint chưa nội suy; xuất
  `checkpoints/vi_siglip2_best/wiseft_best`.
- **Deliverables mirror**: bản export tốt nhất copy sang `artifacts/deliverables/latest`
  (copytree, không symlink — Drive không hỗ trợ symlink).
- Cell 12 so **zero-shot vs finetuned vs WiSE-FT** trên val split; cell 13 in đúng các
  env cần đặt để dùng model (`CVP_EMBEDDING__MODEL=finetuned`,
  `CVP_FINETUNED__CHECKPOINT=…`) + tùy chọn push HF Hub.

**Thời gian** (EPOCHS=8, early-stop, corpus caption vài trăm nghìn cặp):

| Stage | H100 | A100 | L4 |
|---|---|---|---|
| public parquets + train-data assembly (cached) | phút | phút | phút |
| LoRA-LiT training (early stop) | **~30–60 min** | ~1–2 h | ~4–8 h (chạy được, nhưng nên bắt đầu bằng H100) |
| WiSE-FT sweep + eval 3 model | ~10–15 min | ~15–25 min | ~40–60 min |

Text tower nhỏ so với GPU — đây là notebook duy nhất H100 thật sự đáng tiền. Cũng có thể
**bắt đầu trên L4, kết thúc trên H100**: checkpoint nằm trên Drive, run pointer không
quan tâm GPU nào đã ghi nó.

**Nhớ bằng chứng:** mọi đội top AIC/VBS thắng bằng zero-shot ensemble + dịch câu.
Fine-tune là **kênh phụ** cho tên riêng/địa danh tiếng Việt — đo bằng
`python scripts/eval_model.py` (so `--baseline siglip2` vs `--candidate finetuned`,
cùng không gian ảnh siglip2) và chấm trên gói đề dev trước khi bật vào ensemble thi đấu.

---

## 3. Notebook 03 — `03_test_system.ipynb` (kiểm tra + đóng gói + chấm thử)

**Làm gì:** nạp engine (cell 5 đặt `CVP_EMBEDDING__MODEL="siglip2"` — đổi thành
`finetuned`/`ensemble` để test model khác; `CVP_QUERY__PROVIDER="none"` = test offline
không cần Gemini) → 5 câu KIS tiếng Việt hiển thị lưới ảnh inline → smoke TRAKE + AVS →
ghi CSV đúng format Codabench → **validate + package** (`validate_submission_dir`
strict; còn lỗi chặn là KHÔNG tạo zip — mỗi dòng sai là mất một lượt nộp; zip chứa đúng
folder `submission` + MANIFEST sha256) → **chấm GT offline** đúng công thức BTC nếu có
`MyDrive/AIC2025/queries/gt.json` (R@k, k ∈ {1,5,20,50,100}, Final = mean) →
**`RUN_AUTO_AGENT`** (mặc định `False`): dry-run trọn gói thể thức tự động
(`run_auto(..., submit=False)` — đọc đề từ `queries/example/*.txt`, tự suy task theo tên
file, search, CSV, validate, zip; không nộp gì) → tùy chọn serve Streamlit qua proxy Colab.

**Thời gian:** vài phút trên GPU bất kỳ (search ~1s/câu; bật
`search.reranker: blip2_itm|qwen_reranker` cộng thêm vài giây/câu trên L4, dưới 1s trên
H100). Notebook 03 chạy được **ngay sau notebook 01** (zero-shot) nếu bỏ qua training.

**Bộ đề dev khuyên dùng:** gói **89 câu chung kết 2025** (73 KIS / 9 QA / 7 TRAKE) —
upload vào `MyDrive/AIC2025/queries/dev-2025-finals/` + `gt.json`, rồi trên laptop (hoặc cell Colab):

```bash
# một phát: chạy đề → validate → zip → chấm offline (tiết kiệm lượt nộp: 5/ngày, 20 tổng)
python scripts/20_run_queries.py --query-dir queries/dev-2025-finals --zip --gt queries/dev-2025-finals/gt.json
# battery ablation A1–A10 (lane, SuperGlobal, fusion, TRAKE, VLM rerank, cross-rerank, temporal boost…)
python scripts/26_run_ablations.py --query-dir queries/dev-2025-finals --gt queries/dev-2025-finals/gt.json [--only A9 A10]
# latency (mục tiêu: p50 ≤ 200 ms, p95 ≤ 500 ms trên laptop)
python scripts/50_bench_latency.py --n 200 --query-dir queries/dev-2025-finals
```

**Nền cho thể thức tự động 2026:** spec BTC chưa công bố (chắc chắn KHÔNG phải "nộp file
tự động"), nhưng nền machine-callable đã sẵn: `cvp serve [--host 0.0.0.0 --port 8000]`
mở FastAPI với `/health`, `/search/text|image|qa|trake|avs`, `/nearest/{global_id}`,
`/keyframe/{global_id}` — khi BTC ra protocol chỉ cần viết adapter mỏng lên trên.

---

## 4. Chi phí compute unit (Colab Pro+, mức đo thật trên tài khoản này)

Tốc độ đốt: **H100 ≈ 18.05 units/h (đo thật trên tài khoản đang dùng)** · A100 ≈ 8–9 ·
L4 ≈ 4–5 · T4 ≈ 1.5–2. Con số A100/L4/T4 là xấp xỉ phổ biến giữa 2026 — **kiểm tra lại
trong panel Resources của chính session** vì Google đổi giá theo thời điểm/khu vực.

| Plan | Tiết kiệm | Nhanh |
|---|---|---|
| Notebook 01 (index 2 lane + captions) | L4 qua đêm ≈ 25–45 units | A100 ≈ 30–55 units (H100 nhanh hơn nữa nhưng unit tương đương hoặc hơn — chỉ đáng khi gấp) |
| Notebook 02 (training) | A100 ≈ 10–18 units | **H100 ≈ 10–18 units — nhanh 2–3× nên tổng unit KHÔNG đắt hơn A100; luôn ưu tiên H100 cho nb02** |
| Notebook 03 (test/đóng gói) | GPU nào cũng ≈ 1–3 units | — |
| Lane tùy chọn (qwen_embed / jina / metaclip2, mỗi lane) | A100 ≈ 10–25 units | H100 ≈ 15–30 units |

Trọn pipeline mặc định ≈ **50–100 units** — Pro+ (500 units/tháng) dư sức chạy nhiều
vòng rebuild khi BTC phát hành data 2026 (≤ 25/07/2026). Lưu ý sắc bén nhất với Pro+:
**runtime H100 để idle vẫn đốt ~18 units/h** — xong notebook là
**Runtime → Disconnect and delete runtime** ngay.

---

## 5. Hành vi resume sau disconnect (những gì được ĐẢM BẢO)

Quy tắc vàng: **Run all sau khi reconnect = resume.** Cụ thể từng stage:

| Stage | Đơn vị resume | Disconnect mất gì |
|---|---|---|
| cài dependencies | cả cell (check metadata, không cài lại) | ~1–2 min re-check |
| giải nén zip BTC | mỗi zip (marker `.unzipped-*`) | chỉ zip đang dở giải nén lại |
| copy keyframes → local | mỗi thư mục con (tmp + rename) | thư mục đang copy dở |
| tự cắt keyframe K-batch | mỗi video | video đang cắt |
| embedding | mỗi video (`.npy` đã có là skip) | video đang embed |
| captions / OCR / ASR | mỗi video (JSON đã có là skip) | video đang xử lý |
| FAISS index | build lại từ embeddings đã cache | vài giây–phút |
| BM25 text index | signature catalog (+ rebuild khi aux có thay đổi) | một lần build lại |
| training | checkpoint atomic gần nhất (≤ `EVAL_EVERY_STEPS`=200 bước) + **resume chính xác giữa epoch** (RNG-reproducible) | ≤ 200 bước (~1–2 min trên H100) |
| WiSE-FT sweep | chạy lại cuối training | vài phút |
| submissions / chấm GT | mỗi CSV | câu đang chạy |

Hai điều kiện giữ cho bảng trên đúng: artifacts nằm trên **Drive** (mặc định của
cell mount), và **không sửa config training giữa run** — đổi config là đổi sha256,
run pointer sẽ cảnh báo CONFIG MISMATCH thay vì âm thầm trộn hai run (riêng
`MICRO_BATCH`/`GRAD_ACCUM` do probe chỉnh thì miễn nhiễm, xem mục 2).

---

## 6. Khi nào bật lane tùy chọn (qwen_embed / jina / metaclip2)?

Mặc định trước thi đấu trong `configs/settings.yaml` là ensemble
**[siglip2, openclip] 0.55/0.45**. Lưu ý: line-up THẬT SỰ dự thi vòng 3 là
**[finetuned, metaclip2] 0.6/0.4** (`_LINEUP_BATTLE`, `notebooks/09_campaign.ipynb`). Ba lane
tùy chọn đều NẶNG (mỗi lane = re-embed toàn bộ corpus + thêm một index + thêm RAM/VRAM
lúc query), chỉ thêm khi **còn thời gian VÀ đo được lợi ích**:

| Lane | Model | Lưu ý |
|---|---|---|
| `qwen_embed` | `Qwen/Qwen3-VL-Embedding-2B` (MRL cắt 1024-d) | R@1 tiếng Việt zero-shot cao nhất trong model mở; cần `transformers>=4.57`; instruction chỉ áp phía query |
| `jina` | `jinaai/jina-clip-v2` (Matryoshka 64–1024-d) | 89 ngôn ngữ; license **CC-BY-NC** — ổn cho thi đấu học thuật, không dùng thương mại |
| `metaclip2` | `facebook/metaclip-2-worldwide-huge-quickgelu` | multilingual SOTA 2026 (XM3600); nặng tương đương bigG |

Quy trình chuẩn (đo trước, bật sau):

1. Notebook 01: thêm tên lane vào `EMBED_MODELS` (vd `["siglip2", "openclip", "jina"]`)
   → Run all — chỉ lane mới bị embed, các lane cũ skip.
2. Chấm A/B trên gói 89 câu dev bằng scorer chính thức:
   `python scripts/20_run_queries.py --query-dir queries/dev-2025-finals --gt queries/dev-2025-finals/gt.json`
   với `CVP_EMBEDDING__MODEL=ensemble` +
   `CVP_EMBEDDING__ENSEMBLE_MEMBERS='["siglip2","openclip","jina"]'` (so với baseline
   2 lane); hoặc chạy battery `scripts/26_run_ablations.py` (A1 = lane composition).
   Riêng tower fine-tune (cùng không gian ảnh siglip2) so nhanh bằng
   `python scripts/eval_model.py --baseline siglip2 --candidate finetuned`.
3. Thắng có ý nghĩa → tune trọng số (`scripts/23_dump_signals.py` +
   `scripts/21_tune_weights.py`) rồi mới ghim vào cấu hình thi đấu.
4. Kiểm tra latency còn đạt mục tiêu: `python scripts/50_bench_latency.py --n 200`
   (p50 ≤ 200 ms, p95 ≤ 500 ms — chung kết theo nhịp DRES 4–5 phút/câu, nộp sớm điểm
   cao nên tốc độ là điểm số).

Tương tự cho cross-encoder rerank (`CVP_SEARCH__RERANKER=blip2_itm|qwen_reranker`):
đo bằng ablation **A9** + bench latency trước khi bật mặc định.

---

## 7. Troubleshooting

| Triệu chứng | Cách xử lý |
|---|---|
| `Drive mount FAILED` / write test fail | chạy lại cell mount, duyệt popup; kiểm tra quota Drive |
| clone GitHub fail | repo private → thêm secret `GITHUB_TOKEN` (fine-grained PAT); hoặc upload repo vào `MyDrive/AIC2025/Core-Vision_Perfect_V1` (cell 3 tự fallback) |
| numpy/llvmlite conflict sau khi cài | bạn đã tự cài torch — đừng; restart runtime, chạy lại cell deps nguyên bản |
| `model tag mismatch` khi embed | session này nạp checkpoint KHÁC session trước (vd PE-Core hụt mạng → fallback DFN5B) — khôi phục mạng để nạp đúng checkpoint, hoặc `FORCE_EMBED=True` re-embed sạch lane đó |
| TransNetV2 báo `unavailable` | không sao — extraction tự fallback PySceneDetect; muốn TransNetV2 thì chạy lại cell 7 khi có mạng |
| `No map-keyframes for …` trong doctor | L-batch thiếu csv → `frame_idx` nộp sẽ SAI. Rebuild bằng `scripts/05_rebuild_map_keyframes.py` (mục 1b) hoặc chờ csv chính thức |
| OOM khi train | chạy lại cell probe (`PROBE_BATCH=True`) hoặc hạ `MICRO_BATCH` một nấc; effective batch giữ nguyên nhờ grad accumulation |
| Cảnh báo `CONFIG MISMATCH` ở nb02 | cố ý đổi config? — xóa `artifacts/runs/vi_siglip2` để train sạch; không thì revert config để resume |
| Drive `input/output error` giữa run | remount Drive, chạy lại cell; ghi atomic nên không có artifact hỏng |
| `index is STALE` ở nb03 | catalog đổi sau khi build index → chạy lại cell embed/index của nb01 (hoặc `scripts/30_ingest.py`) |
| Embedding chậm khủng khiếp | đang đọc JPG qua Drive FUSE → để `COPY_KEYFRAMES_LOCAL=True` (mặc định) |
| Gemini lỗi / hết quota | tự degrade theo chuỗi fallback → Google Translate → raw query; không chặn pipeline |
