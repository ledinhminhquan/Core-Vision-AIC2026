# 🔁 DATASET_INGESTION — Quy trình nạp dữ liệu chuẩn

> Dữ liệu AIC **lớn và về theo từng đợt** (batch/zip tải dần). Đây là quy trình DUY NHẤT
> cần theo mỗi lần thêm dữ liệu — mọi bước đều idempotent + resumable, chạy lại không hỏng gì.
> **Batch 1 vòng sơ tuyển 2026 ĐÃ PHÁT (08/2026)** — kiểm kê xác minh ở §1b; BTC xác nhận
> sẽ phát thêm **batch 2** sau. Định dạng chi tiết từng file: [DATA_FORMAT.md](DATA_FORMAT.md).
> Tổ chức Drive/Colab: [DRIVE_SETUP.md](DRIVE_SETUP.md).

## TL;DR — 3 lệnh sau mỗi đợt data

```bash
# CVP_PATHS__DATA_ROOT trỏ vào data/ đã merge (mặc định ./data)
python scripts/05_rebuild_map_keyframes.py   # CHỈ khi thiếu map-keyframes (xem §4)
python scripts/30_ingest.py                  # (extract) → catalog → embed → FAISS
python scripts/doctor.py                     # kiểm coverage + index staleness
```

---

## 1. Gói dữ liệu của BTC (layout 2025 — 2026 dự kiến tương tự)

Trên Drive chia sẻ của BTC (tham chiếu AIC 2025):

```
AIC2025/                                       (dataset gốc BTC)
├── Keyframes_L21.zip … Keyframes_L30.zip           keyframes L-batch
│   (L26 tách nhỏ: Keyframes_L26_c/d/e.zip)
├── video_batch_1/
│   ├── clip-features-32-aic25-b1.zip               features CLIP ViT-B/32 (L-batch)
│   ├── media-info-aic25-b1.zip                     metadata YouTube       (L-batch)
│   ├── objects-aic25-b1.zip                        detections Open Images (L-batch)
│   ├── map-keyframes-aic25-b1.zip                  cầu nối frame_idx — ✅ CÓ trong batch 1 2026 (§1b)
│   └── Videos_L21_a.zip … Videos_L30_a.zip         video gốc              (L-batch)
├── video_batch_2/
│   └── Videos_K01.zip … Videos_K20.zip             CHỈ video gốc          (K-batch)
└── query/                                          đề chính thức
```

**Hai loại batch — xử lý khác nhau:**

| Batch | Keyframes? | features / objects / media-info? | Việc phải làm |
|---|---|---|---|
| **L** (L21–L30) | ✅ có (`Keyframes_L*`) | ✅ có (`*-aic25-b1`) | giải nén vào `data/` (+ §4 nếu thiếu map) |
| **K** (K01–K20) | ❌ không | ❌ không | **tự cắt keyframes** — `30_ingest.py` tự làm (TransNetV2 → PySceneDetect), map-keyframes CHÍNH XÁC được sinh kèm |

> BTC có thể phát bổ sung keyframes/features cho K-batch (dạng `*-aic25-b2`) —
> khi đó giải nén như L-batch và ingest với `--no-extract`.

## 1b. ✅ Batch 1 sơ tuyển 2026 — KIỂM KÊ ĐÃ XÁC MINH (soi từng zip, 15/08/2026)

BTC phát **32 zip, 106.7 GiB** (mirror `https://aic-data.ledo.io.vn/<tên-file>.zip`,
danh sách chính thức trong Google Sheet của BTC). PDF BTC xác nhận nguyên văn: *"Đây
cũng là dữ liệu batch 1 của AIC 2025"* — tức đúng layout L-batch ở §1, **VÀ CÓ
map-keyframes** (lỗ hổng §4 KHÔNG xảy ra với batch này):

| Gói | Nội dung đã xác minh |
|---|---|
| `Keyframes_L21..L30` (+`L26_a..e`) | 177.321 jpg, `keyframes/{vid}/{nnn}.jpg` 3 chữ số, 1-based, liên tục, khớp `n` |
| `Videos_L21_a..L30_a` (+`L26_a..e`) | 873 mp4, wrapper `video/` |
| `map-keyframes-aic25-b1` | **873 csv ĐỦ**, đúng 4 cột `n,pts_time,fps,frame_idx`, frame_idx tăng nghiêm ngặt |
| `clip-features-32-aic25-b1` | 873 npy `{vid}.npy`, shape (số keyframe, 512), float16, đã L2-normalize |
| `objects-aic25-b1` | 177.321 json `objects/{vid}/{nnn}.json`, keys `detection_*` (giá trị dạng CHUỖI — parser đã float-hóa) |
| `media-info-aic25-b1` | 873 json YouTube (`title/description/keywords/…`) |

Phủ chéo **873/873 video trùng khớp cả 6 thành phần** (L21–L30; L24 bắt đầu từ
V002 — không phải lỗi); mẫu ngẫu nhiên 12 video: số keyframe = số dòng map csv =
số hàng npy. Tên zip route đúng 32/32 qua `guess_dest` của notebook 01 — **ném
nguyên zip vào `MyDrive/AIC2025/data/` là đủ**. KHÔNG có K-batch trong batch 1.
> **2026 lưu ý:** chủ đề năm nay có thể thêm video sousveillance/egocentric, tổng
> có thể 1000+ giờ (FAQ) — đường tự-cắt-keyframes kiểu K-batch chính là bảo hiểm
> cho mọi gói "chỉ có video"; artifact lạ (metadata lifelog…) thì coi như kênh
> metadata BM25 bổ sung, đừng cố nhét vào các thư mục chuẩn.

## 2. Layout `data/` hợp nhất mà code kỳ vọng + mapping giải nén

Mọi thứ nằm dưới MỘT `data_root` (local: `CVP_PATHS__DATA_ROOT`, mặc định `./data`;
Colab: `MyDrive/AIC2025/data` — trên Drive có thể **để nguyên zip**, cell 5 của
notebook 01 tự nhận dạng theo tên và giải nén đúng chỗ, xem DRIVE_SETUP §1):

```
data/
├── keyframes/{vid}/{nnn}.jpg      nnn = thứ tự 1-based, zero-pad (BTC 3 chữ số, tự cắt tới 5)
├── clip-features-32/{vid}.npy     (N,512) float16 — hàng n-1 ↔ keyframe n
├── map-keyframes/{vid}.csv        n,pts_time,fps,frame_idx — CẦU NỐI NỘP BÀI
├── media-info/{vid}.json          title/description/keywords/watch_url
├── objects/{vid}/{nnn}.json       Open Images V4, ≤100 box/frame
└── videos/{vid}.mp4               K-batch BẮT BUỘC; L-batch tùy chọn (ASR + §4)
```

Video id dạng `^[A-Z]\d{2}_V\d{3}$` (`L21_V001`, `K08_V030` — `cvp.constants.VIDEO_ID_RE`).

### Lệnh giải nén (chạy cho MỖI phần mới tải)

```bash
cd data
# L-batch keyframes (zip cấp cao nhất)              -> data/keyframes/
unzip -q -o /path/Keyframes_L28.zip -d .            # chứa keyframes/L28_V*/...
# L-batch artifacts (video_batch_1)
unzip -q -o /path/clip-features-32-aic25-b1.zip -d .   # -> clip-features-32/
unzip -q -o /path/objects-aic25-b1.zip -d .            # -> objects/
unzip -q -o /path/media-info-aic25-b1.zip -d .         # -> media-info/
unzip -q -o /path/map-keyframes-aic25-b1.zip -d .      # -> map-keyframes/ (✅ có trong Batch 1)
# videos: MỌI Videos zip Batch 1 bọc mp4 trong thư mục video/ (số ít — §1b) →
# giải nén ra chỗ tạm rồi dồn phẳng .mp4 vào data/videos/
unzip -q -o /path/Videos_L28_a.zip -d /tmp/vid_unzip
mv /tmp/vid_unzip/video/*.mp4 videos/
```

(PowerShell: `Expand-Archive -Path <zip> -DestinationPath . -Force` tương đương.)

Quên dồn phẳng cũng không chết: ASR và self-extraction có fallback tự dò thêm
`data/videos/video/` — nhưng layout phẳng vẫn là chuẩn. Mọi gói khác (keyframes/
map/objects/…) cũng có wrapper 1 lớp — đích cuối luôn là `data/<artifact>/{vid}...`.

## 3. QUY TRÌNH CHUẨN sau mỗi đợt data

### 3a. Đường một lệnh (khuyên dùng)

```bash
python scripts/30_ingest.py        # --no-extract nếu đã có keyframes chính thức cho mọi video
python scripts/doctor.py
```

`30_ingest.py` làm đúng thứ tự này (chỉ làm phần còn thiếu):

1. **Tự cắt keyframes + map-keyframes** cho video chưa có keyframes (K-batch).
   Video đã có keyframes → bỏ qua. Tắt bằng `--no-extract`.
2. **Rebuild catalog** — video mới nhận `global_id`; id gán lại trên tập video
   đã sort `(video_id, n)` nên rebuild trên cùng corpus luôn cho cùng id.
3. **Embed** chỉ video mới (cache theo video — thêm 50 video vào corpus 2000
   video thì chỉ embed 50). Model theo `CVP_EMBEDDING__MODEL` (mặc định `siglip2`;
   `ensemble` → embed đủ mọi member; `finetuned` dùng chung không gian siglip2).
4. **Rebuild FAISS index** cho từng lane — index ghi signature của catalog,
   index cũ với catalog mới sẽ **từ chối phục vụ** (fail loud) thay vì trả sai
   keyframe. `--force-index` để ép rebuild kể cả khi tưởng là fresh.

### 3b. Đường từng bước (tương đương, khi cần kiểm soát chi tiết)

```bash
python scripts/01_extract_keyframes.py            # K-batch tự cắt (nếu cần)
python scripts/00_build_catalog.py --force        # manifest.parquet + signature
python scripts/02_embed_and_index.py --all-members   # hoặc --model siglip2
python scripts/03_build_aux_indexes.py --objects-index --text-index
# (local cần extra aux: pip install -e ".[aux]" — easyocr/opencv/scenedetect/timm/einops)
python scripts/doctor.py
```

### 3c. Kênh text + objects (KHÔNG nằm trong `30_ingest.py` — chạy riêng)

```bash
# OCR/ASR/captions nặng GPU — thường chạy trong notebook 01; local khi cần:
python scripts/03_build_aux_indexes.py --ocr --asr --captions [--caption-stride 2] [--videos L28_V001 ...]
# Sau MỌI thay đổi corpus/artifacts: nén objects + persist BM25 (nhanh, CPU):
python scripts/03_build_aux_indexes.py --objects-index --text-index
```

- `--text-index` persist BM25 (`artifacts/text_index/*.json.gz`, ghi signature
  catalog — corpus đổi mà quên rebuild thì engine tự fallback in-memory, chậm hơn
  nhưng đúng). Signature khớp → tự skip; ép lại bằng `--force-text-index`.
- `--objects-index` → §5.

### 3d. `doctor.py` — lưới an toàn

In JSON coverage per-artifact (keyframes / map / ocr / asr / captions / objects /
media-info / embedded per-lane) + cảnh báo:

```
⚠  [siglip2] index is STALE — run scripts/30_ingest.py before searching!
```

Không chắc trạng thái → cứ chạy doctor. STALE → chạy lại `30_ingest.py`.
Smoke test sau ingest: `cvp search "người đàn ông áo trắng" --k 5`
(hoặc app Streamlit — **restart app sau ingest** để nạp index mới).

## 4. LỖ HỔNG map-keyframes + tái tạo bằng `scripts/05_rebuild_map_keyframes.py`

**Cập nhật 15/08/2026: Batch 1 chính thức CÓ ĐỦ 873/873 map-keyframes csv (§1b)
— mục này KHÔNG cần cho batch 1.** Giữ lại làm BẢO HIỂM cho batch sau / gói
"chỉ có video": `frame_idx` nộp bài lấy từ chính file map (nộp sai frame_idx =
0 điểm). Thiếu map, catalog vẫn build nhưng `has_map=False` và `frame_idx` chỉ
là ước lượng `n-1` @25fps — **không dùng để nộp được**.

Tái tạo xấp xỉ từ video gốc (cần cả `keyframes/{vid}/` LẪN `videos/{vid}.mp4`):

```bash
python scripts/05_rebuild_map_keyframes.py [--stride 5] [--overwrite] \
    [--videos-dir DIR] [--keyframes-dir DIR] [--out-dir DIR]
```

Cách hoạt động: dhash mọi keyframe jpg → quét video theo bước `--stride` (decode
nhỏ 160×90) → khớp đơn điệu bằng DP (tên keyframe là thứ tự thời gian nên ràng
buộc monotone khử trùng lặp shot) → tinh chỉnh chính xác trong ±stride frame.
Resumable: csv đã có thì skip (trừ `--overwrite`); một video hỏng không chặn cả sweep.

**Thứ tự bắt buộc:** chạy `05` **TRƯỚC** `30_ingest.py` để catalog nhặt được csv
tái tạo. Nếu đã lỡ ingest trước: chạy `05` xong phải
`python scripts/00_build_catalog.py --force` — vì signature catalog chỉ hash
(video_id, số keyframes), thay/thêm map csv KHÔNG đổi signature nên build thường
sẽ tự skip. Embeddings + FAISS **không cần** làm lại (hàng embedding là vị trí,
không phụ thuộc frame_idx).

**Kết quả là XẤP XỈ** — khi BTC phát gói map-keyframes chính thức: giải nén đè
vào `data/map-keyframes/`, rồi lại `00_build_catalog.py --force` (cùng lý do
signature ở trên) → doctor. K-batch không cần bước này: tự cắt keyframes sinh
map CHÍNH XÁC ngay lúc extract.

## 5. Nén objects thành parquet (`--objects-index`)

BTC phát ~178k file JSON nhỏ (1 file/keyframe) — mở hàng nghìn file lúc query
rất chậm, trên Drive/FUSE là thảm họa. Nén một lần:

```bash
python scripts/03_build_aux_indexes.py --objects-index    # [--overwrite] để rebuild
```

→ `artifacts/objects_index/objects.parquet` (zstd, stream từng video, ghi atomic):
một hàng per `(video_id, n)`, giữ detection `score ≥ 0.3` (đúng sàn lọc của
`ObjectBooster` lúc query nên không mất gì). `ObjectBooster` **tự ưu tiên**
parquet khi tồn tại; không có thì per-file `ObjectStore` vẫn chạy như cũ —
nghĩa là bước này là tối ưu, không phải điều kiện.

## 6. Facts đã kiểm chứng trên data 2025 (đọc trước khi viết bất kỳ code chạm data)

1. **873 video batch-1 (L21–L30) và dãy id CÓ LỖ THỦNG** — không tồn tại:
   `L21_V004`, `L21_V020`, `L24_V001`, `L24_V034`, `L26_V417`.
   → **KHÔNG BAO GIỜ sinh id bằng `range()`** — luôn liệt kê từ thư mục thật
   (catalog làm đúng vậy: scan `keyframes/`, lọc theo `VIDEO_ID_RE`).
2. **Objects JSON: mọi giá trị là CHUỖI** — `detection_scores` là `"0.6421"`,
   box cũng là chuỗi, thứ tự box `[ymin, xmin, ymax, xmax]` chuẩn hóa 0–1; khóa
   entity là `detection_class_entities` (fallback `detection_class_names`).
   `ObjectStore` đã cast `float()` sẵn — code mới chạm JSON thô phải tự cast.
3. **clip-features-32: float16, shape `(N,512)`, hàng `n-1` ↔ keyframe `n`**
   (0-based vs tên file 1-based). `ingest_provided_features` cast fp32 + L2-norm
   và **đối chiếu N với số keyframes trong catalog** — lệch là skip video kèm
   cảnh báo (embed lại bằng model thật thay vì dùng features lệch).
4. **Thứ tự entry trong zip KHÔNG được đảm bảo** — đừng bao giờ tin
   `ZipFile.namelist()`/thứ tự giải nén; luôn sort theo tên. Catalog sort
   `(video_id, n)` nên `global_id` deterministic bất kể nguồn giải nén.
5. `map-keyframes` có `fps` **khác nhau theo từng video** (25.0 là phổ biến,
   không phải bất biến) — mọi quy đổi thời gian phải đọc fps từ csv, cấm hardcode.
6. Một model/lane một index — đổi `CVP_EMBEDDING__MODEL` là rebuild lane đó từ
   đầu; `provided_clip32` (512-d) chỉ đủ cho corpus thuần-L, corpus có K-batch
   phải embed thống nhất bằng model thật (siglip2/openclip/…).

## 7. Checklist NGÀY DATA 2026 ĐỔ BỘ (chậm nhất 25/07/2026)

BTC phát dataset + đề bài + baseline + metrics **không muộn hơn 25/07/2026**
(sơ tuyển tháng 8 trên Codabench — tối đa 20 lượt nộp, ≤5/ngày, nên mọi phút
sau data drop đều quý). Chuẩn bị sẵn để chạy ingest NGAY trong ngày:

- [ ] **Trước ngày đó:** trống ≥300GB (Drive + local), `pytest` xanh, notebook 01
      mở sẵn, GPU Colab còn quota, `GEMINI_API_KEY` còn hạn mức.
- [ ] Tải toàn bộ gói → **kiểm kê đối chiếu §1**: batch nào có keyframes/features,
      batch nào chỉ video? Tên gói lệch layout 2025 (vd `*-aic26-*`, loại artifact
      mới cho egocentric/lifelog) → ghi lại, cập nhật DATA_FORMAT trước khi chế code.
- [ ] Giải nén theo mapping §2 (hoặc ném zip lên `MyDrive/AIC2025/data/` cho
      notebook 01 tự lo). Kiểm tra vài `{vid}` bằng mắt: keyframe jpg mở được,
      npy đúng `(N,512)` nếu có.
- [ ] **Có map-keyframes không?** Không → tải kèm videos và chạy
      `scripts/05_rebuild_map_keyframes.py` trước khi ingest (§4); có → so đếm
      dòng csv với số jpg của vài video.
- [ ] `python scripts/30_ingest.py` (Colab notebook 01 cho corpus lớn — mọi bước
      resumable, đứt phiên thì Run all lại).
- [ ] `python scripts/03_build_aux_indexes.py --objects-index --text-index`
      (+ `--ocr --asr --captions` trên GPU khi tới lượt).
- [ ] `python scripts/doctor.py` — coverage đủ, không STALE, số video khớp kiểm kê.
- [ ] Smoke: `cvp search "..." --k 5` ra kết quả hợp lý;
      `python scripts/50_bench_latency.py` đạt p50 ≤ 200ms / p95 ≤ 500ms.
- [ ] Có đề mẫu/baseline của BTC → chạy thử trọn đường nộp:
      `python scripts/20_run_queries.py --query-dir <đề> --zip` (+`--gt` nếu có
      đáp án) — zip phải chứa folder `submission`, validate pass.

## 8. Hiện trạng dữ liệu local (cập nhật 15/08/2026 — Batch 1 chính thức đã về)

| Gói | Ở đâu | Ghi chú |
|---|---|---|
| 32 zip Batch 1 (Keyframes/Videos L21–L30 + 4 gói aux) | `AIC2026-Info/Datasets1/` local | kiểm kê đầy đủ: §1b |
| **map-keyframes-aic25-b1.zip** | **✅ TRONG BATCH 1 — đủ 873/873 csv** | §4 chỉ còn là bảo hiểm batch sau |
| Gói đề chung kết 2025: 89 câu (73 KIS / 9 QA / 7 TRAKE) | local | bộ dev thật cho `scripts/40_eval_official.py` + `scripts/21_tune_weights.py` |

Drive mặc định: `MyDrive/AIC2025/{data,artifacts}` (đổi bằng `DRIVE_PROJECT_DIR`
trong notebook — xem DRIVE_SETUP).

## 9. Quy tắc vàng (để không bao giờ hỏng dữ liệu)

- **Không sửa tay `global_id` / không tái dùng index cũ với catalog mới** — sau
  mỗi lần thêm data cứ để `30_ingest.py` rebuild; guard signature sẽ chặn nếu quên.
- Thêm video làm **xáo lại `global_id`** (id là vị trí) — vì thế index luôn được
  rebuild, nhưng rebuild rẻ (giây–phút) so với embed (đã cache per-video).
- **Một embedding model cho một index** — không đổi model giữa chừng corpus.
- **map-keyframes quyết định điểm số**: L-batch dùng csv BTC khi có, tạm thời
  tái tạo bằng `scripts/05` (xấp xỉ), K-batch tự extract là chính xác; `doctor`
  cho biết video nào còn thiếu (`with_map_keyframes`).
- Sau khi thay map-keyframes: `00_build_catalog.py --force` (signature không đổi
  nên build thường sẽ skip — §4); KHÔNG cần re-embed.
