# 📊 EVALUATION — chấm điểm chính thức, GT format, và tự kiểm tra hệ thống

Tài liệu này mô tả: cách BTC chấm một bài nộp (công thức đã cài **1:1** trong
`src/cvp/eval/official.py` — pure stdlib, chạy mọi nơi, không cần index/GPU),
định dạng ground-truth JSON để tái lập điểm số đó **offline**, ba mức tự kiểm tra
sau khi build/train, vòng tune trọng số, battery ablation A1–A10, và kỷ luật
lượt nộp vòng loại.

> **Bối cảnh 2026 (đã xác minh):** dataset + đề bài + baseline + metrics phát hành
> **không muộn hơn 25/07/2026**; sơ tuyển 8/2026 trên **Codabench** (2026 chưa mở
> trang — dự kiến đầu–giữa T8 dưới organizer `vnaic`; trang 2025 id `10187` là
> tham chiếu định dạng), kết quả 30/8; sơ tuyển **kèm báo cáo giải pháp** (LaTeX
> kit trong `report/`); chung kết on-site **12–26/9/2026** dùng server kiểu DRES
> (DRES 2.0.4): nộp đúng càng sớm càng nhiều điểm, nộp sai bị trừ.

---

## 1. Công thức chính thức (vòng loại Codabench)

Một bài nộp cho MỘT truy vấn là danh sách xếp hạng ≤ 100 dòng `r_1 … r_100`.
Mỗi dòng nhận một **R-Score**, sau đó grader nhìn 5 cutoff:

```
KIS   : R-Score(r_i) = 1( video_i == GT_video  VÀ  frame_i ∈ [s, e] )
QA    : R-Score(r_i) = 1( video ĐÚNG  VÀ  frame trong cửa sổ  VÀ  normalize(answer_i) == normalize(GT_answer) )
TRAKE : R-Score(r_i) = (1/N) · Σ_j 1( frame_{i,j} ∈ [s_j, e_j] )     nếu đúng video, ngược lại 0

R@k   : max R-Score trong k dòng đầu                (k ∈ {1, 5, 20, 50, 100})
Final : trung bình 5 giá trị R@k                    (điểm MỘT truy vấn)
Run   : trung bình Final trên MỌI truy vấn GT       (truy vấn bỏ/nộp hỏng = 0)
```

Cài đặt: `r_score_kis` / `r_score_qa` / `r_score_trake`, `r_at_k`, `final_score`,
`score_rows` / `score_csv` (một truy vấn) và `score_run` (cả folder) trong
`cvp/eval/official.py`. Dòng thứ >100 bị bỏ qua kèm cảnh báo — giống hệt server.

### Hệ quả cần thuộc lòng khi thiết kế đấu pháp

- **Rank 1 đáng giá 5× rank 100.** Với truy vấn hit-nhị-phân (KIS/QA), Final chỉ
  phụ thuộc `best_rank`:

  | `best_rank` của dòng đúng | Final của truy vấn |
  |---|---|
  | 1 | **1.0** |
  | 2–5 | 0.8 |
  | 6–20 | 0.6 |
  | 21–50 | 0.4 |
  | 51–100 | 0.2 |
  | không có | 0.0 |

  Đẩy hit lên **cao** quan trọng hơn nhiều so với "có mặt đâu đó" — đây chính là
  lý do tồn tại của stage rerank cross-encoder (`search.reranker`, mục 7/A9).
- **Không có phạt cho dòng sai** (vòng loại) → **luôn lấp đủ 100 dòng**. Dòng thừa
  chỉ có thể nâng R@50/R@100, không bao giờ hạ điểm. (`search.display_k: 120`
  mặc định đã dư; writer cắt về 100.) *Chung kết thì NGƯỢC LẠI: DRES trừ điểm
  nộp sai — xem `COMPETITION_PLAYBOOK.md`.*
- **TRAKE có điểm phần.** Chuỗi 4 sự kiện trúng 3 cửa sổ → dòng đó 0.75 (ví dụ BTC:
  3/4). DP DANTE (`search/temporal.py`, `temporal.algo: dante`) tối ưu nguyên chuỗi
  cho đúng metric này. Frame thiếu/không parse được = miss, mẫu số luôn là N.
- **QA chấm answer như văn bản tiếng Việt.** `normalize_answer()` casefold + gộp
  whitespace + bỏ dấu câu cuối chuỗi (NFC), nhưng **GIỮ NGUYÊN dấu tiếng Việt**:
  `"mau xanh" != "màu xanh"` — gõ đáp án CÓ DẤU. Answer khớp khi bằng BẤT KỲ
  phần tử nào trong `answers` của GT. Giới hạn 100 ký tự do server áp —
  `sanitize_answer` (hằng `MAX_QA_ANSWER_CHARS`) tự cắt + cảnh báo ở writer/
  packager/VQA/run_queries.
- **`frame_idx` là số frame trong video GỐC** (tra qua `map-keyframes`), *không phải*
  thứ tự keyframe. **Batch 1 (15/08/2026) ĐÃ PHÁT đủ 873/873 map-keyframes csv**
  (DATASET_INGESTION §1b) — unzip là điểm offline đáng tin ngay. Chỉ batch tương
  lai thiếu csv mới cần `python scripts/05_rebuild_map_keyframes.py` (dhash +
  monotone DP, XẤP XỈ). Sai `frame_idx` = 0 điểm dù tìm đúng khoảnh khắc.
- **AVS/KIS-C/KIS-V chấm local theo công thức KIS** (alias trong `_TASK_ALIASES`:
  `avs`/`kis-v`/`kisv`/`vkis`/`video-kis`/`kis-c`/`tkis`/`vqa`…) — TRỪ KHI entry
  GT mang **`targets`** (vòng 3): `"targets": [{"video_id":..., "range":[s,e]},…]`
  — một item cho MỖI đoạn đúng riêng biệt (có thể khác video). Khi đó scorer
  chấm **coverage@k** = tỉ lệ targets được phủ trong k dòng đầu (final = mean
  coverage@{1,5,20,50,100}) — bản mô phỏng offline gần nhất của danh sách AVS
  ẩn của BTC (công thức thật chưa công bố; AVS 2026 **chưa chắc** thi). Không
  có `targets` → vẫn là proxy KIS "tìm ≥1 đoạn đúng" như cũ.

## 2. Định dạng ground-truth JSON

MỘT file JSON cho mỗi bộ đề, key là **stem của file query** (cũng là stem của CSV
nộp, ví dụ `query-p1-1-kis`). Dạng chuẩn:

```json
{
  "query-p1-1-kis":   {"task": "kis",   "video_id": "L01_V001",
                       "range": [500, 510]},
  "query-p1-2-qa":    {"task": "qa",    "video_id": "L05_V005",
                       "range": [800, 900],
                       "answers": ["màu xanh", "xanh dương"]},
  "query-p1-3-trake": {"task": "trake", "video_id": "L10_V010",
                       "moments": [[95, 105], [145, 155]]}
}
```

`load_ground_truth()` chấp nhận thêm các cách viết tương đương (unify tự động):

| Cách viết | Ý nghĩa |
|---|---|
| `"frame_start": 500, "frame_end": 510` | format repo Core-Vision_HCMC-AI cũ |
| `"center": 505, "epsilon": 5` | kiểu BTC: frame tâm ± epsilon (cửa sổ inclusive) |
| `"ranges": [[500,510],[900,940]]` | **NHIỀU cửa sổ chấp nhận được** trong một video — KIS/QA ăn điểm khi frame rơi vào BẤT KỲ cửa sổ nào (dev set kiểu AVS, đáp án lặp lại trong video). Mỗi phần tử cũng có thể là dict `frame_start/frame_end` hoặc `center/epsilon` |
| TRAKE `"events"` | list dict `{"frame_start","frame_end"}`, list cặp `[s,e]`, hoặc list `{"center","epsilon"}` |
| TRAKE `"centers": [c1..cN], "epsilon": e` | N tâm sự kiện + một epsilon chung |
| `"answer": "màu xanh"` | một đáp án duy nhất thay cho `answers` |
| thiếu `"task"` | suy từ stem (ưu tiên substring `trake → avs → qa → kis`) |

Helper dựng GT từ đề BTC (tâm ± epsilon):

```python
from cvp.eval.official import range_from_center, trake_gt

gt = {
    "query-p1-1-kis": {"task": "kis", "video_id": "L01_V001",
                       **range_from_center(center=505, epsilon=5)},
    "query-p1-3-trake": trake_gt("L10_V010", centers=[100, 150], epsilon=5),
}
```

GT hỏng thì **fail to tiếng**: entry KIS/QA không có cửa sổ dùng được →
`ValueError` ngay lúc load (mọi bài nộp sẽ âm thầm 0 điểm nếu để lọt); QA thiếu
answer → truy vấn đó vào `unscored`. Query không có GT → `unscored` (không bao
giờ crash harness); GT không có CSV → `gt_without_submission` **và vẫn tính 0
vào mẫu số run** — đúng như bỏ trắng một câu trên server.

## 3. Chấm một folder submission

```bash
# script — bảng per-query + per-task + run mean/sum; validate cấu trúc mọi CSV
python scripts/40_eval_official.py --submission-dir ./artifacts/submissions/p1 \
    --gt ./queries/p1_gt.json --json-out ./artifacts/eval/p1_official.json
```

```python
# Python / notebook
from cvp.eval.official import score_run, score_submission_dir
report = score_run("artifacts/submissions/p1", "queries/p1_gt.json")
print(report.mean_final, report.by_task)      # RunReport dataclass
legacy = score_submission_dir("artifacts/submissions/p1", "queries/p1_gt.json")
print(legacy["macro"])                        # dict API cũ: macro/macro_scored
```

Report chứa `per_query` (Final, R@k, `best_rank`, `best_score`, `num_rows` từng
CSV), `by_task` (mean Final theo task), `mean_final`/`sum_final`/`mean_r_at`
tính trên **toàn bộ GT** (mẫu số chính thức), và sổ sách `unscored` +
`gt_without_submission`. `scripts/40` còn validate lại mọi CSV theo contract
writer (id video hợp lệ, frame nguyên ≥0, đúng số cột theo task, TRAKE strictly
increasing, ≤100 dòng) — **exit code 2** khi có CSV mà server BTC sẽ từ chối.

## 4. Ba mức tự kiểm tra

### Mức 1 — encoder: `scripts/eval_model.py` (retrieval R@K trên val split)

Trả lời "fine-tune có thắng zero-shot không?" mà không cần đề chính thức:

```bash
python scripts/eval_model.py --baseline siglip2 --candidate finetuned
```

`--baseline`/`--candidate` là **tên model trong registry** (`siglip2`, `finetuned`,
`openclip`, …). In R@1/R@5/R@10, MRR, median rank cho cả hai + **delta**, trên
split val (tách theo VIDEO, chống rò rỉ) từ `artifacts/train_data/`. Delta R@1/R@5
dương rõ rệt mới đáng bật `finetuned` vào ensemble thi đấu (mọi đội top AIC/VBS
thắng bằng zero-shot ensemble — fine-tune là kênh phụ, đo trước khi bật).
Cell verdict của notebook 02 chạy đúng phép so sánh này tự động.

### Mức 2 — hệ thống: chạy đề + chấm công thức BTC (một lệnh)

End-to-end với đủ stack thật (dịch, ensemble, SuperGlobal, fusion, rerank, VQA):

```bash
# một phát: chạy đề → validate → zip Codabench → chấm offline vs GT dev
python scripts/20_run_queries.py --query-dir ./queries/dev --zip --gt ./queries/dev/gt.json

# hoặc tách rời khi cần chấm lại nhiều lần
python scripts/20_run_queries.py --query-dir ./queries/p1 --out-dir ./artifacts/submissions/p1
python scripts/40_eval_official.py --submission-dir ./artifacts/submissions/p1 --gt gt.json
```

`--gt` in bảng per-query + per-task + `MEAN FINAL` — **luôn chấm offline trước
khi tiêu một lượt nộp** (5 lượt/ngày, 20 lượt tổng). `--no-vqa` bỏ bước gợi ý
answer khi chỉ đo retrieval. Notebook 03 wrap đúng flow này trên Colab (đổi
`CVP_EMBEDDING__MODEL` ở ô cấu hình; thả `gt.json` vào thư mục queries trên
Drive là ô chấm điểm tự chạy). Bộ đề dev thường trực là **gói 89 câu chung kết
2025** (73 KIS / 9 QA / 7 TRAKE — có sẵn trong repo: `queries/dev-2025-finals/`).

### Mức 3 — con người: Streamlit UI

```bash
streamlit run app/streamlit_app.py
```

Tự gõ đề luyện tập, kiểm tra dòng top-1 **bằng mắt** (filmstrip/grid, 5 tab
KIS/QA/TRAKE/AVS/Chat + đồng hồ 5 phút progressive-KIS). Đây là mức duy nhất
kiểm tra vòng lặp tương tác sẽ dùng thật ở chung kết — drill trong
`COMPETITION_PLAYBOOK.md`. Nhớ luật chung kết 2026: query có thể là clip **chỉ
được XEM** (cấm quay/chụp/capture — được phép tự mô tả lại/vẽ/sinh ảnh mồi),
audio có thể bị tắt; 2025: KIS 5 phút với 5 hint nhỏ giọt mỗi phút, VKIS 4
phút/clip 20s.

## 5. Vòng tune trọng số tín hiệu (không tốn GPU, không chạy lại engine)

```bash
# 1) dump RAW score map từng tín hiệu (engine chạy 1 lần/câu)
python scripts/23_dump_signals.py --query-dir ./queries/dev --out-dir ./artifacts/signal_dumps/dev

# 2) tối ưu offline trên map đã cache, chấm bằng metric vòng loại
python scripts/21_tune_weights.py --signals-dir ./artifacts/signal_dumps/dev \
    --gt ./queries/dev/gt.json --method random --trials 120 \
    --out ./artifacts/tuning/best_weights.json
```

`scripts/23` normalise query **giống hệt** batch runner (TRAKE bỏ header/bóc
`E1:`, mirror cả `temporal.event_context=prepend`; QA chỉ lấy phần mô tả) — trọng
số được tune trên đúng text mà engine sẽ thấy. `scripts/21` pin `visual=1.0`,
tune 5 tín hiệu phụ (ocr/asr/caption/metadata/object) bằng random-search
(`--method grid` để quét lưới 5 bậc/chiều), KIS chấm bằng chính
`cvp.eval.official.score_rows`; QA re-rank như KIS, TRAKE qua proxy một-frame
(map cache xếp hạng keyframe đơn). Kết quả in sẵn dòng env để dán vào máy thi:

```
CVP_SEARCH__WEIGHTS__VISUAL=1.0
CVP_SEARCH__WEIGHTS__OCR=0.4123
...
```

## 6. Battery ablation A1–A10: `scripts/26_run_ablations.py`

Một lệnh chạy bộ đề một lần cho MỖI biến thể config (deep-copy settings, không
sửa code), chấm bằng scorer chính thức, in + lưu `ablation_results.json`:

```bash
python scripts/26_run_ablations.py --query-dir queries/dev --gt queries/dev/gt.json \
    [--only A1 A9 A10] [--out artifacts/ablations]
```

| ID | Câu hỏi | Biến thể |
|---|---|---|
| A1 | lane composition | `siglip2-only` vs `ensemble(siglip2+openclip)` |
| A2 | SuperGlobal | `search.rerank` off/on |
| A3 | tín hiệu fusion | visual-only vs all-signals vs `fusion_method: rrf` (full sweep = scripts/21) |
| A4 | thuật toán TRAKE | beam vs dante vs dante+ensemble |
| A5 | fine-tune encoder | *ở notebook 02 (tự in bảng riêng)* |
| A6 | VLM listwise rerank | `search.vlm_rerank` off / gemini (cần `GEMINI_API_KEY`) |
| A7 | AVS diversify | `avs_mmr_lambda` 1.0 / 0.7 / 0.5 |
| A8 | QA per-group | `vqa.answers_per_query` 1 vs 5 |
| A9 | **cross-encoder rerank (mới)** | `search.reranker`: none / `blip2_itm` / `qwen_reranker` |
| A10 | **temporal boost + retry (mới)** | `search.temporal_boost` off/on; `search.low_confidence_retry` on |

Một biến thể fail không dừng battery (ghi `FAILED` + tiếp tục). Bảng này là số
liệu trích thẳng vào báo cáo sơ tuyển và bài SOICT/MTA (`docs/PAPER_NOTES.md`).

## 7. Cổng latency: `scripts/50_bench_latency.py`

Chung kết tính giờ theo phút (KIS 5', VKIS 4') — retrieval phải trả lời dưới
giây để người (hoặc auto-agent) giữ được thời gian. Chạy sau MỖI lần rebuild
artifacts và trước MỖI ngày thi:

```bash
python scripts/50_bench_latency.py --n 200 [--query-dir queries/example] [--warmup 5]
```

Đo p50/p95/p99 wall-clock của `search_text` (hot path KIS). **Target: p50 ≤ 200 ms,
p95 ≤ 500 ms** trên laptop thi — vượt target script tự in checklist nghi phạm
(text index persisted chưa? cache warm chưa? ensemble đủ member? `display_k` quá
lớn?). Service FastAPI (`cvp serve`, endpoint `/health`, `/search/*`) dùng chung
engine nên số đo này đại diện luôn cho track tự động.

## 8. Kỷ luật lượt nộp vòng loại + layout zip

- **Ngân sách: tối đa 20 lượt nộp TỔNG, ≤5 lượt/ngày.** 2025 còn giới hạn khung
  giờ nộp buổi sáng 9:00–11:59 — chuẩn bị tinh thần cho CẢ HAI chế độ cho tới
  khi thể lệ 2026 chốt. Quy tắc sắt: **không nộp khi chưa chấm offline** (mức 2)
  và chưa qua validate (`scripts/40` exit 0).
- **CSV**: UTF-8, **không header**, ≤100 dòng, phân cách dấu phẩy. KIS/KIS-V:
  `video_name, frame_id`; QA: `video_name, frame_id, answer` (answer ≤100 ký tự,
  VN hoặc EN, quote khi chứa dấu phẩy); TRAKE: `video_name, f1, ..., fN`
  (strictly increasing).
- **Zip PHẢI chứa folder tên `submission`** — khớp `submission.package_name`
  trong `configs/settings.yaml`; `package_codabench()` zip đúng layout
  `submission/<stem>.csv`, chỉ đóng gói file của run hiện tại (file cũ trong
  folder không bao giờ đi ké), kèm `MANIFEST.json` (sha256 từng file + zip) để
  audit hậu kiểm. Có bất kỳ error validate nào → **từ chối tạo zip** (lượt nộp
  là tài nguyên quý).
- Sơ tuyển 2026 **yêu cầu kèm báo cáo giải pháp** — bảng ablation (mục 6) +
  score offline (mục 3) là nguồn số liệu; LaTeX kit trong `report/`.

## 9. Đọc số liệu — bảng tra nhanh

| Quan sát | Ý nghĩa / hành động |
|---|---|
| `mean_final` tăng nhưng `mean_r_at[1]` đứng yên | đuôi ranking tốt lên nhưng hit chưa lên đỉnh — bật cross-encoder (`search.reranker: blip2_itm` hoặc `qwen_reranker`, đo bằng A9) hoặc tune weights (mục 5) |
| `best_rank` hay rơi vào 5–20 | ensemble TÌM RA nhưng xếp thấp — đúng ca cross-encoder rerank sinh ra để xử |
| KIS cao, QA thấp | localisation ổn, answer sai — soát DẤU tiếng Việt, ≤100 ký tự, kiểm tra `vqa.frames_per_answer` chưa bị hạ dưới 3 (MỘT call Gemini nhìn CẢ DẢI frame — fix ca "giải toán trong video" 2025; thử nâng 5 khi chữ trải dài nhiều khung), verify bằng mắt ở mức 3 |
| TRAKE hay ra ≈ (N−1)/N | một sự kiện trượt cửa sổ có hệ thống — chỉnh `temporal.max_gap_s`/`min_gap_s`, thử `temporal.event_context: prepend`, diễn đạt lại sự kiện đó |
| Truy vấn "… sau khi …" kém | bật `search.temporal_boost: true` (Vortex before/now/after, đo bằng A10) |
| Ranking phẳng, top-1 điểm thấp | bật `search.low_confidence_retry: true` (reformulate + RRF merge, track tự động) |
| Delta encoder (+) nhưng điểm hệ thống đứng yên | kênh dịch/enhance đang gánh — giữ zero-shot ensemble, coi fine-tune là kênh phụ |
| Nhiều `unscored: no ground-truth entry` | stem CSV ≠ key GT — GT phải key theo đúng stem file query |
| `gt_without_submission` khác rỗng | thiếu CSV cho câu đó — vẫn ăn 0 vào mẫu số, kiểm tra batch runner có bỏ sót file đề |
| Điểm `[avs]` trong `by_task` | GT có `targets` → coverage@k đa đoạn (mục 2); không có → proxy KIS. AVS 2026 chưa chắc thi, giữ behind flag |
| Điểm local đẹp nhưng nghi ngờ `frame_idx` | kiểm `map-keyframes` — thiếu thì `scripts/05_rebuild_map_keyframes.py`, đối chiếu vài frame bằng mắt ở mức 3 |
