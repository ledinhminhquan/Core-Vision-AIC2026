# report.md — Nhật ký nghiên cứu của Cursor (bản sao cách ly Core-Vision_Perfect_V1)

File cộng dồn theo từng phiên (luật an toàn #6 của CURSOR_BRIEF.md). Claude đọc file này
để review/cherry-pick. Mọi khẳng định chưa kiểm được trên máy này đều ghi rõ **[chưa xác minh]**.

---

## Phiên 2026-08-27 21:35 — KHẢO SÁT TRAKE (không sửa code)

### 0. Tóm tắt phiên

- **Test baseline: XANH** — `python -m pytest tests -q` → **763 passed, 1 skipped, 0 failed** (~104s).
- Nhiệm vụ phiên: chỉ khảo sát. **File tạo mới duy nhất: `report.md` (file này). Không sửa file code nào.**
- Đã đọc trọn đường đi TRAKE: `pipeline/run_queries.py` (parse đề) → `search/engine.py::search_trake`
  → `search/temporal.py::trake_search` (pool video + DANTE DP) → `submission/writer.py::write_trake`
  → `pipeline/auto_agent.py::run_auto` (validate/zip/submit). Đã đọc `eval/official.py` +
  `eval/metrics.py` (công thức chấm), `scripts/62/63/23/21/26`, `config.py` + `configs/settings.yaml`
  (cơ chế env override `CVP_SECTION__KEY`, ví dụ `CVP_TEMPORAL__PER_EVENT_TOPK=300`),
  `models/query_processor.py`, `data/catalog.py`, cell bench trong `notebooks/_build_notebooks.py`,
  `tests/test_temporal.py` + fixture `conftest.py`.
- **Kết luận lớn nhất**: TRAKE đang là "công dân hạng hai" của pipeline — KIS được hưởng
  4-5 biến thể query Gemini, BM25 4 kênh, object boost, neighbor boost, SuperGlobal,
  cross-encoder Qwen-8B, VLM rerank ×3, low-confidence retry; **TRAKE chỉ có cosine thô
  của text sự kiện nguyên bản** (không enhancement, không text signals, không rerank, không retry).
  Cộng thêm một **trần đo lường** do lưới keyframe thưa vs cửa sổ GT ±12 frame (xem H1) thì
  0.183 vs 0.729 là giải thích được.

### 1. Sơ đồ luồng TRAKE (vẽ bằng chữ, đã đối chiếu code từng bước)

```
query-pX-N-trake.txt  (BTC: 1 dòng header ngữ cảnh + "E1: …" … "Ek: …")
        │
        ▼
run_auto (pipeline/auto_agent.py)            ← chạy cả pack, mỗi file 1 query
  └─ run_query_file (pipeline/run_queries.py)
       ├─ infer_task(tên file) → "trake"     (ưu tiên chuỗi con trake→avs→qa→kis)
       ├─ load_query_lines: đọc utf-8-sig, strip ký tự Cf vô hình, bỏ dòng rỗng
       ├─ parse_trake_events:
       │    • ≥2 dòng khớp regex "En[:.)-hoặc khoảng trắng]" → CHỈ các dòng đó là event,
       │      strip prefix "En", DROP header (mặc định temporal.event_context=none;
       │      "prepend" thì ghép header vào đầu TỪNG event — knob A/B có sẵn)
       │    • nhãn trùng kiểu E1,E2,E2,E4 (đề thật p1-18, p2-22) vẫn ăn: lấy theo thứ tự dòng
       └─ engine.search_trake(events)   (search/engine.py)
            ├─ lọc event rỗng
            ├─ query_processor.process(TỪNG event)   ← gọi Gemini enhance/translate (có cache)
            ├─ _texts_for(model):
            │    • lane multilingual  → CHỈ p.original (text Việt THÔ, 1 biến thể/event)
            │    • lane English-only  → enhanced ▸ translation ▸ expansions[0]
            │    ⚠ Đội hình trận = finetuned(SigLIP-2 VI) + metaclip2, CẢ HAI multilingual=True
            │      → enhancement/expansions Gemini bị VỨT trên đường TRAKE (vẫn tốn API call)
            ├─ temporal.use_ensemble=true & ≥2 lane sống → event_vecs per member + stores per member
            └─ trake_search(...)   (search/temporal.py)
                 ├─ [1] POOL VIDEO:
                 │     mỗi member: FAISS store.search(event_vecs, per_event_topk=100)/event
                 │     → minmax per event per member × weight → gộp hits
                 │     → pool_videos: mỗi video giữ best-score-per-event,
                 │       xếp theo SUM best per-event, giữ top max_videos=30
                 │     ⚠ video đúng không lọt top-100 FAISS của event nào → LOẠI VĨNH VIỄN → 0đ
                 ├─ [2] DP CHÍNH XÁC TỪNG VIDEO (trên TOÀN BỘ frame của video, không chỉ hits):
                 │     rows = catalog.video_rows(vid) sort theo n  (n, frame_idx, pts_time)
                 │     sim (frames×K):
                 │       • ensemble: Σ_w minmax-per-event(cosine per member); minmax tính trên
                 │         TOÀN BỘ pooled videos (range chết → 0.5 trung tính); thiếu embedding
                 │         của member ở video nào thì renormalize weight các member còn lại
                 │       • đơn lane: cosine thô
                 │     DANTE DP (mặc định, algo=dante):
                 │       DP[j,t] = sim[t,j] − λ·time[t] + max_τ(DP[j−1,τ] + λ·time[τ])
                 │       τ: strictly trước t, min_gap_s=0 ≤ gap ≤ max_gap_s=150, sliding-window
                 │       max O(N·K); sim_floor=0.10 chặn frame yếu (tự nới nếu giết cả cột);
                 │       λ=gap_penalty_per_s=0.0005 ưu tiên chuỗi sít
                 │       → top beam_size=8 chuỗi/video, last-frame phân biệt
                 │       (pts không đơn điệu → fallback beam DP; algo=beam là solver cũ)
                 ├─ [3] monotonize_frame_idxs: frame_idx bằng nhau (map csv có 614 cặp tie)
                 │     → bơm +1; GIẢM thật (map lỗi trộn fallback) → drop chuỗi
                 └─ [4] score = (tổng chuỗi đã phạt gap)/K → gộp mọi video, sort desc, top-100
        │
        ▼
write_trake (submission/writer.py): mỗi dòng "video_id,f1,…,fK";
  ép strictly-increasing, dedup nguyên tuple, ≤100 dòng, ghi atomic
        │
        ▼
validate_file (packager, strict) → zip Codabench → (tùy chọn) DRES submit top-1
  (top1_times: pts giây/event → ms cho DRES; _reconcile_times bảo vệ khớp dòng 1)
```

Các knob TRAKE hiện có (tất cả override được bằng env `CVP_TEMPORAL__*`):
`algo` (dante|beam), `use_ensemble`, `per_event_topk=100`, `max_videos=30`,
`min_gap_s=0`, `max_gap_s=150`, `sim_floor=0.10`, `beam_size=8`,
`gap_penalty_per_s=0.0005`, `event_context` (none|prepend).

### 2. Bất đối xứng KIS vs TRAKE trong chính pipeline (đối chiếu code, là sự thật chứ không phải giả thuyết)

| Tầng | KIS (`search_text`) | TRAKE (`search_trake`) |
|---|---|---|
| Biến thể query | original + enhanced + translation + 2 expansions, max-fuse | **1 biến thể/event: text thô** (lane multilingual) |
| Dense pool | topk=500/lane + SuperGlobal refine | per_event_topk=100/lane/event, **không SuperGlobal** |
| Text signals | BM25 ocr/asr/caption/metadata (weights tuned) | **không có** |
| Object boost / neighbor boost | có / có | **không / không** |
| Cross-encoder (Qwen3-VL-8B, bench bật) | có | **không** |
| VLM rerank Gemini ×3 (bench bật) | có | **không** |
| Low-confidence retry (bench bật) | có | **không** |
| Metric offline | đúng video + frame trong ±125×5 cửa sổ | đúng video + **TỪNG event trong ±12** (điểm phần theo event) |

### 3. Bench offline chấm TRAKE thế nào (nguồn số 0.183)

- GT dựng bởi `scripts/62_build_gt_from_reference.py` từ bài tham chiếu **19.8/23**:
  TRAKE = dòng top-1 của reference, mỗi event là cửa sổ **±12 frame (~0.5s @25fps)**;
  KIS/QA = top-5 frame cùng video, mỗi frame ±125 (~5s). Reference ~86% đúng → GT có nhiễu.
- Cell bench (notebooks lab, `_build_notebooks.py` ~dòng 2380-2432): ensemble
  `finetuned+metaclip2` 60/40 qua env, VLM rerank 48×3, qwen_reranker, low-conf retry,
  weights tuned+shrinkage 50% → `run_auto` → `score_run` → `bench_full.json` (+ xoay `-prev`).
  Các knob rerank/retry đó **không chạm** nhánh TRAKE (chỉ KIS/QA/AVS hưởng).
- Công thức: R-Score dòng = (số event có frame trong cửa sổ)/K nếu đúng video, 0 nếu sai video;
  R@k = max R-Score trong k dòng đầu (k∈{1,5,20,50,100}); Final câu = trung bình 5 R@k.
- Lưu ý phụ: `21_tune_weights` chấm TRAKE bằng **proxy 1-frame** và `23_dump_signals` dump
  TRAKE như 1 câu KIS (nối events bằng ". ") → trọng số fusion tuned KHÔNG tối ưu cho TRAKE
  thật; vô hại hiện nay vì đường TRAKE không dùng các trọng số đó, nhưng đừng đọc nhầm
  per-task "trake" của scripts/21 là điểm TRAKE thật.
- 873 video / ~177K keyframes → **~203 keyframe/video**; tin tức ~15-25 phút → khoảng cách
  keyframe trung bình **~5-7s (~125-175 frame @25fps)** [ước tính từ số liệu brief, chưa xác
  minh phân bố thật vì máy này không có artifacts].

### 4. Giả thuyết XẾP HẠNG vì sao TRAKE 0.183 << KIS 0.729 (kèm cách kiểm chứng)

Xếp theo (khả năng đóng góp vào gap) × (độ tin của bằng chứng hiện có). Mỗi giả thuyết có
cách kiểm chứng rẻ nhất; phần lớn cần artifacts nên sẽ đóng gói vào script chẩn đoán
(Nhiệm vụ 1, bước A) để chủ dự án chạy trên Colab.

**H1 — Trần lượng tử hóa: lưới keyframe thưa vs cửa sổ GT ±12 frame (nghi phạm số 1, một
phần là artifact đo lường).** Submission chỉ nộp được frame_idx CỦA keyframe; nếu tham chiếu
19.8/23 không nộp trên đúng lưới keyframe của ta, xác suất tồn tại keyframe trong cửa sổ 25
frame quanh mỗi mốc GT chỉ ~25/148 ≈ **17-25%** — trùng khớp đáng ngờ với 0.183. Nghĩa là:
kể cả chọn ĐÚNG video và keyframe GẦN NHẤT cho mọi event, điểm bench vẫn kẹt quanh mức này.
*Kiểm chứng*: script chẩn đoán đọc `gt.json` + `manifest.parquet` (Colab): với mỗi mốc GT,
có tồn tại keyframe trong ±12 không → in "trần lý thuyết" per query; so `lab_full/*.csv`
với trần đó. Nếu trần ≈ 0.2 thì ưu tiên #1 là chiến lược nộp frame NGOÀI lưới keyframe
(nội suy giữa 2 keyframe + jitter nhiều dòng, xem H6) chứ không phải retrieval. Cần hỏi
chủ dự án: reference 19.8/23 là của đội mình (cùng lưới) hay đội khác (khác lưới)?
**[chưa xác minh — cần artifacts]**

**H2 — Recall video ở bước pooling: video đúng không vào nổi top-30.** Sai video = 0 tuyệt
đối cho cả câu (metric nhân đôi hình phạt: R@1..R@100 đều 0). Nguyên nhân kép: (a) event
text sau khi DROP header quá chung chung ("khoảnh khắc đầu tiên thấy cắt nấm" khớp trăm
video nấu ăn), (b) per_event_topk=100/177K frame quá nông — video đúng nhưng mỗi frame chỉ
suýt soát top-100 sẽ rớt cả 4 event → sum=0. *Kiểm chứng*: script chẩn đoán log
`pool_videos` output: GT video có trong pool 30 không, đứng hạng mấy, best-per-event score
bao nhiêu; đếm tỷ lệ trên 7 câu TRAKE của pack thử nghiệm. Chi phí sửa nếu đúng: tăng
`per_event_topk`/`max_videos` (thuần config) + H3/H4/H5.

**H3 — Vứt ngữ cảnh video (event_context=none là default).** Header đề thật luôn chứa manh
mối chọn video ("cảnh nấu món cá", "camera an ninh về một lần khám xét", "lắp ráp trong một
xưởng", "Hoàng Thành Thăng Long phía sau người phỏng vấn") — knob `prepend` có sẵn nhưng
chưa từng bench [theo brief không thấy số A/B nào]. *Kiểm chứng*: chạy bench 2 lần
`CVP_TEMPORAL__EVENT_CONTEXT=none|prepend` (hoặc thêm biến thể "prepend chỉ cho bước pooling,
giữ event trần cho DP" — xem Nhiệm vụ 1 mục B2), so per-query TRAKE. Đo được ngay bằng
hạ tầng hiện có, 0 dòng code mới.

**H4 — Lane đa ngôn ngữ bỏ toàn bộ Gemini enhancement/expansions cho TRAKE.** KIS thắng nhờ
max-fuse 4-5 biến thể; TRAKE encode đúng 1 câu thô/event trong khi vẫn TỐN call Gemini xử lý
từng event (kết quả bị vứt vì `_texts_for` chỉ lấy `p.original` khi multilingual). Sự kiện
kiểu "khoảnh khắc đầu tiên X chạm Y" là dạng text mà enhancement Gemini (mô tả thị giác cụ
thể) giúp CLIP-tower nhiều nhất. *Kiểm chứng*: thêm knob `event_query_variants` (mặc định =
hành vi cũ), sim per event = max qua biến thể; A/B bench. Offline test được bằng fixture
(vector giả, xem kế hoạch test).

**H5 — TRAKE mù trước captions dày/ASR/OCR.** Artifacts mới có 177K caption Vintern stride-1
mô tả HÀNH ĐỘNG từng keyframe — tín hiệu bước↔frame lý tưởng (brief mục 5 cũng gợi ý đúng
hướng này), nhưng ma trận sim của DP hiện chỉ có cosine ảnh. Sự kiện tinh vi ("mở lửa",
"gập khăn") caption-BM25 phân biệt tốt hơn CLIP. *Kiểm chứng*: prototype
`caption_signal_weight` blend BM25-caption per event vào sim matrix (config-gated, degrade
khi thiếu artifacts), A/B bench; offline unit test bằng text store fixture nhỏ.

**H6 — Lãng phí ngân sách 100 dòng + không khai thác metric max-over-rows.** Hiện nộp ≤8
chuỗi/video × ~13-30 video: các dòng cùng video gần trùng nhau (chỉ khác last-frame), các
dòng video sai chắc chắn 0đ. Metric lấy MAX theo dòng trong top-k nên 100 dòng là 100 vé số:
chiến lược trội hơn là dồn dòng cho ứng viên tốt với **jitter frame quanh từng event**
(keyframe lân cận ±1..2 VÀ frame nội suy GIỮA hai keyframe — phá luôn trần H1; với đề
"khoảnh khắc ĐẦU TIÊN" nên bias về phía đầu shot). *Kiểm chứng*: mô phỏng offline THUẦN
số học trên Colab: từ chuỗi top hiện có + manifest, sinh biến thể jitter, chấm lại bằng
`score_rows` — không cần chạy lại engine; nếu +lớn thì code `submit_strategy: jitter`
(mặc định legacy).

**H7 — Không có tầng verify/rerank cho chuỗi.** KIS bench bật Qwen-8B + VLM ×3 (+10% H@1
theo tài liệu repo) còn TRAKE dùng thẳng điểm DP. Gemini nhìn dải K frame là verifier rẻ
(≤30 call/câu). *Kiểm chứng*: prototype `trake_vlm_verify` (config-gated, tái dùng plumbing
`vlm_rerank`), A/B bench trên top-10÷20 candidate.

**H8 — Ràng buộc thời gian sai với đề trải dài: max_gap_s=150.** Đề p3-24 (4 địa danh xuất
hiện lần đầu trong phóng sự du lịch) các mốc có thể cách nhau >2.5 phút → chuỗi đúng bị CẤM
về mặt cấu trúc, DP buộc chọn chuỗi sít sai chỗ; λ phạt gap càng đẩy về chuỗi sít.
*Kiểm chứng*: đo phân bố gap thật giữa các mốc GT trong `gt.json` (1 lệnh pandas trên Colab);
nếu có câu >150s → A/B `CVP_TEMPORAL__MAX_GAP_S=300|600`.

**H9 — GT nhiễu ở mẫu quá nhỏ.** Reference 19.8/23 ≈ 86% đúng, pack thử nghiệm chỉ có vài
câu TRAKE (pack dev-2025-finals: 7/89 câu) → 1-2 câu GT sai kéo trung bình xuống rất mạnh
(mỗi câu nặng ~14%). *Kiểm chứng*: chủ dự án xem tay 7 cặp (đề, GT video+mốc) trên Colab
(có keyframe ảnh); đồng thời in per-query final từ `bench_full.json` để biết 0.183 là "đều
thấp" hay "vài câu 0 kéo sập" — hai bức tranh dẫn tới thuốc khác nhau.

**H10 — Bench đo một thứ, BTC chấm một thứ.** ±12 frame là lựa chọn của scripts/62, chưa có
căn cứ BTC 2026 dùng cửa sổ nào; nếu BTC rộng hơn (ví dụ theo shot/moment), điểm thật có thể
cao hơn 0.183 và độ ưu tiên các fix đổi hẳn (H1 nhẹ đi, H2/H3 nặng lên). *Kiểm chứng*: chạy
lại `score_run` với gt epsilon 12/25/50/125 (chỉ cần dựng lại gt.json bằng scripts/62 với
`--trake-epsilon` khác — 3 phút/lần trên Colab) → vẽ độ nhạy điểm TRAKE theo epsilon; đọc
kỹ công bố thể lệ 2026 khi có.

Hai lỗi/nợ nhỏ ghi nhận thêm (không xếp hạng): (i) `63_ensemble_runs` merge TRAKE theo danh
tính NGUYÊN TUPLE — hai lượt chạy hầu như không bao giờ trùng nguyên tuple K frame → RRF
thoái hóa thành xen kẽ 2 danh sách, ít giá trị cho TRAKE (Nhiệm vụ 2 sẽ mở rộng danh tính
mềm hơn: theo video hoặc theo trùng ≥⌈K/2⌉ event-window); (ii) `search_trake` tốn call
Gemini cho từng event dù kết quả bị vứt ở lineup hiện tại (cache đỡ phần nào) — sẽ tự khỏi
nếu làm H4.

### 5. Kế hoạch chi tiết — Nhiệm vụ 1 (TRAKE deep-dive)

Nguyên tắc chung (theo brief): mọi cải tiến có knob config riêng, **mặc định = hành vi cũ**;
mỗi mảnh có unit test fixture nhỏ (pattern `conftest.corpus_with_index` + `_basis` vector
trong `test_temporal.py`); bench-before-adopt do chủ dự án chạy Colab.

**Bước A — Chẩn đoán trước, sửa sau (script mới `scripts/66_trake_diag.py` + test).**
Input: `--gt gt.json --submissions lab_full/ [--pool-log]` (+ artifacts trên Colab).
Output bảng per-query:
1. Trần lưới: mỗi mốc GT có keyframe trong ±12? → trần lý thuyết per query (H1).
2. GT video có trong pool-30 không, hạng mấy (H2) — cần chạy lại phần pooling với engine
   thật trên Colab, hoặc log từ run bench (thêm debug-log gated, không đổi hành vi).
3. Khoảng cách |frame nộp − mốc GT| per event của dòng TỐT NHẤT (đang trượt bao xa).
4. Phân bố gap giữa các mốc GT vs `max_gap_s` (H8).
5. Độ nhạy epsilon 12/25/50/125 (H10).
Phần 1/3/4/5 thuần số học — viết + test offline được ngay bằng fixture tự tạo; phần 2 cần
GPU/artifacts. Đây là bước quyết định thứ tự làm B.

**Bước B — Các cải tiến config-gated, xếp theo kỳ vọng lãi/chi phí** (thứ tự có thể đảo sau bước A):
1. `temporal.submit_strategy: legacy|jitter` (H6, H1): sau khi rank candidate, lấy top-V
   video tốt nhất, phủ 100 dòng bằng biến thể jitter per event (keyframe lân cận + nội suy
   giữa keyframe, bias sớm cho "khoảnh khắc đầu tiên"); bảo toàn strictly-increasing, dedup.
   Sửa: `temporal.py` (sinh biến thể) + `TemporalCfg`; test: property (tăng nghiêm ngặt,
   ≤100, ưu tiên thứ tự score), golden case fixture. KHÔNG cần GPU để test.
2. `temporal.pool_query: events|events+header|events+fullquery` (H3, H2): thêm biến thể
   dùng header/toàn văn CHO RIÊNG bước pooling (DP vẫn chấm bằng event trần — tách "chọn
   video" khỏi "canh mốc"). Sửa: `engine.search_trake` + `temporal.py::pool_videos` nhận
   thêm hits; test stub store.
3. `temporal.event_query_variants: original|all` (H4): sim per event = max qua biến thể
   Gemini đã cache (original/enhanced/expansions), cả nhánh đơn lane lẫn ensemble. Sửa:
   `_texts_for` (engine) + `_cosine_sim`/`_ensemble_video_sims` (temporal, nhận nhiều vec
   per event rồi max trước minmax); test: vector giả 2 biến thể trong đó chỉ biến thể 2
   khớp — bật knob mới tìm thấy chuỗi.
4. Config-only sweep (H2, H8): `per_event_topk` 100→300/500, `max_videos` 30→60,
   `max_gap_s` 150→300/600 — 0 dòng code, chỉ cần bench (đưa vào battery A4 mở rộng của
   `scripts/26_run_ablations.py`).
5. `temporal.caption_signal_weight: 0.0` (H5): blend BM25 caption per event vào sim trước
   DP (minmax per event, degrade sạch khi thiếu text index). Sửa: `temporal.py` +
   `engine.search_trake` (truyền TextSignals per-video theo hàng keyframe); test: fixture
   caption chứa từ khóa event ở đúng frame → chuỗi xoay theo khi bật weight.
6. `temporal.vlm_verify: false` (H7): Gemini chấm dải K frame cho top-N candidate, blend
   điểm; prototype sau cùng vì tốn quota + cần bench mới biết đáng. Test bằng provider stub.

**Bước C — Nghiệm thu**: suite xanh; bench A/B từng knob (một knob một lần chạy, theo luật
bench-before-adopt); dùng `scripts/65_bench_diff.py` (Nhiệm vụ 3) để đọc delta per-query.

### 6. Kế hoạch chi tiết — Nhiệm vụ 2 (adaptive multi-attempt, `scripts/64_adaptive_attempts.py`)

Thiết kế 3 pha chạy offline được hoàn toàn:

1. **Module mới `src/cvp/pipeline/attempts.py`** (logic thuần, test dễ):
   - `query_confidence(rows_meta) -> float` per query từ dữ liệu KHÔNG cần BTC:
     margin top-1 vs median (tái dùng `ranking_confidence` có sẵn trong `run_queries.py`),
     độ đồng thuận lane (Jaccard top-k giữa member maps), độ phân tán khi RRF các biến thể,
     riêng TRAKE thêm: score DP top-1 vs top-2 khác video, số video phủ top-10 dòng.
     → Cần attempt 1 GHI SIDECAR `{stem}.meta.json` (top-k score, per-lane top ids, task):
     thêm knob `submission.write_meta: false` (mặc định off, bench/auto bật) vào
     `run_query_file` — thay đổi nhỏ, gated, không đổi CSV.
   - `build_plan(meta_dir, thresholds) -> attempt_plan.json`: danh sách câu yếu + chế độ
     nỗ lực cao per task (knob đã có sẵn hết: expansions↑, vlm votes↑, topk↑,
     per_event_topk↑, self_consistency↑, low_confidence_retry on…).
   - `merge_attempts([dirA, dirB, …], weights, task_rules) -> merged/`: tổng quát hóa
     `rrf_merge` của scripts/63 lên N lượt CÓ TRỌNG SỐ; danh tính dòng: KIS/QA (video,frame),
     TRAKE thêm chế độ mềm (trùng video + ≥⌈K/2⌉ frame trong ±ε coi là cùng ứng viên, giữ
     bản của lượt rank cao) — vá điểm yếu tuple-identity đã nêu ở mục 4.
2. **`scripts/64_adaptive_attempts.py`** — CLI 3 lệnh: `plan` (đọc out1 → attempt_plan.json),
   `run` (chạy lại CHỈ câu yếu với env override từ plan, ghi out2 — engine_factory tái dùng
   `run_auto` với query_dir lọc theo plan), `merge` (out1+out2[+out3] → merged/ + báo cáo
   câu nào đổi top-1, in markdown). `plan`/`merge` chạy offline thuần CSV/JSON.
3. **Tests** (fixture nhỏ tự tạo, không GPU): confidence trên phân bố phẳng vs nhọn;
   plan thresholds; merge N lượt (thứ tự, dedup, ≤100, trọng số nghiêng đúng phía,
   TRAKE soft-identity); e2e với stub engine như `auto_agent` tests đang làm
   (`engine_factory` inject).
4. **Nghiệm thu trên bench**: lượt 1 full pack → lượt 2 chỉ câu yếu (kỳ vọng <30% số câu)
   → merge; so `bench_full.json` trước/sau bằng 65_bench_diff. Ăn khớp phát hiện của
   round-43: chỉ merge "anh em cùng noise" — lượt 2 cùng config-family nỗ lực cao hơn,
   không merge lineup khác đẳng cấp.

### 7. Việc dở dang / cần chủ dự án bổ sung

- Chưa có trên máy này: `bench_full.json` per-query + `lab_full/` CSVs + `manifest.parquet`
  → H1/H2/H9 mới dừng ở mức lập luận + con số ước tính. Xin copy 3 thứ đó vào repo lab
  (hoặc chạy `scripts/66_trake_diag.py` khi nó ra đời ở phiên tới).
- Cần xác nhận: bài tham chiếu 19.8/23 nộp bằng lưới keyframe nào (của mình hay đội khác)?
  — quyết định trọng số H1.
- Câu hỏi thể lệ 2026: BTC chấm TRAKE bằng cửa sổ cỡ nào / theo moment hay frame? (H10).
- Phiên tới (đề xuất): viết `scripts/66_trake_diag.py` phần offline + unit tests (bước A),
  song song code knob B1 (jitter) vì test được không cần GPU và không đổi hành vi mặc định.

### 8. Trạng thái bàn giao phiên này

- Nghiên cứu: xong toàn bộ phạm vi yêu cầu (luồng TRAKE, 3 scripts, config/env).
- Kết luận: mục 2 (bất đối xứng), mục 4 (10 giả thuyết xếp hạng + cách kiểm chứng).
- File sửa/tạo: **chỉ `report.md`** (tạo mới — lý do: luật an toàn #6 + yêu cầu đề phiên).
- Test: `python -m pytest tests -q` → 763 passed, 1 skipped — XANH (chạy lúc 21:34, trước
  khi tạo report.md; report.md không ảnh hưởng test).
- Không commit git (được phép theo quy trình #5).

---

## Phiên 2026-08-27 21:55 — NHIỆM VỤ 1: cải tiến TRAKE (4 knob config-gated + 23 test mới)

### 0. Tóm tắt phiên

- Triển khai **4 cải tiến TRAKE** từ kế hoạch B của phiên trước, đúng ưu tiên đề bài
  (DP alignment giữ ràng buộc thứ tự thời gian, dùng **caption dày làm tín hiệu bước**;
  cộng thêm biến thể query per event, pool-context, và chiến lược jitter phủ 100 dòng).
- **Mọi hành vi mới nằm sau knob trong `TemporalCfg` + `configs/settings.yaml`,
  mặc định = hành vi cũ nguyên vẹn** (có test khẳng định no-op: identity-map, weight 0,
  scorer không bị gọi, stub engine cũ không nhận kwarg mới).
- Test: **785 passed + 1 skipped, 0 failed** (thêm 23 test mới trong
  `tests/test_trake_upgrades.py`). Một lượt chạy giữa chừng gặp `OSError Errno 22`
  ở `test_notebooks` (builder ghi file .ipynb bị khóa tạm thời trên ổ E: — lỗi môi
  trường, không liên quan thay đổi; chạy lại riêng file đó 65/65 xanh, chạy lại
  full suite xanh sạch; git status xác nhận không notebook nào bị đổi nội dung).

### 1. Thiết kế 4 cải tiến (tất cả trong nhánh TRAKE, không đụng KIS/QA/AVS)

**(a) Caption dày làm tín hiệu bước cho DP — `temporal.caption_signal_weight` (mặc định 0.0)**
Đây là yêu cầu chính của đề bài. Luồng: `engine.search_trake` (khi weight > 0) dựng
closure `caption_scorer_from_signals(text_signals, catalog, event_texts_gốc_tiếng_Việt)`
→ `trake_search` gọi closure cho TỪNG video đã pool → ma trận BM25 caption thô
(n_rows × K) → `_normalized_caption_by_vid` minmax **per event trên toàn bộ pooled
videos** (cột chết/không tín hiệu → 0, KHÁC quy ước 0.5 trung tính của dense — BM25
vắng = không có tín hiệu, không được thưởng) → `sim = sim_dense + w × caption_norm`
NGAY TRƯỚC KHI DANTE/beam DP chạy. DP và mọi ràng buộc thứ tự/gap giữ nguyên — chỉ
ma trận similarity (bước hành động × frame) giàu tín hiệu hơn. BM25 chấm qua method
mới `TextSignals.score_field` (chỉ chấm 1 kênh caption thay vì cả 4 kênh × K lần).
Degrade sạch: thiếu artifacts caption → closure trả None → no-op.

**(b) Biến thể query per event — `temporal.event_query_variants: original|all` (mặc định original)**
Khi `all`: mỗi event encode toàn bộ biến thể đã cache của query processor
(`texts_for_search` — đúng danh sách KIS dùng: original + enhanced + translation +
expansions), vector rows mang `variant_map` (row → event). Similarity per event =
**max qua biến thể** (`_reduce_variants`, phản chiếu `fusion.aggregate_queries(how=max)`
của KIS), áp cho cả nhánh đơn lane lẫn ensemble (`_ensemble_video_sims` reduce trước
minmax; `_pooled_event_hits` cho mọi variant row đổ hit vào event của nó). Sửa được
H4 phiên trước: lane đa ngôn ngữ không còn vứt enhancement Gemini (đằng nào cũng đã
trả tiền API cho nó).

**(c) Header đề dẫn đường bước gom video — `temporal.pool_context: none|prepend` (mặc định none)**
Khi `prepend`: `run_query_file` tách header bằng helper mới `split_trake_query`
(refactor từ `parse_trake_events`, hành vi cũ giữ nguyên từng byte — test pin) và
truyền `context=` vào `engine.search_trake` (chỉ khi knob bật → stub engine cũ không
bao giờ nhận kwarg lạ). Engine encode "<header>. <event>" thành `pool_event_vecs`
(chỉ lane đa ngôn ngữ — header tiếng Việt trên lane English-only là rác), và
`trake_search` dùng chúng **CHỈ cho bước FAISS pooling**; DP vẫn chấm bằng event
trần. Tách "chọn video" (cần ngữ cảnh) khỏi "canh mốc" (cần độ sắc per event) —
sửa H2+H3 mà không loãng alignment như `event_context: prepend` cũ.

**(d) Chiến lược nộp jitter — `temporal.submit_strategy: legacy|jitter` + `jitter_videos` (mặc định legacy, 4)**
Khi `jitter`: sau khi xếp hạng xong, giữ **nguyên văn 8 dòng đầu** (hằng `JITTER_HEAD`
— R@1/R@5 không bao giờ thua legacy theo cấu trúc), rồi phủ ngân sách 100 dòng bằng
biến thể frame quanh chuỗi tốt nhất của tối đa `jitter_videos` video top (theo thứ
tự hạng): trung điểm GIỮA keyframe trước/sau (phá trần lượng tử hóa H1 — frame nộp
không bắt buộc nằm trên lưới keyframe), keyframe liền kề, ưu tiên biến thể SỚM trước
(đề TRAKE hỏi "khoảnh khắc ĐẦU TIÊN"); đuôi legacy giữ sau các block. Mọi biến thể
strictly-increasing, dedup, cap max_results (`jitter_frame_variants` +
`expand_candidates_jitter` là hàm thuần — test property dễ). Metric chính thức lấy
max R-Score theo dòng trong top-k nên dòng thêm quanh ứng viên mạnh là vé số miễn phí.

### 2. File đã sửa/tạo (từng cái + lý do)

| File | Thay đổi |
|---|---|
| `src/cvp/config.py` | +5 field `TemporalCfg`: `event_query_variants`, `pool_context`, `caption_signal_weight` (ge=0), `submit_strategy`, `jitter_videos` (ge=1) — tất cả default = hành vi cũ |
| `configs/settings.yaml` | +5 key tương ứng trong `temporal:` kèm chú thích tiếng Việt cách A/B |
| `src/cvp/search/temporal.py` | Lõi: `_identity_map`/`_n_events_of`/`_reduce_variants`; `Member` tuple 6 thành phần (thêm vmap + pool_vecs) trong `_resolve_members`/`_pooled_event_hits`/`_ensemble_video_sims`; `caption_scorer_from_signals` + `_normalized_caption_by_vid`; `jitter_frame_variants` + `expand_candidates_jitter` + `JITTER_HEAD`; `trake_search` nhận 5 kwarg mới (`event_variant_map`, `pool_event_vecs`, `caption_scorer`, `variant_maps_by_member`, `pool_vecs_by_member`), blend caption trước DP, hook jitter sau xếp hạng. Chữ ký cũ + hành vi mặc định giữ nguyên (positional call sites không đổi) |
| `src/cvp/search/engine.py` | `search_trake` thêm kwarg-only `context`; dựng variant texts/map per member (`_variant_texts_for` — mode original giữ NGUYÊN biểu thức `_texts_for` cũ), encode pool vecs (chỉ lane multilingual, lỗi → fallback), dựng caption closure khi weight > 0; truyền kwargs mới xuống `trake_search` |
| `src/cvp/search/text_signals.py` | +`TextSignals.score_field(field, text, candidates)` — chấm BM25 1 kênh frame-field (caption) thay vì cả 4 kênh; kênh lạ/metadata → `{}` |
| `src/cvp/pipeline/run_queries.py` | +`split_trake_query(lines) -> (header, events)` (logic tách dùng chung); `parse_trake_events` refactor thành wrapper mỏng (hành vi y hệt — test round-9/39 pin vẫn xanh); nhánh TRAKE của `run_query_file` truyền `context=header` CHỈ khi `pool_context=prepend` (tương thích ngược stub engine) |
| `tests/test_trake_upgrades.py` | **MỚI** — 23 test (chi tiết mục 3) |
| `report.md` | Mục phiên này |

### 3. Test (fixture tổng hợp tự dựng, không GPU/network — 23 test mới, tổng suite 785+1)

Đúng yêu cầu đề bài "chọn đúng chuỗi khi thứ tự đúng tồn tại, không tệ hơn baseline khi không":

- **Caption**: dense xếp chuỗi decoy 0.90 > chuỗi thật 0.85 → bật weight 0.5 + caption
  BM25 đúng 2 mốc → chuỗi thật thắng, đúng ns [2,5]/frame [200,500]
  (`test_caption_signal_rescues_chain_dense_ranks_second`); weight 0 (default) →
  scorer KHÔNG BỊ GỌI + kết quả trùng baseline từng score; scorer trả None (thiếu
  artifacts) → trùng baseline; cột event chết normalize về 0 (không phải 0.5);
  factory end-to-end đọc caption artifacts thật qua persisted BM25 index (ma trận
  (6,2) đúng ô, không cross-talk — chú ý tokenize_vi gập dấu nên caption test phải
  chọn từ không trùng sau fold); `score_field` kênh lạ → {}.
- **Biến thể**: chuỗi chỉ biến thể THỨ HAI của event 1 nhìn thấy → legacy trượt,
  bật variant map tìm đúng [1,4] và score > legacy + 0.3; biến thể nhiễu trực giao
  → winner + score y hệt legacy (không tệ hơn); identity map ≡ legacy từng score;
  ensemble 2 member (một member mù) vẫn tìm đúng chuỗi qua variant của member kia.
- **Pool-context**: max_videos=2, video thật hạng 3 theo event-vec → bị cắt (không
  ứng viên nào từ nó); pool_event_vecs trỏ đúng video → vào pool, DP vẫn chấm bằng
  event vec (test dùng junk-basis RIÊNG per video để tránh rò similarity).
- **Jitter**: hàm thuần — all-early đứng đầu, base bị loại, strictly increasing,
  dedup, biên (event ở keyframe đầu/cuối, keyframe kề nhau) không vỡ; expander —
  8 dòng đầu nguyên văn, block chỉ của top-`jitter_videos` video, đuôi legacy còn,
  cap max_results; end-to-end qua `trake_search` — top-1 y hệt legacy, số dòng nhiều
  hơn, có frame giữa-keyframe (f % 100 ≠ 0).
- **Wiring**: `split_trake_query` tách header/events + format thường; refactor
  `parse_trake_events` không đổi hành vi (cả prepend_context); `run_query_file`
  truyền context đúng header khi knob bật, stub kiểu CŨ (không có kwarg context)
  chạy bình thường khi knob tắt; defaults mọi knob + parity settings.yaml.

### 4. Cách bật knob khi bench trên Colab (một knob một lượt, bench-before-adopt)

```bash
# B5 — caption làm tín hiệu bước (CẦN artifacts/captions + text_index caption trên máy bench;
#      thiếu thì tự no-op nên số đo sẽ y hệt baseline — kiểm tra log trước khi kết luận):
CVP_TEMPORAL__CAPTION_SIGNAL_WEIGHT=0.3        # thử 0.2 / 0.3 / 0.5

# B3 — biến thể Gemini per event (0 call API mới nếu cache query ấm — dump/warm sẵn):
CVP_TEMPORAL__EVENT_QUERY_VARIANTS=all

# B2 — header dẫn đường pooling (độc lập với event_context cũ):
CVP_TEMPORAL__POOL_CONTEXT=prepend

# B1 — phủ 100 dòng bằng jitter quanh top video:
CVP_TEMPORAL__SUBMIT_STRATEGY=jitter
CVP_TEMPORAL__JITTER_VIDEOS=4                  # thử 3 / 4 / 6

# Config-only sweep kèm theo (0 dòng code, đã đề xuất phiên trước):
CVP_TEMPORAL__PER_EVENT_TOPK=300  CVP_TEMPORAL__MAX_VIDEOS=60  CVP_TEMPORAL__MAX_GAP_S=300
```

Gợi ý thứ tự bench: (1) jitter — rẻ nhất, không thêm API/GPU, kỳ vọng ăn ngay trần
H1; (2) pool_context + per_event_topk/max_videos — nhắm recall video; (3) variants;
(4) caption weight (cần xác nhận caption artifacts + text_index build cho catalog trận).

### 5. Dự đoán trung thực về mức cải thiện + rủi ro

Dự đoán (offline bench, gt ±12 frame, 7 câu TRAKE — biên độ RẤT rộng vì mẫu nhỏ,
mỗi câu nặng ~14%):

- **Jitter (B1)**: kỳ vọng cao nhất theo phân tích H1/H6 — nếu trần lưới keyframe
  đúng là nút thắt, phủ trung điểm có thể nâng TRAKE **0.183 → ~0.25-0.35**; nếu
  video/DP mới là nút thắt thì gần như 0 thay đổi (thiết kế bảo đảm không âm ở
  R@1/R@5; R@20+ chỉ mất khi đuôi legacy sau hạng ~80 từng ghi điểm mà bị đẩy ra —
  hiếm vì đuôi đó là video hạng thấp).
- **Pool-context (B2)**: nếu chẩn đoán "video đúng không vào pool" chiếm ≥2/7 câu
  → mỗi câu cứu được là +~0.1-0.14 tổng TRAKE; ngược lại ~0. Rủi ro: header kéo
  pooling về video "đúng chủ đề nhưng sai clip" (tin tức lặp b-roll) — vì DP vẫn
  chấm event trần nên hại chủ yếu là chiếm chỗ pool; max_videos tăng kèm sẽ đệm.
- **Variants (B3)**: kỳ vọng vừa (+0.01-0.05 tổng): enhancement giúp event mô tả
  hành động trừu tượng, nhưng max-fusion cũng có rủi ro biến thể lệch nghĩa thắp
  sáng frame sai (test nhiễu chỉ chứng minh trường hợp trực giao; bench là trọng tài).
  Chi phí: ~×3-4 lượt encode text + FAISS per event (vẫn <1s), 0 call Gemini mới khi
  cache ấm.
- **Caption (B5)**: dao động lớn nhất — caption Vintern stride-1 mô tả hành động
  đúng kiểu đề TRAKE, nhưng BM25 tiếng Việt gập dấu trên câu ngắn dễ khớp giả
  ("cắt nấm" khớp mọi video nấu ăn); minmax-per-event trên pool sẽ khuếch tán nếu
  nhiều video cùng khớp. Kỳ vọng +0.02-0.08 nếu artifacts caption phủ đủ; **0 nếu
  máy bench chưa build text_index caption** (degrade âm thầm — phải soi log
  "TextSignals: persisted text_index..."). Chi phí: K lượt BM25/video × ~30 video,
  ~vài trăm ms/câu với persisted index.
- **Tổng hợp thận trọng**: nếu H1 đúng như phân tích, tổ hợp jitter + (pool hoặc
  topk sweep) có cửa đưa TRAKE lên **~0.3±0.1**; nếu bench cho thấy video-recall
  đã tốt và GT quá nhiễu (H9) thì trần thực tế thấp hơn và phải chuyển hướng sang
  script chẩn đoán 66 trước khi tin bất kỳ số nào.

Rủi ro chung: (i) tương tác knob — bật ĐỒNG THỜI nhiều knob chưa được test tổ hợp
trên bench, nên A/B từng cái; (ii) jitter đổi phân phối dòng nộp — nếu BTC chấm
khác giả định max-over-rows (H10) thì lợi thế giảm (không âm, vì head giữ nguyên);
(iii) caption weight cố định 1 giá trị toàn cục — về sau đáng tune bằng harness 21
mở rộng (hiện chưa làm); (iv) `_minmax_1d` degenerate → 0.5 trong pooling ensemble
nghĩa là lane mù vẫn đóng 0.5×weight vào MỌI video — hành vi CŨ, giữ nguyên,
nhưng đáng ghi nhận khi đọc số pooling.

### 6. Việc dở dang / đề xuất phiên tới

- Chưa làm (theo kế hoạch phiên 1, vẫn đứng): script chẩn đoán `scripts/66_trake_diag.py`
  (bước A) — nên làm TRƯỚC khi đọc kết quả bench A/B để tách "trần đo lường" khỏi
  "chất lượng pipeline"; VLM verify chuỗi (B6) và Nhiệm vụ 2 (64_adaptive_attempts).
- Chủ dự án khi bench: bật từng knob một, giữ nguyên seed/config còn lại, so per-query
  bằng bench_full.json; câu hỏi mở phiên 1 (nguồn tham chiếu 19.8/23, thể lệ chấm
  TRAKE 2026) vẫn cần trả lời.
- Test status bàn giao: `python -m pytest tests -q` → **785 passed, 1 skipped, 0 failed**
  (lượt cuối 62.6s). Không commit git.

---

## Phiên 2026-08-27 22:21 — NHIỆM VỤ 2: adaptive multi-attempt (scripts/64 + module attempts)

### 0. Tóm tắt phiên

- Hoàn thành Nhiệm vụ 2 của CURSOR_BRIEF: module `src/cvp/pipeline/attempts.py` +
  CLI `scripts/64_adaptive_attempts.py` (3 lệnh `plan` / `run` / `merge`).
- **`plan` và `merge` chạy offline thuần** (input = thư mục/zip CSV + signal dumps
  tùy chọn); `run` là lệnh duy nhất cần engine (Colab). **Không test nào gọi API.**
- **`scripts/63_ensemble_runs.py` không bị sửa một byte nào**; merge N lượt mới được
  chứng minh bằng test tái tạo đúng output 63 row-for-row khi 2 lượt trọng số bằng nhau.
- Test: **803 passed + 1 skipped, 0 failed** (thêm 18 test mới trong
  `tests/test_adaptive_attempts.py`).

### 1. Thiết kế

**Độ tự tin per query (không cần điểm BTC)** — 3 thành phần, mỗi cái [0,1], trung
bình có trọng số (mặc định margin 0.5 / concentration 0.3 / answer_consensus 0.2,
thành phần vắng thì renormalize phần còn lại):

1. `margin` — `ranking_confidence` (tái dùng nguyên hàm đã test của
   `run_queries`, cũng chính là thống kê low-confidence-retry của engine) trên map
   điểm `visual` từ signal dumps của scripts/23 **khi có**; không có dumps → thành
   phần này vắng, không bịa.
2. `concentration` — tỷ lệ dòng trong head (mặc định 10) ở lại đúng video của dòng
   top-1: ranking tự tin thì cụm quanh một khoảnh khắc, ranking nhiễu thì rải nhiều
   video. Ghi rõ trong docstring: heuristic, có thể bị lừa bởi "tự tin nhầm video"
   — nên mới trung bình nhiều thành phần thay vì tin một cái.
3. `answer_consensus` (chỉ QA) — tỷ lệ câu trả lời đa số trong head;
   `QA_FALLBACK_ANSWER` ("không rõ") tính là KHÔNG có câu trả lời và vẫn nằm ở mẫu
   số (head toàn fallback → 0, nửa fallback → trần 0.5).

Câu có `.txt` trong pack nhưng KHÔNG có CSV (rớt lượt 1 — auto track bỏ câu đó
âm thầm) → confidence 0, luôn đứng đầu danh sách yếu.

**Chọn câu yếu** — `select_weak`: dưới `threshold` (mặc định 0.35), yếu nhất trước,
cắt trần `ceil(max_fraction × tổng)` (mặc định 0.5 — ngân sách lượt 2).

**Kế hoạch lượt 2** — `build_plan` xuất JSON: bảng confidence per query (kèm từng
thành phần để người đọc soi được), danh sách `weak`, và `env` nỗ lực cao gộp từ
`HIGH_EFFORT_ENV` theo ĐÚNG các task xuất hiện trong nhóm yếu (toàn knob CÓ SẴN):
common = `LOW_CONFIDENCE_RETRY=true`, `TOPK=800`, `EXPANSIONS=4`; kis/avs/qa thêm
`VLM_RERANK_VOTES=5`; qa thêm `VQA__SELF_CONSISTENCY=5`; trake thêm
`PER_EVENT_TOPK=300`, `MAX_VIDEOS=60`, `EVENT_QUERY_VARIANTS=all` (knob mới của
Nhiệm vụ 1 — hai nhiệm vụ khớp nhau đúng chỗ). Có ghi chú chi phí: đổi
`EXPANSIONS` làm lệch cache-key của query processor → câu yếu sẽ gọi Gemini
enhance lại ở lượt 2 (chủ đích: nhiều variant hơn, nhưng tốn quota).

**Merge N lượt có trọng số** — `rrf_merge_runs(runs, weights, k)`: union stems,
per stem chạy `merge_rows` giữ NGUYÊN semantics 63 (danh tính dòng KIS/QA/AVS =
`(video, frame)`, cột answer ngoài key và lấy từ lượt xếp hạng tốt hơn — hòa thì
lượt đứng trước thắng; TRAKE = nguyên tuple; sort RRF ổn định; cap 100), tổng quát
hóa thành `score += w_i/(k+rank+1)`. Trọng số ≤ 0 hoặc lệch số lượng → ValueError
fail loud. Kèm `diff_top1` + `render_diff_markdown` (báo cáo "câu nào đổi" cho
lượt 3 theo brief).

**CLI `scripts/64_adaptive_attempts.py`**:
- `plan --run att1/ [--signals-dir dumps/] [--query-dir pack/] [--threshold 0.35]
  [--max-fraction 0.5] [--out plan.json] [--stage-dir weak/]` — in bảng confidence
  (đánh dấu câu YẾU), stage file .txt câu yếu, in env + lệnh chạy tiếp. Offline thuần.
- `run --plan plan.json --out att2/` — set env từ plan TRƯỚC khi `load_settings`,
  stage câu yếu, gọi `run_auto` (cần artifacts/engine — dùng trên Colab; module
  import lazy nên script vẫn import sạch trên máy không torch).
- `merge --runs att1 att2 [att3] [--weights 1 1 0.8] --out merged/ [--report DIFF.md]`
  — offline thuần, in danh sách câu đổi top-1.

### 2. File đã sửa/tạo (từng cái + lý do)

| File | Thay đổi |
|---|---|
| `src/cvp/pipeline/attempts.py` | **MỚI** — toàn bộ logic thuần (confidence, plan, stage, merge, diff); import nhẹ (run_queries lazy trong hàm để giữ module torch-free như auto_agent); hằng `HIGH_EFFORT_ENV` + `DEFAULT_COMPONENT_WEIGHTS` ở một chỗ để chỉnh |
| `scripts/64_adaptive_attempts.py` | **MỚI** — CLI 3 lệnh, theo đúng quy ước script repo (`_bootstrap` try/except như scripts/21, cvp import ở top, heavy import lazy trong `cmd_run`) |
| `tests/test_adaptive_attempts.py` | **MỚI** — 18 test (chi tiết mục 3) |
| `report.md` | Mục phiên này |

Không sửa file nào khác — đặc biệt `scripts/63_ensemble_runs.py` giữ nguyên.

### 3. Test (18 test mới, fixtures nhỏ tự dựng trong tmp_path, 0 API)

- **Điểm tự tin**: margin nhọn (10,1,1,…) → 0.9, phẳng → 0, <10 giá trị → 0,
  None/{} → None (vắng); concentration cụm → 1.0, xen kẽ → 0.5, rỗng → 0, chỉ xét
  head; answer_consensus đa số 7/10 → 0.7, fallback pha loãng mẫu số (6 thật + 4
  "không rõ" → 0.6), toàn fallback → 0, TRAKE không có cột answer → 0;
  `query_confidence` kết hợp + renormalize khi vắng thành phần (số kiểm bằng tay),
  rows rỗng → 0.
- **Chọn câu yếu + plan**: dưới ngưỡng, yếu nhất trước, cap `max_fraction` cắt
  đúng 2/3 câu; `build_plan` phát hiện câu mất CSV (conf 0, component
  `missing_csv`), env gộp đúng theo task yếu (chỉ-kis → không có key TEMPORAL/VQA;
  đủ kis+qa+trake → đủ bộ), mọi key HIGH_EFFORT_ENV phải mang prefix `CVP_`;
  dumps phẳng kéo confidence xuống so với không dumps (margin có tác dụng thật);
  dump hỏng → warn+skip không chết plan; `stage_weak_queries` copy đúng câu yếu,
  stem ma bị bỏ qua.
- **Merge N lượt**: **parity 63** — 2 lượt trọng số bằng nhau tái tạo từng dòng
  output `63_ensemble_runs.rrf_merge` trên cả kis/qa/trake (63 load bằng importlib,
  không sửa); trọng số nghiêng lượt nặng hơn thắng tie; answer lấy từ lượt xếp
  hạng tốt nhất, hòa rank → lượt đứng trước; TRAKE nguyên tuple không dedupe;
  cap 100 + limit tùy chỉnh; ValueError khi weights lệch/≤0; union stems;
  write_merged → load_run roundtrip từng byte; diff_top1 + markdown.
- **CLI e2e offline**: importlib load scripts/64, chạy `main()` với argv giả —
  `plan` ra JSON đúng câu yếu + stage đúng file + env trake đủ; `merge` 2 lượt
  weights [1,2] ra CSV + DIFF.md, lượt nặng hơn thắng top-1, stdout báo "1 câu
  đổi top-1".

### 4. Cách dùng khi bench (Colab)

```bash
# Lượt 1 như bình thường (25_auto_agent / notebook bench) → att1/
python scripts/64_adaptive_attempts.py plan --run att1/ \
    --signals-dir artifacts/signal_dumps/thunghiem --query-dir queries/pack \
    --out artifacts/attempts/plan.json --stage-dir artifacts/attempts/weak
# Đọc bảng confidence in ra — chỉnh --threshold/--max-fraction nếu muốn
python scripts/64_adaptive_attempts.py run --plan artifacts/attempts/plan.json --out att2/
python scripts/64_adaptive_attempts.py merge --runs att1/ att2/ --out merged/ \
    --report merged/DIFF.md          # rồi score_run merged/ như mọi khi
# Lượt 3 (nếu có): merge --runs att1/ att2/ att3/ --weights 1 1 1
```

Luật round-43 vẫn áp dụng (ghi trong docstring script): chỉ merge "anh em cùng
noise" — lượt 2 là CÙNG lineup với nỗ lực cao hơn, không merge lineup khác đẳng cấp.

### 5. Dự đoán trung thực + rủi ro

- Bằng chứng round-43: merge 2 lượt cùng config 0.6587 > 0.6500/0.6413 từng lượt —
  lượt 2 "nỗ lực cao có chủ đích vào đúng câu yếu" có cơ sở ăn hơn merge mù, kỳ vọng
  **+0.01-0.03 mean_final** trên bench với chi phí chỉ ~≤50% số câu chạy lại.
- Rủi ro chính: (i) **confidence không dumps chỉ còn concentration** — heuristic
  thô, có thể chọn nhầm câu khỏe đi chạy lại (vô hại về điểm — merge giữ lượt 1 —
  nhưng tốn quota); khuyến nghị luôn dump signals trước (scripts/23 đằng nào cũng
  chạy cho tuning); (ii) lượt 2 đổi `EXPANSIONS` → Gemini re-enhance các câu yếu
  (quota); (iii) trọng số merge chưa có số liệu — mặc định 1.0 đều là an toàn nhất,
  chỉ nghiêng khi bench chứng minh; (iv) TRAKE merge vẫn danh tính nguyên tuple
  (giữ đúng 63) — hai lượt hiếm khi trùng tuple nên merge TRAKE thiên về xen kẽ;
  danh tính mềm (trùng video + ≥⌈K/2⌉ event gần nhau) ghi nhận là việc tương lai,
  không làm vội trong phiên này vì semantics "giữ frame nào" chưa có bench trọng tài.
- Ngưỡng 0.35 / cap 0.5 là điểm khởi đầu hợp lý chứ chưa tune — bảng confidence
  in ra đủ chi tiết (từng thành phần) để chủ dự án chỉnh bằng mắt sau lượt bench đầu.

### 6. Việc dở dang / đề xuất tiếp theo

- Nhiệm vụ 3 (`scripts/65_bench_diff.py`) chưa làm — nhỏ, nên làm phiên tới để đọc
  delta bench trước/sau các knob Nhiệm vụ 1+2.
- Script chẩn đoán TRAKE (`66_trake_diag.py`, bước A phiên 1) vẫn đứng trong hàng đợi.
- Test status bàn giao: `python -m pytest tests -q` → **803 passed, 1 skipped,
  0 failed** (61.5s). Không commit git.

---

## Phiên 2026-08-27 22:33 — NHIỆM VỤ 3 (bench diff) + TỔNG KIỂM + TỔNG KẾT BÀN GIAO

### 0. Tóm tắt phiên

- **Nhiệm vụ 3 xong**: `scripts/65_bench_diff.py` — so `bench_full.json` vs
  `bench_full-prev.json` (đúng format `RunReport.to_dict` mà notebook Lab ghi) →
  markdown: headline mean_final, bảng theo task, bảng per-query nhóm theo task với
  Δ (regression đứng ĐẦU mỗi nhóm), ghi chú câu unscored (kèm lý do) / mới xuất
  hiện / biến mất; câu vắng ở một bản tính 0.0 khi lấy Δ — đúng quy ước mean_final
  của scorer. Script thuần stdlib, chạy mọi nơi có 2 file JSON; CLI `--dir` hoặc
  `--cur/--prev`, `--out` ghi file; thiếu file/-prev hoặc JSON hỏng → SystemExit
  thân thiện. 7 test mới (`tests/test_bench_diff.py`) toàn fixture tay, offline.
- **Tổng kiểm phản biện toàn bộ diff chiến dịch** (đọc lại từng hunk git diff):
  tìm và SỬA 2 bug edge-case + 2 điểm phòng thủ/tài liệu:
  1. `temporal.trake_search` (nhánh đơn lane, variant map có "lỗ" — event không
     có row nào, ví dụ `[0, 2]`): `np.concatenate([])` crash → giờ event trống
     nhận mảng rỗng như `_pooled_event_hits`; test pin
     `test_variant_map_with_gap_does_not_crash`.
  2. `attempts.select_weak` với `max_fraction=0`: sàn `max(1, …)` cũ vẫn trả 1 câu
     — ngân sách 0 phải nghĩa là KHÔNG chạy lại → sửa cap đúng 0; test pin thêm.
  3. `attempts.merge_rows`: thêm ValueError khi số rankings ≠ số weights (zip cũ
     nuốt lặng phần đuôi nếu gọi trực tiếp không qua `rrf_merge_runs`).
  4. Ghi chú config (config.py + settings.yaml): ĐỪNG bật `pool_context: prepend`
     cùng `event_context: prepend` — event lúc đó đã mang header, text pooling sẽ
     nhân đôi header (không crash, chỉ loãng).
  Soát luật an toàn: không đụng gì ngoài folder, không git remote/push/commit,
  không sửa tay .ipynb (git xác nhận notebooks không đổi), không sửa/xóa test cũ
  nào (chỉ THÊM test), mọi claim chưa kiểm được trong report đều gắn nhãn.
- Test cuối: **811 passed + 1 skipped, 0 failed** (100s).

---

## TỔNG KẾT BÀN GIAO (toàn chiến dịch 27/08/2026, 4 phiên)

### A. Danh sách ĐẦY ĐỦ file thay đổi (mỗi file 1 dòng)

**Sửa (6):**

| File | Tóm tắt |
|---|---|
| `src/cvp/config.py` | +7 knob `TemporalCfg` (event_query_variants, pool_context, caption_signal_weight, submit_strategy, jitter_videos + 2 ghi chú), tất cả default = hành vi cũ |
| `configs/settings.yaml` | +7 key `temporal:` tương ứng, kèm chú thích tiếng Việt cách A/B và cảnh báo kết hợp knob |
| `src/cvp/search/temporal.py` | Lõi TRAKE: variant-map max-fusion per event, pool-vectors riêng cho bước gom video, caption-BM25 blend vào ma trận DP, jitter expander phủ 100 dòng, `caption_scorer_from_signals`; chữ ký/hành vi mặc định giữ nguyên |
| `src/cvp/search/engine.py` | `search_trake` + kwarg-only `context`; dựng variant texts/pool vecs per member, caption closure khi weight>0; mode mặc định giữ NGUYÊN biểu thức `_texts_for` cũ |
| `src/cvp/pipeline/run_queries.py` | +`split_trake_query` (tách header/events dùng chung); `parse_trake_events` thành wrapper mỏng (hành vi y hệt, test round-9/39 pin xanh); truyền `context=header` CHỈ khi knob bật |
| `src/cvp/search/text_signals.py` | +`TextSignals.score_field` — chấm BM25 đúng 1 kênh frame-field (TRAKE caption cần K lần/video, chấm cả 4 kênh là phí) |

**Tạo mới (7):**

| File | Tóm tắt |
|---|---|
| `src/cvp/pipeline/attempts.py` | Module Nhiệm vụ 2: confidence per query (margin/concentration/answer-consensus), plan lượt 2 + env nỗ lực cao, RRF-merge N lượt có trọng số (parity 63), diff top-1 |
| `scripts/64_adaptive_attempts.py` | CLI 3 lệnh plan/run/merge — plan & merge offline thuần, run cần engine (Colab) |
| `scripts/65_bench_diff.py` | Bench delta reporter (Nhiệm vụ 3), stdlib-only, markdown per-query theo task |
| `tests/test_trake_upgrades.py` | 24 test cho 4 knob TRAKE: cứu đúng chuỗi khi tín hiệu tồn tại, no-op tuyệt đối khi knob tắt, property jitter, wiring header |
| `tests/test_adaptive_attempts.py` | 18 test Nhiệm vụ 2: components, chọn câu yếu, plan env, merge N lượt (kèm parity từng dòng với scripts/63), CLI e2e offline |
| `tests/test_bench_diff.py` | 7 test Nhiệm vụ 3: delta/counts/notes, sort regression-first, render markdown, CLI + lỗi thân thiện |
| `report.md` | Nhật ký 4 phiên + tổng kết này |

**Không đụng**: `scripts/63_ensemble_runs.py` (yêu cầu giữ nguyên — parity pin bằng
test), toàn bộ `notebooks/` (.ipynb sinh ra — không sửa tay), mọi test cũ.

### B. Knob mới + giá trị đề xuất khi bench (một knob một lượt, bench-before-adopt)

| Knob (env override) | Default (=cũ) | Đề xuất thử | Nhắm giả thuyết |
|---|---|---|---|
| `CVP_TEMPORAL__SUBMIT_STRATEGY` | legacy | `jitter` (thử đầu tiên — rẻ, 0 API) | H1/H6 trần lưới keyframe + phí 100 dòng |
| `CVP_TEMPORAL__JITTER_VIDEOS` | 4 | 3 / 4 / 6 | — |
| `CVP_TEMPORAL__POOL_CONTEXT` | none | `prepend` (đừng kèm event_context) | H2/H3 recall video |
| `CVP_TEMPORAL__PER_EVENT_TOPK` | 100 | 300 (config-only) | H2 |
| `CVP_TEMPORAL__MAX_VIDEOS` | 30 | 60 (config-only) | H2 |
| `CVP_TEMPORAL__EVENT_QUERY_VARIANTS` | original | `all` (cache Gemini ấm = 0 call mới) | H4 bỏ phí enhancement |
| `CVP_TEMPORAL__CAPTION_SIGNAL_WEIGHT` | 0.0 | 0.2 → 0.3 → 0.5 (CẦN text_index caption trên máy bench, soi log kẻo no-op âm thầm) | H5 caption làm tín hiệu bước |
| `CVP_TEMPORAL__MAX_GAP_S` | 150 | 300 / 600 (config-only, đo gap GT trước) | H8 đề trải dài |

Quy trình đọc kết quả: sau mỗi lượt bench chạy
`python scripts/65_bench_diff.py --dir artifacts/lab --out artifacts/lab/BENCH_DIFF.md`
để thấy per-query tăng/giảm theo task. Vòng adaptive:
`64 plan → 64 run → 64 merge` (mục phiên 22:21 có lệnh đầy đủ).

### C. Những gì CHƯA làm xong

1. **`scripts/66_trake_diag.py`** (bước A kế hoạch phiên 1): đo trần lưới keyframe
   vs GT ±12, tỷ lệ GT-video-lọt-pool, khoảng cách frame nộp ↔ mốc GT, độ nhạy
   epsilon — cần artifacts/Colab; đây là thứ tách "trần đo lường" khỏi "chất
   lượng pipeline" TRƯỚC khi tin số A/B.
2. **TRAKE VLM verify** (B6 phiên 1): Gemini chấm dải K frame của top candidates —
   để sau khi 4 knob hiện tại có số bench.
3. **TRAKE merge soft-identity** trong 64 (trùng video + ≥⌈K/2⌉ event gần nhau =
   cùng ứng viên): giữ nguyên tuple-identity của 63 vì semantics "giữ frame nào"
   chưa có trọng tài; hiện merge TRAKE thiên về xen kẽ 2 lượt.
4. **Số bench thật cho mọi knob**: máy này không có artifacts — toàn bộ dự đoán
   trong report là ước lượng có lập luận, chưa kiểm chứng trên bench.
5. Ba câu hỏi mở cho chủ dự án từ phiên 1: nguồn lưới keyframe của bài tham chiếu
   19.8/23; xin `bench_full.json` + `lab_full/` vào lab để xác minh H1/H2/H9;
   thể lệ chấm TRAKE 2026 của BTC (frame-window hay moment?).

### D. 3 đề xuất giá trị nhất cho chặng tiếp theo

1. **Bench jitter + pool/topk sweep NGAY trong lượt bench kế tiếp** (0 API, 0 GPU
   thêm, 3 env var): nếu H1 đúng như phân tích (0.183 ≈ trần lưới ±12), riêng
   jitter có cửa đưa TRAKE 0.183 → ~0.25-0.35 — đây là tỷ lệ lãi/chi phí tốt nhất
   toàn bộ danh mục; kết quả cũng là bằng chứng trực tiếp xác nhận/bác H1.
2. **Viết `66_trake_diag.py` trước khi tin bất kỳ số A/B nào**: 7 câu TRAKE × GT
   nhiễu ±12 frame là mẫu cực nhỏ — không có bảng chẩn đoán (GT video có lọt pool
   không, trần lưới bao nhiêu, trượt mấy frame) thì một knob "tăng 0.05" trên bench
   có thể chỉ là nhiễu GT; phần lớn script này thuần số học, viết + test offline
   được ngay, chỉ phần pool-log cần một lần chạy Colab.
3. **Đưa vòng adaptive 64 vào notebook Lab như một cell chuẩn** (plan → run câu
   yếu → merge → score_run + 65_bench_diff): biến "3 lượt nộp" của thể lệ thành
   quy trình máy tự động đúng định hướng BTC 2026 ("máy chạy giống chatbot") —
   toàn bộ mảnh đã có và đã test, chỉ còn ghép cell trong `_build_notebooks.py`
   (nhớ chạy builder thay vì sửa tay .ipynb).

Test cuối chiến dịch: `python -m pytest tests -q` → **811 passed, 1 skipped,
0 failed**. Không commit git; không file nào ngoài repo bị đụng.
