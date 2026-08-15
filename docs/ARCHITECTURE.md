# 🏗️ ARCHITECTURE — Kiến trúc, bất biến & hợp đồng module

Tài liệu này mô tả cách các mảnh của **Core Vision Perfect V1** (package `cvp`)
khớp với nhau: bất biến giữ cho toàn hệ nhất quán, dataflow online/offline,
kiến trúc 2 thể thức thi 2026, và chỗ để mở rộng. Bối cảnh bài toán + cơ sở
bằng chứng nằm trong [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md); tài liệu này là
bản đồ kỹ thuật cho người sửa code.

---

## 1. Sơ đồ tầng (layered design)

```
config ──────── configs/settings.yaml → cvp.config.Settings (override: CVP_<SECTION>__<KEY>=...)
  │             constants.py = hợp đồng CỐ ĐỊNH của BTC (≤100 dòng, regex video_id, ≤100 ký tự answer)
  │
data ─────────── catalog (bất biến global_id) · extraction (TransNetV2→PySceneDetect→2s, 3 kf/shot)
  │              · keyframe_align (dhash + monotone DP — lõi rebuild map-keyframes)
  │              · video_frames (decord→cv2, 1 reader/video) · metadata (media-info + objects)
  │              · objects_compact (178k JSON → 1 parquet, CompactObjects drop-in ObjectStore)
  │
models ───────── base.EmbeddingModel (ABC) → siglip2/finetuned · openclip (PE-Core→DFN5B)
  │              · qwen_embed · provided_clip32 · mclip · jina (jina-clip-v2) · metaclip2
  │              · registry (name → constructor, lazy import) · query_processor (Gemini 1-call)
  │              · agent (KIS-C: gộp hint + câu hỏi làm rõ)
  │
index ────────── embedder (resumable/video) → store (FAISS + signature + model_tag)
  │              · text_store (BM25 persisted, O(K·|q|))
  │
search ───────── engine (orchestrator) · fusion (weighted_sum/rrf/neighbor_boost)
  │              · superglobal (đa biến thể, max-fuse) · temporal (DANTE/beam TRAKE)
  │              · temporal_boost (Vortex before/now/after) · avs (MMR) · feedback (Rocchio)
  │              · text_signals (BM25 4 kênh) · object_filter (228 danh từ VI)
  │              · vqa (multi-frame strip) · cross_rerank (BLIP-2 ITM / Qwen3-VL-Reranker)
  │              · vlm_rerank (listwise Gemini/Vintern)
  │
submission ───── writer (CSV đúng thể lệ) · packager (validate + zip 'submission/' + MANIFEST)
  │              · dres_client (DRES v2, urllib thuần, KHÔNG BAO GIỜ tự retry)
  │
eval ─────────── official (công thức BTC, pure stdlib) · metrics (R@K/MRR cho training)
  │
pipeline ─────── ingest (idempotent build) · run_queries (parser đề thật + QA theo nhóm
  │              + low-confidence retry) · auto_agent (thể thức TỰ ĐỘNG 2026)
  │
interfaces ───── service/ (FastAPI: /health /search/* /nearest /keyframe) · cli (cvp serve|search|eval|version)
  │              · app/streamlit_app.py (5 tab + đồng hồ + basket)
training ─────── build_dataset · datamodule · losses · lit_trainer (LoRA-LiT + WiSE-FT) [offline]
auxindex ─────── ocr (EasyOCR) · asr (PhoWhisper) · captioner (Vintern-1B)              [offline]
```

Quy ước chung: import nặng (torch/transformers/faiss) luôn **lazy** bên trong
constructor/hàm — mọi module import sạch trên máy không GPU; suite **526 test CPU**
chạy không cần model/mạng/data thật.

## 2. Bất biến trung tâm: `global_id == dòng manifest == dòng FAISS`

`KeyframeCatalog.build()` gán `global_id` tuần tự trên keyframes đã sort theo
`(video_id, n)` → rebuild trên cùng corpus luôn cho cùng id. `index/embedder.py`
ghi **một `.npy` mỗi video, dòng theo thứ tự `n`**; `IndexStore.build()` stream
các file đó **theo đúng thứ tự video của catalog**, nên:

```
FAISS trả id  ==  catalog.global_id  ==  dòng của ma trận embedding ghép
```

Một FAISS hit map thẳng về `(video_id, n, frame_idx, pts_time, path)` bằng một
lần `catalog.ref(gid)`. Ba tầng guard giữ bất biến này:

1. **Catalog signature** — hash của tập `(video_id, số keyframe)` được ghi vào
   `artifacts/indexes/{model_key}/meta.json`; `store.load(catalog)` **từ chối
   phục vụ** khi lệch (`is_stale`) — sai hàng = sai `frame_idx` = 0 điểm trên
   Codabench/DRES, nên thà chết to còn hơn trôi ngầm. `text_store` và
   `objects_index` cũng lưu + kiểm signature y hệt.
2. **`model_tag` guard** — lane `openclip` có thể resolve ra checkpoint khác nhau
   giữa các máy (PE-Core-bigG 1280-d vs DFN5B 1024-d). Tag của checkpoint đã
   build được lưu trong meta; `SearchEngine.__init__` so với tag đang load và
   **cảnh báo re-embed** khi lệch (search index PE-Core bằng vector DFN5B =
   rác im lặng). Lệch dimension bị bắt ngay khi load.
3. **Hàng embedding là VỊ TRÍ, không phải `n`** — dòng của một frame trong
   `.npy` là `global_id − video_start` (xem `engine._vectors_for`), sống sót
   khi tên keyframe có khoảng trống (`001.jpg, 003.jpg`).

Ở **chế độ ensemble**, bất biến giữ cho TỪNG lane: mỗi member có embeddings +
FAISS index riêng dưới `artifacts/{embeddings,indexes}/{model_key}/`, tất cả
build từ CÙNG một catalog → score map các lane fuse trực tiếp trên `global_id`.
Riêng lane `finetuned` (LiT — image tower đóng băng) **dùng chung index của
`siglip2`** (`registry.index_key_for`): không re-embed, không copy.

Thêm dữ liệu (BTC phát hành theo đợt) ⇒ catalog đổi ⇒ signature đổi ⇒ mọi index
phải rebuild — tự động hoá bằng `python scripts/30_ingest.py` (idempotent, chỉ
làm phần delta).

## 3. Dataflow online — một truy vấn KIS/QA (`engine.search_text`)

```
query VI
 ├─► QueryProcessor.process()      # 1 call Gemini: dịch + mô tả thị giác + N expansions (JSON)
 │                                 # cache đĩa theo (provider, model, query); degrade:
 │                                 # gemini-3.5-flash → gemini-3-flash-preview → gemini-2.5-flash
 │                                 # → Google Translate miễn phí → passthrough (không bao giờ chết)
 ├─► mỗi lane: encode các biến thể phù hợp (lane đa ngữ nhận cả câu VI gốc)
 │     └─► FAISS top-K (search.topk = 500) / biến thể ─► MAX theo biến thể
 │     └─► SuperGlobal (search.rerank) — S1 refine láng giềng trong pool, S2 query
 │         expansion; max-fuse MỌI biến thể; zero-row giữ nguyên điểm dense
 ├─► fuse lane: weighted_sum (0.55 siglip2 / 0.45 openclip) — hoặc chỉ 1 lane
 ├─► trên ứng viên (O(K), không bao giờ O(corpus)):
 │     + BM25+ persisted: ocr/asr/caption/metadata  (search.weights.*)
 │     + object boost (228 danh từ VI, ràng buộc đếm/vị trí — soft, không hard-filter)
 │     → weighted_sum hoặc rrf (search.fusion_method)
 ├─► neighbor boost (±search.neighbor_window frame, cùng video)
 ├─► temporal boost (search.temporal_boost, OFF mặc định) — query có marker
 │     "… sau khi/trước khi …": tách target/context bằng regex, boost theo max cos
 │     của láng giềng MỘT PHÍA với clause context (Vortex 79.6/88)
 ├─► sort → display_k
 ├─► cross-encoder rerank (search.reranker: none|blip2_itm|qwen_reranker) —
 │     pairwise trên top rerank_topk=100; blend (1−w)·minmax(fused) + w·minmax(cross)
 └─► VLM listwise rerank (search.vlm_rerank, OFF) — Gemini/Vintern chấm 0–10
       top vlm_rerank_topk=24, chỉ xáo đầu bảng
```

Ba tầng rerank **bổ trợ, không thay thế nhau**:

| Tầng | Phạm vi | Cơ chế | Chi phí |
|---|---|---|---|
| SuperGlobal | top-500 / lane | thuần vector, không train | ~1 ms |
| `cross_rerank` | top-100 sau fusion | cross-encoder pairwise (BLIP-2 ITM / Qwen3-VL-Reranker-2B) | ~100× 1 FAISS probe / cặp — cần GPU |
| `vlm_rerank` | top-24 | VLM nhìn CẢ danh sách, chấm holistic | 1 call API / query |

Mọi tầng tuỳ chọn đều **exception-wrapped và trả nguyên trạng khi lỗi** — một
provider sập chỉ làm mất một tín hiệu, không bao giờ làm chết truy vấn. Fail
tối đa của reranker = `rerank_topk` vị trí (đuôi bảng không bị đụng).

Các đường online khác (cùng engine):

- **QA** (`pipeline/run_queries.compute_qa_answers`): ứng viên gộp nhóm theo
  (video, ngắt cảnh) → VQA **từng nhóm** (≤ `vqa.answers_per_query` nhóm,
  ≤ `vqa.max_calls_per_query` call); mỗi call gửi **dải `vqa.frames_per_answer`
  frame theo thứ tự thời gian** (fix dạng đề "giải toán diễn ra qua nhiều
  frame" của 2025) — mỗi dòng CSV mang answer đúng nhóm của nó.
- **TRAKE** (`engine.search_trake` → `temporal.trake_search`): FAISS
  `temporal.per_event_topk`/sự kiện → pool ≤ `temporal.max_videos` video → DP
  chính xác trên ma trận (frames × events); ensemble khi `temporal.use_ensemble`
  (lane thiếu tự rút, renormalize). Solver: DANTE O(N·T) (mặc định) hoặc beam.
- **AVS** (`engine.search_avs`, GIỮ BEHIND FLAG — 2026 chưa chắc thi):
  search display_k lớn → MMR mức embedding khử b-roll trùng xuyên video
  + cap `avs_per_video_cap`/video + cách ≥ `avs_min_gap_s`.
- **Feedback** (`engine.search_with_feedback`): Rocchio trong không gian của
  TỪNG lane rồi fuse — 1 FAISS call thêm mỗi lane.
- **Low-confidence retry** (`run_queries.maybe_retry_low_confidence`, đường
  batch/auto, OFF mặc định — `search.low_confidence_retry`): ranking "phẳng"
  (confidence < threshold) → re-search tối đa 3 expansion ĐÃ CACHE (không tốn
  call mới) → RRF-merge, primary trước.

## 4. Dataflow offline — build artifacts (resumable từng bước)

```
videos ─► scripts/01_extract_keyframes.py     # K-batch không có keyframes BTC:
 │          TransNetV2 → PySceneDetect → cửa sổ 2s; 3 kf/shot tại extraction.shot_positions
 │          (mặc định [0.15,0.5,0.85]); dedup MAD; GHI LUÔN map-keyframes/{vid}.csv đúng
 │
 ├─► scripts/05_rebuild_map_keyframes.py      # BẢO HIỂM batch sau (Batch 1 2026 ĐÃ
 │          kèm đủ 873/873 map csv — DATASET_INGESTION §1b); thiếu map là không
 │          nộp nổi frame_idx. dhash mọi keyframe → dhash video theo stride 5 →
 │          monotone DP align → refine ±stride. XẤP XỈ — ưu tiên bản BTC khi phát hành.
 │
keyframes ─► scripts/00_build_catalog.py      # manifest.parquet + signature (mục 2)
 ├─► scripts/02_embed_and_index.py [--all-members]   # .npy/video → FAISS/lane;
 │          provided_clip32 nuốt thẳng clip-features-32 của BTC (không cần GPU)
 ├─► scripts/03_build_aux_indexes.py --ocr --asr --captions   # resumable/video
 │          --text-index      # BM25 persisted: artifacts/text_index/{field}.json.gz
 │          --objects-index   # objects.parquet (ObjectBooster tự ưu tiên khi tồn tại)
 │
tất cả ─► scripts/30_ingest.py                # 1 lệnh idempotent: (extract) → catalog
                                              # → embed → index — chỉ làm việc còn thiếu
```

Kỷ luật chung: mọi write qua `utils.io.atomic_*` (`.tmp` + `os.replace`) — Colab
disconnect không bao giờ để lại artifact viết dở; mọi stage skip việc đã xong.

## 5. Kiến trúc 2 thể thức 2026 — MỘT engine, HAI mặt

AIC 2026 có **thể thức tương tác** (người + trợ lý; dùng agent trong tool là
TUỲ CHỌN) và **thể thức tự động** (pilot, trợ lý đấu trợ lý — BTC **chưa công bố
spec giao thức**). Thiết kế: một `SearchEngine` duy nhất, ba mặt gọi nó:

```
                    ┌──────────────────────────┐
   người vận hành ─►│ app/streamlit_app.py     │ 5 tab + đồng hồ + basket + feedback
                    ├──────────────────────────┤
   máy / BTC ──────►│ cvp.service (FastAPI)    │ GET  /health · /nearest/{gid} · /keyframe/{gid}
                    │  cvp serve --port 8000   │ POST /search/text · /search/image
                    │                          │      /search/qa · /search/trake · /search/avs
                    ├──────────────────────────┤
   query pack ─────►│ pipeline/auto_agent      │ đề → CSV → validate → zip → [DRES top-1]
                    └────────────┬─────────────┘
                                 ▼
                          SearchEngine (mục 3)
```

- **Service** (`pip install -e ".[search,service]"` rồi `cvp serve` hoặc
  `uvicorn cvp.service.app:app`): engine load MỘT lần ở lifespan;
  `create_app(engine=...)` nhận engine dựng sẵn — đó là cách test suite chạy
  mọi endpoint không cần model, và cách auto-agent có thể nhúng service
  in-process. Đây là **nền machine-callable cho thể thức tự động**: BTC công bố
  giao thức gì thì viết adapter mỏng lên các endpoint này (mục 6).
- **Auto-agent** (`python scripts/25_auto_agent.py --query-dir … --out-dir …`):
  suy task từ tên file (`trake` → `avs` → `qa` → `kis`) → gọi đúng engine call
  → re-validate CHÍNH các CSV vừa ghi (`packager.validate_file` — file cũ của
  run trước không đi ké) → zip Codabench (folder trong zip **bắt buộc tên
  `submission`** — `submission.package_name`). Với `--submit`: đẩy top-1 qua
  `dres_client` **chỉ khi** run sạch lỗi AND `submission.auto_submit: true`
  AND `dres_base_url` đã set — nộp sai bị TRỪ ĐIỂM ở chung kết nên gate mặc
  định đóng, và client **không bao giờ tự retry** lệnh bị từ chối.
- **DRES**: client theo API DRES v2 (bản hiện hành 2.0.4); mọi path/field đều
  cấu hình được vì endpoint thật chỉ công bố tại chung kết. Answer DRES v2
  không có field frame — frame đổi sang ms qua `fps`.
- **KIS-C** (`models/agent.py`) là *phong cách hệ thống* 2026 (chưa phải task
  chính thức): gộp MỌI hint tích luỹ thành một query + sinh ≤3 câu hỏi làm rõ
  xếp theo khả năng thu hẹp không gian ứng viên; không có Gemini key thì
  degrade thành nối hint.

## 6. Mở rộng hệ thống

### 6.1 Thêm một lane embedding

1. Viết backend trong `src/cvp/models/` implement hợp đồng `models/base.py`:
   `encode_image(list[PIL.Image])` / `encode_text(list[str])` → `float32 (N, dim)`
   **L2-normalized, chung một không gian** (inner product == cosine); khai báo
   `multilingual` (quyết định lane nhận câu VI gốc hay bản dịch) và `model_tag`.
   Import framework nặng LAZY trong constructor.
2. Đăng ký tên trong `models/registry.py` (`resolve_constructor`); nếu lane
   dùng chung không gian với lane khác (kiểu `finetuned`→`siglip2`) thì map
   trong `index_key_for`.
3. Thêm key config vào section `embedding` (config.py + settings.yaml).
4. Build artifacts: `python scripts/02_embed_and_index.py --all-members` (hoặc
   set `CVP_EMBEDDING__MODEL=<tên>` cho một lane), rồi thêm vào
   `embedding.ensemble_members` + `ensemble_weights`.

`jina` (jina-clip-v2, 89 ngôn ngữ, Matryoshka `embedding.jina_dim`; ⚠ weights
CC-BY-NC) và `metaclip2` (MetaCLIP 2 worldwide-huge, SOTA XM3600 giữa các
two-tower mở 2026) là hai ví dụ mẫu đã đi đủ 4 bước — chỉ **đo trên bộ đề dev
rồi mới cho vào ensemble thi đấu**.

### 6.2 Thêm kênh text cho metadata lifelog (định hướng dữ liệu 2026)

Đề 2026 có thể kèm video sousveillance/egocentric với sidecar mới (GPS, thời
gian, sensor…). BM25 stack ăn thêm kênh mà không đụng engine:

1. Viết collector offline ghi `artifacts/<field>/{video_id}.json` theo hợp đồng
   sẵn có — `{"n_to_text": {...}}` (mức keyframe, kiểu OCR/caption) hoặc
   `{"segments": [{"start","end","text"}]}` (mức thời gian, kiểu ASR).
2. Khai báo field trong `search/text_signals.py`: thêm tên vào `FRAME_FIELDS`
   (key theo `global_id`) hoặc xử lý kiểu `metadata` (key theo `video_id`,
   broadcast cho mọi frame của video) + loader tương ứng.
3. Thêm trọng số vào `SearchWeights` (config.py) + `search.weights`
   (settings.yaml) — `_finalize` của engine tự nhặt field mới vào fusion.
4. Persist BM25 để khỏi rescan corpus: nối field vào bước `--text-index` của
   `scripts/03_build_aux_indexes.py` (text_store lưu đủ sufficient statistics
   + signature).
5. Tune lại trọng số bằng `scripts/23_dump_signals.py` + `scripts/21_tune_weights.py`.

### 6.3 Bind giao thức thể thức tự động (khi BTC công bố spec)

Spec chưa tồn tại — chắc chắn KHÔNG phải "tự nộp file". Kiến trúc chờ sẵn:

- Giao thức kiểu **request/response** (BTC gọi mình): viết router FastAPI mới
  trong `service/` translate payload BTC ⇄ `schemas.py`, mount vào
  `create_app` — engine, guard, VQA giữ nguyên.
- Giao thức kiểu **client** (mình gọi server BTC, kiểu DRES): nhân bản pattern
  `submission/dres_client.py` (urllib thuần, exception-wrapped, mọi path là
  template config, KHÔNG tự retry) và nối vào vòng lặp `pipeline/auto_agent.py`
  — chỗ duy nhất cần sửa là `_submit_top1` + gate.
- Trong lúc chờ: `run_auto(engine_factory=…)` nhận engine/client stub → viết
  test giao thức TRƯỚC khi có server thật, đúng cách suite hiện tại test service.

## 7. Ghi chú hiệu năng

- **FAISS `flatip`** (mặc định) là exact search, thoải mái tới **~1–2M vector**
  trên laptop (corpus 2025 ~300k keyframes; kịch bản "1000+ giờ" 2026 vẫn trong
  tầm). Vượt nữa mới cần `index.type: ivf` (`ivf_nlist: 4096`, `ivf_nprobe: 32`
  — nprobe cao hơn = recall cao hơn, chậm hơn) hoặc `hnsw` (`hnsw_m: 32`,
  `hnsw_ef_search: 128`). `index.use_gpu: true` khi có faiss-gpu.
- **Chuẩn latency** (mục tiêu đã chốt): `search_text` **p50 ≤ 200 ms,
  p95 ≤ 500 ms** trên laptop. Đo sau MỖI lần rebuild artifacts:
  `python scripts/50_bench_latency.py --n 200` (in p50/p95/p99). Chung kết
  DRES 4–5 phút/câu + decay theo thời gian — stack phải trả lời dưới 1 giây để
  người (hoặc auto-agent) giữ được quỹ thời gian.
- **BM25 persisted**: tokenize 1 lần lúc build (`scripts/03 --text-index`);
  query time chỉ chấm ứng viên bằng closed-form BM25+ — hết trễ ở truy vấn đầu.
  Thiếu `text_index/` (artifacts của notebook cũ) → tự fallback in-memory.
- **Objects parquet**: 1 lần đọc file thay cho ~178k JSON nhỏ (thảm hoạ trên
  Drive/mạng); build stream từng video, không giữ quá vài MB RAM.
- **Cross-encoder là stage online đắt nhất** — giữ `rerank_topk ≤ 100`, bật
  trên máy GPU; `qwen_reranker` cần smoke-test trên Colab trước ngày thi
  (xem EVALUATION). VLM rerank = 1 call API/query, tính vào budget mạng venue.
- Đo ablation A1–A10 một lệnh:
  `python scripts/26_run_ablations.py --query-dir queries/dev-2025-finals --gt queries/dev-2025-finals/gt.json`
  (bộ đề chuẩn: gói 89 câu chung kết 2025 — 73 KIS / 9 QA / 7 TRAKE).
- Mọi call ngoài (Gemini/translate/VLM) đều exception-wrapped — mất mạng =
  degrade về visual-only, không bao giờ crash giữa trận.
