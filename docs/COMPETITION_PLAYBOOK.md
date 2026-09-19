# 🏆 COMPETITION_PLAYBOOK — Đấu pháp thi đấu trực tiếp

## ⚔️ Hai hình thức thi 2026 (đọc trước tiên)

1. **Tương tác (truyền thống)** — người + hệ thống. Dùng agent trong công cụ là
   **TÙY CHỌN**, không bắt buộc: đấu pháp chính vẫn là Driver/Spotter/Verifier
   ở các mục dưới; agent (KIS-C tab, gợi ý VQA) chỉ là trợ thủ.
2. **Tự động (pilot 2026)** — trợ lý đấu trợ lý, KHÔNG người can thiệp. BTC
   **CHƯA công bố spec** (chắc chắn KHÔNG phải kiểu "tự nộp file"); nền
   máy-gọi-được của ta đã sẵn: `cvp serve` (FastAPI: `GET /health`,
   `POST /search/text|image|qa|trake|avs`, `GET /nearest/{gid}`, `GET /keyframe/{gid}`)
   + `scripts/25_auto_agent.py`. Khi spec ra chỉ cần viết adapter giao thức trên
   nền service này.
3. **Theo dõi 3 kênh để bắt spec sớm:** Q&A sheet của BTC · Facebook AICHCMC ·
   Codabench organizer `vnaic` (trang 2026 dự kiến mở đầu–giữa T8/2026; trang
   2025 id 10187 là mẫu đối chiếu format).

## 0. Trước ngày thi (checklist)

- [ ] Artifacts sync về laptop, app đạt mục tiêu `< 1s/query` (test bằng nb 03 trước).
- [ ] `CVP_EMBEDDING__MODEL=ensemble` với `[finetuned, openclip]` (đã train) —
      nếu máy yếu: `finetuned` đơn.
- [ ] GEMINI_API_KEY nạp sẵn + **đã test offline fallback** (rút mạng thử 1 query).
- [ ] Cache truy vấn ấm (chạy 20–30 query luyện tập).
- [ ] **Cổng latency**: `python scripts/50_bench_latency.py` đạt p50 ≤ 200 ms /
      p95 ≤ 500 ms trên đúng laptop thi — chạy SAU mỗi lần rebuild artifacts và
      TRƯỚC mỗi vòng thi; script tự gợi ý nguyên nhân khi vượt ngưỡng.
- [ ] Luyện gõ: mô tả cảnh bằng danh từ + màu + hành động, KHÔNG mô tả meta
      ("tìm cảnh..." là thừa — Gemini enhancement tự bỏ, nhưng đừng phí thời gian gõ).
- [ ] Phân vai 3 người: **Driver** (gõ + thao tác), **Spotter** (nhìn lưới kết quả,
      xem context 🎞), **Verifier** (mở YouTube link xác nhận + quyết định nộp).

## 1. KIS — quy trình 5 phút

1. **0:00–0:30** — Gõ mô tả chi tiết nhất có thể (chi tiết PHÂN BIỆT: chữ trên màn hình,
   màu áo, số lượng người, bối cảnh). Search.
2. **0:30–2:00** — Quét lưới. Thấy "gần đúng" → 🎞 context (frame lân cận) hoặc 🔍 similar.
   Không thấy → đổi góc mô tả (đối tượng khác trong cảnh, góc máy, chữ OCR).
3. **2:00–3:30** — Dùng feedback: đánh ✓ các frame gần đúng, ✗ các frame sai hẳn →
   **♻ Refine**. Thử thêm từ khóa OCR (chữ đọc được trong hint) — BM25 kéo rất mạnh.
4. **3:30–5:00** — Chốt: Verifier xác nhận qua YouTube deep-link đúng giây.
   Nộp sớm = điểm cao; nộp sai = trừ điểm ⇒ **chắc ≥80% mới nộp sớm**, còn lại
   điền đủ 100 dòng (hit rank sau vẫn có điểm ở vòng loại).

**Mẹo mô tả:** ưu tiên (1) chữ xuất hiện trên màn hình (OCR), (2) vật thể đếm được
("hai xe máy màu đỏ"), (3) bối cảnh ("trường quay nền xanh", "bờ sông lúc hoàng hôn"),
(4) hành động. Tránh khái niệm trừu tượng.

## 2. KIS-V (xem clip — CẤM quay/chụp/screen-capture)

**Luật 2026:** clip chỉ được **XEM** trên màn hình BTC — cấm quay phim, chụp ảnh,
screen-capture dưới mọi hình thức; **âm thanh có thể bị TẮT** (đừng xây đấu pháp
dựa vào nghe). ĐƯỢC PHÉP: mô tả lại bằng lời, VẼ lại, hoặc dùng AI sinh ảnh từ
mô tả để đối chiếu. (2025: VKIS 4 phút / clip 20 giây.)

Quy trình "xem → mô tả lại → search":
1. Cả đội cùng xem trọn clip; Spotter đọc TO các chi tiết phân biệt (chữ trên
   hình, màu áo, số người/vật, góc máy, bối cảnh); Verifier ghi thứ tự shot.
2. Driver gõ ngay 2–3 biến thể mô tả → search. Chọn khoảnh khắc ĐẶC TRƯNG nhất
   của clip (đầu/cuối shot đặc biệt) thay vì cảnh chung chung.
3. Thấy frame "gần đúng" → **🔍 similar** (query-by-example) để xoay quanh vùng
   ảnh đó — đây là đường bù cho việc không được giữ clip.
4. Nếu vẽ/sinh ảnh: trước hết dùng ảnh để CẢ ĐỘI thống nhất mô tả; ở mức API đã
   có uploader **🖼 KIS-V — tìm bằng ẢNH** ngay trong tab KIS của app (vẽ/sinh ảnh → tải lên → search, fuse toàn bộ ensemble), hoặc POST `/search/image` của service.

## 3. QA — 2 bước

1. Tìm frame như KIS (bằng phần MÔ TẢ CẢNH của câu hỏi, không phải câu hỏi).
2. Đáp án: đọc trực tiếp từ frame (phóng to) + bấm **💡 Suggest answers (VQA)** để
   Gemini đề xuất — **luôn kiểm tra bằng mắt trước khi nộp**, VQA chỉ là gợi ý.
   Đáp án ngắn gọn ≤100 ký tự, đúng chính tả có dấu.

## 4. TRAKE

1. Tách đề thành các sự kiện NGẮN, mỗi dòng một sự kiện, đúng thứ tự.
2. Search sequences → xem hàng thumbnail từng chuỗi (mỗi cột 1 sự kiện).
3. Sai một sự kiện giữa chuỗi? Sửa mô tả sự kiện đó cụ thể hơn rồi chạy lại.
4. Kiểm tra chuỗi tăng dần hợp lý về thời gian (cột thời gian dưới mỗi thumbnail).

## 5. AVS (nếu xuất hiện)

Tab AVS: mô tả tổng quát → hệ tự đa dạng hóa (≤3 dòng/video, cách ≥10s).
Mục tiêu là PHỦ nhiều video/đoạn khác nhau — đừng dồn 100 dòng vào 1 video.

## 6. KIS-C / progressive KIS

Tab 💬: mỗi khi ban tổ chức nhả thêm gợi ý → **Add hint** (trợ lý tự GỘP mọi gợi ý
thành 1 truy vấn — không bao giờ chỉ search gợi ý mới nhất). Đọc các "câu hỏi làm rõ"
trợ lý đề xuất — đó là những chi tiết nên chờ/đoán từ gợi ý sau.

## 6b. ⚡ CHECKLIST TUẦN THI ĐẦU (buổi 4, 18/08/2026 — đợt 1 TỐI THỨ SÁU trên B1)

1. **Email đội trưởng (chậm nhất thứ Ba)**: tài khoản hệ thống thi RIÊNG của BTC (không phải Codabench) + thư mục Drive chung (BẬT notification) chứa baseline + video hướng dẫn.
2. **Tải file spec nộp bài + validator CSV của BTC** → chạy validator lên zip do `packager.py` sinh ra; lệch gì sửa writer/packager NGAY (tên file CSV phải khớp tên file query từng đợt, vd `query-p1-1-kis.csv`).
3. **Nộp THI THỬ trong tuần** — bắt buộc, để lỗi kỹ thuật chết ở vòng thử.
4. Thi thật đợt 1: ~20-25 câu (đa số KIS + ~4 QA + ~1 TRAKE), CHỈ dữ liệu B1 (nấu ăn + học/luyện thi). Xét TỔNG 3 đợt — đợt 1 điểm thấp vẫn gỡ được, đừng hoảng.
5. Leaderboard **ẨN ~50% điểm** (giữ như 2025) — tin scorer offline, không đốt lượt nộp để "sửa rank".
6. Kiểm frame thủ công: mở video bằng **Media Player Classic → Ctrl+G** (hiện frame hiện tại; frame đánh số TỪ 1, theo presentation time; lệch ±1 frame vẫn được chấm đúng). Độ rộng đoạn đáp án: ngắn ~4-10s, điển hình ~1 phút, tối đa ~5 phút.
7. Query có thể mô tả ÂM THANH và thực thể NGOÀI video (vd "Donald Trump" khi video chỉ nói "Tổng thống Mỹ") — Verifier google thực thể lạ rồi mớm từ khóa THỊ GIÁC cho Driver; Gemini enhancement đã tự mở rộng thực thể (round-18).

## 7. Vòng sơ tuyển (hệ thống riêng của BTC) — kỷ luật nộp bài

- **THỂ LỆ CHÍNH THỨC: 3 lượt/gói, lượt CUỐI tính điểm, sai format vẫn tốn lượt** → LUÔN chạy validator + chấm offline trước:
  `python scripts/40_eval_official.py --submission-dir ... --gt ...` (đúng công thức BTC).
- Nộp bằng zip từ `scripts/20_run_queries.py --zip` — packager đã validate từng dòng
  (regex video id, ≤100 dòng, TRAKE tăng dần, QA ≤100 ký tự); zip lỗi là KHÔNG tạo.
- Có bộ đề dev + đáp án (đề practice 2025 = 89 câu): tune trọng số trước đợt nộp
  (`scripts/23` + `scripts/21`), dán các dòng `CVP_SEARCH__WEIGHTS__*` nó in ra.
- 🆕 2026-07-08: batch runner đọc được **đề nguyên bản của BTC** (TRAKE có dòng ngữ
  cảnh + tiền tố `E1:`; QA một dòng có "Hỏi …?") — KHÔNG cần sửa tay file đề nữa;
  GT tự soạn giờ nhận cả `"ranges": [[s1,e1],[s2,e2]]` (nhiều cửa sổ chấp nhận được).

### 7.1 Bán kết: nộp ĐÚNG 1 FILE (trong khung thời gian)

- Mỗi lượt nộp = **MỘT file zip duy nhất**, bên trong BẮT BUỘC có folder
  `submission/` chứa các CSV per-query — đúng mặc định của packager
  (`submission.package_name: submission`); nộp lẻ CSV hay zip sai layout = 0 điểm.
- CSV: UTF-8, **KHÔNG header**, **≤100 dòng**, phân cách phẩy. KIS:
  `video_name,frame_id` · QA: `video_name,frame_id,answer` (answer ≤100 ký tự,
  VI hoặc EN, quote nếu chứa phẩy) · TRAKE: `video_name,f1,...,fN`
  (`writer.py`/`packager.py` enforce toàn bộ — spec 2025, cổng BTC ghi 2026 giữ format).
- Điểm mỗi câu = trung bình trên k∈{1,5,20,50,100} của **max R-Score trong top-k**
  → luôn điền đủ 100 dòng, xếp hạng tốt ăn điểm gấp bội.
- Hạn mức CHÍNH THỨC 2026: **3 lượt MỖI GÓI truy vấn — lượt nộp CUỐI CÙNG là lượt được chấm** (đừng bao giờ 'nộp thử' bằng lượt cuối; nộp sai định dạng vẫn bị trừ); 2025 còn giới hạn khung giờ
  sáng 9:00–11:59 — chuẩn bị đấu pháp cho CẢ HAI chế độ (dồn chấm offline từ hôm
  trước, nộp bản tốt nhất đầu khung giờ).
- **Bán kết còn phải nộp BÁO CÁO giải pháp** (văn bản) — LaTeX kit có sẵn trong
  `report/` (`ai_conquer2026.cls`); viết song song với các đợt nộp, đừng dồn sát hạn.
- Mốc: dataset + đề + baseline + metrics phát hành **≤ 25/07/2026**; kết quả
  sơ tuyển **30/8**; chung kết on-site **12–26/9/2026**.

## 8. Thể thức TỰ ĐỘNG 2026 (assistant vs assistant)

- `python scripts/25_auto_agent.py --query-dir <đề>` chạy trọn: nhận diện dạng đề theo tên
  file → search/TRAKE/AVS → QA tự trả lời theo nhóm ứng viên → validate → zip.
- Bật nộp thẳng DRES: đặt `submission.dres_base_url` + env `DRES_USER/DRES_PASSWORD`
  + `submission.auto_submit: true` — client KHÔNG BAO GIỜ tự retry lệnh bị từ chối (bị trừ điểm).
- Diễn tập trước ở nhà bằng bộ đề practice (RUN_AUTO_AGENT trong notebook 03).
- ⚠️ Spec thể thức tự động **CHƯA công bố** (assistant-vs-assistant, chắc chắn
  không phải "tự nộp file") — phần trên là baseline "chạy trọn gói đề"; khi BTC
  ra giao thức, viết adapter trên nền `cvp serve` (xem mục ⚔️ đầu file). Cho track
  này cân nhắc bật `CVP_SEARCH__LOW_CONFIDENCE_RETRY=true` (tự reformulate +
  RRF-merge khi ranking phẳng — không có người để đổi query thay máy).

## 9. Chung kết — vũ khí bí mật

- **Đồng hồ 5 phút** hiện trong tab KIS-C — nộp sớm điểm cao, dưới 60s màu đỏ.
  (🆕 2026-07-08: đồng hồ tự nhảy theo thời gian thật từng giây —
  `st.fragment(run_every)` — không cần bấm gì để nó cập nhật nữa.)
- **VLM rerank** (`CVP_SEARCH__VLM_RERANK=true`): bật khi mạng ổn — top-24 được Gemini
  chấm lại, +~10% H@1 theo UIT; lỗi API tự về thứ tự cũ nên bật không rủi ro.
- **Cross-encoder rerank** (`CVP_SEARCH__RERANKER=blip2_itm` hoặc `qwen_reranker`):
  chấm lại top-100 (`search.rerank_topk`) từng cặp (query, ảnh) — `blip2_itm`
  offline cần GPU (blueprint Unified-IMMR 76.4/88); `qwen_reranker` =
  Qwen3-VL-Reranker-8B (model 01/2026). Blend `search.rerank_weight: 0.5`.
  **Chỉ bật sau khi A9 (scripts/26) cho số dương** trên bộ đề dev.
- **Temporal boost** (`CVP_SEARCH__TEMPORAL_BOOST=true`): câu "… sau khi / trước
  khi …" được tách target/context bằng regex (không gọi LLM, deterministic) và
  cộng điểm láng giềng đúng hướng thời gian (công thức Vortex 79.6/88) — bật sau
  khi đo A10.
- **Low-confidence retry** (`CVP_SEARCH__LOW_CONFIDENCE_RETRY=true`) — dành cho
  track TỰ ĐỘNG: ranking phẳng (confidence < `low_confidence_threshold` 0.25) →
  tự reformulate + RRF-merge. Thi tương tác để tắt (người đổi query tốt hơn).
- **📺 Group by video (VISIONE-style)**: checkbox trong app — gom lưới kết quả
  theo video khi cần quét ngữ cảnh nhanh hoặc nghi các hit dồn về một video.
- **Ablations 1 lệnh**: `python scripts/26_run_ablations.py --query-dir queries/dev-2025-finals
  --gt queries/dev-2025-finals/gt.json [--only A9 A10]` — chấm bằng scorer chính thức, quyết
  định knob nào được bật bằng SỐ, không cảm tính.
- Checkpoint drift: nếu app cảnh báo "index was built by X but loaded Y" → máy này
  tải fallback khác checkpoint đã build index. Đừng thi trên máy đó — rebuild hoặc đổi máy.

## 10. Khi mọi thứ hỏng

| Sự cố | Phản xạ |
|---|---|
| Mất mạng | Hệ tự chạy raw-query (SigLIP-2 đọc tiếng Việt) — cứ thi tiếp, thêm từ khóa OCR |
| Gemini lỗi/limit | Chuỗi model tự fallback `gemini-3.7-flash → gemini-3.5-flash → gemini-flash-latest`, hết chuỗi → Google Translate → raw query — không cần làm gì |
| App crash | `streamlit run` lại (~30s load); index/catalog bất biến nên không mất gì |
| Query bí | Đổi chiến thuật: tìm bằng OCR text / object đếm được / metadata chương trình |
| Chậm | Giảm `Results shown`; tắt rerank: `CVP_SEARCH__RERANK=false` (khởi động lại) |
| Truy vấn đầu tiên chậm | Thiếu text_index bền vững → chạy `scripts/03_build_aux_indexes.py --text-index` |
