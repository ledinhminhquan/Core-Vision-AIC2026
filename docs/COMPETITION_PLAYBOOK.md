# 🏆 COMPETITION_PLAYBOOK — Đấu pháp thi đấu trực tiếp

## 0. Trước ngày thi (checklist)

- [ ] Artifacts sync về laptop, app chạy `< 1s/query` (test bằng nb 03 trước).
- [ ] `CVP_EMBEDDING__MODEL=ensemble` với `[finetuned, openclip]` (đã train) —
      nếu máy yếu: `finetuned` đơn.
- [ ] GEMINI_API_KEY nạp sẵn + **đã test offline fallback** (rút mạng thử 1 query).
- [ ] Cache truy vấn ấm (chạy 20–30 query luyện tập).
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

## 2. KIS-V (xem clip)

Cả đội cùng xem; Spotter đọc to các chi tiết phân biệt; Driver gõ ngay 2–3 biến thể
mô tả. Chọn khoảnh khắc ĐẶC TRƯNG nhất của clip (đầu/cuối shot đặc biệt) thay vì
cảnh chung chung.

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

## 7. Vòng sơ tuyển (Codabench) — kỷ luật nộp bài

- **5 lượt/ngày, 20 lượt tổng (thể lệ 2026)** → LUÔN chấm offline trước:
  `python scripts/40_eval_official.py --submission-dir ... --gt ...` (đúng công thức BTC).
- Nộp bằng zip từ `scripts/20_run_queries.py --zip` — packager đã validate từng dòng
  (regex video id, ≤100 dòng, TRAKE tăng dần, QA ≤100 ký tự); zip lỗi là KHÔNG tạo.
- Có bộ đề dev + đáp án (đề practice 2025 = 89 câu): tune trọng số trước đợt nộp
  (`scripts/23` + `scripts/21`), dán các dòng `CVP_SEARCH__WEIGHTS__*` nó in ra.
- 🆕 2026-07-08: batch runner đọc được **đề nguyên bản của BTC** (TRAKE có dòng ngữ
  cảnh + tiền tố `E1:`; QA một dòng có "Hỏi …?") — KHÔNG cần sửa tay file đề nữa;
  GT tự soạn giờ nhận cả `"ranges": [[s1,e1],[s2,e2]]` (nhiều cửa sổ chấp nhận được).

## 8. Thể thức TỰ ĐỘNG 2026 (assistant vs assistant)

- `python scripts/25_auto_agent.py --query-dir <đề>` chạy trọn: nhận diện dạng đề theo tên
  file → search/TRAKE/AVS → QA tự trả lời theo nhóm ứng viên → validate → zip.
- Bật nộp thẳng DRES: đặt `submission.dres_base_url` + env `DRES_USER/DRES_PASSWORD`
  + `submission.auto_submit: true` — client KHÔNG BAO GIỜ tự retry lệnh bị từ chối (bị trừ điểm).
- Diễn tập trước ở nhà bằng bộ đề practice (RUN_AUTO_AGENT trong notebook 03).

## 9. Chung kết — vũ khí bí mật

- **Đồng hồ 5 phút** hiện trong tab KIS-C — nộp sớm điểm cao, dưới 60s màu đỏ.
  (🆕 2026-07-08: đồng hồ tự nhảy theo thời gian thật từng giây —
  `st.fragment(run_every)` — không cần bấm gì để nó cập nhật nữa.)
- **VLM rerank** (`CVP_SEARCH__VLM_RERANK=true`): bật khi mạng ổn — top-24 được Gemini
  chấm lại, +~10% H@1 theo UIT; lỗi API tự về thứ tự cũ nên bật không rủi ro.
- Checkpoint drift: nếu app cảnh báo "index was built by X but loaded Y" → máy này
  tải fallback khác checkpoint đã build index. Đừng thi trên máy đó — rebuild hoặc đổi máy.

## 10. Khi mọi thứ hỏng

| Sự cố | Phản xạ |
|---|---|
| Mất mạng | Hệ tự chạy raw-query (SigLIP-2 đọc tiếng Việt) — cứ thi tiếp, thêm từ khóa OCR |
| Gemini limit | Fallback tự động Google Translate — không cần làm gì |
| App crash | `streamlit run` lại (~30s load); index/catalog bất biến nên không mất gì |
| Query bí | Đổi chiến thuật: tìm bằng OCR text / object đếm được / metadata chương trình |
| Chậm | Giảm `Results shown`; tắt rerank: `CVP_SEARCH__RERANK=false` (khởi động lại) |
| Truy vấn đầu tiên chậm | Thiếu text_index bền vững → chạy `scripts/03_build_aux_indexes.py --text-index` |
