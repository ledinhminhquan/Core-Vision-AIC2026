# report2.md — Nhật ký đợt 2 (Cursor, nền round-72 `9f4e55c`)

Báo cáo CỘNG DỒN theo Luật an toàn §0.5: mỗi phiên ghi nghiên cứu, kết luận,
từng file sửa/tạo + lý do, trạng thái test, việc dở dang.

---

# PHIÊN 1 — 27/08/2026 · KHẢO SÁT QA (không sửa code)

**Phạm vi phiên**: đọc trọn đường QA, vẽ luồng, xếp hạng điểm chết, lập kế hoạch
Nhiệm vụ A + thiết kế B/C/D/E. File duy nhất được tạo: `report2.md` (file này).

**Baseline**: `python -m pytest tests -q` → **XANH** (817 test thu thập; 816 pass + 1 skip,
~76 s máy local). Không file nào khác bị đụng.

**Nguồn đã đọc trọn**: `src/cvp/search/vqa.py` · `src/cvp/pipeline/run_queries.py` ·
`src/cvp/pipeline/auto_agent.py` · `src/cvp/submission/writer.py` ·
`src/cvp/eval/official.py` · `scripts/62_build_gt_from_reference.py` ·
`src/cvp/config.py` (VqaCfg/SearchCfg) · `src/cvp/pipeline/attempts.py` (nhánh QA) ·
cell bench trong `notebooks/_build_notebooks.py` (~dòng 2360-2432) ·
9 file `*-qa.txt` trong `queries/dev-2025-finals/` · `tests/test_vqa_multiframe.py`.

---

## 1 · SƠ ĐỒ LUỒNG QA TỪNG BƯỚC (đường auto-track = đường bench)

Cấu hình bench#1 thực tế (cell bench nb04): ensemble `finetuned+metaclip2` 60/40,
weights tuned + shrinkage 50%, `qwen_reranker`, VLM rerank 48×3, low-confidence
retry BẬT, `CVP_VQA__SELF_CONSISTENCY=3`. Mặc định settings.yaml: `self_consistency=1`
— nghĩa là **bench và trận vote 3, còn ai chạy config mặc định chỉ vote 1**.

```
query-p*-N-qa.txt
 │ load_query_lines (run_queries.py:47): utf-8-sig, strip_invisible (bỏ Cf), bỏ dòng rỗng
 ▼
parse_query_lines("qa", lines) (run_queries.py:185)
 │ 1 dòng  → split_qa_line (:151): cắt tại "Hỏi"/"Câu hỏi" VIẾT HOA CUỐI CÙNG;
 │           không marker → cắt câu cuối nếu kết "?" hoặc mở đầu Hãy/Hảy/Cho biết/Đếm/Kể tên;
 │           vẫn không → (cả dòng, cả dòng)  ← passthrough, câu hỏi lẫn trong retrieval text
 │ ≥2 dòng → question = dòng cuối nếu trông nghi vấn; description = các dòng còn lại
 ▼
retrieval_text → engine.search_text (engine.py:401)
 │ QueryProcessor: Gemini enhance + translation + 2 expansions, max-fuse per lane
 │ dense topk=500/lane + SuperGlobal; BM25 ocr/asr/caption/metadata + object boost
 │ → fusion weighted_sum (weights tuned) + neighbor_boost(0.10, ±2) [+temporal_boost nếu bật]
 │ → qwen_reranker (bench) → VLM rerank 48×3 (bench) → cắt display_k=120 SearchResult
 │ maybe_retry_low_confidence (run_queries.py:299, bench BẬT):
 │   ranking phẳng (conf<0.25) → search_prepared các biến thể cache → RRF-merge
 ▼
compute_qa_answers(results, question, vqa, settings) (run_queries.py:366)
 │ group_candidates (:222): bucket theo video, cắt nhóm khi gap >10 s HOẶC >250 frame
 │   (tham số CỨNG, không knob); nhóm xếp theo rank tốt nhất
 │ budget = min(answers_per_query=5, max_calls_per_query=5) NHÓM đầu tiên; mỗi nhóm:
 │   ├ asr_context (vqa.py:50): thoại ±20 s quanh pts của dòng tốt nhất, ≤800 ký tự
 │   ├ _group_strip (:351): ≤frames_per_answer=3 ảnh trải ĐỀU nhóm theo thứ tự thời gian
 │   ├ vqa.answer_group (vqa.py:242): Gemini strip — model answer_model=gemini-3.1-pro-preview,
 │   │   wall 90 s, chain Pro→3.7-flash→3.5-flash→flash-latest; ảnh thumbnail ≤768px;
 │   │   self_consistency vote (bench=3): majority theo Counter(casefold CHUỖI THÔ);
 │   │   hòa → đáp án gặp trước thắng. Cả 3 vote fail → Vintern local 1 frame giữa → ""
 │   └ stamp answer lên MỌI dòng của nhóm
 │ Dòng thuộc nhóm ngoài budget / nhóm trả "" → THỪA KẾ answer của nhóm tốt nhất có answer
 │ Dòng nào vẫn "" → QA_FALLBACK_ANSWER = "không rõ"
 ▼
write_qa (writer.py:107): mỗi dòng (video, frame_idx, answer)
 │ sanitize_answer (:50): NFC, \n\t→space, gộp space, CẮT 100 ký tự, quote CSV nếu chứa , hoặc "
 │ dedup (video, frame, answer.casefold()); trần 100 dòng (display 120 → 20 dòng cuối rơi)
 ▼
validate_file → package_codabench → [DRES: submit_qa CHỈ DÒNG 1 nếu bật] (auto_agent.py)
```

**Chấm điểm** (offline mirror của BTC — `official.py`):

- `r_score_qa` (:498) = `1` ⟺ **video khớp** AND **frame ∈ một cửa sổ GT** AND
  **normalize_answer(answer) ∈ normalize_answer(GT answers)**. Sai một trong ba = 0 cho dòng đó.
- `normalize_answer` (:469) = NFC + casefold + gộp whitespace + **bỏ dấu câu CUỐI chuỗi**
  (`.,;:!?…"'`”’)]}»`) — **giữ dấu tiếng Việt**, **KHÔNG bỏ dấu câu ĐẦU/GIỮA chuỗi**.
- Final câu = mean(R@1, R@5, R@20, R@50, R@100), R@k = max R-Score trong k dòng đầu.
  → một dòng đúng-cả-ba ở rank ≤5 đã cho 0.8; **0 tròn nghĩa là KHÔNG dòng nào trong
  100 dòng thỏa đồng thời cả ba điều kiện** — thông tin rất mạnh để chẩn đoán.
- GT offline (`scripts/62`): video = top-1 của bài tham chiếu 19.8/23; cửa sổ = top-5 frame
  cùng video, mỗi frame ±125 (~5 s); answers = MỌI answer khác nhau của các dòng video đó
  trong reference. Reference ≈86% đúng → GT answer/moment có thể nhiễu ở 2-3 câu toàn pack.

**Dạng đáp án gặp trong 9 câu QA pack dev 2025** (định hướng canonicalization):
địa danh/tên riêng (xã, khu du lịch, nghệ sĩ, virus, tên món) ×6, **con số** (đếm "miếng",
chữ số hàng chục) ×2, **văn bản dài** (2 câu thơ — dễ chạm trần 100 ký tự) ×1. Một câu có
đáp án nằm trong CHỮ TRÊN MÀN HÌNH màu xanh (OCR), một câu đáp án từ THOẠI (ASR).

---

## 2 · DANH SÁCH XẾP HẠNG ĐIỂM CHẾT (một câu QA ăn 0)

Xếp theo (xác suất là thủ phạm của 2 câu 0 điểm bench#1) × (mức tàn phá) × (độ rẻ để cứu).
Lớp: **M** = moment sai · **A** = moment đúng answer sai nội dung · **F** = answer đúng nội
dung sai ĐỊNH DẠNG. Cột cuối: cách kiểm chứng RẺ NHẤT (67 = `scripts/67_qa_diag.py` sẽ xây).

| # | Mã | Điểm chết | Vì sao nghi ngờ | Kiểm chứng rẻ nhất |
|---|----|-----------|-----------------|---------------------|
| 1 | **A1** | **Thừa kế fallback theo budget nhóm**: chỉ 5 nhóm đầu được VQA (`answers_per_query=5`); MỌI dòng nhóm 6+ mang answer của NHÓM 1 (cảnh khác → answer gần như chắc sai). Nếu khoảnh khắc đúng nằm rank ~20-100 (R@20/50/100 lẽ ra vớt được 0.6), answer thừa kế giết sạch → 0 tròn. Khớp chữ ký "0 tuyệt đối dù retrieval kiểu KIS hiếm khi trượt cả 100 dòng". | Cơ chế đọc thẳng từ code (`compute_qa_answers` :428-431); bench#1 QA 0/0/0.8 | 67: dòng trúng cửa sổ GT nhưng answer == answer dòng top-1 (video khác) → cờ `INHERITED`; đếm bao nhiêu % dòng CSV mang cùng một answer |
| 2 | **M1** | **Retrieval trượt hẳn**: description QA trừu tượng hơn KIS (vd câu toán p2-21 gần như zero visual cue), question tail (đôi khi chứa cue thị giác, vd "chữ xanh lá dưới bản đồ" p2-3) bị BỎ khỏi retrieval text → GT video không vào nổi 120 dòng. | Câu QA khó vốn dĩ lệch phân bố so với KIS; 2/3 câu 0 | 67: có dòng nào video==GT không; rank đầu tiên của GT video; nếu không có → M1 chắc chắn |
| 3 | **F1** | **Số vs chữ**: model trả "sáu"/"hai" trong khi GT ghi "6"/"2" (hoặc ngược). 2/9 câu dev là câu đếm/chữ số; normalize BTC lẫn offline KHÔNG quy đổi. | Loại câu hỏi hiện hữu trong pack; vote casefold cũng vỡ vì biến thể (xem A5) | 67: match lại bằng canonicalize num↔chữ trên dòng trúng GT — strict=0 nhưng canon=1 → F1; fixture unit test |
| 4 | **M2** | **Lưới keyframe thưa vs cửa sổ ±125**: đúng video nhưng keyframe gần nhất lệch >5 s khỏi moment GT (grid trung bình ~5-7 s/keyframe) → mọi dòng cùng video đều hụt cửa sổ. | Cùng họ với H1 TRAKE đợt 1 (đã nghi grid); QA cửa sổ rộng hơn (±125 vs ±12) nên xác suất thấp hơn | 67: min khoảng cách |frame − cửa sổ| trên các dòng cùng video → in "suýt trúng, hụt X frame" |
| 5 | **A2** | **Model trả lời sai thật** trên frame đúng (đọc chữ nhỏ sai, đếm sai, hallucinate tên) — kể cả 3 vote đồng thuận vẫn có thể đồng thuận SAI. | VQA đếm/OCR-nhỏ là điểm yếu VLM kinh điển | 67: bảng (question, GT answer, answer của ta trên dòng trúng GT) — mắt người phán 1 phút/câu; sửa cần knob neighbor-vote + prompt, không đo thêm được offline |
| 6 | **F3** | **Ngoặc kép/nháy quanh tiêu đề, tên riêng**: `normalize_answer` chỉ rstrip → `"Khung trời mơ ước"` mất ngoặc CUỐI nhưng GIỮ ngoặc ĐẦU → không bao giờ khớp `khung trời mơ ước`. Câu hỏi pack dev trích tiêu đề trong ngoặc → model hay lặp lại ngoặc. | Đọc thẳng từ `_TRAILING_PUNCT` (official.py:94) — lỗ hổng thật của cả scorer offline lẫn khả năng cao của BTC | Unit test `normalize_answer('"x"') != 'x'` (1 phút); 67: canon strip-ngoặc-hai-đầu bắt được lớp này |
| 7 | **F2** | **Đơn vị & khoảng trắng**: "300 kg" vs "300kg" vs "300 ki-lô-gam"; "88%" vs "88 %". Vote thô cũng tách phiếu vì đúng lỗi này (bằng chứng round-40: '300 kg' ↔ '30 kg'). | Câu vaccine 88%, câu 200g trong pack dev | 67: canon unit-fold; unit test |
| 8 | **A5** | **Vote 3 mảnh không đa số**: 3 vote ra 3 biến thể tương đương khác format → Counter không có đa số thật, lấy phiếu GẶP TRƯỚC → như tung xu. Không knob nào cứu khi tắt canonicalize. | Cơ chế đọc thẳng (vqa.py:267-270) | Fixture: mock 3 vote ["2","hai","02"] → xem winner đổi theo thứ tự; 67 không thấy được (CSV chỉ giữ 1 answer/nhóm) |
| 9 | **F5** | **Trần 100 ký tự**: đáp án dài (2 câu thơ p1-19) bị `sanitize_answer` nối dòng rồi CẮT giữa chừng → khác GT vĩnh viễn. | `MAX_QA_ANSWER_CHARS=100` + câu thơ trong pack | 67: cờ answer chạm trần (len==100 sau strip) + so len GT answer |
| 10 | **F4** | **Preamble/trang trí**: "Đáp án: X", "Câu trả lời là X", markdown `**X**` — prompt cấm nhưng model Pro thỉnh thoảng vẫn thêm, nhất là khi có ASR context dài. | Hành vi VLM phổ biến; chưa có guard nào trong code | 67: regex preamble trên toàn bộ answer của lab_full (rẻ, chạy 1 lần thấy ngay tần suất) |
| 11 | **A6** | **VQA chết cả câu** (key/quota/timeout ×3 vote ×5 nhóm, Vintern local cũng fail) → 100 dòng "không rõ" → 0 chắc chắn. | Câu 0.8 chứng tỏ VQA sống ÍT NHẤT một câu; vẫn phải loại trừ per-câu | 67: % dòng "không rõ" per câu (100% → A6, án tại hồ sơ) |
| 12 | **M4** | **split_qa_line vỡ dạng lạ**: passthrough (question=cả dòng) → retrieval text ôm cả câu hỏi (nhiễu CLIP) và VQA nhận câu hỏi thừa mô tả; hoặc description bị cắt cụt do "Hỏi" giả. | 9/9 file dev parse ĐÚNG khi tôi mô phỏng tay các regex — rủi ro nằm ở pack MỚI | 67 (có `--queries`): in (description, question) từng câu để mắt người duyệt 30 giây |
| 13 | **F6** | **Dấu câu GIỮA chuỗi**: "màu đỏ, trắng" vs "màu đỏ trắng"; "TP. HCM" vs "TP HCM". normalize chỉ xử lý đuôi. | Dạng đáp án liệt kê/viết tắt có trong phân bố câu hỏi | 67: canon strip-all-punct tier |
| 14 | **F8** | **Mất dấu tiếng Việt**: "mau xanh" ≠ "màu xanh" — BTC lẫn offline đều giữ dấu. Model tiếng Việt hiếm khi mất dấu, nhưng OCR-đọc-theo có thể. | Xác suất thấp | 67: canon accent-fold tier CHỈ để chẩn đoán (không bao giờ nộp bản không dấu) |
| 15 | **M3** | **Moment đúng nhưng rank >100/120**: writer cắt 100, engine cắt display_k=120 → R@100 cũng trượt. | KIS mạnh → hiếm; nhưng câu khó có thể | 67 chỉ thấy 100 dòng → gián tiếp (GT video vắng bóng); xác chứng sâu cần rank dump trên Colab (đắt — để sau) |
| 16 | **A3** | **Strip 3 frame trải đều bỏ lỡ frame bằng chứng** (nhóm dài >3 keyframe, chữ chỉ hiện 1 frame); thumbnail 768px làm mờ chữ nhỏ. | Cơ chế đọc từ `_group_strip` + `img.thumbnail((768,768))` | Cần ảnh thật → Colab; 67 in size nhóm trúng GT (nhóm ≥6 keyframe = rủi ro cao) làm bằng chứng gián tiếp |
| 17 | **A4** | **ASR context đánh lạc hướng**: thoại ±20 s có thể thuộc tin KHÁC (bản tin chuyển mục nhanh) → model tin thoại hơn hình. | Heuristic; chưa có bằng chứng | Đắt (cần re-call) — ghi nhận; knob neighbor-vote giảm rủi ro chéo |
| 18 | **M5** | **Low-confidence retry xáo đầu ranking**: RRF-merge có thể đẩy dòng đúng khỏi top-1/5 ở câu vốn đã đúng. | Knob bật ở bench; hai mặt | So sánh 2 lần bench retry on/off qua `scripts/65_bench_diff.py` (Colab, 1 lệnh — nhưng tốn 1 lượt bench; ưu tiên thấp) |
| 19 | **F9** | **GT offline nhiễu**: answers lấy từ reference 19.8 — nếu reference sai/khác format ở câu đó thì mọi answer đúng đều 0 offline (đo nhầm, pipeline vô tội). | Reference ~86% đúng | 67 in GT answers per câu — chủ dự án đối chiếu video thật 2 phút/câu; hỏi: reference là của đội mình hay đội khác? |

**Kết luận chẩn đoán sơ bộ** (chưa có artifacts để chạy 67 trên máy này): với retrieval
mạnh cỡ KIS, kịch bản "0 tròn" hợp lý nhất là **A1 (thừa kế answer sai trên dòng trúng
GT rank sâu)** và/hoặc **F1/F2/F3 (đúng nội dung, vỡ định dạng)** — cả hai đều SỬA ĐƯỢC
offline không cần model tốt hơn; M1 dành cho câu toán-đọc-đề. `67_qa_diag` sẽ phân xử
bằng dữ liệu thật, và 3 knob của Nhiệm vụ A nhắm thẳng ba lớp này.

---

## 3 · KẾ HOẠCH CHI TIẾT NHIỆM VỤ A (phiên sau thực thi)

### 3.1 · `src/cvp/search/answer_norm.py` (MỚI — thuần stdlib, dùng chung script + knob)

- `strip_decoration(s)`: bỏ preamble ("Đáp án:", "Trả lời:", "Câu trả lời là…"),
  markdown `**…**`/`\`…\``, ngoặc kép/nháy/«»/“” Ở CẢ HAI ĐẦU, gộp whitespace, NFC.
- `fold_numbers_vi(s)`: chữ số tiếng Việt ↔ digit hai chiều, chuẩn về DIGIT
  (không/một/hai…mười/mươi/trăm, "hai mươi bảy"→"27", "mốt/lăm/tư" biến thể); thập phân
  "3,5"→"3.5"; KHÔNG đụng số trong tên riêng có chữ hoa liền kề.
- `fold_units(s)`: dán số-đơn vị về "N unit" chuẩn hóa ("300kg"→"300 kg", "88 %"→"88%",
  "ki-lô-gam/kilogam"→"kg", "phần trăm"→"%"… bảng nhỏ, mở rộng dần theo 67).
- `canonical_key(s, tier)`: tier 1 = strip_decoration + casefold + bỏ dấu câu hai đầu;
  tier 2 = tier 1 + fold_numbers_vi + fold_units + bỏ dấu câu giữa chuỗi;
  tier 3 (CHỈ chẩn đoán) = tier 2 + bỏ dấu tiếng Việt.
- Bất biến phải test: idempotent; không bao giờ trả chuỗi rỗng từ input không rỗng
  (fallback về input đã NFC/casefold); tier 0 = `normalize_answer` chính chủ (import lại,
  không chép công thức).

### 3.2 · `scripts/67_qa_diag.py` (MỚI — offline thuần, không engine, không API)

```
python scripts/67_qa_diag.py --run <lab_full dir|zip> --gt <gt-thunghiem.json> \
    [--queries <pack dir>] [--out qa_diag.md] [--json-out qa_diag.json] [--head 10]
```

- Tái dùng: `attempts.load_run` (đọc dir/zip y hệt 63/64), `official.load_ground_truth`,
  `normalize_answer`, `_entry_ranges`/`_entry_answers` (qua API công khai nếu đủ, nếu
  không thì hàm nhỏ nội bộ đọc entry canonical — KHÔNG chép công thức chấm).
- Per câu QA trong GT, phân loại theo cây quyết định:
  1. Không dòng nào `video==GT` → **MOMENT_MISS/no-video**; nếu có video nhưng mọi frame
     hụt cửa sổ → **MOMENT_MISS/near** + min khoảng cách frame (bắt M1 vs M2), rank đầu
     tiên của GT video.
  2. Có dòng trúng cửa sổ: strict match (tier 0) trên các dòng trúng → **OK** (đối chiếu
     final trong bench_full.json nếu có).
  3. Không strict nhưng tier 1/2 match → **FORMAT_MISS** (+ tier nào cứu được → map về
     F1/F2/F3/F6; tier 3 match → cờ F8 riêng).
  4. Còn lại → **ANSWER_MISS**; cờ phụ: `INHERITED` (answer dòng trúng == answer dòng
     top-1 mà top-1 khác video → A1), `NO_ANSWER` ("không rõ" → A6), else A2.
- Diagnostics kèm per câu: % "không rõ", số answer phân biệt trong head-10 (đồng thuận),
  answer chạm trần 100 ký tự, preamble regex hit, GT answers echo, (có `--queries`)
  description/question sau split để duyệt M4.
- Output: bảng markdown xếp theo lớp + JSON máy đọc; tổng kết đếm theo lớp → "đầu tư
  knob nào trước" có số liệu.
- **Test fixture** (`tests/test_qa_diag.py`): tmp_path dựng gt.json + 5 CSV nhỏ phủ đủ
  5 verdict (OK / MOMENT_MISS ×2 kiểu / ANSWER_MISS+INHERITED / FORMAT_MISS số-vs-chữ
  / NO_ANSWER); import script qua importlib như test 64 đang làm; assert verdict + cờ.

### 3.3 · Knob 1 — `vqa.answer_canonicalize` (bool, mặc định **false**)

- Điểm sửa: `VqaAssistant.answer_group` (vqa.py:267-270). Khi bật: vote theo
  `canonical_key(vote, tier=2)`; lớp thắng = nhiều phiếu nhất; đại diện nộp bài = dạng
  RAW xuất hiện nhiều nhất trong lớp thắng (hòa → gặp trước, giữ determinism). Khi tắt:
  giữ nguyên từng byte hành vi Counter cũ.
- Tests (mock instance method `_ask_gemini_strip`, không API): votes ["2","hai","ba"]
  → knob on thắng lớp {2,hai} và nộp "2"; knob off giữ hành vi cũ; votes toàn "không rõ"
  không bị canonical hóa thành rỗng.

### 3.4 · Knob 2 — `vqa.answer_neighbor_frames` (int ≥0, mặc định **0**)

- Điểm sửa: `compute_qa_answers`. Khi >0, mỗi nhóm trong budget hỏi thêm tối đa N strip
  LỆCH: strip lấy từ các dòng lân cận cùng video quanh nhóm (ưu tiên keyframe sát ngoài
  hai mép nhóm có mặt trong `results`; không đủ thì chia đôi nhóm). Phiếu của MỌI strip
  đổ chung một rổ vote (canonical nếu knob 1 bật) → moment lệch nhẹ vẫn hội tụ về một
  đáp án; đồng thời `max_calls_per_query` được cho RĂNG THẬT: tổng lời gọi
  (strip × self_consistency) bị cắt tại đó, không âm thầm nhân (vá đúng tinh thần
  "fail to tiếng" của 4 bản vá audit đợt 1 — hiện `max_calls_per_query` chỉ đếm NHÓM,
  không đếm CALL).
- Chi phí bench: neighbor=1 → ×2 call Gemini per nhóm. Ghi rõ trong settings.yaml.
- Tests: stub vqa đếm số lần `answer_group`/strip nhận được; neighbor=0 → y hệt cũ
  (bit-identical answers); neighbor=1 → 2 strip/nhóm, majority chéo strip thắng;
  budget call bị tôn trọng.

### 3.5 · Knob 3 — `vqa.consistency_rerank` (bool, mặc định **false**)

- Cần độ ổn định phiếu per nhóm → thêm `VqaAssistant.answer_group_stats(...)` trả
  `(answer, agree_ratio, n_votes)`; `answer_group` cũ = wrapper lấy phần tử đầu (stub/
  legacy không có method mới vẫn chạy qua duck-typing như code hiện tại).
- Hàm THUẦN `consistency_rerank(results, answers, stats) -> (results, answers)` trong
  `run_queries.py`: chỉ khi nhóm top BẤT NHẤT (agree < 0.5, ví dụ 3 phiếu 3 kiểu sau
  canonical) VÀ tồn tại nhóm khác trong budget ĐỒNG THUẬN TUYỆT ĐỐI (agree == 1.0) →
  đẩy trọn khối dòng của nhóm ổn định lên trước khối nhóm bất nhất (thứ tự nội bộ giữ
  nguyên, các dòng khác giữ nguyên). Ngưỡng là hằng số module có docstring, không sinh
  thêm knob phụ.
- Wiring `run_query_file`: nhánh QA gọi bản `_with_stats`; **`record()`/top1_times phải
  chạy SAU rerank** (nếu không, DRES nhận timestamp của dòng cũ — `_reconcile_times`
  hiện tại sẽ tự vứt times khi lệch, nhưng ta chủ động ghi đúng thay vì dựa lưới an toàn).
- Tests: stub scripted stats — nhóm 1 (3 phiếu 3 kiểu) vs nhóm 2 (3/3 cùng đáp) → on:
  khối nhóm 2 lên đầu, record times theo dòng mới; off: thứ tự bit-identical.

### 3.6 · Config + tài liệu + trình tự

- `config.py::VqaCfg` thêm 3 field (default off/0) + `configs/settings.yaml` thêm 3 dòng
  có comment chi phí/ý nghĩa. Không đổi bất kỳ default hành vi nào.
- Trình tự phiên sau: (1) answer_norm + tests → (2) 67_qa_diag + tests → (3) knob 1 →
  (4) knob 2 → (5) knob 3, mỗi bước `pytest tests -q` xanh rồi mới sang bước kế; report2
  cập nhật cuối phiên.
- Đề xuất giá trị bench#2 (chủ dự án A/B trên Colab): `answer_canonicalize=true` (an toàn
  nhất, không thêm call), `answer_neighbor_frames=1` riêng cho QA nếu quota chịu ×2,
  `consistency_rerank=true` chạy thử sau khi 67 xác nhận lớp lỗi tồn tại.

---

## 4 · THIẾT KẾ NHIỆM VỤ B/C/D/E (phác thảo đủ để thực thi, chưa code)

### B — `scripts/66_trake_diag.py` (định lượng H1/H2/H6 của report đợt 1)

- Offline: `--gt gt.json --map-dir <map-keyframes/> [--run <lab TRAKE CSVs>] --out md`.
- **H1 trần lưới**: per câu per event, tồn tại keyframe `frame_idx ∈ [s,e]`? Trần điểm
  per câu = điểm của TỔ HỢP keyframe tốt nhất thỏa strictly-increasing (DP tham lam nhỏ
  trên các ứng viên trong/gần cửa sổ) → bảng per câu + mean trần toàn pack; verdict H1
  đúng nếu mean trần ≈ 0.2 (điểm hiện tại đã sát trần → đầu tư nộp NGOÀI lưới/jitter,
  không phải retrieval).
- **H2 recall video**: từ lab CSVs — GT video có xuất hiện trong 100 dòng không + rank
  đầu tiên; verdict H2 đúng nếu recall thấp (<~70%) → đầu tư pool/context knob.
- **H6 max-over-rows**: per câu, `r_score_trake` từng dòng → so max-over-100-dòng vs
  dòng-1; hiệu số trung bình = lợi ích tiềm năng của jitter/đa dạng hóa hàng (đã có knob
  `submit_strategy: jitter` đợt 1 chờ bench). Fixture: map csv + gt + CSV tí hon tự dựng.

### C — `search.neighbor_consistency_boost` (thuật toán thuần, KIS/QA — không đụng TRAKE)

- Khác `fusion.neighbor_boost` hiện có (điểm của ứng viên LOANG SANG hàng xóm — gai đơn
  độc vẫn phát tán): boost mới thưởng ứng viên ĐƯỢC hàng xóm chống lưng — gai đơn độc
  hàng xóm ~0 nên không được gì, cao nguyên được cộng.
- Công thức đề xuất: trong `_finalize` sau neighbor_boost hiện có, trên fused map đã
  minmax n̂: `support(g) = Σ_{d=1..W} λ^d · (n̂(g−d)+n̂(g+d))/2` (λ=0.5, W=knob window
  mặc định 2, biên video qua `_spans_for`); `fused += β · support` với β =
  `neighbor_consistency_boost` (float, mặc định **0.0 = off**, guard `β≤0 → return sớm`
  bit-identical). Chỉ chạm `_finalize` → search_text/search_prepared/feedback (KIS/QA/AVS)
  hưởng, TRAKE (đường `search_trake` riêng) không đụng. AVS đi chung search_text — chấp
  nhận, ghi chú trong settings.yaml.
- Tests: map tổng hợp — cao nguyên (0.8/0.85/0.8) thắng gai (0.9 lẻ loi) khi bật; thứ tự
  y nguyên từng byte khi 0.0; không rò qua biên video.

### D — `src/cvp/pipeline/row_budget.py` + `search.row_strategy` (KIS/QA, mặc định `legacy`)

- `diversify_tail`: giữ nguyên T dòng đầu (hằng đề xuất 30); phần đuôi thay vì đuôi
  ranking loãng → round-robin neighbor keyframe (±1..±2) của các top-row THUỘC VIDEO
  KHÁC NHAU trong đầu bảng (triết lý jitter TRAKE: metric max-over-rows, dòng đuôi là vé
  số nên phủ QUANH các đỉnh khác video thay vì rải rác); QA: dòng thêm mang answer của
  nhóm nguồn; (mở rộng sau khi có answer_norm) — với câu trả lời dạng SỐ, thêm dòng
  song-format ("6"/"sáu") trên cùng frame — `write_qa` đã chủ đích cho phép cùng frame
  khác answer.
- Hàm thuần trên (rows, refs lân cận từ results) → test offline không cần engine; wire
  tại `run_query_file` trước write_kis/write_qa; `legacy` = bit-identical.

### E — `scripts/68_profile_engine.py` + tối ưu không-đổi-kết-quả

- Micro-bench offline trên fixture synthetic (thiếu artifacts tự sinh): tokenize
  (`text_utils`), BM25 persisted scoring (text_store fixture tmp), `fusion.weighted_sum`
  + `neighbor_boost` (dict 500-5 000 key), DP TRAKE (`temporal` sim matrix ngẫu nhiên cố
  seed). Output bảng md (op, N, ms/op, ước lượng cho đêm 23 câu).
- Tối ưu chỉ nhận khi có test khẳng định OUTPUT KHÔNG ĐỔI trên fixture ngẫu nhiên seed
  cố định (ứng viên: vectorize neighbor_boost/support bằng numpy trên mảng gid liên
  tục; cache `video_span`; tránh re-sort thừa trong `_finalize`) — mỗi cái một commit
  logic riêng trong phiên, đo trước/sau bằng chính 68.

---

## 5 · HỒ SƠ PHIÊN 1 (theo Luật §0.5)

- **Nghiên cứu**: toàn bộ đường QA + scorer + GT builder + bench config (chi tiết ở §1-2).
- **Kết luận chính**: (1) 0-tròn QA gần như chắc chắn KHÔNG phải "model ngu" đơn thuần —
  cơ chế thừa kế answer theo budget (A1) và lớp định dạng F1/F2/F3 là các thủ phạm sửa
  được rẻ; (2) `normalize_answer` có lỗ hổng ngoặc-đầu-chuỗi thật (F3) — vá được bằng
  canonicalize khi vote + kiểm chứng bằng 67; (3) vote 3 mảnh casefold thô tự vỡ trên
  biến thể định dạng (A5) — knob 1 xử; (4) `max_calls_per_query` hiện không giới hạn
  CALL thật (chỉ đếm nhóm) — sẽ được cho răng thật trong knob 2, theo bài học "fail to
  tiếng".
- **File sửa/tạo**: chỉ TẠO `report2.md` (file này) — đúng phạm vi phiên khảo sát.
- **Test status**: XANH — 817 collected, 816 pass + 1 skip, không sửa test nào.
- **Việc dở dang / cần chủ dự án**:
  1. Repo local KHÔNG có artifacts — để chạy `67_qa_diag` thật cần kéo từ Drive về (hoặc
     chạy trên Colab): `artifacts/lab/lab_full/` (+ `lab_full-prev/` nếu muốn so),
     `queries/gt-thunghiem.json`, `artifacts/lab/bench_full.json`.
  2. Câu hỏi: reference 19.8/23 là bài của ĐỘI MÌNH hay đội khác? (quyết định độ tin của
     GT answer format — F9.)
  3. Câu hỏi: BTC có công bố quy tắc chuẩn hóa answer khi chấm không (case, dấu câu,
     số/chữ)? Nếu có ví dụ chấm vòng nháp, gửi để 67 học đúng luật thay vì phỏng đoán.
- **Phiên kế tiếp**: thực thi §3 theo trình tự 3.6 (answer_norm → 67 → knob 1 → 2 → 3),
  suite xanh sau từng bước.

---

# PHIÊN 2 — 27-28/08/2026 · NHIỆM VỤ A: QA OVERHAUL (đã thực thi trọn)

Thực hiện đúng kế hoạch §3 (một chỉnh sửa nhỏ so với thiết kế, ghi ở 6.3).
**Test: XANH — 855 test (854 pass + 1 skip; +38 test mới), không sửa/xóa test cũ.**

## 6.1 · File TẠO MỚI (4)

| File | Lý do |
|---|---|
| `src/cvp/search/answer_norm.py` | Module chuẩn hóa answer dùng chung (script 67 + knob vote + test): `strip_decoration` (preamble "Đáp án:", markdown, ngoặc HAI ĐẦU — vá đúng lỗ hổng F3 mà scorer chỉ rstrip), `fold_numbers_vi` (parser chữ số tiếng Việt 0-999: mươi/mười/trăm/linh/lẻ + biến thể mốt/tư/lăm/nhăm — "không" và "tư/lăm/mốt" đứng lẻ KHÔNG fold để bảo vệ "không rõ"/"tư vấn"), `fold_units` (300kg→"300 kg", "88 %"→"88%", ki-lô-gam→kg, phần trăm→%), số 0 đầu ("06"≡"6"), thập phân "3,5"≡"3.5", nghìn "1.000"≡"1000", `canonical_key` 3 tier (tier 3 bỏ dấu = CHỈ chẩn đoán), `majority_vote` (nhánh legacy tái tạo BYTE-IDENTICAL semantics Counter cũ; nhánh canonical vote theo lớp, đại diện = dạng raw phổ biến nhất lớp thắng). Thuần stdlib. |
| `scripts/67_qa_diag.py` | Chẩn đoán offline (không engine/API): đọc run dir/zip (`attempts.load_run`) + gt.json (`official.load_ground_truth`), phân loại từng câu QA: NO_SUBMISSION / MOMENT_MISS (cờ NO_VIDEO hoặc NEAR_MISS + số frame hụt + rank đầu của video GT) / OK / FORMAT_MISS (tier 1/2/3, cờ ACCENT_ONLY) / ANSWER_MISS (cờ INHERITED = chữ ký điểm chết A1, NO_ANSWER). Kèm per câu: %fallback, đồng thuận head, answer chạm trần 100 ký tự, answer còn trang trí, `--queries` echo (description, question) + cờ PARSE_PASSTHROUGH (M4). Xuất console + markdown + JSON. Hằng `QA_FALLBACK` là bản sao có chủ đích của `run_queries.QA_FALLBACK_ANSWER` (import run_queries kéo engine stack) — test khóa hai hằng bằng nhau. |
| `tests/test_qa_overhaul.py` | 24 test: answer_norm (số/đơn vị/ngoặc/idempotent/không-rỗng; chứng minh lỗ hổng ngoặc-đầu của `normalize_answer` là thật và tier 1 vá được), knob 1 (off = giữ nguyên vote chuỗi thô kể cả tie-break, on = gộp lớp "hai"+"2" thắng "ba" — mock `_ask_gemini_strip`, không API), knob 2 (off = KHÔNG đụng API phiếu + 1 call/nhóm y cũ; on = vote chéo strip lân cận, gộp canonical "sáu/6/06", cạn budget → WARNING to + không âm thầm; stub cũ không có API phiếu → tự về đường cũ), knob 3 (top bất nhất + nhóm dưới đồng thuận → thăng khối; top có đa số thật / nhóm fallback / nhóm 1-phiếu → KHÔNG đụng; end-to-end `run_query_file`: CSV đảo đúng khối và `top1_times` ghi theo dòng top MỚI; off = thứ tự + times y cũ). |
| `tests/test_qa_diag.py` | 14 test: từng verdict/cờ trên fixture tự dựng (INHERITED, NEAR_MISS + khoảng cách, tier 1/2/3…), answer tràn cột không quote vẫn nhận nội dung, CLI end-to-end (md + JSON + console, câu KIS không bị lôi vào), fail-loud khi --run sai / GT không có câu QA, và khóa `QA_FALLBACK == run_queries.QA_FALLBACK_ANSWER`. |

## 6.2 · File SỬA (4)

| File | Sửa gì + lý do |
|---|---|
| `src/cvp/config.py` | `VqaCfg` +3 field mặc định OFF: `answer_canonicalize: bool=False`, `answer_neighbor_frames: int=0 (0..4)`, `consistency_rerank: bool=False` — luật §0.3. |
| `configs/settings.yaml` | Mục `vqa:` thêm 3 knob + comment chi phí ("strip thêm ăn budget max_calls_per_query, cạn budget sẽ WARN to") và ý nghĩa từng knob. |
| `src/cvp/search/vqa.py` | Tách vòng thu phiếu thành method public `answer_group_votes` (phiếu RAW ≤ self_consistency; gemini-only) — `answer_group` gọi lại nó, nhánh off giữ NGUYÊN VĂN hai dòng Counter cũ; khi `answer_canonicalize` bật → `majority_vote(canonicalize=True)`. Không đổi chữ ký/hành vi mặc định. |
| `src/cvp/pipeline/run_queries.py` | (1) `compute_qa_answers` thành wrapper mỏng của `compute_qa_answers_with_stats` — mặc định tạo Y HỆT chuỗi call cũ (answer_group → suggest, 1 call/nhóm), stats chỉ được ghi thêm; (2) đường phiếu gộp: khi `answer_neighbor_frames>0` hoặc `consistency_rerank` bật VÀ vqa có `answer_group_votes` → hỏi strip chính + strips lân cận (`_neighbor_strips`: dòng cùng video ngoài nhóm, gần tâm nhóm nhất — chỉ dùng `results`, offline-testable) rồi `majority_vote` trên MỌI phiếu; strip thêm phân bổ round-robin (`_round_robin_extras`) trong phần budget `max_calls_per_query` còn lại sau strips chính — cạn budget → `log.warning` TO đúng bài học "fail to tiếng"; (3) `QaGroupStat` + `plan_consistency_rerank` (ngưỡng hằng số module có docstring: top thiếu đa số <0.5, nhóm thăng phải đồng thuận 100% với ≥2 phiếu, không phải "không rõ") + wiring nhánh QA của `run_query_file`: đảo khối dòng rồi RE-record `top1_times` theo dòng top mới (không dựa lưới an toàn `_reconcile_times`). |

## 6.3 · Lệch so với thiết kế phiên 1 (ghi trung thực)

- §3.5 dự kiến thêm method `answer_group_stats` trên VqaAssistant — thực tế chỉ cần
  `answer_group_votes` (stats tính ở compute, nơi có đủ ngữ cảnh strip). Bề mặt API nhỏ hơn.
- `max_calls_per_query` GIỮ nguyên nghĩa cũ (số nhóm) khi mọi knob off; chỉ khi
  `answer_neighbor_frames>0` nó kiêm thêm vai trần TỔNG strip-call (chính + lân cận) —
  đúng tinh thần "răng thật" nhưng không đổi hành vi mặc định.
- Phát hiện thêm khi viết fixture: "06" ≠ "6" trong bản nháp canonical → thêm
  `_LEADING_ZERO_RE` (chỉ số đứng lẻ, "l01" không bị đụng vì không có \b giữa hai ký tự từ).

## 6.4 · Cách bật khi bench (Colab, cell bench nb04 hoặc env trận)

```python
# An toàn nhất, không thêm call API — đề xuất bật NGAY bench#2:
os.environ["CVP_VQA__ANSWER_CANONICALIZE"] = "true"
# Vote chéo strip lân cận — chi phí ×(1+N) call Gemini per nhóm; PHẢI nới budget:
os.environ["CVP_VQA__ANSWER_NEIGHBOR_FRAMES"] = "1"
os.environ["CVP_VQA__MAX_CALLS_PER_QUERY"]   = "10"   # 5 nhóm × (1+1) strip
# Thử nghiệm sau khi 67 xác nhận lớp lỗi (đảo thứ tự dòng → đo bằng bench diff):
os.environ["CVP_VQA__CONSISTENCY_RERANK"] = "true"
```

Chẩn đoán trước khi chọn knob (chạy được ngay trên Colab với artifacts Drive):

```bash
python scripts/67_qa_diag.py --run artifacts/lab/lab_full \
    --gt queries/gt-thunghiem.json --queries <pack đề> \
    --out artifacts/lab/qa_diag.md --json-out artifacts/lab/qa_diag.json
```

## 6.5 · Dự đoán trung thực + rủi ro

- **67_qa_diag**: giá trị chính là THAY phỏng đoán bằng số liệu — nó quyết định knob nào
  đáng bật. Rủi ro: GT từ reference 19.8 nhiễu (~14%) → verdict per câu có thể oan/sót ở
  1-2 câu; INHERITED là heuristic chữ ký (trùng answer + khác video) — có thể dương tính
  giả khi hai cảnh thật sự cùng đáp án.
- **answer_canonicalize** (kỳ vọng cao nhất/chi phí 0): chỉ cứu được điểm khi đáp án ĐÚNG
  đã nằm trong phiếu nhưng thua vote vì biến thể định dạng (A5) — nó KHÔNG sửa được định
  dạng của chính chuỗi nộp đi (nếu mọi phiếu đều "sáu" mà GT là "6" thì vẫn 0; lớp đó cần
  đến answer-variant rows, đã ghi vào thiết kế Nhiệm vụ D). Ước lượng: cứu 0-1 câu QA trên
  bench 3 câu; giá trị thật rõ hơn ở pack 2026 nhiều câu số/đơn vị.
- **answer_neighbor_frames**: nhắm A2/A3 (frame lệch, chữ chỉ hiện ở frame lân cận).
  Rủi ro: strip lân cận có thể thuộc CẢNH KHÁC cùng video → phiếu nhiễu; đã giảm bằng
  round-robin budget + majority (phiếu strip chính vẫn tham gia). Chi phí API ×2 khi N=1 —
  cân quota đêm thi. Đề xuất chỉ bật cho QA khi 67 chỉ ra A2/A3 đáng kể.
- **consistency_rerank**: bảo thủ (chỉ hành động khi top VÔ đa số và nhóm dưới đồng thuận
  100% ≥2 phiếu) nên tần suất kích hoạt thấp; khi kích hoạt đúng sẽ cứu R@1/R@5. Rủi ro
  còn lại: nhóm đồng thuận nhưng SAI (model tự tin nhầm trên cảnh dễ đọc) → mất điểm
  R@1 của câu vốn đúng; vì thế mặc định off và chỉ nên bật sau khi bench diff xác nhận.
- Rủi ro chung: đường phiếu (khi knob bật) đổi thứ tự/số lần gọi Gemini → kết quả bench
  không so sánh bit-với-bit với lượt cũ ngay cả khi điểm bằng — dùng `scripts/65_bench_diff`
  để đọc delta per câu thay vì so tay.

## 6.6 · Trạng thái & việc dở dang

- Suite: **XANH** (855 = 854 pass + 1 skip; 76-48s tùy máy). Knob off = bit-identical
  (test khóa cả thứ tự dòng, times, số call API).
- Chưa làm (đúng phạm vi phiên): Nhiệm vụ B (66_trake_diag), C (neighbor_consistency_boost),
  D (row_budget), E (68_profile), F (tổng kiểm bàn giao).
- Vẫn chờ chủ dự án như Phiên 1: kéo `lab_full/` + `gt-thunghiem.json` (+ `bench_full.json`)
  về repo/Colab để chạy 67 thật; trả lời nguồn gốc reference 19.8; luật chuẩn hóa answer
  của BTC nếu có.

---

# PHIÊN 3 — 28/08/2026 · NHIỆM VỤ B (66_trake_diag) + C (neighbor_consistency_boost)

**Test: XANH — 872 test (871 pass + 1 skip; +17 test mới), không sửa/xóa test cũ.**

## 7.1 · Nhiệm vụ B — `scripts/66_trake_diag.py` (TẠO MỚI, offline thuần)

Định lượng 3 giả thuyết TRAKE của report đợt 1 bằng số liệu thay vì phỏng đoán:

```bash
python scripts/66_trake_diag.py --gt queries/gt-thunghiem.json \
    --map-dir data/map-keyframes --run artifacts/lab/lab_full \
    --out artifacts/lab/trake_diag.md --json-out artifacts/lab/trake_diag.json
```

- **H1 — trần lượng tử hóa lưới** (`grid_ceiling`): DP O(K·P) chọn tổ hợp keyframe
  strictly-increasing (đúng ràng buộc writer) trúng nhiều cửa sổ GT nhất → trần điểm
  per câu = max R-Score đạt được nếu retrieval/align HOÀN HẢO trên lưới hiện có. DP xử
  đúng cả ca hai event chung một cửa sổ chỉ chứa 1 keyframe (ceiling 0.5, không phải 1.0)
  và ca cửa sổ GT đảo thứ tự thời gian. Kèm per-event ✓/✗ (chỉ ra event nghẽn),
  `median_gap` của lưới, cờ `feasible=False` khi video ít keyframe hơn số event.
- **H2 — recall video ở pool** (cần `--run`): video GT có mặt trong 100 dòng nộp? rank
  đầu tiên? Ghi TRUNG THỰC trong báo cáo: đây là cận DƯỚI của recall pool nội bộ (pool
  thật chỉ quan sát được khi chạy engine — CSV là thứ duy nhất có offline).
- **H6 — lợi ích max-over-rows** (cần `--run`): per câu: điểm dòng 1 vs dòng tốt nhất
  (+rank), Final đầy đủ, và **headroom = trần H1 − dòng tốt nhất** — chính là phần điểm
  mà jitter/đa dạng hóa hàng (knob `temporal.submit_strategy=jitter` đợt 1) còn với tới.
- Verdict tự động theo ngưỡng hằng số có docstring: H1 ĐÚNG khi trần trung bình <0.5
  (đầu tư nộp frame ngoài lưới), SAI khi ≥0.9 (lỗi ở retrieval/align); H2 ĐÚNG khi
  recall <70%; H6 ĐÚNG khi mean(best−row1) >0.05. Output console + markdown + JSON.
- Test (`tests/test_trake_diag.py`, 11 test): DP các ca hit/khe-lưới/chung-cửa-sổ/đảo
  thứ tự/infeasible, `diagnose_query` end-to-end trên fixture map+gt+CSV tự dựng
  (row1 sai video 0đ, row2 2/3 → benefit +0.667, headroom 0 vì chạm trần), verdicts
  theo ngưỡng, CLI ghi md+json, bỏ H2/H6 khi thiếu `--run`, fail-loud khi map-dir sai /
  GT không có TRAKE.

## 7.2 · Nhiệm vụ C — knob `search.neighbor_consistency_boost` (mặc định 0.0 = OFF)

- **Công thức** (nghiên cứu ghi trong docstring `fusion.neighbor_consistency_boost`):
  - Khác bản chất với `neighbor_boost` hiện có (loang điểm của ứng viên RA hàng xóm —
    gai đơn độc vẫn tưới được vùng quanh nó): boost mới chỉ CỘNG cho ứng viên từ điểm
    mà hàng xóm TỰ kiếm được — gai có hàng xóm câm lặng nhận đúng 0.
  - Chuẩn hóa `n̂(g)=s(g)/max(s)` thay vì min-max: fused score vốn ≥0 với sàn gần 0;
    min-max sẽ zero hóa vai cao nguyên khi vai trùng min của danh sách ứng viên ngắn —
    xóa đúng tín hiệu cần gom.
  - `support(g) = Σ_{d=1..window} decay^d · (n̂(g−d)+n̂(g+d))/2` — trung bình hai phía
    (chuỗi một phía chỉ được nửa), hàng xóm ngoài map/ngoài biên video đóng góp 0
    (chính là hình phạt gai); KHÔNG vượt biên video (dùng spans như neighbor_boost).
  - `out(g) = s(g) + weight · max(s) · support/Σdecay^d` — support chuẩn hóa về [0,1]
    nên `weight` chặn trần lift theo tỷ lệ cố định của điểm top, bất kể window/decay.
    Hệ quả đã pin bằng test: decay là tham số HÌNH DẠNG (chỉ tác động khi hàng xóm
    gần/xa khác mức — decay nhỏ tập trung hàng xóm gần).
- **Vị trí cắm**: `engine._finalize`, TRƯỚC `neighbor_boost` có chủ đích — smoothing
  loang gai ra hàng xóm sẽ ngụy tạo đúng cái cao nguyên đang được kiểm tra; spans tính
  MỘT lần dùng chung hai boost. Chỉ chạm `_finalize` → search_text / search_prepared /
  search_with_feedback (KIS/QA/AVS); `search_trake` đường riêng, KHÔNG đụng (đúng đề bài).
  Ghi chú trung thực: AVS đi chung `search_text` nên hưởng theo; đường ảnh
  (`search_image`) không qua `_finalize` nên không hưởng.
- **Config**: `SearchCfg.neighbor_consistency_boost=0.0` (off) + `_window=2` +
  `_decay=0.5 (gt 0, le 1)`; settings.yaml kèm gợi ý bench thử 0.10-0.20.
- **Test** (`tests/test_neighbor_consistency.py`, 6 test): công thức khớp số tay từng
  gid (tâm/vai cao nguyên vs gai — cao nguyên thắng gai 0.90, gai GIỮ NGUYÊN giá trị);
  off trả về CHÍNH object đầu vào (`is`) cho weight≤0/window=0/map rỗng/toàn 0; không
  rò qua biên video (hai video kề gid); decay shape; gid thiếu span không được boost;
  **integration trên SearchEngine THẬT** (corpus synthetic + fake encoder): off →
  hàm không được gọi + hai lần chạy off cho từng (gid, score) bằng nhau tuyệt đối,
  on → gọi đúng 1 lần với đúng (weight, window, decay) từ config.

## 7.3 · File sửa/tạo phiên này

| File | Loại | Lý do |
|---|---|---|
| `scripts/66_trake_diag.py` | TẠO | Nhiệm vụ B (§7.1) |
| `tests/test_trake_diag.py` | TẠO | 11 test fixture cho B |
| `src/cvp/search/fusion.py` | SỬA | thêm hàm `neighbor_consistency_boost` (§7.2) |
| `src/cvp/search/engine.py` | SỬA | cắm knob vào `_finalize` trước `neighbor_boost`; spans tính một lần |
| `src/cvp/config.py` | SỬA | `SearchCfg` +3 field consistency boost (default off) |
| `configs/settings.yaml` | SỬA | mục `search:` +3 dòng knob kèm comment |
| `tests/test_neighbor_consistency.py` | TẠO | 6 test công thức + integration engine |

## 7.4 · Cách dùng khi bench + dự đoán trung thực

- **66_trake_diag chạy TRƯỚC** (cần map-keyframes + gt + lab_full trên Colab/Drive):
  verdict H1 quyết định đầu tư TRAKE tiếp theo — nếu trần lưới ~0.2 như nghi ngờ đợt 1
  thì jitter (`temporal.submit_strategy=jitter`, đã có từ đợt 1) là đòn chính, không
  phải retrieval; headroom H6 cho biết jitter còn bao nhiêu điểm để nhặt.
- **neighbor_consistency_boost**: bật A/B bằng
  `CVP_SEARCH__NEIGHBOR_CONSISTENCY_BOOST=0.15` (giữ window/decay mặc định 2/0.5).
  Kỳ vọng: giúp các câu KIS/QA mà đáp án đúng là cụm keyframe liền kề bị gai
  cross-video đè ở top — dạng lỗi "top-1 sai video nhưng top-5 có video đúng" (bench#1
  KIS có 5 câu 0.8 = R@1 trượt). Rủi ro trung thực: (1) tin tức hay tái dùng b-roll —
  một cụm b-roll sai video cũng là "cao nguyên" và cũng được cộng; (2) moment đúng chỉ
  được MỘT keyframe đại diện (lưới thưa) thì boost không giúp mà còn nâng cụm khác;
  (3) tương tác với weights tuned — bench diff per câu (scripts/65) trước khi kết luận.
  Vì thế mặc định off, chỉ bật sau A/B.

## 7.5 · Việc dở dang

- Nhiệm vụ D (row_budget) + E (68_profile) + F (tổng kiểm bàn giao) — các phiên sau.
- Các mục chờ chủ dự án giữ nguyên như Phiên 1/2 (artifacts để chạy 66/67 thật).

---

# PHIÊN 4 — 28/08/2026 · NHIỆM VỤ D (row_budget) + E (68_profile + tối ưu)

**Test: XANH — 897 test (896 pass + 1 skip; +25 test mới), không sửa/xóa test cũ.**

## 8.1 · Nhiệm vụ D — `src/cvp/pipeline/row_budget.py` + knob `search.row_strategy`

Triết lý jitter TRAKE (round-72) áp cho KIS/QA: metric lấy max theo dòng nên mỗi
dòng là một vé số; đuôi ranking từ rank ~40 thường loãng — tiêu budget đó CÓ CHỦ
ĐỊCH thay vì đổ nốt ranking.

- `variant_frames(frame, grid)`: vé số quanh một keyframe — trung điểm sớm/muộn
  TRƯỚC (lưới thưa ~5-7 s, cửa sổ GT ±5 s → frame GIỮA hai keyframe với tới cửa
  sổ mà không keyframe nào chạm), rồi keyframe kề trước/sau. Frame ngoài lưới →
  [] (không bịa offset).
- `diversify_tail(rows, grid_fn, head_keep, budget, variants_per_anchor)`:
  giữ NGUYÊN VĂN `head_keep` dòng đầu (R@1/5/20 bất khả xâm phạm) → anchor = dòng
  tốt nhất của TỪNG video phân biệt trong head → variant round-robin qua anchors
  (video nào cũng được phủ, theo thứ tự rank) → phần budget còn lại giữ đuôi gốc.
  Variants chiếm tối đa NỬA đuôi (`_VARIANT_TAIL_SHARE=0.5`, hằng số có docstring)
  — ranking giữa bảng vẫn còn tín hiệu thật, không bị vé số đè hết. Dedup với mọi
  dòng gốc. QA: dòng variant THỪA KẾ answer của anchor (cùng moment cùng đáp án).
  Không dựng nổi variant nào → trả CHÍNH list gốc + log INFO.
- `catalog_grid_fn(catalog)`: adapter duy nhất chạm engine — cache lưới per video
  từ `KeyframeCatalog.load()`; stub engine không catalog / load hỏng → grid None
  → biến thành no-op có tiếng, không bao giờ crash query.
- Wiring `run_query_file._maybe_diversify_rows`: CHỈ task kis + qa; **AVS loại
  trừ có chủ đích** (MMR của AVS đa dạng hóa CHÉO video — nhét variant cùng video
  vào là phá đúng cái nó vừa làm); TRAKE có jitter riêng. `legacy` (mặc định) =
  không đụng một byte.
- Config: `row_strategy: legacy|diversify_tail` + `row_strategy_head=30` +
  `row_strategy_variants=4 (1..8)`; settings.yaml có comment đầy đủ.
- Test (`tests/test_row_budget.py`, 13 test): variant order/mép lưới/ngoài lưới;
  head nguyên văn + round-robin đúng thứ tự + đuôi gốc theo sau + trần nửa-đuôi +
  budget; dedup variant trùng dòng gốc; QA payload thừa kế; no-grid/list ngắn trả
  CHÍNH object; catalog_grid_fn cache + sống sót load hỏng; end-to-end
  `run_query_file` KIS/QA (CSV đúng từng dòng, `top1_times` giữ nguyên vì dòng 1
  bất biến), off = bit-identical, AVS không bao giờ bị diversify.

## 8.2 · Nhiệm vụ E — `scripts/68_profile_engine.py` + 4 tối ưu bất biến kết quả

Micro-bench offline fixture synthetic (không cần artifacts; `--quick` cho smoke):
tokenize / BM25 persisted / minmax / weighted_sum / aggregate variants / hai
neighbor boost / DANTE DP, kích thước mô phỏng Batch-1 (500 ứng viên/field,
fused 5000 gid, 30 video × 200 frame × 5 event TRAKE).

**Số đo máy dev (median 9 lượt), TRƯỚC → SAU tối ưu:**

| op | trước | sau | ghi chú |
|---|---|---|---|
| aggregate_queries (max) 3×2000 gid | 20.23 ms | **1.05 ms (~19×)** | một lượt running-max thay list-per-key + np.max — chạy MỖI lane MỖI query |
| fusion_minmax 5000 gid | 0.74 ms | **0.46 ms** | vectorize (v−lo)/(hi−lo), cùng phép IEEE |
| fusion_weighted_sum dense 5000+4×500 | 2.04 ms | **1.76 ms** | hưởng từ minmax |
| bm25_scores_for 500 key × 8 token | 0.85 ms | 0.81 ms | hoist (k1+1.0) — vi mô, bit-safe |
| video_ids 5000 gid @177k hàng | 1.48 ms | **0.90 ms** | ndarray cache thay pandas .iloc (đo pattern riêng) |
| neighbor_boost / consistency / dante_dp | 10.8 / 7.2 / 83.4 ms | không đổi | xem "cân nhắc rồi bỏ" dưới |
| tokenize_vi 2000 câu | 38.6 ms | không đổi | đường build-time, xem dưới |

4 tối ưu (mỗi cái có test so với BẢN SAO ĐÓNG BĂNG thuật toán cũ trên fixture
seed — bằng TUYỆT ĐỐI từng bit float và từng thứ tự key, `tests/test_perf_identity.py`):

1. `fusion.minmax` — vectorize numpy, cùng biểu thức (v−lo)/(hi−lo) IEEE-754.
2. `fusion.aggregate_queries` nhánh "max" (mặc định `multi_query_agg: max`) —
   running-max một lượt; thứ tự key first-encounter + giá trị `float(np.max)` y hệt.
   Nhánh "mean" GIỮ NGUYÊN code cũ (running mean sẽ đổi bit — không đáng).
3. `TextIndexField.scores_for` — hoist `(k1+1.0)` khỏi vòng trong per-(key,doc,token):
   biểu thức con thuần, cây phép tính còn lại y nguyên → bit-identical.
4. `KeyframeCatalog.video_ids` — cache cột video_id thành ndarray (invalidate trong
   `build()` cùng chỗ `_video_index`), gather fancy-index; giữ nguyên semantics
   iloc (index âm wrap, quá biên IndexError — test pin cả hai).

**Cân nhắc rồi BỎ (ghi trung thực)**: (a) DANTE DP 83 ms/30 video — vòng deque
monotonic tuần tự theo bản chất, vectorize được thì phải đổi cấu trúc thuật toán
→ rủi ro sai lệch > lợi (TRAKE chỉ ~7 câu/đêm, tổng <1 s); (b) `tokenize_vi`
char-cache — 38 ms/2000 câu chỉ đau ở BUILD trên Colab (177k doc ≈ vài giây),
còn per-query không đáng kể, trong khi cache per-char qua NFD từng ký tự có rủi
ro lệch Unicode ở chuỗi tổ hợp lạ — không đáng đánh đổi bất biến; (c) hoist
`b/avgdl` trong BM25 — `(b·dl)/avgdl ≠ (b/avgdl)·dl` từng bit → từ chối.
Kết luận cho đêm thi: tổng chi phí thuần-Python một query KIS/QA sau tối ưu
~25 ms — API (Gemini enhance/VQA/rerank) mới là nơi thời gian nằm; script 68 tồn
tại để khẳng định điều đó bằng số trên chính máy chạy trận (`--out profile.md`).

## 8.3 · File sửa/tạo phiên này

| File | Loại | Lý do |
|---|---|---|
| `src/cvp/pipeline/row_budget.py` | TẠO | Nhiệm vụ D (§8.1) |
| `src/cvp/pipeline/run_queries.py` | SỬA | `_maybe_diversify_rows` + wiring KIS/QA (AVS loại trừ); import MAX_SUBMISSION_ROWS |
| `src/cvp/config.py` | SỬA | `SearchCfg` +3 field row_strategy (default legacy) |
| `configs/settings.yaml` | SỬA | mục `search:` +3 dòng row_strategy kèm comment |
| `tests/test_row_budget.py` | TẠO | 13 test D |
| `scripts/68_profile_engine.py` | TẠO | Nhiệm vụ E: micro-bench (§8.2) |
| `src/cvp/search/fusion.py` | SỬA | tối ưu 1+2 (minmax, aggregate max) |
| `src/cvp/index/text_store.py` | SỬA | tối ưu 3 (hoist k1+1) |
| `src/cvp/data/catalog.py` | SỬA | tối ưu 4 (video_ids ndarray cache + reset trong build) |
| `tests/test_perf_identity.py` | TẠO | 12 test bất biến + smoke CLI 68 |

## 8.4 · Cách bật khi bench + dự đoán trung thực

```python
os.environ["CVP_SEARCH__ROW_STRATEGY"] = "diversify_tail"   # A/B với legacy
# (giữ head/variants mặc định 30/4; các tối ưu E LUÔN bật — chúng bất biến kết quả)
```

- **row_strategy**: chỉ ảnh hưởng R@50/R@100 (head 30 dòng đầu bất biến → R@1/5/20
  không thể xấu đi). Kỳ vọng: cứu các câu mà moment đúng nằm NGOÀI top-30 hoặc
  frame trúng nằm giữa hai keyframe; mất mát tiềm năng: câu mà dòng đúng vốn nằm
  rank 31-100 của ranking gốc và bị variant đẩy quá 100 (trần nửa-đuôi giới hạn
  đúng rủi ro này ở ≤35 dòng cuối). Với GT offline cửa sổ ±125, trung điểm lưới
  hầu như luôn nằm trong cửa sổ nếu keyframe kề đã trúng — lợi thật sự nằm ở
  cửa sổ CHẤM CHẶT hơn của BTC (chưa xác minh độ rộng thật).
- **Tối ưu E**: không có rủi ro điểm (bit-identical, test pin); rủi ro duy nhất
  là hồi quy hiệu năng ở shape lạ — 68 chạy được mọi nơi để kiểm chứng lại.

## 8.5 · Việc dở dang

- Nhiệm vụ F (tổng kiểm đối kháng + TỔNG KẾT BÀN GIAO) — phiên cuối.
- Các mục chờ chủ dự án giữ nguyên (artifacts cho 66/67; nguồn gốc reference 19.8;
  luật chuẩn hóa answer BTC).

---

# PHIÊN 5 — 28/08/2026 · NHIỆM VỤ F: TỔNG KIỂM ĐỐI KHÁNG + BÀN GIAO

## 9.1 · Kết quả tự review mọi diff đợt 2 (con mắt phản biện)

Đã đọc lại TOÀN BỘ diff (8 file src sửa + 5 file src/scripts mới + 6 file test
mới) đối chiếu luật an toàn + bài học "fail to tiếng". **Tìm thấy và SỬA 3 lỗi
lớp hành-vi-âm-thầm** (cùng lớp với 4 bản vá audit đợt 1), mỗi bản sửa kèm test pin:

1. `scripts/66_trake_diag.py`: `--run` gõ nhầm đường dẫn → `load_run` trả rỗng
   ÂM THẦM → mọi câu bị báo "CSV rỗng", verdict H2/H6 sai lệch không dấu vết
   (67 có guard này, 66 thiếu — bất nhất giữa hai script chị em). → SystemExit
   to tiếng + `test_cli_fails_loud_on_missing_run_dir`.
2. `scripts/67_qa_diag.py`: `--queries` gõ nhầm → phần echo (description,
   question) biến mất không dấu vết (mỗi stem chỉ check `qf.exists()`).
   → SystemExit + `test_cli_missing_queries_dir_fails_loud`.
3. `run_queries._maybe_diversify_rows`: knob `diversify_tail` bật trên engine
   KHÔNG có catalog → không dựng nổi variant nào mà chỉ log INFO — knob bật
   nhưng vô tác dụng lặng lẽ. → `log.warning` nêu thẳng "ranking giữ nguyên
   như legacy" + `test_run_query_file_no_catalog_warns_loud_and_keeps_rows`.

**Soát nhưng KHÔNG sửa (ghi nhận trung thực):**

- `fusion.aggregate_queries` nhánh max mới: nếu input chứa NaN, bản cũ
  (`np.max`) lan truyền NaN, bản mới (`v > cur`) giữ giá trị gặp trước — khác
  nhau CHỈ trên input rác (cosine/BM25 không sinh NaN; garbage-in cả hai đằng
  nào cũng hỏng). Không đổi code để giữ tốc độ; test identity dùng giá trị hữu
  hạn đúng phân bố thật.
- `neighbor_consistency_boost` với điểm ÂM lẫn dương: hàng xóm âm bị guard
  `left > 0 or right > 0` lọc một phía nhưng vẫn cộng `(left+right)/2` khi phía
  kia dương (giảm support — semantics hợp lý); fused score thực tế ≥ 0 nên
  nhánh này không chạm được trong production.
- Toàn bộ nhánh knob-off đã có test bit-identical từ các phiên trước (thứ tự
  dòng CSV, top1_times, số call API, object identity của score map) — không
  phát hiện thêm đường nào knob-off còn chạm hành vi cũ.
- Luật an toàn: không đụng `.ipynb`, không sửa/xóa test cũ (chỉ THÊM test vào
  file test MỚI của đợt này), không network trong test, mọi hành vi mới sau
  knob mặc định off, không git remote/push.

## 9.2 · TỔNG KẾT BÀN GIAO ĐỢT 2

### Bảng MỌI file thay đổi (so với nền round-72 `9f4e55c`)

| File | Loại | Một dòng |
|---|---|---|
| `src/cvp/search/answer_norm.py` | MỚI | Chuẩn hóa answer 3 tier (số↔chữ VN, đơn vị, ngoặc/preamble, 0 đầu, thập phân/nghìn) + `majority_vote` (nhánh legacy tái tạo byte-identical) — dùng chung knob vote + script 67 + test. |
| `src/cvp/pipeline/row_budget.py` | MỚI | `diversify_tail`: đầu ranking nguyên văn, đuôi = variant frame lân cận (trung điểm lưới trước) round-robin theo video top; adapter `catalog_grid_fn` chịu được stub. |
| `scripts/66_trake_diag.py` | MỚI | Định lượng H1 (trần lưới, DP strictly-increasing) / H2 (recall video trong CSV) / H6 (max-over-rows + headroom) + verdict tự động; offline, fail-loud mọi đường dẫn. |
| `scripts/67_qa_diag.py` | MỚI | Phân loại từng câu QA fail: MOMENT_MISS (NO_VIDEO/NEAR+khoảng cách) / ANSWER_MISS (INHERITED = chữ ký A1, NO_ANSWER) / FORMAT_MISS (tier 1-3) / OK; kèm %fallback, đồng thuận, trần 100 ký tự, echo parse. |
| `scripts/68_profile_engine.py` | MỚI | Micro-bench 8 hot path thuần Python trên fixture synthetic (kích thước Batch-1), bảng md/JSON, `--quick` cho smoke. |
| `src/cvp/search/vqa.py` | SỬA | Tách `answer_group_votes` (phiếu RAW public); vote theo lớp tương đương khi `vqa.answer_canonicalize` bật; nhánh off giữ nguyên văn logic Counter cũ. |
| `src/cvp/pipeline/run_queries.py` | SỬA | `compute_qa_answers_with_stats` (đường phiếu gộp + neighbor strips + budget có răng + warning cạn budget), `plan_consistency_rerank` + wiring re-record times, `_maybe_diversify_rows` (KIS/QA, AVS loại trừ, warning không catalog). |
| `src/cvp/search/fusion.py` | SỬA | Thêm `neighbor_consistency_boost` (cao nguyên thắng gai, s/max, decay chuẩn hóa, không vượt biên video); tối ưu bit-identical `minmax` (vectorize) + `aggregate_queries` nhánh max (1 lượt, ~19×). |
| `src/cvp/search/engine.py` | SỬA | Cắm consistency boost vào `_finalize` TRƯỚC neighbor_boost (đọc bằng chứng thô); spans tính một lần dùng chung. |
| `src/cvp/config.py` | SỬA | +9 field knob (VqaCfg ×3, SearchCfg ×6) — TẤT CẢ mặc định off/legacy. |
| `configs/settings.yaml` | SỬA | Mirror 9 knob kèm comment chi phí/ý nghĩa/giá trị bench gợi ý. |
| `src/cvp/index/text_store.py` | SỬA | Hoist `(k1+1.0)` khỏi vòng trong BM25 — bit-identical (biểu thức con thuần). |
| `src/cvp/data/catalog.py` | SỬA | `video_ids` gather qua ndarray cache (invalidate trong `build()`), giữ nguyên semantics iloc (âm wrap, quá biên IndexError). |
| `tests/test_qa_overhaul.py` | MỚI | 24 test: answer_norm + 3 knob QA (off bit-identical / on đúng ca từng lớp lỗi). |
| `tests/test_qa_diag.py` | MỚI | 15 test: mọi verdict/cờ 67 + CLI end-to-end + fail-loud + khóa hằng fallback. |
| `tests/test_trake_diag.py` | MỚI | 12 test: DP trần lưới (khe lưới/chung cửa sổ/đảo thứ tự/infeasible), diagnose end-to-end, verdicts, CLI + fail-loud. |
| `tests/test_neighbor_consistency.py` | MỚI | 6 test: công thức khớp số tay, off trả CHÍNH object, biên video, decay shape, integration engine thật. |
| `tests/test_row_budget.py` | MỚI | 14 test: variant order, head nguyên văn, dedup, trần nửa-đuôi, QA payload, catalog adapter, e2e KIS/QA/AVS-loại-trừ, warning không catalog. |
| `tests/test_perf_identity.py` | MỚI | 12 test: 4 tối ưu so với bản sao đóng băng thuật toán cũ — bằng tuyệt đối từng bit + smoke CLI 68. |
| `report2.md` | MỚI | Nhật ký cộng dồn 5 phiên (file này). |

### Bảng knob mới (9 knob, TẤT CẢ mặc định = hành vi cũ)

| Knob | Mặc định | Đề xuất bench#2 | Lớp lỗi nhắm (mã report §2) |
|---|---|---|---|
| `vqa.answer_canonicalize` | false | **true ngay** (0 call thêm) | F1/F2/F3 định dạng + A5 vote vỡ |
| `vqa.answer_neighbor_frames` | 0 | 1, KÈM `max_calls_per_query=10` (chi phí ×2 call) | A2 model sai / A3 strip hụt frame bằng chứng |
| `vqa.consistency_rerank` | false | true SAU khi 67 xác nhận lớp lỗi | A1 thừa kế + A2 khi top bất nhất |
| `search.neighbor_consistency_boost` | 0.0 | 0.15 (A/B qua 65_bench_diff) | M1 gai cross-video đè R@1 KIS/QA |
| `search.neighbor_consistency_window` | 2 | giữ 2 | (tham số hình dạng của knob trên) |
| `search.neighbor_consistency_decay` | 0.5 | giữ 0.5 | (nt) |
| `search.row_strategy` | legacy | `diversify_tail` (A/B; chỉ chạm R@50/100) | M2 lưới thưa — frame giữa hai keyframe |
| `search.row_strategy_head` | 30 | giữ 30 | (đầu bảng bất khả xâm phạm) |
| `search.row_strategy_variants` | 4 | giữ 4 | (số vé số mỗi video top) |

Lệnh env mẫu đã ghi ở §6.4/§7.4/§8.4. Chẩn đoán chạy TRƯỚC khi chọn knob:
`python scripts/67_qa_diag.py --run <lab_full> --gt <gt.json> --queries <pack>` và
`python scripts/66_trake_diag.py --gt <gt.json> --map-dir <map-keyframes> --run <lab_full>`.

### Kiểm kê test (đếm bằng `pytest --collect-only`, chuẩn từng con)

| File test mới | Số test |
|---|---|
| tests/test_qa_overhaul.py | 24 |
| tests/test_qa_diag.py | 15 |
| tests/test_trake_diag.py | 12 |
| tests/test_neighbor_consistency.py | 6 |
| tests/test_row_budget.py | 14 |
| tests/test_perf_identity.py | 12 |
| **Tổng test MỚI đợt 2** | **83** |

Suite: **817 (nền round-72) → 900** (899 pass + 1 skip có sẵn), 817 + 83 = 900
khớp tuyệt đối — không test cũ nào bị sửa/xóa. Đối soát số khai từng phiên:
P2 +38 (24+14 tại thời điểm đó) · P3 +17 (11+6) · P4 +25 (13+12) · P5 +3
(pin cho 3 bản sửa F: 12−11, 15−14, 14−13) → 38+17+25+3 = 83 ✓.

### Việc CHƯA xong (trung thực)

1. **66/67 chưa từng chạy trên dữ liệu thật** — repo local không có artifacts;
   mọi verdict/классификация đang chờ `lab_full/` + `gt-thunghiem.json` +
   `map-keyframes/` (Drive/Colab). Đây là việc PHẢI làm trước khi bật knob nào.
2. **Chưa bench A/B knob nào** — 6 knob hành vi (3 QA + boost + row_strategy)
   cần lượt bench#2 với `65_bench_diff` đọc delta per câu.
3. **Answer-variant rows chưa làm** (thiết kế trong §3/§8: nộp song format
   số↔chữ trên cùng frame trúng — writer đã chủ đích cho phép): chờ 67 xác nhận
   FORMAT_MISS có thật rồi mới đáng ~20 dòng code + test.
4. Câu hỏi chủ dự án còn treo: reference 19.8 của đội mình hay đội khác (độ
   tin GT answer); BTC công bố luật chuẩn hóa answer không.

### 3 đề xuất GIÁ TRỊ NHẤT cho chặng tiếp theo

1. **Chạy 67 + 66 trên artifacts thật TRƯỚC MỌI QUYẾT ĐỊNH** (30 phút Colab):
   bảng verdict sẽ chỉ đích danh knob nào đáng bật và TRAKE nên đầu tư jitter
   hay retrieval — mọi thứ đợt 2 xây đều để phục vụ quyết định-bằng-số này.
2. **Bench#2 bật `answer_canonicalize` + giữ nguyên phần còn lại**, so bằng
   65_bench_diff; nếu 67 báo FORMAT_MISS ≥1 câu → làm ngay answer-variant rows
   (mục 3 việc chưa xong — trần điểm cao nhất trên mỗi dòng code bỏ ra).
3. **Một lượt bench A/B riêng cho cặp `neighbor_consistency_boost=0.15` +
   `row_strategy=diversify_tail`** (cùng họ "điểm từ hàng lân cận", không đụng
   API): nếu H1 của 66 xác nhận trần lưới thấp thì cặp này + jitter TRAKE là
   đường điểm rẻ nhất còn lại trước đêm thi.

**Trạng thái cuối đợt 2: suite XANH 900 test · 9 knob mới mặc định off ·
5 file src/scripts mới + 8 file sửa + 6 file test mới · report2.md cộng dồn 5 phiên.**
