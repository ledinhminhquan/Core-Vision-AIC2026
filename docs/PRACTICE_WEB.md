# Web luyện tập 24/7 (HF Space) — hướng dẫn triển khai từ A đến Z

Mục tiêu: đội vào **https://huggingface.co/spaces/<user>/cvp-practice** luyện
tập BẤT KỲ LÚC NÀO, không cần phiên Colab. Cùng SPA + đăng nhập đội như web
trận đấu; khác biệt: engine chạy CPU (search ~2–5s), không Qwen/VLM rerank,
không VQA (hai thứ đó cần GPU/ảnh gốc — vẫn là việc của Colab đêm thi).

## Bước 0 — chuẩn bị (một lần)

1. Tài khoản Hugging Face nâng **PRO ($9/tháng)** — từ 7/2026 tạo Docker
   Space cần PRO; phần cứng CPU Basic (2 vCPU/16GB) vẫn miễn phí.
2. Tạo **HF token WRITE** (Settings → Access Tokens) — dùng cho bước upload.
3. Đã có GitHub PAT fine-grained (Contents: Read) như Colab đang dùng.

## Bước 1 — đóng gói dữ liệu (chạy MỘT lần trong phiên nb03 đang sống)

Sau Run all (env + artifacts local sẵn), chạy 2 cell mới:

```python
# 1a · thumbnails 320px webp (~25 phút, ghi thẳng vào Drive để tái dùng)
!python /content/Core-Vision_Perfect_V1/scripts/60_make_thumbs.py \
    --out /content/drive/MyDrive/AIC2025/artifacts/thumbs \
    --upload <hf-user>/aic26-thumbs --token hf_XXXX
```

```python
# 1b · artifacts cho engine CPU (catalog/index/embeddings siglip2/BM25/objects/checkpoint)
!python /content/Core-Vision_Perfect_V1/scripts/61_push_practice_artifacts.py \
    --repo <hf-user>/aic26-artifacts --token hf_XXXX
```

Cả hai repo dataset được tạo **PRIVATE tự động** — dữ liệu BTC không bao giờ
public. (Bonus tức thì: từ phiên Colab sau, ô staging tự thấy `thumbs/` và web
trận đấu cũng tải lưới ảnh nhanh gấp ~10 lần.)

## Bước 2 — tạo Space

1. huggingface.co → New Space → tên `cvp-practice` → SDK **Docker** →
   **Private** → CPU Basic (free).
2. Upload 3 file từ `deploy/hf-space/` của repo:
   - `Dockerfile`
   - `app.py`
   - `SPACE_README.md` → đổi tên thành `README.md` (giữ phần đầu `sdk: docker`).
3. Settings → **Variables and secrets**:

   | Loại | Tên | Giá trị |
   |---|---|---|
   | Secret | `GITHUB_TOKEN` | PAT GitHub (Contents: Read) |
   | Secret | `HF_TOKEN` | HF token (Read đủ) |
   | Secret | `TEAM_PASS` | mật khẩu đội tự chọn |
   | Secret | `GEMINI_API_KEY` | (tùy chọn) bật Gemini dịch/enhance |
   | Variable | `TEAM_USER` | `aic2026-222` |
   | Variable | `ARTIFACTS_REPO` | `<hf-user>/aic26-artifacts` |
   | Variable | `THUMBS_REPO` | `<hf-user>/aic26-thumbs` |

4. Space tự build (~10 phút) → khởi động (~3–6 phút tải dữ liệu + model) →
   màn hình đăng nhập đội hiện ra. Gửi link + mật khẩu vào nhóm kín.

## Vận hành

- **Ngủ/đánh thức:** phần cứng free ngủ sau **48h không ai vào**; ai mở link
  là tự dậy (chờ vài phút). Muốn không bao giờ ngủ: tạo monitor miễn phí
  (UptimeRobot / cron-job.org) ping URL Space mỗi 12 giờ — trang `/` mở nên
  ping không cần đăng nhập.
- **Cập nhật code:** Space → Settings → **Factory rebuild** (clone lại repo
  GitHub mới nhất). Cập nhật dữ liệu: chạy lại bước 1 rồi **Restart** Space.
- **Đổi mật khẩu đội:** sửa secret `TEAM_PASS` → Restart.
- **Xuất bài khi luyện:** nút Export vẫn ghi CSV vào đĩa Space (mất khi
  restart) — luyện tập là để luyện mắt và quy trình; bài nộp thật luôn làm
  trên phiên Colab trận đấu.

## Bảo mật — 4 lớp

1. Space **Private** (chỉ tài khoản HF của bạn thấy trang quản trị).
2. **Đăng nhập đội bắt buộc** trên chính web (cookie HttpOnly 24h) — ảnh,
   search, export đều 401 nếu chưa đăng nhập; link lộ ra ngoài cũng vô dụng.
3. Hai dataset dữ liệu **Private** — chỉ HF_TOKEN của bạn đọc được.
4. GitHub PAT chỉ Read, nằm trong build-secret — không lộ vào image layer.
