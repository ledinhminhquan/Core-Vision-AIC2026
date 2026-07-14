# report/ — Báo cáo giải pháp vòng sơ tuyển AIC 2026

Vòng sơ tuyển **bắt buộc nộp kèm báo cáo giải pháp** (FAQ BTC). Thư mục này là bộ khung
LaTeX hoàn chỉnh: `main.tex` đã có nội dung thật cho mọi mục, chỉ chờ **điền số liệu**
sau khi BTC phát hành dữ liệu 2026 (chậm nhất 25/07/2026) và ta chạy xong các script đo.

## Cấu trúc

| File | Nguồn gốc | Vai trò |
|---|---|---|
| `main.tex` | viết mới cho Perfect V1 | nội dung báo cáo (tiếng Việt) |
| `ai_conquer2026.cls` | copy từ template AIVN (`AIO_HCMC AI/Project Report/`) | class: font/babel tiếng Việt, màu, hộp code, biblatex |
| `tvietlistings.sty` | copy từ template AIVN | bảng `literate` để listings hiển thị đủ dấu tiếng Việt |
| `vipythonhighlight.sty` | copy từ template AIVN | highlight Python + inline `\pyth` |
| `references.bib` | copy từ template + **đã bổ sung** citations thật (SigLIP-2, PE-Core, BLIP-2, SuperGlobal, MMR, LoRA, WiSE-FT, DRES, fusion TOIS'23) | thư mục tham khảo |
| `figures/` | trống | đặt `logo.pdf` (logo đội, tùy chọn) và `pipeline.pdf` (Hình 1) |

**KHÔNG sửa `.cls`/`.sty`** — mọi tùy biến (logo, title, author) đều nằm trong `main.tex`.

## Cách build

Class dùng babel `vietnamese` + fontenc T1/T5 + `biblatex` với `backend=bibtex` →
**biên dịch bằng `pdflatex`** (KHÔNG dùng xelatex/lualatex — xem Lưu ý bên dưới):

```powershell
cd report
pdflatex main.tex
bibtex   main          # backend=bibtex nên chạy 'bibtex', KHÔNG phải 'biber'
pdflatex main.tex
pdflatex main.tex      # lần 3 chốt TOC + cross-ref
```

Hoặc một lệnh với latexmk:

```powershell
latexmk -pdf -bibtex main.tex
```

Trên Overleaf: upload cả 5 file + thư mục `figures/`, đặt compiler = **pdfLaTeX**,
TeX Live 2024 trở lên.

### Yêu cầu TeX

- **TeX Live 2023+ bản full** (hoặc MiKTeX auto-install). Các gói ngoài bộ chuẩn mà class
  cần: `biblatex` + `biblatex-ieee`, `tcolorbox`, `menukeys`, `dirtree`, `epigraph`,
  `svg`, `hyphsubst`, `listingsutf8`, `vntex` (cấp fontenc T5 + macro dấu tiếng Việt
  `\ohorn`, `\abreve`... mà `tvietlistings.sty` dùng), `fontawesome5` (tùy chọn — thiếu
  thì class tự fallback).
- Babel phải đủ mới để hiểu modifier `vietnamese.licr` (class truyền
  `\PassOptionsToPackage{vietnamese.licr}{babel}`) — TeX Live 2023+ đạt; TeX Live cũ hơn
  sẽ báo lỗi option lạ ngay dòng đầu.

### Lưu ý đã biết của template (đọc trước khi đổ lỗi cho main.tex)

1. **Chỉ pdfLaTeX.** `vipythonhighlight.sty` nạp cứng `inputenc` + fontenc **T5**;
   dưới XeLaTeX/LuaLaTeX sẽ cảnh báo/lỗi và font Việt vỡ. Nhánh `fontspec + Consolas`
   trong class chỉ dành cho khối `\cmdmono`, không đủ để cứu cả tài liệu.
2. **Logo:** class mặc định chèn `Figures/logo.pdf` vào title. `main.tex` đã override
   an toàn — nếu `figures/logo.pdf` không tồn tại thì bỏ qua, vẫn compile sạch.
   Muốn có logo đội: thả file PDF vào `figures/logo.pdf` là xong.
3. **`\includesvg`** (gói `svg`) cần Inkscape + `-shell-escape`. `main.tex` không dùng —
   xuất hình sang PDF rồi `\includegraphics` cho lành.
4. Đánh số mục theo template là **La Mã** (I., II., ...); đừng ngạc nhiên.

## Checklist điền sau khi có dữ liệu 2026 / sau các run

Tìm chuỗi `TODO` trong `main.tex` (11 chỗ). Cụ thể:

| Chỗ | Lấy số từ đâu |
|---|---|
| Tên đội + thành viên (`\author`) | đăng ký với BTC |
| Hình 1 — sơ đồ pipeline | vẽ từ `docs/PROJECT_CONTEXT.md` mục 2 → `figures/pipeline.pdf`, thay khối `\fbox` bằng `\includegraphics[width=\linewidth]{figures/pipeline.pdf}` |
| Bảng ablation A1–A10 (mục V) | `python scripts/26_run_ablations.py --query-dir queries/dev --gt queries/dev/gt.json` → bảng in ra + `artifacts/ablations/` (A5 lấy từ bảng của notebook 02; quét trọng số đầy đủ A3 từ `scripts/21_tune_weights.py`) |
| Kết quả train LoRA-LiT + WiSE-FT (mục IV) | notebook 02 (winner α + val R@5 vs baseline từ `scripts/eval_model.py`) |
| Bảng latency p50/p95/p99 (mục V) | `python scripts/50_bench_latency.py --n 200` trên laptop thi đấu, sau khi build artifacts thật |
| Bài học thực chiến + kết quả sơ tuyển (mục VI, VII, Tóm tắt) | sau các lượt nộp Codabench tháng 8/2026 |

Lưu ý: repo đã kèm sẵn `queries/example` VÀ **`queries/dev-2025-finals/`** (89 truy vấn
chung kết 2025 nguyên bản: 73 KIS / 9 QA / 7 TRAKE). Chỉ còn `gt.json` là phải tự soạn
cho những câu đội đã xác minh đáp án (format: docstring `cvp/eval/official.py`).

Quy tắc: **số nào chưa đo thì để nguyên TODO**, không ước lượng. Bảng ablation lấy đúng
nhãn cấu hình như script in ra để người chấm đối chiếu được với repo
(GitHub: https://github.com/ledinhminhquan/Core-Vision_Perfect_V1).

## Đồng bộ nội dung

`main.tex` được viết khớp `docs/PROJECT_CONTEXT.md` (source of truth) tại thời điểm
2026-07-14. Nếu PROJECT_CONTEXT đổi (spec hình thức tự động được BTC công bố, AVS được
xác nhận thi/không thi, đổi model lane...), cập nhật lại mục tương ứng trong báo cáo —
đặc biệt mục I (thể lệ) và mục III (thành phần).
