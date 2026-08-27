"""Chẩn đoán QA offline (Nhiệm vụ A đợt 2): mỗi câu QA fail rơi vào lớp nào?

Đọc CSV của một lượt chạy (``lab_full/`` hoặc .zip) + gt.json (dựng bởi
scripts/62) và phân loại TỪNG câu QA theo cây quyết định:

    NO_SUBMISSION  không có CSV / CSV rỗng
    MOMENT_MISS    không dòng nào trúng (video GT vắng bóng, hoặc có video
                   nhưng mọi frame hụt cửa sổ — in khoảng cách "suýt trúng")
    OK             ≥1 dòng trúng cửa sổ VÀ answer khớp chuẩn BTC (tier 0)
    FORMAT_MISS    answer ĐÚNG NỘI DUNG nhưng sai định dạng — khớp sau khi
                   canonicalize (tier 1: ngoặc/preamble; tier 2: số↔chữ VN,
                   đơn vị, dấu giữa chuỗi; tier 3: CHỈ khi bỏ dấu tiếng Việt
                   mới khớp — bản nộp vẫn 0 điểm, cờ riêng)
    ANSWER_MISS    moment đúng, answer sai nội dung; cờ phụ:
                   INHERITED (answer thừa kế từ nhóm top khác video — điểm
                   chết A1 của report2), NO_ANSWER (toàn "không rõ")

Kèm diagnostics per câu: % dòng fallback, đồng thuận head, answer chạm trần
100 ký tự, answer còn trang trí (preamble/ngoặc/markdown), và (với
``--queries``) echo (description, question) sau bước split để soát lỗi parse.

Offline thuần: không engine, không API — chạy được trên máy trần lẫn Colab.

    python scripts/67_qa_diag.py --run artifacts/lab/lab_full \
        --gt queries/gt-thunghiem.json --out qa_diag.md --json-out qa_diag.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

try:  # chạy từ scripts/ — tests import qua importlib thì bỏ qua
    from _bootstrap import init  # noqa: F401 — chỉ cần side-effect sys.path
except ImportError:  # pragma: no cover
    pass

from cvp.constants import MAX_QA_ANSWER_CHARS
from cvp.eval.official import (
    _entry_ranges,
    load_ground_truth,
    normalize_answer,
)
from cvp.pipeline.attempts import load_run
from cvp.search.answer_norm import canonical_key, looks_decorated

# Bản sao của cvp.pipeline.run_queries.QA_FALLBACK_ANSWER — import run_queries
# kéo cả engine stack (numpy/faiss) vào một script cố tình offline-thuần.
# tests/test_qa_diag.py khóa hai hằng số này bằng nhau.
QA_FALLBACK = "không rõ"

TIER_LABEL = {
    1: "tier1: ngoặc/preamble/markdown",
    2: "tier2: số↔chữ VN, đơn vị, dấu câu giữa chuỗi",
    3: "tier3: CHỈ khớp khi bỏ dấu tiếng Việt (bản nộp vẫn 0đ)",
}


def _row_answer(row: Sequence[str]) -> str:
    """Answer của một dòng QA — nối mọi cell sau frame (dấu phẩy không quote)."""
    return ",".join(str(c) for c in row[2:]) if len(row) > 2 else ""


def _frame_of(row: Sequence[str]) -> int | None:
    try:
        return int(str(row[1]).strip())
    except (TypeError, ValueError, IndexError):
        return None


def _window_distance(frame: int, windows: Sequence[tuple[int, int]]) -> int:
    """Số frame hụt cửa sổ gần nhất (0 = trúng)."""
    best = None
    for s, e in windows:
        d = s - frame if frame < s else (frame - e if frame > e else 0)
        best = d if best is None else min(best, d)
    return best if best is not None else 10 ** 9


def _consensus_head(rows: Sequence[Sequence[str]], head: int) -> float:
    """Tỷ lệ đa số của answer (bỏ fallback) trong ``head`` dòng đầu, [0,1]."""
    fb = normalize_answer(QA_FALLBACK)
    answers = [normalize_answer(_row_answer(r)) for r in rows[:head] if len(r) > 2]
    answers = [a for a in answers if a]
    if not answers:
        return 0.0
    real = [a for a in answers if a != fb]
    if not real:
        return 0.0
    return Counter(real).most_common(1)[0][1] / len(answers)


def classify_qa(stem: str, rows: Sequence[Sequence[str]] | None,
                entry: Mapping[str, Any], head: int = 10) -> dict[str, Any]:
    """Phân loại một câu QA — thuần dữ liệu, không I/O (unit-test được)."""
    gt_video = str(entry.get("video_id", "")).strip()
    windows = _entry_ranges(entry)
    raw_answers = entry.get("answers") or ([entry["answer"]] if entry.get("answer") else [])
    raw_answers = [str(a) for a in raw_answers if str(a).strip()]
    diag: dict[str, Any] = {"stem": stem, "gt_video": gt_video,
                            "gt_answers": raw_answers, "flags": []}

    if not rows:
        diag.update(verdict="NO_SUBMISSION",
                    note="không có CSV / CSV rỗng — câu này 0 điểm chắc chắn")
        return diag
    if not raw_answers or not windows:
        diag.update(verdict="GT_UNUSABLE",
                    note="GT thiếu answers hoặc cửa sổ frame — không phân loại được")
        return diag
    # Round-73: GT answer chuẩn hóa về rỗng ("?", "...") KHÔNG được vào tập so
    # khớp — official._entry_answers cũng lọc rỗng, nếu không một dòng answer
    # rỗng-sau-chuẩn-hóa sẽ được phán OK trong khi r_score_qa chấm 0.
    gt_norm = {n for n in (normalize_answer(a) for a in raw_answers) if n}
    if not gt_norm:
        diag.update(verdict="GT_UNUSABLE",
                    note="mọi GT answer chuẩn hóa về rỗng — official coi là "
                         "'gt missing answer', không phân loại được")
        return diag

    n_rows = len(rows)
    fb_norm = normalize_answer(QA_FALLBACK)
    all_answers = [_row_answer(r) for r in rows]
    diag["pct_fallback"] = round(
        sum(1 for a in all_answers if normalize_answer(a) == fb_norm) / n_rows, 3)
    diag["consensus_head"] = round(_consensus_head(rows, head), 3)
    diag["len_cap_rows"] = sum(1 for a in all_answers
                               if len(a) >= MAX_QA_ANSWER_CHARS - 2)
    diag["decorated_answers"] = sum(1 for a in set(all_answers)
                                    if a and looks_decorated(a))

    video_rows = [i for i, r in enumerate(rows) if str(r[0]).strip() == gt_video]
    if not video_rows:
        diag.update(verdict="MOMENT_MISS",
                    note=f"video GT {gt_video} không xuất hiện trong {n_rows} dòng "
                         f"(top-1 = {str(rows[0][0]).strip()})")
        diag["flags"].append("NO_VIDEO")
        return diag

    hits, near = [], []
    for i in video_rows:
        frame = _frame_of(rows[i])
        if frame is None:
            continue
        d = _window_distance(frame, windows)
        (hits if d == 0 else near).append((i, d))
    if not hits:
        min_d = min((d for _i, d in near), default=None)
        first_rank = video_rows[0] + 1
        diag.update(verdict="MOMENT_MISS",
                    note=f"đúng video (dòng đầu rank {first_rank}) nhưng mọi frame hụt "
                         f"cửa sổ — hụt gần nhất {min_d} frame")
        diag["flags"].append("NEAR_MISS")
        diag["min_frame_distance"] = min_d
        diag["first_video_rank"] = first_rank
        return diag

    hit_ranks = [i + 1 for i, _d in hits]
    diag["hit_ranks"] = hit_ranks[:10]
    strict = [i for i, _d in hits
              if (na := normalize_answer(_row_answer(rows[i]))) and na in gt_norm]
    if strict:
        diag.update(verdict="OK", best_rank=min(strict) + 1,
                    note=f"dòng trúng + answer khớp chuẩn BTC tại rank {min(strict) + 1}")
        return diag

    hit_answers: list[str] = []
    for i, _d in hits:
        a = _row_answer(rows[i])
        if a and a not in hit_answers:
            hit_answers.append(a)
    diag["hit_answers"] = hit_answers[:5]

    for tier in (1, 2, 3):
        gt_keys = {canonical_key(a, tier) for a in raw_answers}
        matched = [a for a in hit_answers if canonical_key(a, tier) in gt_keys]
        if matched:
            diag.update(verdict="FORMAT_MISS", format_tier=tier,
                        note=f"answer đúng nội dung, sai định dạng — {TIER_LABEL[tier]}; "
                             f"ví dụ: nộp {matched[0]!r} vs GT {raw_answers[0]!r}")
            if tier == 3:
                diag["flags"].append("ACCENT_ONLY")
            return diag

    diag["verdict"] = "ANSWER_MISS"
    notes = [f"moment đúng (rank {hit_ranks[0]}) nhưng answer sai nội dung"]
    top_ans_norm = normalize_answer(_row_answer(rows[0]))
    top_video = str(rows[0][0]).strip()
    if top_video != gt_video and any(
            normalize_answer(a) == top_ans_norm and top_ans_norm for a in hit_answers):
        diag["flags"].append("INHERITED")
        notes.append("answer trên dòng trúng == answer nhóm top-1 KHÁC video "
                     "(chữ ký thừa kế fallback theo budget — điểm chết A1)")
    if hit_answers and all(normalize_answer(a) == fb_norm for a in hit_answers):
        diag["flags"].append("NO_ANSWER")
        notes.append('mọi dòng trúng đều "không rõ" — VQA không trả lời được nhóm này')
    diag["note"] = "; ".join(notes)
    return diag


def _echo_parse(stem: str, query_dir: Path) -> dict[str, str] | None:
    """(description, question) sau bước split — soát điểm chết M4 bằng mắt."""
    qf = query_dir / f"{stem}.txt"
    if not qf.exists():
        return None
    # Import muộn: run_queries kéo engine stack — chỉ trả giá khi --queries.
    from cvp.pipeline.run_queries import load_query_lines, parse_query_lines

    lines = load_query_lines(qf)
    if not lines:
        return None
    desc, question = parse_query_lines("qa", lines)
    out = {"description": desc, "question": question or ""}
    if question is not None and desc.strip() == (question or "").strip():
        out["warning"] = "PASSTHROUGH: không tách được câu hỏi khỏi mô tả (M4)"
    return out


def render_markdown(diags: list[dict[str, Any]], run_label: str, gt_label: str) -> str:
    order = ["MOMENT_MISS", "ANSWER_MISS", "FORMAT_MISS", "NO_SUBMISSION",
             "GT_UNUSABLE", "OK"]
    counts = Counter(d["verdict"] for d in diags)
    lines = [f"# QA diag — {run_label} vs {gt_label}", ""]
    lines.append("| verdict | số câu |")
    lines.append("|---|---|")
    for v in order:
        if counts.get(v):
            lines.append(f"| {v} | {counts[v]} |")
    lines += ["", "| câu | verdict | cờ | ghi chú |", "|---|---|---|---|"]
    for d in sorted(diags, key=lambda d: (order.index(d["verdict"]), d["stem"])):
        flags = ",".join(d.get("flags") or []) or "—"
        lines.append(f"| {d['stem']} | {d['verdict']} | {flags} | {d.get('note', '')} |")
    lines.append("")
    for d in sorted(diags, key=lambda d: d["stem"]):
        lines.append(f"## {d['stem']} — {d['verdict']}")
        for key in ("gt_video", "gt_answers", "hit_ranks", "hit_answers", "best_rank",
                    "min_frame_distance", "first_video_rank", "format_tier",
                    "pct_fallback", "consensus_head", "len_cap_rows",
                    "decorated_answers", "parse"):
            if d.get(key) not in (None, [], ""):
                lines.append(f"- {key}: {d[key]}")
        if d.get("note"):
            lines.append(f"- note: {d['note']}")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="thư mục/zip CSV của lượt chạy")
    ap.add_argument("--gt", required=True, help="gt.json (scripts/62)")
    ap.add_argument("--queries", default=None,
                    help="pack đề .txt — echo (description, question) để soát parse")
    ap.add_argument("--out", default=None, help="file markdown báo cáo")
    ap.add_argument("--json-out", default=None, help="file JSON máy đọc")
    ap.add_argument("--head", type=int, default=10, help="số dòng đầu tính đồng thuận")
    args = ap.parse_args()

    if not Path(args.run).exists():
        raise SystemExit(f"--run không tồn tại: {args.run}")
    # Round-75 (tổng kiểm F): --queries gõ nhầm từng làm phần echo parse biến
    # mất KHÔNG dấu vết (mỗi stem chỉ check qf.exists()) — fail to tiếng.
    if args.queries and not Path(args.queries).is_dir():
        raise SystemExit(f"--queries không tồn tại: {args.queries}")
    run = load_run(args.run)
    gt = load_ground_truth(args.gt)
    qa_stems = sorted(s for s, e in gt.items() if e.get("task") == "qa")
    if not qa_stems:
        raise SystemExit(f"GT {args.gt} không có câu QA nào — sai file?")

    diags: list[dict[str, Any]] = []
    for stem in qa_stems:
        d = classify_qa(stem, run.get(stem), gt[stem], head=args.head)
        if args.queries:
            echo = _echo_parse(stem, Path(args.queries))
            if echo:
                d["parse"] = echo
                if "warning" in echo:
                    d.setdefault("flags", []).append("PARSE_PASSTHROUGH")
        diags.append(d)

    counts = Counter(d["verdict"] for d in diags)
    print(f"QA diag: {len(diags)} câu — " +
          ", ".join(f"{v}={n}" for v, n in sorted(counts.items())))
    for d in diags:
        flags = f" [{','.join(d['flags'])}]" if d.get("flags") else ""
        print(f"  {d['stem']:28s} {d['verdict']:13s}{flags} {d.get('note', '')}")

    md = render_markdown(diags, Path(args.run).name, Path(args.gt).name)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md, encoding="utf-8")
        print(f"\nBáo cáo markdown → {out}")
    if args.json_out:
        jout = Path(args.json_out)
        jout.parent.mkdir(parents=True, exist_ok=True)
        jout.write_text(json.dumps({"summary": dict(counts), "queries": diags},
                                   ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"JSON → {jout}")


if __name__ == "__main__":
    main()
