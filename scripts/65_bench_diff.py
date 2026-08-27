"""Bench delta reporter (Nhiệm vụ 3): bench_full.json vs bench_full-prev.json.

Đọc hai RunReport JSON (``cvp.eval.official.RunReport.to_dict`` — notebook Lab
ghi ``bench_full.json`` và xoay bản trước sang ``bench_full-prev.json``) và in
bảng markdown: headline mean_final, bảng theo task, bảng per-query nhóm theo
task với Δ tăng/giảm (regression nổi lên ĐẦU mỗi nhóm), kèm ghi chú câu
unscored / mới xuất hiện / biến mất. Câu vắng ở một bản tính 0.0 khi lấy Δ —
đúng quy ước mean_final của scorer (bỏ câu = 0 điểm).

    python scripts/65_bench_diff.py --dir artifacts/lab
    python scripts/65_bench_diff.py --cur bench_full.json --prev bench_full-prev.json \\
        --out artifacts/lab/BENCH_DIFF.md

Chỉ dùng stdlib — chạy được ở bất cứ đâu có 2 file JSON, không cần artifacts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:  # UTF-8 stdout trên Windows (side effect lúc import); tests bỏ qua được
    import _bootstrap  # noqa: F401
except ImportError:  # pragma: no cover — tests import via importlib
    pass

EPS = 1e-9


def _infer_task(stem: str) -> str:
    """trake→avs→qa→kis substring priority — same rule as the whole repo."""
    s = stem.lower()
    return next((t for t in ("trake", "avs", "qa", "kis") if t in s), "kis")


def _fmt(x: float | None) -> str:
    return "—" if x is None else f"{x:.3f}"


def _fmt_delta(d: float) -> str:
    return f"{d:+.3f}"


def diff_reports(cur: dict, prev: dict) -> dict:
    """Structured delta between two RunReport dicts.

    Returns:
        {"mean_final": {prev, cur, delta}, "scored": {prev, cur},
         "by_task": [{task, prev, cur, delta}...],
         "queries": [{stem, task, prev, cur, delta, note}...]  (sorted by
             task rồi Δ tăng dần — regression đứng đầu mỗi nhóm),
         "counts": {improved, worsened, unchanged, new, gone}}

    ``prev``/``cur`` per query là ``final`` của câu đó, None khi bản ấy không
    chấm được câu này (mất CSV / unscored — lý do nằm trong ``note``); Δ luôn
    tính với vắng = 0.0.
    """
    cur_q = cur.get("per_query") or {}
    prev_q = prev.get("per_query") or {}
    cur_uns = cur.get("unscored") or {}
    prev_uns = prev.get("unscored") or {}

    stems = sorted(set(cur_q) | set(prev_q) | set(cur_uns) | set(prev_uns))
    queries: list[dict] = []
    counts = {"improved": 0, "worsened": 0, "unchanged": 0, "new": 0, "gone": 0}
    for stem in stems:
        c, p = cur_q.get(stem), prev_q.get(stem)
        task = str((c or p or {}).get("task") or _infer_task(stem))
        cf = float(c["final"]) if c else None
        pf = float(p["final"]) if p else None
        delta = (cf or 0.0) - (pf or 0.0)
        notes: list[str] = []
        if c is None:
            notes.append(f"bản mới unscored: {cur_uns[stem]}" if stem in cur_uns
                         else "biến mất ở bản mới")
        if p is None:
            notes.append(f"bản cũ unscored: {prev_uns[stem]}" if stem in prev_uns
                         else "mới xuất hiện")
        if c is not None and p is not None:
            if delta > EPS:
                counts["improved"] += 1
            elif delta < -EPS:
                counts["worsened"] += 1
            else:
                counts["unchanged"] += 1
        elif c is not None:
            counts["new"] += 1
        else:
            counts["gone"] += 1
        queries.append({"stem": stem, "task": task, "prev": pf, "cur": cf,
                        "delta": round(delta, 6), "note": "; ".join(notes)})
    queries.sort(key=lambda q: (q["task"], q["delta"], q["stem"]))

    tasks = sorted(set(cur.get("by_task") or {}) | set(prev.get("by_task") or {}))
    by_task = []
    for t in tasks:
        tp = (prev.get("by_task") or {}).get(t)
        tc = (cur.get("by_task") or {}).get(t)
        by_task.append({"task": t, "prev": tp, "cur": tc,
                        "delta": round((tc or 0.0) - (tp or 0.0), 6)})

    mp = float(prev.get("mean_final") or 0.0)
    mc = float(cur.get("mean_final") or 0.0)
    return {
        "mean_final": {"prev": mp, "cur": mc, "delta": round(mc - mp, 6)},
        "scored": {
            "prev": f"{prev.get('num_scored', 0)}/{prev.get('num_gt', 0)}",
            "cur": f"{cur.get('num_scored', 0)}/{cur.get('num_gt', 0)}",
        },
        "by_task": by_task,
        "queries": queries,
        "counts": counts,
    }


def render_markdown(diff: dict, cur_name: str = "bench_full.json",
                    prev_name: str = "bench_full-prev.json") -> str:
    """Human-readable markdown for the diff structure of :func:`diff_reports`."""
    mf = diff["mean_final"]
    n = diff["counts"]
    lines = [
        f"# Bench diff — {cur_name} vs {prev_name}",
        "",
        f"**mean_final: {_fmt(mf['prev'])} → {_fmt(mf['cur'])} ({_fmt_delta(mf['delta'])})**"
        f" · scored {diff['scored']['prev']} → {diff['scored']['cur']}"
        f" · ↑{n['improved']} ↓{n['worsened']} ={n['unchanged']}"
        f" mới:{n['new']} mất:{n['gone']}",
        "",
        "| task | prev | cur | Δ |",
        "|---|---|---|---|",
    ]
    for row in diff["by_task"]:
        lines.append(f"| {row['task']} | {_fmt(row['prev'])} | {_fmt(row['cur'])} "
                     f"| {_fmt_delta(row['delta'])} |")
    for task in sorted({q["task"] for q in diff["queries"]}):
        rows = [q for q in diff["queries"] if q["task"] == task]
        lines += ["", f"## {task} ({len(rows)} câu — Δ xấu nhất trước)", "",
                  "| query | prev | cur | Δ | ghi chú |", "|---|---|---|---|---|"]
        for q in rows:
            lines.append(f"| {q['stem']} | {_fmt(q['prev'])} | {_fmt(q['cur'])} "
                         f"| {_fmt_delta(q['delta'])} | {q['note']} |")
    return "\n".join(lines) + "\n"


def _load_json(path: Path, label: str) -> dict:
    if not path.is_file():
        raise SystemExit(
            f"Thiếu file {label}: {path} — cần đủ 2 bản bench để diff "
            "(lần bench đầu tiên chưa có bench_full-prev.json là bình thường)."
        )
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except ValueError as e:
        raise SystemExit(f"File {label} hỏng ({path}): {e}") from e


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=None,
                    help="thư mục chứa bench_full.json + bench_full-prev.json (vd artifacts/lab)")
    ap.add_argument("--cur", default=None, help="đường dẫn bench mới (mặc định bench_full.json)")
    ap.add_argument("--prev", default=None,
                    help="đường dẫn bench cũ (mặc định bench_full-prev.json)")
    ap.add_argument("--out", default=None, help="ghi markdown vào file này (vẫn in ra stdout)")
    args = ap.parse_args()

    if args.dir:
        cur_path = Path(args.dir) / "bench_full.json"
        prev_path = Path(args.dir) / "bench_full-prev.json"
    else:
        cur_path = Path(args.cur or "bench_full.json")
        prev_path = Path(args.prev or "bench_full-prev.json")

    diff = diff_reports(_load_json(cur_path, "bench mới"), _load_json(prev_path, "bench cũ"))
    md = render_markdown(diff, cur_path.name, prev_path.name)
    print(md)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md, encoding="utf-8")
        print(f"→ đã ghi {out}")


if __name__ == "__main__":
    main()
