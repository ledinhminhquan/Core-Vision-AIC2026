"""Adaptive multi-attempt (Nhiệm vụ 2): "lượt sau giỏi hơn lượt trước".

Ba lệnh, hai lệnh đầu chạy OFFLINE thuần (không engine, không API):

  plan   Đọc lượt 1 (thư mục/zip CSV) + signal dumps (scripts/23, nếu có) →
         độ tự tin per query → attempt_plan.json: danh sách câu yếu + env
         nỗ lực cao cho đúng các task yếu (+ stage câu yếu nếu có --stage-dir).
  run    Chạy lại CHỈ các câu yếu theo plan với env nỗ lực cao (CẦN artifacts
         + engine — dùng trên Colab; offline hãy dừng ở plan/merge).
  merge  RRF-merge N lượt có trọng số (mở rộng scripts/63 — 63 giữ nguyên;
         merge 2 lượt trọng số bằng nhau tái tạo đúng output 63) → merged/
         + báo cáo câu nào đổi top-1.

Ví dụ:
    python scripts/64_adaptive_attempts.py plan --run artifacts/submissions/att1 \
        --signals-dir artifacts/signal_dumps/thunghiem --query-dir queries/pack \
        --out artifacts/attempts/plan.json --stage-dir artifacts/attempts/weak_queries
    python scripts/64_adaptive_attempts.py run --plan artifacts/attempts/plan.json \
        --out artifacts/submissions/att2
    python scripts/64_adaptive_attempts.py merge \
        --runs artifacts/submissions/att1 artifacts/submissions/att2 \
        --weights 1.0 1.0 --out artifacts/submissions/merged \
        --report artifacts/submissions/merged/DIFF.md

Lưu ý round-43 vẫn là luật: chỉ merge "anh em cùng noise" (cùng lineup, khác
nỗ lực/seed) — không merge hai đội hình khác đẳng cấp.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

try:  # script executed from scripts/ — falls through when imported by tests
    from _bootstrap import init
except ImportError:  # pragma: no cover — tests import via importlib
    init = None  # type: ignore[assignment]

from cvp.pipeline.attempts import (
    build_plan,
    diff_top1,
    load_run,
    render_diff_markdown,
    rrf_merge_runs,
    stage_weak_queries,
    write_merged,
)


def cmd_plan(args: argparse.Namespace) -> None:
    plan = build_plan(
        args.run,
        signals_dir=args.signals_dir,
        query_dir=args.query_dir,
        threshold=args.threshold,
        max_fraction=args.max_fraction,
        head=args.head,
    )
    if args.stage_dir:
        plan["stage_dir"] = str(args.stage_dir)
        if not args.query_dir:
            raise SystemExit("--stage-dir cần --query-dir (nguồn file .txt để copy)")
        staged = stage_weak_queries(args.query_dir, plan["weak"], args.stage_dir)
        print(f"Staged {len(staged)}/{len(plan['weak'])} câu yếu → {args.stage_dir}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")

    weak = set(plan["weak"])
    print(f"\nĐộ tự tin per query (ngưỡng yếu < {plan['threshold']}):")
    for stem, q in plan["queries"].items():
        mark = "  ← YẾU" if stem in weak else ""
        comps = ", ".join(f"{k}={v:.2f}" for k, v in q["components"].items())
        print(f"  {stem:28s} {q['task']:5s} conf={q['confidence']:.3f}  ({comps}){mark}")
    print(f"\n{len(weak)}/{len(plan['queries'])} câu vào lượt 2. Env nỗ lực cao:")
    for key, val in plan["env"].items():
        print(f"  {key}={val}")
    print("  (CVP_QUERY__EXPANSIONS đổi cache-key query processor → các câu yếu "
          "sẽ gọi Gemini enhance lại — chủ đích, nhưng tốn quota.)")
    print(f"\nPlan → {out}")
    print(f"Tiếp theo: python scripts/64_adaptive_attempts.py run --plan {out} --out <att2_dir>")


def cmd_run(args: argparse.Namespace) -> None:
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    weak = plan.get("weak") or []
    if not weak:
        print("Plan không có câu yếu — không cần lượt 2.")
        return
    query_dir = args.query_dir or plan.get("query_dir")
    if not query_dir:
        raise SystemExit("Không biết pack đề gốc: truyền --query-dir (hoặc plan phải có query_dir)")
    # Round-72 (Cursor-lab audit): --out trỏ vào chính thư mục lượt 1 sẽ ghi đè
    # mất bài gốc mà bước merge cần — từ chối thẳng.
    _run_dir = plan.get("run_dir")
    if _run_dir and Path(args.out).resolve() == Path(_run_dir).resolve():
        raise SystemExit(
            "--out đang trỏ vào chính thư mục lượt 1 (plan.run_dir) — sẽ ghi đè "
            "mất bài gốc mà bước merge cần. Dùng một thư mục MỚI cho lượt 2.")
    stage_dir = args.stage_dir or plan.get("stage_dir") or (Path(args.out) / "_weak_queries")
    staged = stage_weak_queries(query_dir, weak, stage_dir)
    if not staged:
        raise SystemExit(f"Không stage được câu yếu nào từ {query_dir} — sai pack?")

    # Env nỗ lực cao phải vào TRƯỚC load_settings (cvp.config đọc env lúc load).
    for key, val in (plan.get("env") or {}).items():
        os.environ[key] = str(val)
    if init is None:  # pragma: no cover — chạy từ scripts/ mới có _bootstrap
        raise SystemExit("Chạy lệnh này từ repo: python scripts/64_adaptive_attempts.py run …")
    settings = init(args.settings)
    from cvp.pipeline.auto_agent import run_auto  # heavy: engine stack

    report = run_auto(stage_dir, Path(args.out), settings, submit=False)
    total = len(report.written) + len(report.failed)
    print(f"Lượt 2: {len(report.written)}/{total} câu yếu có CSV → {args.out}")
    for stem, why in sorted(report.failed.items()):
        print(f"  ✗ vẫn hỏng {stem}: {why}")
    print(f"Tiếp theo: python scripts/64_adaptive_attempts.py merge "
          f"--runs {plan['run_dir']} {args.out} --out <merged_dir>")


def cmd_merge(args: argparse.Namespace) -> None:
    runs = [load_run(p) for p in args.runs]
    weights = args.weights if args.weights else None
    merged = rrf_merge_runs(runs, weights=weights, k=args.k)
    written = write_merged(merged, args.out)
    diffs = diff_top1(runs[0], merged)
    print(f"Merged {len(args.runs)} lượt (weights={weights or [1.0] * len(runs)}) "
          f"→ {len(written)} CSV tại {args.out}")
    print(f"{len(diffs)} câu đổi top-1 so với lượt gốc"
          + (":" if diffs else "."))
    for d in diffs:
        print(f"  {d['stem']}: {','.join(d['base_top1'])} → {','.join(d['merged_top1'])}")
    if args.report:
        rp = Path(args.report)
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(render_diff_markdown(diffs), encoding="utf-8")
        print(f"Báo cáo diff → {rp}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("plan", help="lượt 1 → độ tự tin per query → attempt_plan.json")
    p.add_argument("--run", required=True, help="thư mục/zip CSV của lượt 1")
    p.add_argument("--signals-dir", default=None, help="dump scripts/23 (thêm margin)")
    p.add_argument("--query-dir", default=None,
                   help="pack đề gốc — phát hiện câu KHÔNG có CSV (rớt lượt 1)")
    p.add_argument("--threshold", type=float, default=0.35, help="ngưỡng yếu")
    p.add_argument("--max-fraction", type=float, default=0.5,
                   help="trần tỷ lệ câu chạy lại (ngân sách lượt 2)")
    p.add_argument("--head", type=int, default=10, help="số dòng đầu xét concentration/consensus")
    p.add_argument("--out", default="attempt_plan.json")
    p.add_argument("--stage-dir", default=None, help="copy .txt câu yếu vào đây")
    p.set_defaults(func=cmd_plan)

    r = sub.add_parser("run", help="chạy lại CHỈ câu yếu với env nỗ lực cao (cần engine)")
    r.add_argument("--plan", required=True)
    r.add_argument("--query-dir", default=None, help="pack đề gốc (mặc định lấy từ plan)")
    r.add_argument("--stage-dir", default=None)
    r.add_argument("--out", required=True)
    r.add_argument("--settings", default=None)
    r.set_defaults(func=cmd_run)

    m = sub.add_parser("merge", help="RRF-merge N lượt có trọng số")
    m.add_argument("--runs", nargs="+", required=True, help="các thư mục/zip CSV, LƯỢT GỐC TRƯỚC")
    m.add_argument("--weights", nargs="*", type=float, default=None,
                   help="trọng số per lượt (mặc định 1.0 đều)")
    m.add_argument("--k", type=int, default=60)
    m.add_argument("--out", required=True)
    m.add_argument("--report", default=None, help="ghi bảng diff markdown vào đây")
    m.set_defaults(func=cmd_merge)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
