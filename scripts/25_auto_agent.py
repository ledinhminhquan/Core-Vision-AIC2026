"""Automatic-track agent: query pack → search → validated CSVs → Codabench zip.

Optionally pushes each query's top-1 to a DRES endpoint (finals) — gated by
``submission.auto_submit`` + ``submission.dres_base_url`` in settings and the
``--submit`` flag (credentials via env DRES_USER / DRES_PASSWORD).

Example:
    python scripts/25_auto_agent.py --query-dir ./queries/auto --out-dir ./artifacts/submissions/auto
"""

import argparse
from pathlib import Path

from _bootstrap import init

from cvp.pipeline.auto_agent import run_auto


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--settings", default=None)
    ap.add_argument("--query-dir", required=True)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--submit", action="store_true",
                    help="push top-1 per query to DRES (requires submission.auto_submit: true)")
    args = ap.parse_args()

    settings = init(args.settings)
    out_dir = Path(args.out_dir) if args.out_dir else settings.paths.art("submissions", "auto")
    report = run_auto(Path(args.query_dir), out_dir, settings, submit=args.submit)

    total = len(report.written) + len(report.failed)
    print(f"Answered {len(report.written)}/{total} query files → {out_dir}")
    for stem, why in sorted(report.failed.items()):
        print(f"  ✗ DROPPED {stem}: {why}   ← query này sẽ 0 điểm nếu không xử lý!")
    for issue in report.issues:
        print(f"  {issue}")
    if report.zip_path:
        print(f"Packaged: {report.zip_path}")
    else:
        print("NOT packaged — fix the errors above and re-run.")
    for stem, res in report.submitted:
        print(f"  DRES {stem}: {'ACCEPTED' if res.ok else 'REJECTED'}"
              f"{' verdict=' + res.verdict if res.verdict else ''} ({res.status}) {res.message}")


if __name__ == "__main__":
    main()
