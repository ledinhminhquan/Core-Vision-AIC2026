"""Batch-run an organiser query pack → submission CSVs (Codabench-ready).

Examples:
    python scripts/20_run_queries.py --query-dir ./queries/p1 --out-dir ./artifacts/submissions/p1
    # one-stop prelim command: run + validate + zip + score offline vs dev GT
    python scripts/20_run_queries.py --query-dir ./queries/dev --zip --gt ./queries/dev/gt.json
"""

import argparse
from pathlib import Path

from _bootstrap import init

from cvp.pipeline.run_queries import run_query_folder


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--settings", default=None)
    ap.add_argument("--query-dir", required=True)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--no-vqa", action="store_true", help="skip VQA answer suggestions")
    ap.add_argument("--zip", action="store_true",
                    help="validate + package the CSVs into a Codabench zip after writing")
    ap.add_argument("--gt", default=None,
                    help="ground-truth JSON: score this run with the OFFICIAL formulas "
                         "right after writing (submissions are rationed 5/day, 20 total "
                         "— always score offline first)")
    args = ap.parse_args()

    settings = init(args.settings)
    out_dir = Path(args.out_dir) if args.out_dir else settings.paths.art("submissions")
    written = run_query_folder(settings, Path(args.query_dir), out_dir, with_vqa=not args.no_vqa)
    print(f"Wrote {len(written)} submission CSVs → {out_dir}")

    if args.zip:
        from cvp.submission.packager import has_errors, package_codabench

        zip_path = out_dir / f"{settings.submission.package_name}.zip"
        # Package ONLY this run's CSVs — stale files in out_dir must not ride along.
        issues = package_codabench(out_dir, zip_path,
                                   package_name=settings.submission.package_name,
                                   files=written)
        for issue in issues:
            print(f"  {issue}")
        if has_errors(issues):
            print("Zip refused — fix the errors above and re-run with --zip.")
        else:
            print(f"Packaged: {zip_path} (+ MANIFEST.json)")

    if args.gt:
        from cvp.eval.official import score_run

        report = score_run(out_dir, Path(args.gt))
        print(f"\nOffline official score vs {args.gt}:")
        for stem, qs in sorted(report.per_query.items()):
            print(f"  {stem:36s} final={qs.final:.3f}  "
                  + " ".join(f"R@{k}={v:.2f}" for k, v in sorted(qs.r_at.items())))
        for task, mean in sorted(report.by_task.items()):
            print(f"  [{task}] mean final = {mean:.4f}")
        print(f"  MEAN FINAL = {report.mean_final:.4f}  ({report.num_scored}/{report.num_gt} scored)")


if __name__ == "__main__":
    main()
