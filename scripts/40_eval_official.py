"""Score a submission folder with the official organiser formulas (offline Codabench).

Example:
    python scripts/40_eval_official.py --submission-dir ./artifacts/submissions/p1 \
        --gt ./queries/p1_gt.json --json-out ./artifacts/eval/p1_official.json

Prints a per-query table, per-task means and the overall run score (mean + sum).
Every CSV is also structurally validated against the submission contract
(``cvf.submission.writer`` rules). Exit codes: 0 = OK, 2 = at least one
submission CSV is malformed (the organiser server would reject it).
"""

import argparse
import sys
from pathlib import Path

import _bootstrap  # noqa: F401  — side-effect: puts src/ on sys.path

from cvf.eval.official import K_VALUES, RunReport, load_ground_truth, score_run, validate_csv
from cvf.utils.io import atomic_write_json
from cvf.utils.logging import setup_logging


def _print_report(report: RunReport) -> None:
    stems = list(report.per_query) + list(report.unscored) + report.gt_without_submission
    name_w = max([len(s) for s in stems] + [5]) + 2
    header = (
        f"{'query':<{name_w}}{'task':<7}"
        + "".join(f"{'R@' + str(k):>7}" for k in K_VALUES)
        + f"{'Final':>8}  best"
    )
    print(header)
    print("-" * len(header))
    for stem in sorted(report.per_query):
        qs = report.per_query[stem]
        cells = "".join(f"{qs.r_at[k]:>7.2f}" for k in K_VALUES)
        best = qs.best_rank if qs.best_rank is not None else "-"
        print(f"{stem:<{name_w}}{qs.task:<7}{cells}{qs.final:>8.3f}  {best}")
    for stem in sorted(report.unscored):
        print(f"{stem:<{name_w}}UNSCORED: {report.unscored[stem]}")
    for stem in report.gt_without_submission:
        print(f"{stem:<{name_w}}MISSING submission (counts as 0)")
    print("-" * len(header))
    for task, mean in sorted(report.by_task.items()):
        print(f"per-task mean [{task:<7}]: {mean:.4f}")
    print(
        f"run score: mean final {report.mean_final:.4f} | sum {report.sum_final:.4f} "
        f"over {report.num_gt} GT queries ({report.num_scored} scored)"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--submission-dir", required=True, help="folder of {query-stem}.csv files")
    ap.add_argument("--gt", required=True, help="ground-truth JSON (see cvf.eval.official docstring)")
    ap.add_argument("--json-out", default=None, help="write the full report as JSON here")
    args = ap.parse_args()
    setup_logging("INFO")

    sub_dir = Path(args.submission_dir)
    gt = load_ground_truth(Path(args.gt))
    report = score_run(sub_dir, gt)

    malformed: dict[str, list[str]] = {}
    for csv_path in sorted(sub_dir.glob("*.csv")):
        entry = gt.get(csv_path.stem) or {}
        task = entry.get("task")
        problems = validate_csv(csv_path, task if isinstance(task, str) else None)
        if problems:
            malformed[csv_path.name] = problems

    _print_report(report)

    if args.json_out:
        payload = report.to_dict()
        payload["malformed"] = malformed
        atomic_write_json(Path(args.json_out), payload)
        print(f"report written -> {args.json_out}")

    if malformed:
        print(f"MALFORMED submission CSVs ({len(malformed)}):", file=sys.stderr)
        for name, problems in sorted(malformed.items()):
            for p in problems:
                print(f"  {name}: {p}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
