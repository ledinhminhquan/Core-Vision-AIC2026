"""Console entry point: ``cvp <command>``.

Deliberately thin — heavy operational flows stay in ``scripts/`` (they carry
the resume/validate orchestration and are the documented competition path).
The CLI covers what you want from any terminal without remembering paths:

    cvp serve   [--host 0.0.0.0 --port 8000]   # HTTP/JSON retrieval service
    cvp search  "câu truy vấn" [--k 10]        # quick smoke search
    cvp eval    --submission-dir D --gt gt.json # official-formula offline scoring
    cvp version                                 # package + lane summary
"""

from __future__ import annotations

import argparse
import sys


def _cmd_serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ImportError:
        print("uvicorn missing — install the service extra:  pip install -e '.[service]'",
              file=sys.stderr)
        return 2
    uvicorn.run("cvp.service.app:get_app", factory=True,
                host=args.host, port=args.port, log_level="info")
    return 0


def _cmd_search(args: argparse.Namespace) -> int:
    from cvp.config import load_settings
    from cvp.search.engine import SearchEngine

    settings = load_settings(args.settings)
    engine = SearchEngine(settings)
    results = engine.search_text(args.query, display_k=args.k)
    if not results:
        print("(no results)")
        return 1
    for rank, r in enumerate(results, 1):
        ref = r.ref
        print(f"{rank:3d}. {ref.video_id},{ref.frame_idx}  "
              f"score={r.score:.4f}  t={ref.pts_time:.1f}s  n={ref.n}")
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    """Score a submission folder with the OFFICIAL formulas (no engine load)."""
    from pathlib import Path

    from cvp.eval.official import score_run

    report = score_run(Path(args.submission_dir), Path(args.gt))
    for stem in sorted(report.per_query):
        qs = report.per_query[stem]
        print(f"  {stem:<28s} {qs.task:<6s} final={qs.final:.3f} "
              f"best_rank={qs.best_rank if qs.best_rank is not None else '-'}")
    for stem, why in sorted(report.unscored.items()):
        print(f"  {stem:<28s} UNSCORED: {why}")
    for stem in report.gt_without_submission:
        print(f"  {stem:<28s} MISSING submission (counts 0)")
    print(f"mean final = {report.mean_final:.4f} over {report.num_gt} GT queries "
          f"({report.num_scored} scored)")
    return 0


def _cmd_version(_args: argparse.Namespace) -> int:
    from importlib.metadata import PackageNotFoundError, version

    try:
        v = version("cvp")
    except PackageNotFoundError:
        v = "(not installed — running from source)"
    print(f"Core-Vision Perfect V1 — package cvp {v}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="cvp", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("serve", help="run the HTTP/JSON retrieval service")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(fn=_cmd_serve)

    p = sub.add_parser("search", help="one-shot text search (smoke test)")
    p.add_argument("query")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--settings", default=None)
    p.set_defaults(fn=_cmd_search)

    p = sub.add_parser("eval", help="offline official-formula scoring of a CSV folder")
    p.add_argument("--submission-dir", required=True)
    p.add_argument("--gt", required=True)
    p.set_defaults(fn=_cmd_eval)

    p = sub.add_parser("version", help="print package version")
    p.set_defaults(fn=_cmd_version)

    args = ap.parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":
    raise SystemExit(main())
