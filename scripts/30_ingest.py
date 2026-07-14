"""Full idempotent build: (extract) → catalog → embed → index.

Run after every dataset drop. Only missing work is done.
"""

import argparse

from _bootstrap import init

from cvf.pipeline.ingest import run_ingest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--settings", default=None)
    ap.add_argument("--no-extract", action="store_true", help="skip keyframe self-extraction")
    ap.add_argument("--force-index", action="store_true")
    args = ap.parse_args()

    settings = init(args.settings)
    report = run_ingest(settings, extract=not args.no_extract, force_index=args.force_index)
    print(
        f"Ingest done: {report.keyframes} keyframes / {report.videos} videos | "
        f"extracted {report.extracted_videos} | embedded {report.embedded} | indexed {report.indexed}"
    )


if __name__ == "__main__":
    main()
