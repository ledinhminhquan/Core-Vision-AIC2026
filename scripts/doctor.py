"""Corpus health check — coverage of every artifact + index staleness."""

import argparse
import json

from _bootstrap import init

from cvf.pipeline.ingest import doctor


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--settings", default=None)
    args = ap.parse_args()

    settings = init(args.settings)
    report = doctor(settings)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    for name, m in report["members"].items():
        if m["index_stale"]:
            print(f"\n⚠  [{name}] index is STALE — run scripts/30_ingest.py before searching!")


if __name__ == "__main__":
    main()
