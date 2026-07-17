"""Warm the query-enhancement disk cache before a competition round.

The playbook says "cache ấm 20–30 query" — venue networks are slow and the
FIRST Gemini call per query costs a full round-trip. This script runs every
query of a pack (or a list of strings) through ``QueryProcessor.process`` so
translations/enhancements/expansions are already ON DISK when the round
starts; the engine then answers from cache even if the network dies mid-round.

    python scripts/51_warm_cache.py --query-dir queries/dev-2025-finals
    python scripts/51_warm_cache.py --queries "câu 1" "câu 2"

No models, no index, no GPU — only the (cheap) query-processing path runs.
"""

import argparse
from pathlib import Path

from _bootstrap import init


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--settings", default=None)
    ap.add_argument("--query-dir", default=None, help="folder of organiser *.txt files")
    ap.add_argument("--queries", nargs="*", default=None, help="explicit query texts")
    args = ap.parse_args()

    settings = init(args.settings)
    texts: list[str] = list(args.queries or [])
    if args.query_dir:
        for f in sorted(Path(args.query_dir).glob("*.txt")):
            t = f.read_text(encoding="utf-8-sig").strip().replace("\n", " ")
            if t:
                texts.append(t)
    if not texts:
        ap.error("give --query-dir and/or --queries")

    from cvp.models.query_processor import QueryProcessor

    qp = QueryProcessor(settings)
    ok = degraded = 0
    for i, t in enumerate(texts, 1):
        processed = qp.process(t)
        if processed.provider_used == settings.query.provider:
            ok += 1
        else:
            degraded += 1
        print(f"[{i}/{len(texts)}] {processed.provider_used:<9s} {t[:70]}")
    print(f"\nWarmed {len(texts)} queries: {ok} full-quality, {degraded} degraded "
          f"(degraded results are NOT cached — rerun once the network is healthy).")


if __name__ == "__main__":
    main()
