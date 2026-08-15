"""Warm the query-enhancement disk cache before a competition round.

The playbook says "cache ấm 20–30 query" — venue networks are slow and the
FIRST Gemini call per query costs a full round-trip. This script runs the
EXACT strings the engine will later process (the cache key is the exact query
text!) through ``QueryProcessor.process`` so they are already ON DISK when the
round starts:

* KIS/AVS → all stripped non-empty lines joined (``parse_query_lines``),
* QA      → the DESCRIPTION part only (the question never reaches the
            processor at round time),
* TRAKE   → EACH parsed event separately (``parse_trake_events`` — headers
            dropped, ``E1:`` prefixes stripped).

Warming any other string (e.g. the whole file as one blob) would report
success while the round-time lookups still MISS (review finding C1).

    python scripts/51_warm_cache.py --query-dir queries/dev-2025-finals
    python scripts/51_warm_cache.py --queries "câu 1" "câu 2"

No models, no index, no GPU — only the (cheap) query-processing path runs.
Requires ``query.provider: gemini`` — with any other provider there is
nothing to warm (full-quality results are the only ones cached).
"""

import argparse
from pathlib import Path

from _bootstrap import init

from cvp.pipeline.run_queries import infer_task, parse_query_lines, parse_trake_events


def round_time_texts(query_file: Path, event_context: str = "none") -> list[str]:
    """The exact processor inputs ``run_query_file`` will produce for one file.

    Mirrors run_query_file: utf-8-sig read, stripped non-empty lines, task from
    the FILENAME; TRAKE yields one string per event (each is processed
    separately at round time), everything else yields the retrieval text.
    """
    from cvp.pipeline.run_queries import load_query_lines

    lines = load_query_lines(query_file)   # THE round-time loader — never drift
    if not lines:
        return []
    task = infer_task(query_file.name)
    if task == "trake":
        return parse_trake_events(lines, prepend_context=(event_context == "prepend"))
    retrieval_text, _question = parse_query_lines(task, lines)
    return [retrieval_text] if retrieval_text.strip() else []


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--settings", default=None)
    ap.add_argument("--query-dir", default=None, help="folder of organiser *.txt files")
    ap.add_argument("--queries", nargs="*", default=None, help="explicit query texts")
    args = ap.parse_args()

    settings = init(args.settings)
    if settings.query.provider != "gemini":
        raise SystemExit(
            f"query.provider is {settings.query.provider!r} — only full-quality "
            "Gemini results are cached, so warming would be a no-op. Set "
            "CVP_QUERY__PROVIDER=gemini (and GEMINI_API_KEY) first."
        )

    texts: list[str] = list(args.queries or [])
    if args.query_dir:
        for f in sorted(Path(args.query_dir).glob("*.txt")):
            texts.extend(round_time_texts(f, settings.temporal.event_context))
    if not texts:
        ap.error("give --query-dir and/or --queries")

    from cvp.models.query_processor import QueryProcessor

    qp = QueryProcessor(settings)
    ok = degraded = 0
    for i, t in enumerate(texts, 1):
        processed = qp.process(t)
        if processed.provider_used == "gemini":
            ok += 1
        else:
            degraded += 1
        print(f"[{i}/{len(texts)}] {processed.provider_used:<9s} {t[:70]}")
    print(f"\nWarmed {len(texts)} round-time strings: {ok} full-quality (cached), "
          f"{degraded} degraded (NOT cached — rerun once the network is healthy).")


if __name__ == "__main__":
    main()
