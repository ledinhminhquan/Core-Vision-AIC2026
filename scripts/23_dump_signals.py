"""Dump RAW per-signal score maps for a query pack → weight-tuning input.

For each query file the engine runs once and the pre-fusion signal maps
(visual / ocr / asr / caption / metadata / object) are written as one JSON per
query. Query files are normalised EXACTLY like the batch runner
(``cvp.pipeline.run_queries``): organiser TRAKE files (header + "E1:…") dump
against the cleaned event texts, single-line QA files against the description
part only. Output shape is what scripts/21_tune_weights.py expects:

    {"<query-stem>": {"visual": {"<video_id>,<frame_idx>": raw_score, ...}, ...}}

Then: python scripts/21_tune_weights.py --signals-dir <out> --gt <gt.json> ...

Example:
    python scripts/23_dump_signals.py --query-dir ./queries/dev-2025-finals \
        --out-dir ./artifacts/signal_dumps/dev
"""

import argparse
from pathlib import Path

from _bootstrap import init

from cvp.utils.io import atomic_write_json


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--settings", default=None)
    ap.add_argument("--query-dir", required=True)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--topk", type=int, default=None, help="dense candidates per query")
    args = ap.parse_args()

    settings = init(args.settings)
    query_dir = Path(args.query_dir)
    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else settings.paths.art("signal_dumps", query_dir.name)
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    from cvp.pipeline.run_queries import infer_task, parse_query_lines, parse_trake_events
    from cvp.search.engine import SearchEngine  # heavy import after arg parsing

    engine = SearchEngine(settings)
    files = sorted(query_dir.glob("*.txt"))
    if not files:
        raise SystemExit(f"No .txt query files in {query_dir}")

    from cvp.pipeline.run_queries import load_query_lines

    for qf in files:
        lines = load_query_lines(qf)       # THE round-time loader — never drift
        if not lines:
            print(f"{qf.stem}: EMPTY query file — skipped")
            continue
        task = infer_task(qf.name)
        # SAME normalisation as the batch runner (shared parse_query_lines —
        # round-3 fix: multi-paragraph KIS/AVS joins all lines, multi-line QA
        # drops only the trailing question line): tuned weights must be
        # optimised on exactly the query text the runtime will search.
        if task == "trake":
            # Mirror the runtime's A/B knob too (round-4 fix): with
            # temporal.event_context=prepend the engine searches header-prefixed
            # event text, so the dump must match.
            prepend = getattr(settings.temporal, "event_context", "none") == "prepend"
            query = ". ".join(parse_trake_events(lines, prepend_context=prepend))
        else:
            query, _question = parse_query_lines(task, lines)
        _results, dump = engine.search_text_debug(query, topk=args.topk)
        payload: dict[str, dict[str, float]] = {}
        for signal, id_scores in dump.items():
            row_scores: dict[str, float] = {}
            for gid, score in id_scores.items():
                ref = engine.catalog.ref(gid)
                row_scores[f"{ref.video_id},{ref.frame_idx}"] = float(score)
            if row_scores:
                payload[signal] = row_scores
        atomic_write_json(out_dir / f"{qf.stem}.json", {qf.stem: payload}, indent=None)
        print(f"{qf.stem}: {', '.join(f'{k}({len(v)})' for k, v in payload.items()) or 'EMPTY'}")

    print(f"\nDumped {len(files)} queries → {out_dir}")
    print("Next: python scripts/21_tune_weights.py --signals-dir "
          f"{out_dir} --gt <gt.json> --trials 60")


if __name__ == "__main__":
    main()
