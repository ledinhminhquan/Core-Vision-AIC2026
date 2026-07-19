"""Latency benchmark: p50 / p95 / p99 per query over the live engine.

Finals are DRES-style with 4–5-minute budgets per query — the retrieval stack
must answer in well under a second so the humans (or the auto-agent) keep the
time. Run this after every artifacts rebuild and before every competition day.

    python scripts/50_bench_latency.py [--n 200] [--query-dir queries/example]
        [--queries "câu 1" "câu 2" ...] [--warmup 5]

Reports per-query wall-clock percentiles for `search_text` (the KIS hot path).
Targets (AIO plan, adopted): text p50 ≤ 200 ms, p95 ≤ 500 ms on the laptop.
"""

import argparse
import statistics
import time
from pathlib import Path

from _bootstrap import init

DEFAULT_QUERIES = [
    "một người đàn ông mặc áo sơ mi trắng đứng trước bản đồ thời tiết",
    "đoàn xe diễu hành trên đường phố với cờ đỏ sao vàng",
    "cận cảnh bàn tay đang gói bánh chưng bằng lá dong",
    "nữ phát thanh viên áo dài xanh ngồi tại bàn tin thời sự",
    "các vận động viên chạy điền kinh trên sân vận động buổi tối",
]


def _percentile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return float("nan")
    idx = min(len(sorted_vals) - 1, max(0, round(q * (len(sorted_vals) - 1))))
    return sorted_vals[idx]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--settings", default=None)
    ap.add_argument("--n", type=int, default=200, help="total timed searches")
    ap.add_argument("--warmup", type=int, default=5, help="untimed warmup searches")
    ap.add_argument("--queries", nargs="*", default=None, help="explicit query texts")
    ap.add_argument("--query-dir", default=None,
                    help="folder of organiser *.txt files to draw queries from")
    args = ap.parse_args()

    settings = init(args.settings)
    queries = list(args.queries or [])
    if args.query_dir:
        for f in sorted(Path(args.query_dir).glob("*.txt")):
            text = f.read_text(encoding="utf-8-sig").strip().replace("\n", " ")
            if text:
                queries.append(text)
    if not queries:
        queries = DEFAULT_QUERIES

    from cvp.search.engine import SearchEngine  # heavy import after arg parsing

    engine = SearchEngine(settings)
    # Warm EVERY distinct query once (review R3-C13): with query.provider=gemini
    # the first hit of each query pays a full enhancement round-trip — leaving
    # most queries cold would poison p95/p99 with network latency, not search
    # latency. --warmup adds extra repeat passes over the head on top.
    for q in queries:
        engine.search_text(q)
    for q in queries[: args.warmup]:
        engine.search_text(q)

    times_ms: list[float] = []
    for i in range(args.n):
        q = queries[i % len(queries)]
        t0 = time.perf_counter()
        engine.search_text(q)
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    s = sorted(times_ms)
    p50, p95, p99 = (_percentile(s, x) for x in (0.50, 0.95, 0.99))
    print(f"search_text over n={len(s)} runs, {len(queries)} distinct queries")
    print(f"  p50 = {p50:8.1f} ms   (target ≤ 200 ms)")
    print(f"  p95 = {p95:8.1f} ms   (target ≤ 500 ms)")
    print(f"  p99 = {p99:8.1f} ms")
    print(f"  mean = {statistics.fmean(s):7.1f} ms   min = {s[0]:.1f}   max = {s[-1]:.1f}")
    if p50 > 200 or p95 > 500:
        print("⚠ Above target — check: persisted text index built? cache warm? "
              "ensemble members on this machine? display_k too large?")


if __name__ == "__main__":
    main()
