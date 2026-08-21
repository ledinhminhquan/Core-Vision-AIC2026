"""Round-43: RRF-merge two machine runs — "lần chạy 2 học lần chạy 1".

Two same-config runs differ only by LLM sampling noise (live: 9.4 vs 9.0);
reciprocal-rank-fusing their rankings is a variance-reduction ensemble at the
whole-pipeline level, the big sibling of the round-40 votes. Row identity:
(video, frame) for KIS/QA/AVS — a QA row keeps the answer of whichever run
ranked it higher — and (video, frame-tuple) for TRAKE.

    python scripts/63_ensemble_runs.py --a <dirA-or-zip> --b <dirB-or-zip> \
        --out merged/   [--k 60]
"""

from __future__ import annotations

import argparse
import csv
import io
import tempfile
import zipfile
from pathlib import Path


def _load(src: str) -> dict[str, list[list[str]]]:
    p = Path(src)
    if p.suffix == ".zip":
        td = Path(tempfile.mkdtemp())
        zipfile.ZipFile(p).extractall(td)
        p = td
    return {f.stem: [r for r in csv.reader(io.StringIO(f.read_text(encoding="utf-8-sig"))) if r]
            for f in sorted(p.rglob("*.csv"))}


def rrf_merge(a: list[list[str]], b: list[list[str]], k: int,
              task: str) -> list[list[str]]:
    score: dict[tuple, float] = {}
    keep: dict[tuple, list[str]] = {}
    best_rank: dict[tuple, int] = {}
    for ranking in (a, b):
        for rank, row in enumerate(ranking):
            # Row identity: (video, frame) for kis/qa/avs — the QA answer
            # column stays OUT of the key so the two runs' differing answers
            # for the same frame dedupe (better-ranked run's answer wins).
            # TRAKE identity is the whole (video, frame-sequence) tuple.
            key = tuple(row) if task == "trake" else tuple(row[:2])
            score[key] = score.get(key, 0.0) + 1.0 / (k + rank + 1)
            if key not in best_rank or rank < best_rank[key]:
                best_rank[key] = rank
                keep[key] = row
    order = sorted(score, key=lambda kk: -score[kk])
    return [keep[kk] for kk in order][:100]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=60)
    args = ap.parse_args()

    ra, rb = _load(args.a), _load(args.b)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for stem in sorted(set(ra) | set(rb)):
        task = next((t for t in ("trake", "avs", "qa", "kis") if t in stem), "kis")
        merged = rrf_merge(ra.get(stem, []), rb.get(stem, []), args.k, task)
        with open(out / f"{stem}.csv", "w", encoding="utf-8", newline="") as f:
            csv.writer(f, lineterminator="\n").writerows(merged)
    print(f"merged {len(set(ra) | set(rb))} stems → {out}")


if __name__ == "__main__":
    main()
