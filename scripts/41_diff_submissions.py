"""Diff two submission runs — see exactly what a config/weight change moved.

Between tuning iterations (scripts/21), lane flips or reranker experiments the
question is always "what actually CHANGED in the output?". This prints, per
query stem present in either folder:

* top-1 (video, frame) identical or not — the cell worth 5× everything else,
* overlap of the top-20 rows (how much the head reshuffled),
* row-count changes, stems only in one run.

    python scripts/41_diff_submissions.py --a artifacts/submissions/runA \
        --b artifacts/submissions/runB [--k 20]

Pure stdlib; safe to run mid-competition (read-only).
"""

import argparse
import csv
from pathlib import Path

import _bootstrap  # noqa: F401 — sys.path + UTF-8 stdout on Windows


def read_rows(csv_path: Path) -> list[tuple[str, ...]]:
    """Headerless submission CSV → list of row tuples (blank lines skipped)."""
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        return [tuple(c.strip() for c in row)
                for row in csv.reader(f) if row and any(c.strip() for c in row)]


def diff_stems(dir_a: Path, dir_b: Path, k: int = 20) -> list[dict]:
    """One record per stem in either folder — the script's printable payload."""
    stems_a = {p.stem: p for p in sorted(dir_a.glob("*.csv"))}
    stems_b = {p.stem: p for p in sorted(dir_b.glob("*.csv"))}
    out: list[dict] = []
    for stem in sorted(set(stems_a) | set(stems_b)):
        if stem not in stems_a or stem not in stems_b:
            out.append({"stem": stem, "status": "only-in-A" if stem in stems_a else "only-in-B"})
            continue
        rows_a, rows_b = read_rows(stems_a[stem]), read_rows(stems_b[stem])
        top_a = rows_a[0][:2] if rows_a else None
        top_b = rows_b[0][:2] if rows_b else None
        head_a = {r[:2] for r in rows_a[:k]}
        head_b = {r[:2] for r in rows_b[:k]}
        denom = max(1, min(len(head_a), len(head_b)))
        out.append({
            "stem": stem,
            "status": "same-top1" if top_a == top_b else "TOP1-CHANGED",
            "top_a": top_a, "top_b": top_b,
            "overlap_k": len(head_a & head_b) / denom,
            "rows": (len(rows_a), len(rows_b)),
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--a", required=True, help="submission folder A (baseline)")
    ap.add_argument("--b", required=True, help="submission folder B (candidate)")
    ap.add_argument("--k", type=int, default=20, help="head size for overlap")
    args = ap.parse_args()

    records = diff_stems(Path(args.a), Path(args.b), k=args.k)
    changed = missing = 0
    for r in records:
        if r["status"] in ("only-in-A", "only-in-B"):
            missing += 1
            print(f"  {r['stem']:<34s} {r['status']}")
            continue
        mark = " " if r["status"] == "same-top1" else "!"
        if mark == "!":
            changed += 1
        print(f"{mark} {r['stem']:<34s} top1 {str(r['top_a']):<24s} -> "
              f"{str(r['top_b']):<24s} overlap@{args.k}={r['overlap_k']:.2f} "
              f"rows={r['rows'][0]}/{r['rows'][1]}")
    print(f"\n{len(records)} stems: {changed} top1-changed, {missing} unmatched. "
          "Score both with scripts/40 (or cvp eval) before choosing a side.")


if __name__ == "__main__":
    main()
