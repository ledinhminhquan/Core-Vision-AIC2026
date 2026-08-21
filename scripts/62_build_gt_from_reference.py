"""Round-43: build a gt.json from a HIGH-SCORING reference submission.

After a round, a near-perfect submission (e.g. the shared 19.8/23 trial-round
bundle) is the best ground truth we will ever get for that pack: its top-1
row per query is treated as the correct (video, moment). The resulting
gt.json feeds cvp.eval.official / scripts/21_tune_weights — the first real
offline tuning signal of the season.

Noise caveat (stated, not hidden): a 19.8/23 reference is ~86% perfect, so
2-3 queries carry wrong GT. Weight tuning averages over the whole pack and
tolerates that; never treat a single query's offline score as gospel.

    python scripts/62_build_gt_from_reference.py \
        --reference "…/submission (3).zip" --queries "…/THUNGHIEM-bo-de-thi" \
        --out queries/trial/gt.json
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import tempfile
import zipfile
from pathlib import Path


def _read_rows(p: Path) -> list[list[str]]:
    return [r for r in csv.reader(io.StringIO(p.read_text(encoding="utf-8-sig")))
            if r]


def build_entry(stem: str, rows: list[list[str]], kis_eps: int,
                trake_eps: int, top_frames: int) -> dict:
    task = next((t for t in ("trake", "avs", "qa", "kis") if t in stem), "kis")
    top_video = rows[0][0]
    if task == "trake":
        frames = [int(x) for x in rows[0][1:]]
        return {"task": "trake", "video_id": top_video,
                "moments": [[f - trake_eps, f + trake_eps] for f in frames]}
    # KIS/QA/AVS: every top-N frame of the SAME video is an acceptable window —
    # references routinely pin several neighbouring frames of the true moment.
    frames = [int(r[1]) for r in rows[:top_frames] if r[0] == top_video]
    entry: dict = {"task": "qa" if task == "qa" else "kis", "video_id": top_video,
                   "ranges": [[f - kis_eps, f + kis_eps] for f in frames]}
    if task == "qa":
        answers = []
        for r in rows:
            if r[0] == top_video and len(r) > 2 and r[2].strip():
                if r[2] not in answers:
                    answers.append(r[2])
        if answers:
            entry["answers"] = answers
    return entry


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference", required=True, help="reference submission .zip")
    ap.add_argument("--queries", required=True, help="query pack dir (*.txt)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--kis-epsilon", type=int, default=125,
                    help="±frames accepted around each reference frame (~5s@25fps)")
    ap.add_argument("--trake-epsilon", type=int, default=12,
                    help="±frames per TRAKE event (alignment is tight)")
    ap.add_argument("--top-frames", type=int, default=5)
    args = ap.parse_args()

    td = Path(tempfile.mkdtemp())
    zipfile.ZipFile(args.reference).extractall(td)
    ref = {p.stem: _read_rows(p) for p in sorted(td.rglob("*.csv"))}
    stems = sorted(p.stem for p in Path(args.queries).glob("*.txt"))

    gt: dict = {}
    for stem in stems:
        rows = ref.get(stem)
        if not rows:
            print(f"  ⚠ reference thiếu {stem} — bỏ qua (câu này không chấm được)")
            continue
        gt[stem] = build_entry(stem, rows, args.kis_epsilon,
                               args.trake_epsilon, args.top_frames)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(gt, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"gt.json: {len(gt)}/{len(stems)} câu → {out}")


if __name__ == "__main__":
    main()
