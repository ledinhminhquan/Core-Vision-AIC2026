"""Chẩn đoán TRAKE offline (Nhiệm vụ B đợt 2): định lượng H1/H2/H6 của report đợt 1.

Ba giả thuyết vì sao TRAKE 0.183 << KIS 0.729 (report/cursor_lab_report.md §4),
giờ đo bằng số thay vì phỏng đoán:

    H1  Trần lượng tử hóa lưới keyframe: submission chỉ nộp được frame_idx CỦA
        keyframe; với cửa sổ GT ±12 frame, kể cả chọn ĐÚNG video và tổ hợp
        keyframe TỐT NHẤT (ràng buộc strictly-increasing của writer), điểm tối
        đa là bao nhiêu? → DP nhỏ trên lưới map-keyframes per câu.
    H2  Recall video ở bước pool: video GT có lọt vào 100 dòng nộp không, rank
        đầu tiên bao nhiêu? (cận DƯỚI của recall pool nội bộ — CSV là thứ duy
        nhất quan sát được offline; pool thật chỉ đo được khi chạy engine.)
    H6  Lợi ích max-over-rows: điểm dòng tốt nhất vs dòng 1, Final đầy đủ vs
        Final chỉ-dòng-1, và headroom = trần H1 − dòng tốt nhất đã nộp (phần
        điểm jitter/đa dạng hóa hàng còn với tới được).

Offline thuần (không engine/API). Đường dẫn qua args để chạy thẳng trên Colab:

    python scripts/66_trake_diag.py --gt queries/gt-thunghiem.json \
        --map-dir data/map-keyframes --run artifacts/lab/lab_full \
        --out artifacts/lab/trake_diag.md
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

try:  # chạy từ scripts/ — tests import qua importlib thì bỏ qua
    from _bootstrap import init  # noqa: F401 — side-effect sys.path
except ImportError:  # pragma: no cover
    pass

from cvp.eval.official import (
    _entry_events,
    final_score,
    load_ground_truth,
    r_score_trake,
)
from cvp.pipeline.attempts import load_run

# Ngưỡng verdict (hằng số có chủ đích, không knob): H1 — trần trung bình dưới
# mức này nghĩa là chính LƯỚI keyframe chặn điểm (đầu tư nộp frame ngoài lưới /
# jitter trung điểm), trên ~0.9 nghĩa là lưới vô tội, lỗi nằm ở retrieval/DP.
H1_GRID_BOUND = 0.5
H1_GRID_INNOCENT = 0.9
# H2 — dưới mức này bước pool/retrieval đói video đúng (nới per_event_topk,
# max_videos, pool_context) trước khi bàn chuyện align.
H2_RECALL_LOW = 0.7
# H6 — chênh (best − row1) trung bình trên mức này = các dòng sâu đã cứu điểm
# thật → chiến lược lấp 100 dòng (jitter) đáng tiền.
H6_BENEFIT_REAL = 0.05


def _read_map_frames(path: Path) -> list[int]:
    """map-keyframes/{vid}.csv → sorted unique frame_idx (dòng hỏng bỏ qua)."""
    frames: set[int] = set()
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            try:
                frames.add(int(float(row["frame_idx"])))
            except (KeyError, TypeError, ValueError):
                continue
    return sorted(frames)


def grid_ceiling(frames: Sequence[int],
                 events: Sequence[tuple[int, int] | None]) -> dict[str, Any]:
    """Trần điểm H1: tổ hợp keyframe strictly-increasing trúng nhiều event nhất.

    DP O(K·P): ``dp[p]`` = số event trúng tối đa khi event hiện tại được gán
    keyframe thứ ``p`` (frame tăng nghiêm ngặt ↔ chỉ số ``p`` tăng nghiêm
    ngặt vì ``frames`` đã sort + unique). Event nào cửa sổ None (GT hỏng) chỉ
    có thể trượt — mẫu số vẫn là K, khớp ``r_score_trake``.

    Returns dict: ``ceiling`` ∈ [0,1] (0.0 kèm ``feasible=False`` khi video có
    ít keyframe hơn số event — writer không thể nhận dòng hợp lệ nào từ lưới),
    ``per_event`` = list bool (cửa sổ CÓ keyframe nào không, bỏ ràng buộc thứ
    tự — chỉ ra event nghẽn), ``hits`` = số event trúng của tổ hợp tốt nhất.
    """
    k = len(events)
    p_count = len(frames)
    per_event = [
        ev is not None and any(ev[0] <= f <= ev[1] for f in frames)
        for ev in events
    ]
    if k == 0:
        return {"ceiling": 0.0, "hits": 0, "per_event": [], "feasible": False}
    if p_count < k:
        # Round-73: dòng nộp NGẮN vẫn hợp lệ (writer chỉ đòi ≥1 frame tăng dần)
        # và r_score_trake zip nó với các event ĐẦU — trần là DP trên p_count
        # event đầu, mẫu số vẫn K. Trước đây báo 0.0 "không dựng nổi dòng hợp
        # lệ" — sai cả hai vế.
        if p_count == 0:
            return {"ceiling": 0.0, "hits": 0, "per_event": per_event,
                    "feasible": False}
        sub = grid_ceiling(frames, events[:p_count])
        return {"ceiling": sub["hits"] / k, "hits": sub["hits"],
                "per_event": per_event, "feasible": False}

    neg = float("-inf")
    dp = [0] * p_count
    for j, ev in enumerate(events):
        hit = [1 if ev is not None and ev[0] <= f <= ev[1] else 0 for f in frames]
        if j == 0:
            dp = [hit[p] for p in range(p_count)]
            continue
        best_before = neg
        new = [neg] * p_count
        for p in range(p_count):
            if p >= 1 and dp[p - 1] > best_before:
                best_before = dp[p - 1]
            if best_before > neg:
                new[p] = best_before + hit[p]
        dp = new
        # event j chỉ được gán từ vị trí j trở đi — các ô đầu vẫn -inf, đúng ý.
    hits = max((v for v in dp if v > neg), default=0)
    hits = max(0, int(hits))
    return {"ceiling": hits / k, "hits": hits, "per_event": per_event, "feasible": True}


def median_grid_gap(frames: Sequence[int]) -> float | None:
    """Khoảng cách median giữa hai keyframe liền kề (frame gốc) — bối cảnh H1."""
    if len(frames) < 2:
        return None
    return float(statistics.median(b - a for a, b in zip(frames, frames[1:])))


def diagnose_query(stem: str, entry: Mapping[str, Any], map_dir: Path,
                   rows: Sequence[Sequence[str]] | None) -> dict[str, Any]:
    """Toàn bộ số liệu H1/H2/H6 cho MỘT câu TRAKE (thuần dữ liệu, test được)."""
    video = str(entry.get("video_id", "")).strip()
    events = _entry_events(entry)
    d: dict[str, Any] = {"stem": stem, "video": video, "k_events": len(events)}

    map_path = map_dir / f"{video}.csv"
    if map_path.is_file():
        frames = _read_map_frames(map_path)
        d["n_keyframes"] = len(frames)
        d["median_gap"] = median_grid_gap(frames)
        d.update({f"h1_{k}": v for k, v in grid_ceiling(frames, events).items()})
    else:
        d["h1_note"] = f"thiếu map-keyframes/{video}.csv — không tính được trần H1"

    if rows is not None:
        if rows:
            d["h2_video_in_rows"] = any(str(r[0]).strip() == video for r in rows)
            d["h2_first_rank"] = next(
                (i + 1 for i, r in enumerate(rows) if str(r[0]).strip() == video), None)
            scores = [r_score_trake([str(c).strip() for c in r], dict(entry))
                      for r in rows]
            best = max(scores)
            d["h6_row1"] = round(scores[0], 4)
            d["h6_best_row"] = round(best, 4)
            d["h6_best_rank"] = scores.index(best) + 1 if best > 0 else None
            d["h6_benefit"] = round(best - scores[0], 4)
            d["h6_final_full"] = round(final_score(scores), 4)
            if "h1_ceiling" in d:
                d["h6_headroom"] = round(max(0.0, d["h1_ceiling"] - best), 4)
        else:
            d["h2_video_in_rows"] = False
            d["h2_note"] = "CSV rỗng"
    return d


def verdict_h1(ceilings: Sequence[float]) -> str:
    if not ceilings:
        return "H1: KHÔNG ĐỦ DỮ LIỆU (thiếu map-keyframes cho mọi câu)."
    mean_c = sum(ceilings) / len(ceilings)
    if mean_c < H1_GRID_BOUND:
        return (f"H1: ĐÚNG — trần lưới trung bình {mean_c:.3f} (<{H1_GRID_BOUND}): "
                "chính lưới keyframe chặn điểm; ưu tiên nộp frame NGOÀI lưới "
                "(jitter trung điểm, temporal.submit_strategy=jitter) hơn là retrieval.")
    if mean_c >= H1_GRID_INNOCENT:
        return (f"H1: SAI — trần lưới trung bình {mean_c:.3f} (≥{H1_GRID_INNOCENT}): "
                "lưới đủ dày, điểm thấp do retrieval/align — đầu tư pool + DP.")
    return (f"H1: MỘT PHẦN — trần lưới trung bình {mean_c:.3f} "
            f"([{H1_GRID_BOUND}, {H1_GRID_INNOCENT})): lưới ăn một phần điểm; "
            "kết hợp jitter VÀ cải thiện align.")


def verdict_h2(recalls: Sequence[bool]) -> str:
    if not recalls:
        return "H2: KHÔNG ĐỦ DỮ LIỆU (không có --run hoặc không câu nào chấm được)."
    rate = sum(recalls) / len(recalls)
    if rate < H2_RECALL_LOW:
        return (f"H2: ĐÚNG — video GT chỉ xuất hiện trong CSV ở {rate:.0%} câu "
                f"(<{H2_RECALL_LOW:.0%}): bước pool đói video đúng — nới "
                "per_event_topk/max_videos/pool_context. (Đo trên CSV nộp = cận "
                "DƯỚI của recall pool.)")
    return (f"H2: SAI (trong phạm vi đo được) — video GT có mặt ở {rate:.0%} câu; "
            "pool không phải nút nghẽn chính. (Cận dưới: pool thật chỉ rộng hơn.)")


def verdict_h6(benefits: Sequence[float], headrooms: Sequence[float]) -> str:
    if not benefits:
        return "H6: KHÔNG ĐỦ DỮ LIỆU (cần --run)."
    mean_b = sum(benefits) / len(benefits)
    parts = []
    if mean_b > H6_BENEFIT_REAL:
        parts.append(f"H6: ĐÚNG — max-over-rows đã cứu trung bình +{mean_b:.3f} "
                     f"điểm/câu so với dòng 1 (> {H6_BENEFIT_REAL})")
    else:
        parts.append(f"H6: chênh best−row1 trung bình chỉ {mean_b:.3f} — các dòng "
                     "sâu hiện tại gần như không cứu điểm")
    if headrooms:
        mean_h = sum(headrooms) / len(headrooms)
        parts.append(f"headroom tới trần lưới còn trung bình {mean_h:.3f}/câu — "
                     "phần này là đích của jitter/đa dạng hóa hàng"
                     if mean_h > 0.01 else
                     "dòng tốt nhất đã CHẠM trần lưới — muốn thêm điểm phải nộp "
                     "frame ngoài lưới")
    return "; ".join(parts) + "."


def render_markdown(diags: list[dict[str, Any]], verdicts: dict[str, str],
                    gt_label: str, run_label: str | None) -> str:
    lines = [f"# TRAKE diag — gt={gt_label}" + (f", run={run_label}" if run_label else ""), ""]
    lines.append("## H1 — trần lượng tử hóa lưới keyframe (cửa sổ GT ±ε, writer strictly-increasing)")
    lines.append("")
    lines.append("| câu | video | K | keyframes | gap median | event có keyframe | trần H1 |")
    lines.append("|---|---|---|---|---|---|---|")
    for d in diags:
        if "h1_ceiling" in d:
            marks = "".join("✓" if b else "✗" for b in d.get("h1_per_event", []))
            gap = d.get("median_gap")
            # Round-73: <2 keyframe parse được → median_gap=None; in "—" thay
            # vì TypeError giết cả báo cáo sau khi console đã in.
            gap_s = "—" if gap is None else f"{gap:.0f}"
            lines.append(
                f"| {d['stem']} | {d['video']} | {d['k_events']} | {d.get('n_keyframes', '?')} "
                f"| {gap_s} | {marks} | {d['h1_ceiling']:.3f}"
                f"{'' if d.get('h1_feasible', True) else ' (lưới < số event — trần theo dòng nộp NGẮN)'} |")
        else:
            lines.append(f"| {d['stem']} | {d['video']} | {d['k_events']} | — | — | — "
                         f"| {d.get('h1_note', '')} |")
    lines += ["", f"**{verdicts['h1']}**", ""]

    if any("h2_video_in_rows" in d for d in diags):
        lines.append("## H2 — recall video GT trong CSV nộp (cận dưới recall pool)")
        lines.append("")
        lines.append("| câu | video GT trong CSV? | rank đầu |")
        lines.append("|---|---|---|")
        for d in diags:
            if "h2_video_in_rows" in d:
                lines.append(f"| {d['stem']} | {'✓' if d['h2_video_in_rows'] else '✗'} "
                             f"| {d.get('h2_first_rank') or '—'} |")
        lines += ["", f"**{verdicts['h2']}**", ""]

        lines.append("## H6 — lợi ích max-over-rows + headroom tới trần lưới")
        lines.append("")
        lines.append("| câu | row1 | best row (rank) | benefit | Final đầy đủ | headroom |")
        lines.append("|---|---|---|---|---|---|")
        for d in diags:
            if "h6_row1" in d:
                rank = f" (r{d['h6_best_rank']})" if d.get("h6_best_rank") else ""
                head = d.get("h6_headroom")
                lines.append(
                    f"| {d['stem']} | {d['h6_row1']:.3f} | {d['h6_best_row']:.3f}{rank} "
                    f"| +{d['h6_benefit']:.3f} | {d['h6_final_full']:.3f} "
                    f"| {head if head is not None else '—'} |")
        lines += ["", f"**{verdicts['h6']}**", ""]
    else:
        lines += ["_H2/H6: bỏ qua — không có --run (CSV lượt chạy)._", ""]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", required=True, help="gt.json (scripts/62)")
    ap.add_argument("--map-dir", required=True, help="thư mục map-keyframes/ (*.csv)")
    ap.add_argument("--run", default=None, help="thư mục/zip CSV lượt chạy (bật H2/H6)")
    ap.add_argument("--out", default=None, help="file markdown báo cáo")
    ap.add_argument("--json-out", default=None, help="file JSON máy đọc")
    args = ap.parse_args()

    map_dir = Path(args.map_dir)
    if not map_dir.is_dir():
        raise SystemExit(f"--map-dir không tồn tại: {map_dir}")
    # Round-75 (tổng kiểm F): một --run gõ nhầm từng trả {} ÂM THẦM → mọi câu
    # "CSV rỗng" và verdict H2/H6 sai lệch mà không ai hay — fail to tiếng.
    if args.run and not Path(args.run).exists():
        raise SystemExit(f"--run không tồn tại: {args.run}")
    gt = load_ground_truth(args.gt)
    trake_stems = sorted(s for s, e in gt.items() if e.get("task") == "trake")
    if not trake_stems:
        raise SystemExit(f"GT {args.gt} không có câu TRAKE nào — sai file?")
    run = load_run(args.run) if args.run else None

    diags = [diagnose_query(stem, gt[stem], map_dir,
                            run.get(stem, []) if run is not None else None)
             for stem in trake_stems]

    verdicts = {
        "h1": verdict_h1([d["h1_ceiling"] for d in diags if "h1_ceiling" in d]),
        "h2": verdict_h2([d["h2_video_in_rows"] for d in diags
                          if "h2_video_in_rows" in d]),
        "h6": verdict_h6([d["h6_benefit"] for d in diags if "h6_benefit" in d],
                         [d["h6_headroom"] for d in diags if "h6_headroom" in d]),
    }

    print(f"TRAKE diag: {len(diags)} câu")
    for d in diags:
        ceiling = f"trần={d['h1_ceiling']:.3f}" if "h1_ceiling" in d else "trần=?"
        extra = ""
        if "h6_best_row" in d:
            extra = (f" best={d['h6_best_row']:.3f}@r{d.get('h6_best_rank') or '—'}"
                     f" benefit=+{d['h6_benefit']:.3f}")
        print(f"  {d['stem']:28s} {ceiling}{extra}")
    for v in verdicts.values():
        print(f"\n{v}")

    md = render_markdown(diags, verdicts, Path(args.gt).name,
                         Path(args.run).name if args.run else None)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md, encoding="utf-8")
        print(f"\nBáo cáo markdown → {out}")
    if args.json_out:
        jout = Path(args.json_out)
        jout.parent.mkdir(parents=True, exist_ok=True)
        jout.write_text(json.dumps({"verdicts": verdicts, "queries": diags},
                                   ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"JSON → {jout}")


if __name__ == "__main__":
    main()
