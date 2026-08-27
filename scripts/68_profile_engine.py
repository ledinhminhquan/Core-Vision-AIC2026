"""Hồ sơ hiệu năng engine (Nhiệm vụ E đợt 2): micro-benchmark hot paths offline.

Đo các đường nóng THUẦN PYTHON của một truy vấn (phần API/GPU không thuộc phạm
vi — chúng do quota/network quyết) trên fixture synthetic tự sinh, KHÔNG cần
artifacts: tokenize, BM25 persisted scoring, min-max + weighted-sum fusion,
aggregate biến thể query, neighbor boosts, DP TRAKE (DANTE). Mỗi op chạy
``--repeats`` lần (bỏ lượt đầu — warmup), báo median + best (ms).

Kích thước mặc định mô phỏng Batch-1 thật: ~500 ứng viên/BM25 field, fused map
~5000 gid, video ~200 keyframe, TRAKE 30 video × 5 event. Đêm thi 23 câu: tổng
chi phí thuần-Python phải là con số KHÔNG đáng kể so với API — bảng này chứng
minh (hoặc bác bỏ) điều đó bằng số.

    python scripts/68_profile_engine.py --out artifacts/lab/profile.md
    python scripts/68_profile_engine.py --quick      # bản nhanh cho CI/smoke
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np

try:  # chạy từ scripts/ — tests import qua importlib thì bỏ qua
    from _bootstrap import init  # noqa: F401 — side-effect sys.path
except ImportError:  # pragma: no cover
    pass

from cvp.config import TemporalCfg
from cvp.index.text_store import TextIndexField
from cvp.search import fusion
from cvp.search.temporal import dante_best_sequences
from cvp.utils.text import tokenize_vi

_WORDS = ("người phụ nữ đàn ông áo đỏ xanh vàng đi xe máy đạp chạy bộ chợ "
          "đường phố nhà cửa sông núi biển thuyền cá bán hàng nấu ăn món "
          "bánh trẻ em học sinh trường lớp cây hoa quả chó mèo".split())


def _sentences(rng: np.random.Generator, n: int, length: int = 12) -> list[str]:
    return [" ".join(rng.choice(_WORDS, size=length)) for _ in range(n)]


def _score_map(rng: np.random.Generator, keys: np.ndarray) -> dict[int, float]:
    return {int(k): float(v) for k, v in zip(keys, rng.random(len(keys)))}


def _bench(fn, repeats: int) -> dict[str, float]:
    """Median/best wall-ms của ``fn()`` qua ``repeats`` lượt (+1 warmup bỏ đi)."""
    fn()  # warmup (JIT dict resize, cache import, ...)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    return {"median_ms": round(statistics.median(times), 3),
            "best_ms": round(min(times), 3), "repeats": repeats}


def build_ops(quick: bool) -> list[tuple[str, str, "callable"]]:
    """(tên op, mô tả kích thước, thunk) — mọi fixture sinh sẵn NGOÀI vùng đo."""
    rng = np.random.default_rng(73)
    scale = 0.2 if quick else 1.0

    # tokenize: 2000 câu ~12 từ (một kho caption nhỏ / nhiều query).
    n_sent = int(2000 * scale) or 50
    sents = _sentences(rng, n_sent)

    # BM25 field synthetic: 5000 doc, vocab từ _WORDS (+ suffix), 500 key chấm.
    n_docs = int(5000 * scale) or 100
    docs: dict[int | str, list] = {}
    df_counts: dict[str, int] = {}
    total_len = 0
    for k in range(n_docs):
        toks = list(rng.choice(_WORDS, size=15))
        tf: dict[str, int] = {}
        for t in toks:
            tf[t] = tf.get(t, 0) + 1
        for t in tf:
            df_counts[t] = df_counts.get(t, 0) + 1
        docs[k] = [[len(toks), tf]]
        total_len += len(toks)
    field = TextIndexField(name="bench", n_docs=n_docs, avgdl=total_len / n_docs,
                           df=df_counts, docs=docs)
    q_tokens = list(rng.choice(_WORDS, size=8))
    cand_keys = [int(x) for x in rng.choice(n_docs, size=min(500, n_docs), replace=False)]

    # Fusion maps: dense 5000 gid + 4 field thưa 500 gid; spans 25 video × 200.
    n_gids = int(5000 * scale) or 200
    gids = np.arange(n_gids)
    dense = _score_map(rng, gids)
    sparse_maps = [_score_map(rng, rng.choice(n_gids, size=min(500, n_gids),
                                              replace=False)) for _ in range(4)]
    weights = [1.0, 0.35, 0.30, 0.25, 0.15]
    per_video = 200
    spans = {int(g): (int(g) // per_video * per_video, per_video) for g in gids}
    fused_sample = fusion.weighted_sum([dense, *sparse_maps], weights)
    variant_maps = [_score_map(rng, rng.choice(n_gids, size=min(2000, n_gids),
                                               replace=False)) for _ in range(3)]

    # TRAKE DP: 30 video × (200 frame, 5 event).
    n_videos = int(30 * scale) or 6
    tcfg = TemporalCfg()
    sims = [np.asarray(rng.random((per_video, 5)), dtype=np.float32)
            for _ in range(n_videos)]
    times_axis = np.arange(per_video, dtype=np.float64) * 5.0   # keyframe ~5s

    return [
        ("tokenize_vi", f"{n_sent} câu ×12 từ",
         lambda: [tokenize_vi(s) for s in sents]),
        ("bm25_scores_for", f"{len(cand_keys)} key × 8 token (field {n_docs} doc)",
         lambda: field.scores_for(q_tokens, cand_keys)),
        ("fusion_minmax", f"map {n_gids} gid",
         lambda: fusion.minmax(dense)),
        ("fusion_weighted_sum", f"dense {n_gids} + 4×500 gid",
         lambda: fusion.weighted_sum([dense, *sparse_maps], weights)),
        ("aggregate_queries_max", "3 biến thể × 2000 gid",
         lambda: fusion.aggregate_queries(variant_maps, how="max")),
        ("neighbor_boost", f"fused {len(fused_sample)} gid, ±2",
         lambda: fusion.neighbor_boost(dict(fused_sample), spans)),
        ("neighbor_consistency_boost", f"fused {len(fused_sample)} gid, w=0.15",
         lambda: fusion.neighbor_consistency_boost(dict(fused_sample), spans,
                                                   weight=0.15)),
        ("dante_dp", f"{n_videos} video × (200 frame × 5 event)",
         lambda: [dante_best_sequences(s, times_axis, tcfg) for s in sims]),
    ]


def render_markdown(rows: list[dict], quick: bool) -> str:
    lines = [f"# Hồ sơ hiệu năng engine (68_profile_engine{' --quick' if quick else ''})",
             "", "| op | kích thước | median ms | best ms | lượt |", "|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['op']} | {r['size']} | {r['median_ms']} "
                     f"| {r['best_ms']} | {r['repeats']} |")
    total = sum(r["median_ms"] for r in rows)
    lines += ["", f"**Tổng median một lượt qua mọi op: {total:.1f} ms** — chi phí "
                  "thuần-Python per query ở cỡ này; API/GPU (Gemini, rerank, faiss) "
                  "mới là phần chi phối đêm thi.", ""]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repeats", type=int, default=7, help="số lượt đo mỗi op")
    ap.add_argument("--quick", action="store_true", help="fixture nhỏ (smoke/CI)")
    ap.add_argument("--out", default=None, help="file markdown")
    ap.add_argument("--json-out", default=None, help="file JSON máy đọc")
    args = ap.parse_args()

    repeats = max(1, int(args.repeats) if not args.quick else min(args.repeats, 3))
    rows: list[dict] = []
    print(f"68_profile_engine: {('QUICK' if args.quick else 'FULL')} × {repeats} lượt")
    for name, size, thunk in build_ops(args.quick):
        stat = _bench(thunk, repeats)
        rows.append({"op": name, "size": size, **stat})
        print(f"  {name:28s} {size:36s} median {stat['median_ms']:8.3f} ms")

    md = render_markdown(rows, args.quick)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md, encoding="utf-8")
        print(f"\nBáo cáo → {out}")
    if args.json_out:
        jout = Path(args.json_out)
        jout.parent.mkdir(parents=True, exist_ok=True)
        jout.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"JSON → {jout}")


if __name__ == "__main__":
    main()
