"""Ablation battery A1–A10 (docs/PAPER_NOTES.md) — one command, one table.

Runs the query pack once per configuration flip, scores each run with the
OFFICIAL scorer, and prints/saves a results table the paper can quote. Each
ablation is a pure env/config change — no code edits, exactly as PAPER_NOTES
planned. Needs artifacts + a query dir + a GT json (the 89-query 2025 finals
pack is the standing dev set).

    python scripts/26_run_ablations.py --query-dir queries/dev-2025-finals --gt queries/dev-2025-finals/gt.json
        [--only A1 A2 ...] [--out artifacts/ablations]

A5 (training) lives in notebook 02 (it prints its own table); A3's full weight
sweep lives in scripts/21 — here they get single representative rows.
"""

import argparse
import copy
import json
import time
from pathlib import Path

from _bootstrap import init

from cvp.config import Settings

# Each variant: (label, {dotted.key: value}) applied on a DEEP COPY of settings.
ABLATIONS: dict[str, list[tuple[str, dict]]] = {
    "A1": [  # lane composition
        ("siglip2-only", {"embedding.model": "siglip2"}),
        ("ensemble(siglip2+openclip)", {"embedding.model": "ensemble"}),
    ],
    "A2": [  # SuperGlobal
        ("superglobal-off", {"search.rerank": False}),
        ("superglobal-on", {"search.rerank": True}),
    ],
    "A3": [  # fusion signals (representative rows; full sweep = scripts/21)
        ("visual-only", {"search.weights.ocr": 0.0, "search.weights.asr": 0.0,
                         "search.weights.caption": 0.0, "search.weights.metadata": 0.0,
                         "search.weights.object": 0.0}),
        ("all-signals(default)", {}),
        ("rrf-fusion", {"search.fusion_method": "rrf"}),
    ],
    "A4": [  # TRAKE algorithm
        ("trake-beam", {"temporal.algo": "beam"}),
        ("trake-dante", {"temporal.algo": "dante", "temporal.use_ensemble": False}),
        ("trake-dante-ensemble", {"temporal.algo": "dante", "temporal.use_ensemble": True}),
    ],
    "A6": [  # VLM rerank (needs GEMINI_API_KEY for the 'on' row)
        ("vlm-rerank-off", {"search.vlm_rerank": False}),
        ("vlm-rerank-gemini", {"search.vlm_rerank": True,
                               "search.vlm_rerank_provider": "gemini"}),
    ],
    "A7": [  # AVS diversification
        ("avs-greedy(λ=1.0)", {"search.avs_mmr_lambda": 1.0}),
        ("avs-mmr(λ=0.7)", {"search.avs_mmr_lambda": 0.7}),
        ("avs-mmr(λ=0.5)", {"search.avs_mmr_lambda": 0.5}),
    ],
    "A8": [  # QA answers
        ("qa-single-global-answer", {"vqa.answers_per_query": 1}),
        ("qa-per-group(default)", {"vqa.answers_per_query": 5}),
    ],
    # 2026 additions worth measuring the same way:
    "A9": [  # cross-encoder rerank
        ("cross-rerank-off", {"search.reranker": "none"}),
        ("cross-rerank-blip2", {"search.reranker": "blip2_itm"}),
        ("cross-rerank-qwen", {"search.reranker": "qwen_reranker"}),
    ],
    "A10": [  # temporal boost + retry
        ("temporal-boost-off", {"search.temporal_boost": False}),
        ("temporal-boost-on", {"search.temporal_boost": True}),
        ("low-conf-retry-on", {"search.low_confidence_retry": True}),
    ],
}


def _apply(settings: Settings, overrides: dict) -> Settings:
    s = copy.deepcopy(settings)
    for dotted, value in overrides.items():
        obj = s
        *path, last = dotted.split(".")
        for part in path:
            obj = getattr(obj, part)
        setattr(obj, last, value)
    return s


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--settings", default=None)
    ap.add_argument("--query-dir", required=True)
    ap.add_argument("--gt", required=True, help="ground-truth json (official.py formats)")
    ap.add_argument("--only", nargs="*", default=None, help="subset, e.g. A1 A4 A9")
    ap.add_argument("--out", default=None, help="output dir (default artifacts/ablations)")
    args = ap.parse_args()

    base = init(args.settings)
    out_root = Path(args.out) if args.out else base.paths.art("ablations")
    out_root.mkdir(parents=True, exist_ok=True)

    # Heavy imports after arg parsing (lazy-import convention).
    from cvp.eval.official import score_run
    from cvp.pipeline.run_queries import run_query_folder

    chosen = {k: v for k, v in ABLATIONS.items()
              if not args.only or k in set(args.only)}

    rows: list[dict] = []
    for aid, variants in chosen.items():
        for label, overrides in variants:
            s = _apply(base, overrides)
            run_dir = out_root / f"{aid}-{label}".replace("(", "_").replace(")", "_")
            run_dir.mkdir(parents=True, exist_ok=True)
            t0 = time.perf_counter()
            try:
                written = run_query_folder(s, Path(args.query_dir), run_dir)
                # Score ONLY this variant run's CSVs — stale files from earlier
                # battery runs in the reused folder would contaminate the
                # ablation table (review R3-C11, same trap scripts/20 guards).
                import shutil
                import tempfile

                with tempfile.TemporaryDirectory(prefix="cvp_abl_") as td:
                    for p in written:
                        shutil.copy2(p, Path(td) / p.name)
                    report = score_run(Path(td), Path(args.gt))
                final = float(report.mean_final)
                status = "ok"
            except Exception as e:  # noqa: BLE001 — one variant must not stop the battery
                final, status = float("nan"), f"FAILED: {e}"
            dt = time.perf_counter() - t0
            rows.append({"ablation": aid, "variant": label, "final": final,
                         "wall_s": round(dt, 1), "status": status})
            print(f"{aid:4s} {label:32s} final={final:.4f}  ({dt:.0f}s)  {status}")

    out_json = out_root / "ablation_results.json"
    out_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_json}")


if __name__ == "__main__":
    main()
