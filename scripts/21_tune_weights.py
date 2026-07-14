"""Tune ``search.weights`` on cached per-signal score maps (no engine, no GPU).

The raw per-signal scores are dumped once with ``scripts/23_dump_signals.py``;
this harness then re-normalizes, re-fuses and re-scores thousands of weight
vectors per second against the official qualifier metric and reports the best.

Signals dir (``--signals-dir``): one or more ``*.json`` files, merged. Each is::

    {
      "<query_id>": {
        "visual":   {"<video_id>,<frame_idx>": <raw score>, ...},
        "ocr":      {...}, "asr": {...}, "caption": {...},
        "metadata": {...}, "object": {...}
      },
      ...
    }

Row keys are ``"<video_id>,<frame_idx>"`` (submission row identity). Scores are
RAW engine outputs — the harness min-max normalizes per query per signal, the
same convention as ``cvp.search.fusion.weighted_sum``.

Ground truth (``--gt``) — a subset of what ``cvp.eval.official`` accepts::

    {
      "<query_id>": {"task": "kis", "video_id": "L21_V001",
                     "frame_start": 4500, "frame_end": 4700},
      "<query_id>": {"task": "qa",  "video_id": "...", "range": [s, e],
                     "answers": ["..."]},
      "<query_id>": {"task": "trake", "video_id": "...",
                     "moments": [[s1, e1], [s2, e2], ...]}
    }

(``segments``/``moments`` are accepted aliases for ``events``; the KIS/QA
window may be spelled ``frame_start``/``frame_end``, ``range: [s, e]`` or
``center``+``epsilon``. QA ``answer``/``answers`` are ignored here — see
below.)

KIS rows are scored with ``cvp.eval.official.score_rows`` when that module is
available (workstream B3), else with the identical qualifier formulas in
``cvp.eval.metrics``. QA is re-ranked as KIS (fusion weights cannot change the
answer column) and TRAKE through a single-frame proxy (frame inside any event
window) because the cached maps rank keyframes, not frame tuples.

Example:
    python scripts/21_tune_weights.py --signals-dir ./artifacts/signal_dumps/dev \\
        --gt ./queries/dev_gt.json --method random --trials 120 \\
        --out ./artifacts/tuning/best_weights.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import math
import random
from pathlib import Path
from typing import Callable, Iterator

try:  # script executed from scripts/ — falls through when imported by tests
    from _bootstrap import init
except ImportError:  # pragma: no cover — tests import via importlib
    init = None  # type: ignore[assignment]

log = logging.getLogger(__name__)

# Signals the engine can dump; "visual" is the dense lane and stays pinned at
# its baseline weight so the others are tuned relative to it.
SIGNALS = ("visual", "ocr", "asr", "caption", "metadata", "object")
PINNED_SIGNAL = "visual"
# Grid steps per auxiliary dimension (--method grid): 5 steps, denser near the
# small values where auxiliary weights actually live.
GRID_STEPS = (0.0, 0.15, 0.3, 0.5, 0.8)
MAX_ROWS = 100  # submission cap — ranks past 100 never score

Rows = list[tuple[str, int]]
Scorer = Callable[[str, Rows, dict], float]


# ── signal loading & fusion ──────────────────────────────────────────────────


def load_signals(signals_dir: Path) -> dict[str, dict[str, dict[str, float]]]:
    """Merge every ``*.json`` under ``signals_dir`` → {query: {signal: {row: score}}}."""
    files = sorted(Path(signals_dir).glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No *.json signal dumps under {signals_dir}")
    out: dict[str, dict[str, dict[str, float]]] = {}
    for f in files:
        data = json.loads(f.read_text(encoding="utf-8"))
        for query, sig_maps in data.items():
            slot = out.setdefault(str(query), {})
            for signal, rows in (sig_maps or {}).items():
                merged = slot.setdefault(str(signal), {})
                merged.update({str(k): float(v) for k, v in (rows or {}).items()})
    return out


def _minmax(scores: dict[str, float]) -> dict[str, float]:
    """Min-max to [0,1]; constant maps → 1.0 (same convention as cvp.search.fusion)."""
    if not scores:
        return {}
    lo, hi = min(scores.values()), max(scores.values())
    if hi - lo < 1e-9:
        return {k: 1.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


def normalize_signals(
    signals: dict[str, dict[str, dict[str, float]]],
) -> dict[str, dict[str, dict[str, float]]]:
    """Min-max every (query, signal) map once so weight trials are cheap."""
    return {q: {s: _minmax(m) for s, m in sig.items()} for q, sig in signals.items()}


def _parse_row_key(key: str) -> tuple[str, int]:
    video_id, _, frame = key.rpartition(",")
    return video_id, int(float(frame))


def rank_rows(
    query_signals_norm: dict[str, dict[str, float]],
    weights: dict[str, float],
    top: int = MAX_ROWS,
) -> Rows:
    """Weighted-sum fusion over normalized maps → top ranked (video_id, frame_idx)."""
    combined: dict[str, float] = {}
    for signal, rows in query_signals_norm.items():
        w = float(weights.get(signal, 0.0))
        if w == 0.0 or not rows:
            continue
        for key, val in rows.items():
            combined[key] = combined.get(key, 0.0) + w * val
    ordered = sorted(combined.items(), key=lambda kv: (-kv[1], kv[0]))  # deterministic
    return [_parse_row_key(k) for k, _ in ordered[:top]]


# ── scoring ──────────────────────────────────────────────────────────────────


def _gt_window(gt: dict) -> tuple[int, int]:
    """KIS/QA window from 'frame_start'/'frame_end', 'range' or 'center'+'epsilon'."""
    if gt.get("frame_start") is not None and gt.get("frame_end") is not None:
        return int(gt["frame_start"]), int(gt["frame_end"])
    rng = gt.get("range")
    if isinstance(rng, (list, tuple)) and len(rng) == 2:
        return int(rng[0]), int(rng[1])
    if gt.get("center") is not None and gt.get("epsilon") is not None:
        center, eps = int(gt["center"]), int(gt["epsilon"])
        return center - eps, center + eps
    raise ValueError(f"GT entry lacks a usable frame window (keys: {sorted(gt)})")


def _gt_windows(gt: dict) -> list[tuple[int, int]]:
    """ALL acceptable KIS/QA windows — supports the multi-window ``ranges``
    spelling of ``cvp.eval.official`` (a row hits when its frame lands in ANY
    window); falls back to the single :func:`_gt_window`."""
    raw = gt.get("ranges")
    if isinstance(raw, (list, tuple)):
        out: list[tuple[int, int]] = []
        for ev in raw:
            try:
                if isinstance(ev, dict):
                    if ev.get("frame_start") is not None and ev.get("frame_end") is not None:
                        out.append((int(ev["frame_start"]), int(ev["frame_end"])))
                    elif ev.get("center") is not None and ev.get("epsilon") is not None:
                        c, e = int(ev["center"]), int(ev["epsilon"])
                        out.append((c - e, c + e))
                elif len(ev) == 2:
                    out.append((int(ev[0]), int(ev[1])))
            except (TypeError, ValueError):
                continue
        if out:
            return out
    return [_gt_window(gt)]


def _gt_events(gt: dict) -> list[tuple[int, int]]:
    """TRAKE event windows from 'events' / 'moments' / 'segments' spellings."""
    out: list[tuple[int, int]] = []
    for ev in gt.get("events") or gt.get("moments") or gt.get("segments") or []:
        if isinstance(ev, dict):
            out.append((int(ev["frame_start"]), int(ev["frame_end"])))
        else:
            out.append((int(ev[0]), int(ev[1])))
    return out


def _local_score(task: str, rows: Rows, gt: dict) -> float:
    """Official qualifier formulas via cvp.eval.metrics (KIS/QA + TRAKE proxy)."""
    from cvp.eval.metrics import qualifier_score

    task = str(task).lower()
    if task in ("kis", "qa"):
        windows = _gt_windows(gt)  # ≥1 windows; a row hits when inside ANY
        video = str(gt["video_id"])
        flags = [v == video and any(s <= f <= e for s, e in windows) for v, f in rows]
        return qualifier_score(flags)
    if task == "trake":
        # Proxy: cached maps rank single keyframes, so a row counts when its
        # frame lands inside ANY event window of the right video.
        video = str(gt["video_id"])
        events = _gt_events(gt)
        flags = [v == video and any(a <= f <= b for a, b in events) for v, f in rows]
        return qualifier_score(flags)
    raise ValueError(f"Unknown task {task!r} in ground truth")


def resolve_scorer() -> tuple[Scorer, str]:
    """Prefer cvp.eval.official for KIS; fall back to cvp.eval.metrics.

    QA/TRAKE always go through :func:`_local_score`: cached single-frame maps
    carry neither answers nor per-event frame tuples, so only the KIS part of
    the official scorer is replayable.
    """
    try:
        from cvp.eval import official
    except ImportError:
        return _local_score, "cvp.eval.metrics (cvp.eval.official not available)"
    fn = getattr(official, "score_rows", None)
    if not callable(fn):
        return _local_score, "cvp.eval.metrics (official.score_rows not found)"

    def scorer(task: str, rows: Rows, gt: dict) -> float:
        if str(task).lower() != "kis":
            return _local_score(task, rows, gt)
        str_rows = [[video, str(frame)] for video, frame in rows]
        return float(fn("kis", str_rows, gt).final)

    return scorer, "cvp.eval.official.score_rows (kis) + cvp.eval.metrics proxy (qa/trake)"


def evaluate_weights(
    weights: dict[str, float],
    signals_norm: dict[str, dict[str, dict[str, float]]],
    gt: dict[str, dict],
    scorer: Scorer,
) -> tuple[float, dict[str, float]]:
    """(mean qualifier score, per-task mean) of one weight vector."""
    per_task: dict[str, list[float]] = {}
    for query, sig in signals_norm.items():
        entry = gt.get(query)
        if entry is None:
            continue
        task = str(entry.get("task", "kis")).lower()
        rows = rank_rows(sig, weights)
        per_task.setdefault(task, []).append(scorer(task, rows, entry))
    all_scores = [s for scores in per_task.values() for s in scores]
    mean = sum(all_scores) / len(all_scores) if all_scores else 0.0
    return mean, {t: sum(v) / len(v) for t, v in per_task.items()}


# ── candidate weight vectors ─────────────────────────────────────────────────


def sample_weights(
    base: dict[str, float], active: list[str], rng: random.Random
) -> dict[str, float]:
    """One random candidate: pinned visual, aux dims jittered around base or uniform."""
    out = {PINNED_SIGNAL: 1.0}
    for signal in active:
        if rng.random() < 0.5:  # log-normal jitter around the current setting
            w = max(float(base.get(signal, 0.25)), 0.02) * math.exp(rng.gauss(0.0, 0.7))
        else:  # fully random restart — escapes a bad baseline
            w = rng.random()
        out[signal] = round(min(max(w, 0.0), 1.5), 4)
    return out


def grid_weights(active: list[str]) -> Iterator[dict[str, float]]:
    """Full grid over the auxiliary signals present in the dumps (5 steps/dim)."""
    for combo in itertools.product(GRID_STEPS, repeat=len(active)):
        yield {PINNED_SIGNAL: 1.0, **dict(zip(active, combo))}


# ── tuning loop ──────────────────────────────────────────────────────────────


def tune(
    signals: dict[str, dict[str, dict[str, float]]],
    gt: dict[str, dict],
    base_weights: dict[str, float],
    trials: int = 60,
    method: str = "random",
    seed: int = 0,
    scorer: Scorer | None = None,
    scorer_name: str | None = None,
) -> dict:
    """Search weight vectors over cached signal maps; return a report dict."""
    if scorer is None:
        scorer, scorer_name = resolve_scorer()
    signals = {q: s for q, s in signals.items() if q in gt}
    skipped = sorted(set(gt) - set(signals))
    if not signals:
        raise ValueError("No overlap between signal dumps and ground-truth queries")
    signals_norm = normalize_signals(signals)

    active = sorted(
        {s for sig in signals_norm.values() for s in sig} - {PINNED_SIGNAL}
    )
    base = {s: float(base_weights.get(s, 0.0)) for s in SIGNALS}
    base_score, base_per_task = evaluate_weights(base, signals_norm, gt, scorer)

    if method == "grid":
        candidates: Iterator[dict[str, float]] = grid_weights(active)
    elif method == "random":
        rng = random.Random(seed)
        candidates = (sample_weights(base, active, rng) for _ in range(trials))
    else:
        raise ValueError(f"Unknown method {method!r} (expected 'random' or 'grid')")

    best_weights, best_score, best_per_task = dict(base), base_score, base_per_task
    seen: set[tuple[float, ...]] = set()
    evaluated = 0
    for cand in candidates:
        full = {**base, **cand}  # inactive signals keep their baseline value
        key = tuple(full[s] for s in SIGNALS)
        if key in seen:
            continue
        seen.add(key)
        score, per_task = evaluate_weights(full, signals_norm, gt, scorer)
        evaluated += 1
        if score > best_score:
            best_weights, best_score, best_per_task = full, score, per_task
    return {
        "method": method,
        "scorer": scorer_name or "custom",
        "queries": len(signals_norm),
        "skipped_queries": skipped,
        "active_signals": active,
        "trials_evaluated": evaluated,
        "baseline": {"weights": base, "score": round(base_score, 6),
                     "per_task": {t: round(v, 6) for t, v in base_per_task.items()}},
        "best": {"weights": {s: round(w, 4) for s, w in best_weights.items()},
                 "score": round(best_score, 6),
                 "per_task": {t: round(v, 6) for t, v in best_per_task.items()}},
        "delta": round(best_score - base_score, 6),
        "per_task_delta": {
            t: round(best_per_task.get(t, 0.0) - base_per_task.get(t, 0.0), 6)
            for t in sorted(set(base_per_task) | set(best_per_task))
        },
    }


# ── CLI ──────────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--settings", default=None)
    ap.add_argument("--signals-dir", required=True,
                    help="dir of per-query signal dumps (scripts/23_dump_signals.py output)")
    ap.add_argument("--gt", required=True, help="ground-truth JSON (format in docstring)")
    ap.add_argument("--query-dir", default=None,
                    help="optional: only tune queries whose <id>.txt exists here")
    ap.add_argument("--trials", type=int, default=60, help="random-search trials")
    ap.add_argument("--method", choices=("random", "grid"), default="random")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="report JSON path")
    args = ap.parse_args()

    if init is None:  # pragma: no cover
        raise RuntimeError("Run this script from the repo (scripts/21_tune_weights.py)")
    settings = init(args.settings)
    from cvp.utils.io import atomic_write_json, read_json

    signals = load_signals(Path(args.signals_dir))
    gt = {str(k): v for k, v in read_json(Path(args.gt)).items()}
    if args.query_dir:
        wanted = {p.stem for p in Path(args.query_dir).glob("*.txt")}
        signals = {q: s for q, s in signals.items() if q in wanted}

    report = tune(
        signals, gt,
        base_weights=settings.search.weights.model_dump(),
        trials=args.trials, method=args.method, seed=args.seed,
    )

    out = Path(args.out) if args.out else settings.paths.art("tuning", "best_weights.json")
    atomic_write_json(out, report)

    print(f"Scorer: {report['scorer']}")
    print(f"Queries: {report['queries']}  (skipped without dumps: {len(report['skipped_queries'])})")
    print(f"Baseline {report['baseline']['score']:.4f} -> best {report['best']['score']:.4f} "
          f"(delta {report['delta']:+.4f}) over {report['trials_evaluated']} candidates")
    for task, delta in report["per_task_delta"].items():
        print(f"  {task}: {report['baseline']['per_task'].get(task, 0.0):.4f} -> "
              f"{report['best']['per_task'].get(task, 0.0):.4f} ({delta:+.4f})")
    print("Best weights: " + json.dumps(report["best"]["weights"], ensure_ascii=False))
    print("Apply via configs/settings.yaml search.weights, or env overrides:")
    for signal, w in report["best"]["weights"].items():
        print(f"  CVP_SEARCH__WEIGHTS__{signal.upper()}={w}")
    print(f"Report -> {out}")


if __name__ == "__main__":
    main()
