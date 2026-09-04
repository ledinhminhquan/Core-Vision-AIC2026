"""Adaptive multi-attempt support — "lượt sau giỏi hơn lượt trước" (Nhiệm vụ 2).

Three pure-offline pieces, orchestrated by ``scripts/64_adaptive_attempts.py``:

1. **Per-query confidence** from a finished run's OWN outputs — no organiser
   scores needed:

   * ``margin`` — how far the top of the dumped ``visual`` signal map towers
     over its bulk (``ranking_confidence`` — the same statistic the engine's
     low-confidence retry uses), when scripts/23 signal dumps are available;
   * ``concentration`` — share of the CSV head that stays on the top-1 row's
     video (a confident hit clusters around one moment; flat noise scatters
     across videos). Heuristic, can be fooled by a confidently-WRONG video —
     that is why components are averaged, never trusted alone;
   * ``answer_consensus`` (QA only) — majority share of the answer column over
     the CSV head; the :data:`~cvp.pipeline.run_queries.QA_FALLBACK_ANSWER`
     placeholder counts as NO answer, so a run of fallbacks scores 0.

2. **Attempt-2 plan** — the weak-query list (below threshold, capped by a
   budget fraction) plus high-effort env overrides for exactly the tasks that
   appear among the weak queries. Every override is an EXISTING settings knob.

3. **Weighted N-run RRF merge** — the same row-identity semantics as
   ``scripts/63_ensemble_runs.py`` (KIS/QA/AVS: ``(video, frame)``, answer of
   the better-ranked run wins; TRAKE: the whole frame tuple), generalized to N
   runs with per-run weights. ``scripts/63`` itself is UNTOUCHED; a test pins
   that the equal-weight 2-run merge reproduces its output row-for-row.

Heavy imports (``cvp.pipeline.run_queries`` pulls the engine stack) happen
lazily inside functions — this module imports clean on machines without
faiss/torch, like ``cvp.pipeline.auto_agent``.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import math
import shutil
import tempfile
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from cvp.constants import MAX_SUBMISSION_ROWS
from cvp.submission.packager import infer_task

log = logging.getLogger(__name__)

Rows = list[list[str]]

# Component weights of the combined confidence; absent components (no signal
# dumps, non-QA task) drop out and the rest renormalize.
DEFAULT_COMPONENT_WEIGHTS: dict[str, float] = {
    "margin": 0.5,
    "concentration": 0.3,
    "answer_consensus": 0.2,
}

# High-effort knobs for the attempt-2 rerun — EXISTING settings keys only
# (env form, values parsed as YAML by cvp.config). "common" always applies;
# per-task blocks apply when that task appears among the weak queries.
# NOTE: changing CVP_QUERY__EXPANSIONS alters the query-processor cache key,
# so weak queries are RE-enhanced by Gemini on attempt 2 — intended (more
# variants), but it does cost fresh API calls.
HIGH_EFFORT_ENV: dict[str, dict[str, str]] = {
    "common": {
        "CVP_SEARCH__LOW_CONFIDENCE_RETRY": "true",
        "CVP_SEARCH__TOPK": "800",
        "CVP_QUERY__EXPANSIONS": "4",
    },
    "kis": {"CVP_SEARCH__VLM_RERANK_VOTES": "5"},
    "avs": {"CVP_SEARCH__VLM_RERANK_VOTES": "5"},
    "qa": {
        "CVP_VQA__SELF_CONSISTENCY": "5",
        "CVP_SEARCH__VLM_RERANK_VOTES": "5",
    },
    "trake": {
        "CVP_TEMPORAL__PER_EVENT_TOPK": "300",
        "CVP_TEMPORAL__MAX_VIDEOS": "60",
        "CVP_TEMPORAL__EVENT_QUERY_VARIANTS": "all",
    },
}


@dataclass
class QueryConfidence:
    """One query's self-assessed confidence after a run (no organiser scores)."""

    stem: str
    task: str
    confidence: float                       # [0, 1] — combined
    components: dict[str, float] = field(default_factory=dict)
    n_rows: int = 0


# ── run / dump loading ───────────────────────────────────────────────────────


def load_run(src: str | Path) -> dict[str, Rows]:
    """Submission CSVs of one run (folder or .zip) → ``{stem: rows}``.

    Same reading semantics as ``scripts/63_ensemble_runs._load``: utf-8-sig,
    empty rows dropped, ``rglob`` so Codabench-style subfolders work.
    """
    p = Path(src)
    if p.suffix == ".zip":
        td = Path(tempfile.mkdtemp(prefix="cvp_run_"))
        zipfile.ZipFile(p).extractall(td)
        p = td
    return {f.stem: [r for r in csv.reader(io.StringIO(f.read_text(encoding="utf-8-sig"))) if r]
            for f in sorted(p.rglob("*.csv"))}


def load_signal_dumps(signals_dir: str | Path) -> dict[str, dict[str, dict[str, float]]]:
    """Merge scripts/23 dumps → ``{stem: {signal: {"video,frame": score}}}``.

    Unlike the tuning harness (scripts/21, which must fail loud), a corrupt
    dump here only costs one query its ``margin`` component — warn and skip.
    """
    out: dict[str, dict[str, dict[str, float]]] = {}
    root = Path(signals_dir)
    for f in sorted(root.glob("*.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            log.warning("Signal dump %s unreadable (%s) — its query loses the "
                        "margin component", f.name, e)
            continue
        for stem, sig_maps in (data or {}).items():
            slot = out.setdefault(str(stem), {})
            for signal, rows in (sig_maps or {}).items():
                merged = slot.setdefault(str(signal), {})
                merged.update({str(k): float(v) for k, v in (rows or {}).items()})
    return out


# ── confidence components ────────────────────────────────────────────────────


def margin_confidence(signal_map: dict[str, float] | None) -> float | None:
    """``ranking_confidence`` over one dumped signal map (None = unavailable)."""
    if not signal_map:
        return None
    from cvp.pipeline.run_queries import ranking_confidence  # lazy: engine stack

    return float(ranking_confidence(list(signal_map.values())))


def video_concentration(rows: Rows, head: int = 10) -> float:
    """Share of the first ``head`` rows staying on the top-1 row's video, [0,1]."""
    head_rows = [r for r in rows[:head] if r]
    if not head_rows:
        return 0.0
    top_video = head_rows[0][0]
    return sum(1 for r in head_rows if r[0] == top_video) / len(head_rows)


def answer_consensus(rows: Rows, head: int = 10) -> float:
    """Majority share of the QA answer column over the CSV head, [0,1].

    Fallback answers count as NO answer (they mean the VQA produced nothing),
    and they stay in the denominator — a head full of fallbacks scores 0, a
    half-fallback head caps at 0.5.
    """
    from cvp.pipeline.run_queries import QA_FALLBACK_ANSWER  # lazy: engine stack

    answers = [r[2].strip().casefold() for r in rows[:head]
               if len(r) > 2 and r[2].strip()]
    if not answers:
        return 0.0
    real = [a for a in answers if a != QA_FALLBACK_ANSWER.casefold()]
    if not real:
        return 0.0
    top_share = Counter(real).most_common(1)[0][1]
    return top_share / len(answers)


def qa_fallback_only(rows: Rows) -> bool:
    """True when a QA CSV carries no real answer — every answered row is the
    fallback answer (or the answer column is empty). Such a query scored 0
    (VQA produced nothing: Gemini storm / daily quota) and is worth a second
    attempt."""
    from cvp.pipeline.run_queries import QA_FALLBACK_ANSWER  # lazy: engine stack

    answers = [r[2].strip().casefold() for r in rows if len(r) > 2 and r[2].strip()]
    return not answers or all(a == QA_FALLBACK_ANSWER.casefold() for a in answers)


def rescue_fallback_qa(run_dir: str | Path) -> list[str]:
    """Round-93 — attempt 2 as a RESCUE, not a re-run: delete the QA CSVs of
    ``run_dir`` whose rows are all fallback answers so a resumed ``run_auto``
    re-answers exactly those queries. KIS/TRAKE rows and QA queries that
    already have an answer are untouched, so the rescued pack can never score
    below the original (a 'không rõ' row scores 0 either way). Returns the
    stems removed."""
    p = Path(run_dir)
    removed: list[str] = []
    for f in sorted(p.glob("query-*-qa.csv")):
        rows = [r for r in csv.reader(io.StringIO(f.read_text(encoding="utf-8-sig"))) if r]
        if qa_fallback_only(rows):
            f.unlink()
            removed.append(f.stem)
    return removed


def query_confidence(
    stem: str,
    rows: Rows,
    *,
    task: str | None = None,
    signals: dict[str, dict[str, float]] | None = None,
    head: int = 10,
    component_weights: dict[str, float] | None = None,
) -> QueryConfidence:
    """Combine the available components into one [0,1] confidence.

    Args:
        stem: query/CSV stem (task inferred from it when ``task`` is None).
        rows: the run's submission rows for this query (ranked best first).
        signals: this query's scripts/23 dump ``{signal: {row: score}}``;
            only ``visual`` feeds the margin component. None = no dumps.
        head: rows considered by concentration/consensus.
        component_weights: overrides for :data:`DEFAULT_COMPONENT_WEIGHTS`.
    """
    task = (task or infer_task(stem)).lower()
    comps: dict[str, float] = {}
    m = margin_confidence((signals or {}).get("visual"))
    if m is not None:
        comps["margin"] = round(m, 6)
    comps["concentration"] = round(video_concentration(rows, head), 6)
    if task == "qa":
        comps["answer_consensus"] = round(answer_consensus(rows, head), 6)
    weights = {**DEFAULT_COMPONENT_WEIGHTS, **(component_weights or {})}
    avail = {k: v for k, v in comps.items() if weights.get(k, 0.0) > 0}
    if not rows or not avail:
        conf = 0.0
    else:
        w_sum = sum(weights[k] for k in avail)
        conf = sum(v * weights[k] for k, v in avail.items()) / w_sum
    return QueryConfidence(stem=stem, task=task, confidence=round(conf, 6),
                           components=comps, n_rows=len(rows))


# ── weak-query selection + plan ──────────────────────────────────────────────


# ── Round-82 EVIDENCE NOTE (30/08) ──────────────────────────────────────────
# Validated offline on two full bench runs (23 câu, official per-query scores):
#   Pearson(query_confidence, điểm thật) = +0.10 và −0.04  → KHÔNG có sức dự báo;
#   ngưỡng mặc định 0.35 đánh dấu 0/23 câu yếu; câu conf=0.4 điểm 1.0 và câu
#   conf=1.0 điểm 0.0 lẫn lộn. Đồng thời 2 lượt cùng stack ĐỒNG THUẬN 21/23 câu
#   và THUA GIỐNG NHAU ở các câu khó → thất bại là hệ thống, không phải ngẫu
#   nhiên. Kết luận vận hành: KHÔNG dựa vào select_weak để chọn câu đánh lại;
#   lượt 2 = chạy lại TRỌN pack bằng ĐỘI HÌNH KHÁC (nb03 LINEUP="diverse",
#   shard nhiều máy ảo) rồi rrf_merge_runs — phần merge là phần đã được kiểm
#   chứng (0.6587 > cả hai lượt gốc). select_weak giữ lại cho nghiên cứu.
def select_weak(
    confidences: dict[str, QueryConfidence],
    threshold: float = 0.35,
    max_fraction: float = 0.5,
) -> list[str]:
    """Stems below ``threshold``, weakest first, capped at
    ``ceil(max_fraction · total)`` — the attempt-2 budget guard."""
    ordered = sorted(confidences.values(), key=lambda c: (c.confidence, c.stem))
    weak = [c.stem for c in ordered if c.confidence < threshold]
    # ceil keeps small packs alive (1 query × 0.5 → cap 1); an explicit 0
    # budget must mean "no rerun", so no max(1, …) floor here.
    cap = math.ceil(max_fraction * len(confidences))
    return weak[:cap] if cap > 0 else []


def plan_env(weak_tasks: set[str]) -> dict[str, str]:
    """Merged high-effort env for the tasks present among the weak queries."""
    env = dict(HIGH_EFFORT_ENV["common"])
    for task in sorted(weak_tasks):
        env.update(HIGH_EFFORT_ENV.get(task, {}))
    return env


def build_plan(
    run_dir: str | Path,
    *,
    signals_dir: str | Path | None = None,
    query_dir: str | Path | None = None,
    threshold: float = 0.35,
    max_fraction: float = 0.5,
    head: int = 10,
) -> dict:
    """Attempt-2 plan from a finished run.

    Queries whose ``.txt`` exists in ``query_dir`` but which produced NO CSV
    (engine exception, empty query) get confidence 0 — the automatic track
    forfeits them silently otherwise, and a rerun is their only chance.
    """
    runs = load_run(run_dir)
    signals = load_signal_dumps(signals_dir) if signals_dir else {}
    stems: set[str] = set(runs)
    if query_dir:
        stems |= {p.stem for p in Path(query_dir).glob("*.txt")}

    confidences: dict[str, QueryConfidence] = {}
    for stem in sorted(stems):
        rows = runs.get(stem)
        if rows is None:
            confidences[stem] = QueryConfidence(
                stem=stem, task=infer_task(stem), confidence=0.0,
                components={"missing_csv": 0.0}, n_rows=0)
        else:
            confidences[stem] = query_confidence(
                stem, rows, signals=signals.get(stem), head=head)

    weak = select_weak(confidences, threshold=threshold, max_fraction=max_fraction)
    weak_tasks = {confidences[s].task for s in weak}
    return {
        "run_dir": str(run_dir),
        "query_dir": str(query_dir) if query_dir else None,
        "threshold": threshold,
        "max_fraction": max_fraction,
        "head": head,
        "queries": {
            s: {"task": c.task, "confidence": c.confidence,
                "components": c.components, "rows": c.n_rows}
            for s, c in sorted(confidences.items())
        },
        "weak": weak,
        "env": plan_env(weak_tasks),
    }


def stage_weak_queries(query_dir: str | Path, weak: Sequence[str],
                       out_dir: str | Path) -> list[Path]:
    """Copy the weak queries' ``.txt`` into a fresh staging folder — the
    attempt-2 ``run_auto`` input. Missing files are skipped with a warning
    (their stems came from the run, not the pack)."""
    query_dir = Path(query_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    staged: list[Path] = []
    for stem in weak:
        src = query_dir / f"{stem}.txt"
        if not src.is_file():
            log.warning("stage_weak_queries: %s.txt not in %s — skipped", stem, query_dir)
            continue
        dst = out / src.name
        shutil.copy2(src, dst)
        staged.append(dst)
    return staged


# ── weighted N-run RRF merge ─────────────────────────────────────────────────


def merge_rows(
    rankings: Sequence[Rows],
    weights: Sequence[float],
    k: int,
    task: str,
    limit: int = MAX_SUBMISSION_ROWS,
    qa_keep_all_answers: bool = False,
) -> Rows:
    """Weighted RRF over N rankings of ONE query.

    Row identity mirrors ``scripts/63_ensemble_runs.rrf_merge``: KIS/QA/AVS
    dedupe on ``(video, frame)`` — the answer column stays OUT of the key and
    the best-ranked run's row (earlier run on ties) supplies it; TRAKE identity
    is the whole ``(video, frame-tuple)``. With two rankings and equal weights
    this reproduces scripts/63 row-for-row (pinned by test).
    """
    if len(rankings) != len(weights):
        raise ValueError(f"{len(rankings)} rankings but {len(weights)} weights "
                         "— zip would silently drop the tail")
    score: dict[tuple, float] = {}
    keep: dict[tuple, list[str]] = {}
    best_rank: dict[tuple, int] = {}
    for ranking, w in zip(rankings, weights):
        for rank, row in enumerate(ranking):
            key = tuple(row) if task == "trake" else tuple(row[:2])
            score[key] = score.get(key, 0.0) + w / (k + rank + 1)
            if key not in best_rank or rank < best_rank[key]:
                best_rank[key] = rank
                keep[key] = row
    order = sorted(score, key=lambda kk: -score[kk])
    out = [keep[kk] for kk in order][:limit]
    if task == "qa" and qa_keep_all_answers:
        out = _qa_hedge_rows(out, rankings, limit)
    return out


_QA_HEDGE_HEAD = 20      # frames whose alternative answers are worth a ticket
_QA_HEDGE_MAX = 10       # extra rows — they REPLACE the tail, never the head


def _qa_hedge_rows(merged: Rows, rankings: Sequence[Rows], limit: int) -> Rows:
    """Round-86b QA hedge (audit-shaped): two runs answering the SAME frame
    differently (A100-vs-G4 measured 2/7 QA answers agreeing) — append the
    other run's REAL answer as an extra row for frames in the merged head,
    replacing tail rows only (bounded: ≤ _QA_HEDGE_MAX). Fallback/empty
    answers never hedge; answers equal under the official normalizer never
    duplicate. The metric takes the max over rows, so the head is untouched
    and the cost is ≤ 10 tail tickets.
    """
    from cvp.eval.official import normalize_answer
    from cvp.pipeline.run_queries import QA_FALLBACK_ANSWER

    fb = normalize_answer(QA_FALLBACK_ANSWER)

    def _norm(row: Sequence[str]) -> str:
        return normalize_answer(row[2] if len(row) > 2 else "")

    alts: dict[tuple[str, str], list[list[str]]] = {}
    for ranking in rankings:
        for row in ranking:
            n = _norm(row)
            if not n or n == fb:
                continue
            alts.setdefault((row[0], row[1]), []).append([row[0], row[1], row[2]])
    extra: list[list[str]] = []
    seen = {(r[0], r[1], _norm(r)) for r in merged}
    for row in merged[:_QA_HEDGE_HEAD]:
        for alt in alts.get((row[0], row[1]), []):
            k = (alt[0], alt[1], _norm(alt))
            if k not in seen:
                seen.add(k)
                extra.append(alt)
            if len(extra) >= _QA_HEDGE_MAX:
                break
        if len(extra) >= _QA_HEDGE_MAX:
            break
    if not extra:
        return merged
    keep_n = max(_QA_HEDGE_HEAD, limit - len(extra))
    return (list(merged[:keep_n]) + extra)[:limit]


def rrf_merge_runs(
    runs: Sequence[dict[str, Rows]],
    *,
    weights: Sequence[float] | None = None,
    k: int = 60,
    limit: int = MAX_SUBMISSION_ROWS,
    qa_keep_all_answers: bool = False,
) -> dict[str, Rows]:
    """Weighted N-run RRF over whole runs (union of stems).

    Args:
        runs: ``{stem: rows}`` per attempt, PRIMARY FIRST — rank ties keep the
            earlier run's row, exactly like scripts/63's a-then-b order.
        weights: per-run RRF weights (default 1.0 each — noise-sibling merge).
        k: RRF constant.
        limit: row cap per query (organiser: 100).
    """
    if not runs:
        return {}
    ws = list(weights) if weights is not None else [1.0] * len(runs)
    if len(ws) != len(runs):
        raise ValueError(f"{len(runs)} runs but {len(ws)} weights — must match 1:1")
    if any(w <= 0 for w in ws):
        raise ValueError(f"run weights must be > 0, got {ws}")
    stems = sorted(set().union(*(set(r) for r in runs)))
    return {
        stem: merge_rows([r.get(stem, []) for r in runs], ws, k,
                         infer_task(stem), limit, qa_keep_all_answers)
        for stem in stems
    }


def write_merged(merged: dict[str, Rows], out_dir: str | Path) -> list[Path]:
    """Write merged rankings as submission CSVs (63's writer semantics)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for stem in sorted(merged):
        path = out / f"{stem}.csv"
        with open(path, "w", encoding="utf-8", newline="") as f:
            csv.writer(f, lineterminator="\n").writerows(merged[stem])
        written.append(path)
    return written


def diff_top1(base: dict[str, Rows], merged: dict[str, Rows]) -> list[dict]:
    """Which queries changed their top-1 row versus the primary run."""
    out: list[dict] = []
    for stem in sorted(set(base) | set(merged)):
        b = (base.get(stem) or [[]])[0]
        m = (merged.get(stem) or [[]])[0]
        if b != m:
            out.append({"stem": stem, "base_top1": list(b), "merged_top1": list(m)})
    return out


def render_diff_markdown(diffs: list[dict]) -> str:
    """Human-readable top-1 diff table for the merge report."""
    if not diffs:
        return "Không câu nào đổi top-1 sau merge.\n"
    lines = ["| query | top-1 lượt gốc | top-1 sau merge |", "|---|---|---|"]
    for d in diffs:
        lines.append("| {} | {} | {} |".format(
            d["stem"],
            ",".join(d["base_top1"]) or "(trống)",
            ",".join(d["merged_top1"]) or "(trống)",
        ))
    return "\n".join(lines) + "\n"
