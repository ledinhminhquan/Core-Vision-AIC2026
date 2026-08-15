"""Official HCMC AI Challenge scoring — the ground-truth evaluation harness.

Implements the EXACT formulas from the organisers' deck ("HCMC AI Challenge",
Evaluation Metric slides) so local runs predict Codabench scores:

    KIS   : R-Score(r_i) = 1(v_i == GT_v  and  id_i in [s, e])
    VQA   : R-Score(r_i) = 1(v_i == GT_v  and  id_i in [s, e]  and  a_i == GT_a)
    TRAKE : R-Score(r_i) = (1/N) * sum_j 1(id_{i,j} in [s_j, e_j])   if v_i == GT_v, else 0
    R@k   : max R-Score within the first k ranked rows, k in {1, 5, 20, 50, 100}
    Final : mean of the five R@k values (per query)
    Run   : mean (and sum) of the per-query Final scores

Ground-truth JSON format — ONE file per query set, keyed by the query filename
stem (``query-p{r}-{n}-{kis|qa|trake}``, see ``cvp.pipeline.run_queries``), which
is also the submission CSV stem. Canonical form:

    {
      "query-p1-1-kis":   {"task": "kis",   "video_id": "L01_V001",
                           "range": [500, 510]},
      "query-p1-2-qa":    {"task": "qa",    "video_id": "L05_V005",
                           "range": [800, 900],
                           "answers": ["màu xanh", "xanh dương"]},
      "query-p1-3-trake": {"task": "trake", "video_id": "L10_V010",
                           "moments": [[95, 105], [145, 155]]}
    }

Accepted equivalents (unified by :func:`load_ground_truth`, and understood by
the row scorers directly, so either format can be passed straight in):

* ``"frame_start": 500, "frame_end": 510`` — Core-Vision_HCMC-AI repo format.
* ``"center": 505, "epsilon": 5`` — organiser-style centre frame + epsilon
  (the GT window is the inclusive range ``[center-epsilon, center+epsilon]``).
* ``"ranges": [[500, 510], [900, 940], ...]`` — MULTIPLE acceptable windows in
  one video (KIS/QA row scores 1 when its frame lands inside ANY of them).
  Useful for AVS-style dev sets and for KIS answers that repeat in a video;
  each entry may also be a ``{"frame_start","frame_end"}`` or
  ``{"center","epsilon"}`` dict.
* TRAKE ``"events"``: list of ``{"frame_start", "frame_end"}`` dicts (HCMC-AI
  format), of ``[s, e]`` pairs, or of ``{"center", "epsilon"}`` dicts; or
  ``"centers": [c1..cN], "epsilon": eps`` for one shared epsilon.
* ``"answer": "màu xanh"`` — a single acceptable answer instead of ``answers``.
* AVS coverage: ``"targets": [{"video_id": "L01_V001", "range": [500, 510]},
  {"video_id": "L07_V003", "center": 900, "epsilon": 10}, ...]`` — one item per
  DISTINCT correct segment (possibly across videos). Entries with ``targets``
  are scored as coverage@k (fraction of targets hit within the first k rows)
  instead of the single-window max-R-Score — our closest offline analogue of
  the organisers' hidden AVS list.
* A missing ``"task"`` is inferred from the query stem (trake→avs→qa→kis
  substring priority, the same rule as ``cvp.pipeline.run_queries``).

Answer comparison is deliberately forgiving on formatting (casefold, whitespace
collapse, trailing punctuation) but Vietnamese diacritics are PRESERVED: the
organisers grade the Vietnamese text itself, so "mau xanh" != "màu xanh".

Everything here is pure stdlib (no numpy, no index required) so it stays
unit-testable offline and reusable inside notebooks.
"""

from __future__ import annotations

import csv
import logging
import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cvp.constants import ALL_TASKS, MAX_SUBMISSION_ROWS, TASK_KIS, TASK_QA, TASK_TRAKE

log = logging.getLogger(__name__)

# The five cutoffs averaged by the official Final Score.
K_VALUES: tuple[int, ...] = (1, 5, 20, 50, 100)

# Common spellings seen in hand-written GT files -> canonical task names.
# AVS / KIS-C / KIS-V submissions share the KIS row format (video_id,frame_idx),
# so they score under the KIS formula here.
_TASK_ALIASES = {
    "vqa": TASK_QA,
    "tkis": TASK_KIS,
    "kis-t": TASK_KIS,
    "kis-c": TASK_KIS,
    "kis-v": TASK_KIS,
    "kisv": TASK_KIS,
    "vkis": TASK_KIS,
    "video-kis": TASK_KIS,
    "textual-kis": TASK_KIS,
    "avs": TASK_KIS,
}

_WS_RE = re.compile(r"\s+")
_TRAILING_PUNCT = ".,;:!?…\"'`”’)]}»"


# ── Result dataclasses ───────────────────────────────────────────────────────


@dataclass
class QueryScore:
    """Official score of ONE query's ranked submission rows.

    Attributes:
        r_at: R@k for each official cutoff k in :data:`K_VALUES` (int keys).
        final: mean of the five R@k values — the per-query Final Score.
        task: normalised task the rows were scored under (kis|qa|trake).
        best_rank: 1-based rank of the first row attaining the best R-Score
            (None when every row scores 0).
        best_score: the best per-row R-Score in the (truncated) list.
        num_rows: number of rows actually scored (after the 100-row cap).
    """

    r_at: dict[int, float]
    final: float
    task: str = ""
    best_rank: int | None = None
    best_score: float = 0.0
    num_rows: int = 0

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe dict (R@k keys become strings)."""
        return {
            "task": self.task,
            "final": self.final,
            "r_at": {str(k): v for k, v in self.r_at.items()},
            "best_rank": self.best_rank,
            "best_score": self.best_score,
            "num_rows": self.num_rows,
        }


@dataclass
class RunReport:
    """Official scores for a whole submission folder against one GT file.

    ``mean_final`` / ``mean_r_at`` / ``by_task`` average over EVERY GT entry,
    so missing, malformed or unscorable submissions count as 0.0 in the
    denominator — exactly like skipping a query on the submission server.

    Attributes:
        per_query: successfully scored queries, keyed by CSV/query stem.
        mean_final: run score = mean per-query Final over all GT entries.
        by_task: mean Final per normalised task (over that task's GT entries).
        sum_final: sum of the scored queries' Final scores.
        mean_r_at: each R@k averaged over all GT entries (int keys).
        num_gt: number of GT entries (the official denominator).
        num_scored: number of successfully scored submissions.
        unscored: stem -> reason for every CSV that could not be scored.
        gt_without_submission: GT stems with no submission CSV at all.
    """

    per_query: dict[str, QueryScore]
    mean_final: float
    by_task: dict[str, float]
    sum_final: float = 0.0
    mean_r_at: dict[int, float] = field(default_factory=dict)
    num_gt: int = 0
    num_scored: int = 0
    unscored: dict[str, str] = field(default_factory=dict)
    gt_without_submission: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe dict (R@k keys become strings)."""
        return {
            "mean_final": self.mean_final,
            "sum_final": self.sum_final,
            "mean_r_at": {str(k): v for k, v in self.mean_r_at.items()},
            "by_task": dict(self.by_task),
            "num_gt": self.num_gt,
            "num_scored": self.num_scored,
            "per_query": {stem: qs.to_dict() for stem, qs in self.per_query.items()},
            "unscored": dict(self.unscored),
            "gt_without_submission": list(self.gt_without_submission),
        }


# ── Task / GT helpers ────────────────────────────────────────────────────────


def normalize_task(task: Any) -> str | None:
    """'VQA\\n' -> 'qa', ' TKIS ' -> 'kis'; None for anything not a known task.

    Never guesses: a GT entry whose task cannot be recognised must be surfaced
    (unscored / ValueError), not silently scored with the KIS formula.
    """
    t = str(task).strip().lower()
    t = _TASK_ALIASES.get(t, t)
    return t if t in ALL_TASKS else None


def infer_task(filename: str) -> str:
    """Task from a query/CSV filename — mirrors ``cvp.pipeline.run_queries.infer_task``.

    Substring priority trake -> avs -> qa -> kis; 'kis' when nothing matches.
    """
    stem = filename.lower()
    for task in ("trake", "avs", "qa", "kis"):
        if task in stem:
            return task
    return "kis"


def range_from_center(center: int, epsilon: int) -> dict[str, int]:
    """Centre frame + epsilon -> the inclusive GT window used everywhere here."""
    return {"frame_start": int(center) - int(epsilon), "frame_end": int(center) + int(epsilon)}


def trake_gt(video_id: str, centers: Sequence[int], epsilon: int) -> dict[str, Any]:
    """Build a TRAKE GT entry from the N event centre frames and one epsilon."""
    return {
        "task": TASK_TRAKE,
        "video_id": video_id,
        "events": [range_from_center(c, epsilon) for c in centers],
    }


def _entry_range(gt: Mapping[str, Any]) -> tuple[int, int] | None:
    """The KIS/QA window [s, e] from any accepted GT spelling (None if absent)."""
    try:
        if gt.get("frame_start") is not None and gt.get("frame_end") is not None:
            return int(gt["frame_start"]), int(gt["frame_end"])
        rng = gt.get("range")
        if isinstance(rng, Sequence) and not isinstance(rng, str) and len(rng) == 2:
            return int(rng[0]), int(rng[1])
        if gt.get("center") is not None and gt.get("epsilon") is not None:
            c, e = int(gt["center"]), int(gt["epsilon"])
            return c - e, c + e
    except (TypeError, ValueError):
        return None
    return None


def _entry_ranges(gt: Mapping[str, Any]) -> list[tuple[int, int]]:
    """ALL acceptable KIS/QA windows of one GT entry.

    ``"ranges": [[s1,e1], [s2,e2], ...]`` (pairs or per-window dicts) declares
    several acceptable segments in the video — a row scores when its frame
    lands inside ANY of them. Without ``ranges`` this degrades to the single
    :func:`_entry_range` window, so existing GT files behave exactly as before.
    Unparseable windows are dropped (they could only ever be misses).
    """
    raw = gt.get("ranges")
    if isinstance(raw, Sequence) and not isinstance(raw, str):
        out = [w for w in (_one_event(ev) for ev in raw) if w is not None]
        if out:
            return out
    single = _entry_range(gt)
    return [single] if single is not None else []


def _one_event(ev: Any) -> tuple[int, int] | None:
    """One TRAKE moment window from a dict / pair; None when unparseable."""
    try:
        if isinstance(ev, Mapping):
            if ev.get("frame_start") is not None and ev.get("frame_end") is not None:
                return int(ev["frame_start"]), int(ev["frame_end"])
            if ev.get("center") is not None and ev.get("epsilon") is not None:
                c, e = int(ev["center"]), int(ev["epsilon"])
                return c - e, c + e
        elif isinstance(ev, Sequence) and not isinstance(ev, str) and len(ev) == 2:
            return int(ev[0]), int(ev[1])
    except (TypeError, ValueError):
        return None
    return None


def _entry_events(gt: Mapping[str, Any]) -> list[tuple[int, int] | None]:
    """TRAKE per-moment windows from any accepted GT spelling.

    Unparseable moments stay in the list as None so the denominator is always
    the number of GT moments (they can only ever be misses).
    """
    # "segments" is the spelling scripts/21_tune_weights.py documents as an
    # accepted alias — the two scorers must agree on the GT dialect.
    raw = gt.get("events") or gt.get("moments") or gt.get("segments")
    if raw is None and gt.get("centers") is not None and gt.get("epsilon") is not None:
        try:
            eps = int(gt["epsilon"])
        except (TypeError, ValueError):
            return []
        out: list[tuple[int, int] | None] = []
        for c in gt["centers"]:
            try:
                out.append((int(c) - eps, int(c) + eps))
            except (TypeError, ValueError):
                out.append(None)
        return out
    if not isinstance(raw, Sequence) or isinstance(raw, str):
        return []
    return [_one_event(ev) for ev in raw]


def _entry_answers(gt: Mapping[str, Any]) -> list[str]:
    """All acceptable GT answers, normalised, empties dropped."""
    raw: Any = gt.get("answers")
    if raw is None:
        raw = gt.get("answer", "")
    if isinstance(raw, str) or not isinstance(raw, Sequence):
        raw = [raw]
    normalised = [normalize_answer(a) for a in raw]
    return [a for a in normalised if a]


def _entry_targets(entry: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Canonical AVS target list from a GT entry, or [] when absent.

    AVS coverage form (round-3 enhancement): the entry carries
    ``"targets": [{"video_id": ..., <any window spelling>}, ...]`` — one item
    per DISTINCT correct segment (they may live in different videos). Each
    target accepts the same window spellings as a KIS entry (``range``,
    ``ranges``, ``frame_start``/``frame_end``, ``center``+``epsilon``).
    Malformed targets raise (silent 0-scoring GT is operator error).
    """
    raw = entry.get("targets")
    if raw is None:
        return []
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or not raw:
        raise ValueError("'targets' must be a non-empty list of {video_id, window} objects")
    out: list[dict[str, Any]] = []
    for i, t in enumerate(raw):
        if not isinstance(t, Mapping) or not str(t.get("video_id", "")).strip():
            raise ValueError(f"targets[{i}] must be an object with a 'video_id'")
        ranges = _entry_ranges(t)
        if not ranges:
            raise ValueError(
                f"targets[{i}] has no usable frame window — need 'range', 'ranges', "
                "'frame_start'/'frame_end' or 'center'/'epsilon'"
            )
        declared = t.get("ranges")
        if isinstance(declared, Sequence) and not isinstance(declared, (str, bytes)) \
                and len(ranges) < len(declared):
            # A typo'd window would otherwise silently shrink the acceptance
            # region — same class of GT error as no window at all (R3-C4).
            raise ValueError(
                f"targets[{i}]: {len(declared) - len(ranges)} of {len(declared)} "
                "declared windows are unparseable — fix the GT entry"
            )
        out.append({"video_id": str(t["video_id"]).strip(),
                    "ranges": [[s, e] for s, e in ranges]})
    return out


def _canonical_entry(stem: str, entry: Mapping[str, Any]) -> dict[str, Any]:
    """Unify one GT entry into the canonical internal form (module docstring)."""
    e: dict[str, Any] = dict(entry)
    targets = _entry_targets(entry)
    if targets:
        e["targets"] = targets
    raw_task = entry.get("task")
    if raw_task is None or not str(raw_task).strip():
        e["task"] = normalize_task(infer_task(stem))
    else:
        # Keep an unknown task string visible so it is reported, never guessed.
        e["task"] = normalize_task(raw_task) or raw_task
    rng = _entry_range(entry)
    if rng is not None:
        e["frame_start"], e["frame_end"] = rng
    ranges = _entry_ranges(entry)
    if "ranges" in entry and ranges:  # normalise the multi-window spelling
        e["ranges"] = [[s, end] for s, end in ranges]
        if rng is None:  # keep the single-window keys usable for legacy readers
            e["frame_start"], e["frame_end"] = ranges[0]
    events = _entry_events(entry)
    if events:
        e["events"] = [
            {"frame_start": ev[0], "frame_end": ev[1]}
            if ev is not None
            else {"frame_start": None, "frame_end": None}
            for ev in events
        ]
    if "answers" not in e and "answer" in e:
        e["answers"] = [e["answer"]]
    return e


def load_ground_truth(path: str | Path) -> dict[str, dict[str, Any]]:
    """Read + unify a ground-truth JSON file (both accepted formats).

    Args:
        path: JSON file mapping query stems to GT entries (module docstring).

    Returns:
        ``{stem: canonical_entry}`` — every entry has a ``task`` (normalised
        when recognisable), ``frame_start``/``frame_end`` when a window is
        derivable, ``events`` as a list of window dicts, and ``answers`` list.

    Raises:
        FileNotFoundError: when the file does not exist.
        ValueError: when the JSON is not an object of objects, or a KIS/QA
            entry has no usable frame window (no ``range`` / ``ranges`` /
            ``frame_start``+``frame_end`` / ``center``+``epsilon``) — every
            submission would silently score 0 against it, and malformed GT is
            operator error that must be loud.
    """
    from cvp.utils.io import read_json  # local import keeps this module stdlib-light

    raw = read_json(path)
    if not isinstance(raw, Mapping):
        raise ValueError(f"GT file must be a JSON object keyed by query stem: {path}")
    out: dict[str, dict[str, Any]] = {}
    for stem, entry in raw.items():
        if not isinstance(entry, Mapping):
            raise ValueError(f"GT entry {stem!r} must be a JSON object, got {type(entry).__name__}")
        canonical = _canonical_entry(str(stem), entry)
        if (canonical.get("task") in (TASK_KIS, TASK_QA)
                and not _entry_ranges(canonical) and not canonical.get("targets")):
            # AVS coverage entries carry their windows inside `targets` instead.
            raise ValueError(
                f"GT entry {stem!r} has no usable frame window — need 'range', 'ranges', "
                "'frame_start'/'frame_end' or 'center'/'epsilon' (or AVS 'targets')"
            )
        if _task_for(str(stem), canonical) == TASK_TRAKE and not _entry_events(canonical):
            # Without this a TRAKE entry under an unknown key would load fine
            # and score EVERY submission 0.0 — malformed GT must be loud.
            raise ValueError(
                f"GT entry {stem!r} is TRAKE but has no usable events — need 'events', "
                "'moments', 'segments' or 'centers'+'epsilon'"
            )
        out[str(stem)] = canonical
    return out


# ── R-Score (per-row) ────────────────────────────────────────────────────────


def normalize_answer(answer: Any) -> str:
    """Casefold + collapse whitespace + drop trailing punctuation; accents kept."""
    s = unicodedata.normalize("NFC", str(answer)).casefold().strip()
    s = _WS_RE.sub(" ", s)
    return s.rstrip(_TRAILING_PUNCT + " ")


def _video_match(submitted: Any, gt_video_id: Any) -> bool:
    return str(submitted).strip() == str(gt_video_id).strip()


def _frame_in_range(frame: Any, start: Any, end: Any) -> bool:
    try:
        return int(start) <= int(str(frame).strip()) <= int(end)
    except (TypeError, ValueError):
        return False  # unparseable frame / missing range can only ever be a miss


def r_score_kis(row: Sequence[Any], gt: Mapping[str, Any]) -> int:
    """row = (video_id, frame_idx) -> 1 iff video matches AND frame in a GT window.

    A GT entry usually has ONE window ([s, e]); with ``"ranges"`` it may carry
    several acceptable windows — the frame only needs to land inside any one.
    """
    if len(row) < 2 or not _video_match(row[0], gt.get("video_id")):
        return 0
    return int(any(_frame_in_range(row[1], s, e) for s, e in _entry_ranges(gt)))


def r_score_qa(row: Sequence[Any], gt: Mapping[str, Any]) -> int:
    """row = (video_id, frame_idx, answer) -> 1 iff ALL THREE are right.

    The answer matches when it equals ANY of the acceptable GT answers after
    :func:`normalize_answer` (diacritics preserved).
    """
    if len(row) < 3 or not r_score_kis(row[:2], gt):
        return 0
    return int(normalize_answer(row[2]) in set(_entry_answers(gt)))


def r_score_trake(row: Sequence[Any], gt: Mapping[str, Any]) -> float:
    """row = (video_id, [f1..fN]) or flat (video_id, f1, ..., fN) -> [0, 1].

    0.0 for the wrong video; otherwise the fraction of the N GT moments whose
    submitted frame lands inside its own [s_j, e_j] window. Missing or
    unparseable frames simply count as misses — the denominator is always N.
    """
    if len(row) < 2 or not _video_match(row[0], gt.get("video_id")):
        return 0.0
    frames = row[1] if len(row) == 2 and isinstance(row[1], (list, tuple)) else row[1:]
    events = _entry_events(gt)
    if not events:
        return 0.0
    hits = sum(
        1
        for frame, ev in zip(frames, events)
        if ev is not None and _frame_in_range(frame, ev[0], ev[1])
    )
    return hits / len(events)


def _row_hits_target(row: Sequence[Any], target: Mapping[str, Any]) -> bool:
    """True when a (video, frame) row lands inside one AVS target's window.

    Accepts BOTH the canonical ``{"ranges": [[s,e],…]}`` form produced by
    :func:`_entry_targets` and the raw docstring spellings (``range``,
    ``frame_start``/``frame_end``, ``center``+``epsilon``) — a caller passing
    hand-written GT straight in must not get silent zeros (review R3-C3).
    """
    if len(row) < 2 or not _video_match(row[0], target.get("video_id")):
        return False
    return any(_frame_in_range(row[1], s, e) for s, e in _entry_ranges(target))


def coverage_at_k(rows: Sequence[Sequence[Any]],
                  targets: Sequence[Mapping[str, Any]], k: int) -> float:
    """AVS coverage@k = fraction of GT targets hit by ANY of the first k rows.

    The official 2025/2026 AVS formula is unpublished (the organisers score
    against a hidden multi-segment list); this is our closest offline
    analogue: it rewards COVERING many distinct correct segments — which the
    plain KIS proxy (max R-Score of a single window) cannot express.
    """
    if not targets:
        return 0.0
    head = list(rows[:k])
    hit = sum(1 for t in targets if any(_row_hits_target(r, t) for r in head))
    return hit / len(targets)


# ── Ranked-list score ────────────────────────────────────────────────────────


def r_at_k(scores: Sequence[float], k: int) -> float:
    """R@k = max R-Score within the first k ranked rows (0.0 for an empty list)."""
    top = scores[:k]
    return max(top) if top else 0.0


def final_score(scores: Sequence[float]) -> float:
    """Final Score = mean of R@k over k in {1, 5, 20, 50, 100}."""
    return sum(r_at_k(scores, k) for k in K_VALUES) / len(K_VALUES)


def _score_row(row: Sequence[str], task: str, gt: Mapping[str, Any]) -> float:
    if task == TASK_TRAKE:
        return r_score_trake(row, gt)
    if task == TASK_QA:
        # Tolerate stray commas in the answer column (our writer quotes them,
        # but hand-edited CSVs may not) by re-joining everything after the frame.
        merged = [row[0], row[1], ",".join(row[2:])] if len(row) >= 3 else row
        return float(r_score_qa(merged, gt))
    return float(r_score_kis(row, gt))


def _read_rows(csv_path: str | Path) -> list[list[str]]:
    """Headerless submission CSV -> list of raw string rows (blank lines skipped)."""
    with Path(csv_path).open("r", encoding="utf-8-sig", newline="") as f:
        return [row for row in csv.reader(f) if row and any(c.strip() for c in row)]


# ── Query-level scoring (new dataclass API) ──────────────────────────────────


def score_rows(task: str, rows: list[list[str]], gt: dict) -> QueryScore:
    """Score one query's ranked rows with the official formulas.

    Args:
        task: task name (aliases like 'VQA'/'tkis'/'avs' accepted).
        rows: ranked submission rows, best first, as string cells.
        gt: the query's GT entry (either accepted format, module docstring).

    Returns:
        A :class:`QueryScore`. Rows past the official 100-row budget are
        ignored, mirroring the submission server.

    Raises:
        ValueError: when the task is unrecognisable, or a QA entry has no
            non-empty acceptable answer — both would otherwise mis-score
            silently.
    """
    t = normalize_task(task)
    if t is None:
        raise ValueError(f"unknown task {task!r} (expected one of {ALL_TASKS})")
    if t == TASK_QA and not gt.get("targets") and not _entry_answers(gt):
        # Coverage entries never need answers (review R3-C2).
        raise ValueError("gt missing answer: QA entry needs a non-empty 'answer'/'answers'")
    if len(rows) > MAX_SUBMISSION_ROWS:
        log.warning("%d rows submitted; only the first %d are scored.", len(rows), MAX_SUBMISSION_ROWS)
        rows = rows[:MAX_SUBMISSION_ROWS]
    # AVS coverage form: a GT entry carrying `targets` is scored as coverage@k
    # over the DISTINCT correct segments (round-3 enhancement) — the plain
    # single-window path below cannot express "cover as many as possible".
    targets = _entry_targets(gt)
    if targets:
        best_rank = next((i + 1 for i, r in enumerate(rows)
                          if any(_row_hits_target(r, tg) for tg in targets)), None)
        cov = {k: coverage_at_k(rows, targets, k) for k in K_VALUES}
        return QueryScore(
            r_at=cov,
            final=sum(cov.values()) / len(K_VALUES),
            task="avs",
            best_rank=best_rank,
            best_score=cov[max(K_VALUES)],
            num_rows=len(rows),
        )
    scores = [_score_row(row, t, gt) for row in rows]
    best = max(scores) if scores else 0.0
    return QueryScore(
        r_at={k: r_at_k(scores, k) for k in K_VALUES},
        final=final_score(scores),
        task=t,
        best_rank=scores.index(best) + 1 if best > 0 else None,
        best_score=best,
        num_rows=len(scores),
    )


def score_csv(csv_path: str | Path, task: str, gt: dict) -> QueryScore:
    """Score ONE headerless submission CSV — see :func:`score_rows`."""
    return score_rows(task, _read_rows(csv_path), gt)


# ── Run-level scoring ────────────────────────────────────────────────────────


def score_run(submission_dir: Path, gt_path: Path | str | Mapping[str, Any]) -> RunReport:
    """Score every ``*.csv`` in ``submission_dir`` against the GT file.

    The task of each CSV is taken from its GT entry when present, otherwise
    inferred from the filename (trake→avs→qa→kis substring priority, the same
    rule as ``cvp.pipeline.run_queries``). One malformed CSV never sinks the
    report — it lands in ``unscored`` and counts 0.0 in the run means.

    Args:
        submission_dir: folder of ``{query-stem}.csv`` submission files.
        gt_path: ground-truth JSON path, or an already-loaded GT mapping.

    Returns:
        A :class:`RunReport` (per-query scores + official-style run means).

    Raises:
        FileNotFoundError: when ``submission_dir`` does not exist — a typo
            must not read as "your run scored 0.0".
    """
    submission_dir = Path(submission_dir)
    if not submission_dir.is_dir():
        raise FileNotFoundError(f"submissions dir not found: {submission_dir}")
    if isinstance(gt_path, Mapping):
        gt_map = {
            str(stem): _canonical_entry(str(stem), entry)
            for stem, entry in gt_path.items()
            if isinstance(entry, Mapping)
        }
    else:
        gt_map = load_ground_truth(gt_path)

    csv_paths = sorted(submission_dir.glob("*.csv"))
    if not csv_paths and gt_map:
        log.warning("No *.csv in %s but GT has %d entries.", submission_dir, len(gt_map))

    per_query: dict[str, QueryScore] = {}
    unscored: dict[str, str] = {}
    for csv_path in csv_paths:
        stem = csv_path.stem
        entry = gt_map.get(stem)
        if entry is None:
            unscored[stem] = "no ground-truth entry"
            continue
        task = _task_for(stem, entry)
        if task is None:
            unscored[stem] = f"unknown task {entry.get('task')!r}"
            continue
        # Coverage entries carry their own windows and ignore answers — the
        # QA-answer gate must not reject them (review R3-C2).
        if task == TASK_QA and not entry.get("targets") and not _entry_answers(entry):
            unscored[stem] = "gt missing answer"
            continue
        try:
            per_query[stem] = score_csv(csv_path, task, entry)
        except Exception as e:  # noqa: BLE001 — one malformed CSV must not sink the report
            log.warning("Failed to score %s: %s", csv_path, e)
            unscored[stem] = f"error: {e}"

    n_gt = len(gt_map)
    sum_final = sum(qs.final for qs in per_query.values())
    mean_final = sum_final / n_gt if n_gt else 0.0
    mean_r_at = {
        k: (sum(qs.r_at[k] for qs in per_query.values()) / n_gt) if n_gt else 0.0
        for k in K_VALUES
    }
    # Coverage entries group under their OWN task ("avs") so the by_task table
    # matches per_query[stem].task and never contaminates the [kis] mean
    # (review R3-C1: normalize_task aliases avs→kis, so keying off _task_for
    # alone could never produce the [avs] row the docs promise).
    task_of_gt = {
        stem: ("avs" if entry.get("targets") else (_task_for(stem, entry) or "unknown"))
        for stem, entry in gt_map.items()
    }
    by_task: dict[str, float] = {}
    for t in sorted(set(task_of_gt.values())):
        stems = [s for s, tt in task_of_gt.items() if tt == t]
        by_task[t] = sum(per_query[s].final for s in stems if s in per_query) / len(stems)

    report = RunReport(
        per_query=per_query,
        mean_final=mean_final,
        by_task=by_task,
        sum_final=sum_final,
        mean_r_at=mean_r_at,
        num_gt=n_gt,
        num_scored=len(per_query),
        unscored=unscored,
        gt_without_submission=sorted(s for s in gt_map if s not in per_query and s not in unscored),
    )
    log.info(
        "Scored %d/%d GT queries -> run mean final %.4f (sum %.4f)",
        report.num_scored, report.num_gt, report.mean_final, report.sum_final,
    )
    return report


def _task_for(stem: str, entry: Mapping[str, Any]) -> str | None:
    """Task of one GT entry: explicit task first, else inferred from the stem."""
    raw = entry.get("task")
    if raw is None or not str(raw).strip():
        return normalize_task(infer_task(stem))
    return normalize_task(raw)


# ── Submission validation (mirrors cvp.submission.writer's contract) ────────


def validate_rows(task: str, rows: list[list[str]]) -> list[str]:
    """Structural validation of parsed CSV rows against the organiser contract.

    Reuses the row validators of ``cvp.submission.writer`` (read-only import),
    so a CSV that passes here is one the writer could have produced: valid
    video ids, non-negative integer frames, per-task column counts, strictly
    increasing TRAKE frames, and at most 100 rows.

    Returns:
        A list of human-readable problems — empty means the rows are valid.
    """
    from cvp.submission.writer import _validate_frame_idx, _validate_video_id

    t = normalize_task(task) or TASK_KIS
    problems: list[str] = []
    if len(rows) > MAX_SUBMISSION_ROWS:
        problems.append(f"{len(rows)} rows exceed the {MAX_SUBMISSION_ROWS}-row limit")
    for i, row in enumerate(rows, start=1):
        if t == TASK_QA and len(row) < 3:
            problems.append(f"row {i}: QA needs video_id,frame_idx,answer ({len(row)} cols)")
            continue
        if t == TASK_TRAKE and len(row) < 2:
            problems.append(f"row {i}: TRAKE needs video_id,f1[,f2...] ({len(row)} cols)")
            continue
        if t == TASK_KIS and len(row) != 2:
            problems.append(f"row {i}: KIS needs exactly video_id,frame_idx ({len(row)} cols)")
            continue
        try:
            _validate_video_id(str(row[0]).strip())
        except ValueError as e:
            problems.append(f"row {i}: {e}")
        frame_cells = row[1:] if t == TASK_TRAKE else [row[1]]
        frames: list[int] = []
        for cell in frame_cells:
            try:
                frames.append(_validate_frame_idx(int(str(cell).strip())))
            except (TypeError, ValueError):
                problems.append(f"row {i}: bad frame_idx {cell!r}")
        if t == TASK_TRAKE and len(frames) == len(frame_cells):
            if any(b <= a for a, b in zip(frames, frames[1:])):
                problems.append(f"row {i}: TRAKE frames not strictly increasing: {frames}")
    return problems


def validate_csv(csv_path: str | Path, task: str | None = None) -> list[str]:
    """Validate one submission CSV; the task defaults to filename inference."""
    path = Path(csv_path)
    t = normalize_task(task) if task is not None else None
    if t is None:
        t = normalize_task(infer_task(path.name)) or TASK_KIS
    try:
        rows = _read_rows(path)
    except Exception as e:  # noqa: BLE001 — unreadable file is itself the finding
        return [f"unreadable: {e}"]
    return validate_rows(t, rows)


# ── Legacy dict-shaped API (ported from Core-Vision_HCMC-AI) ─────────────────


def score_submission_csv(csv_path: str | Path, gt_entry: Mapping[str, Any]) -> dict[str, Any]:
    """Score ONE headerless submission CSV against its GT entry (dict result).

    Returns ``{"final", "r_at_k": {str(k): R@k}, "best_rank", "best_score",
    "num_rows"}``. ``best_rank`` is the 1-based rank of the first row attaining
    the best R-Score (None when every row scores 0). Rows past the official
    100-row budget are ignored, mirroring the submission server.

    Raises:
        ValueError: when the GT entry's task is unrecognisable (after
            stripping/case-folding and the usual aliases) or when a QA entry
            lacks a non-empty answer — both would otherwise mis-score silently.
    """
    # Task-less entries are inferred from the CSV stem (module docstring),
    # exactly like score_run — defaulting to KIS here scored wrong-answer QA
    # rows as hits and every TRAKE entry as 0. An EXPLICIT but unrecognisable
    # task still fails loud inside score_csv.
    raw_task = gt_entry.get("task")
    if raw_task is None or not str(raw_task).strip():
        raw_task = normalize_task(infer_task(Path(csv_path).name)) or TASK_KIS
    qs = score_csv(csv_path, raw_task, dict(gt_entry))
    return {
        "final": qs.final,
        "r_at_k": {str(k): v for k, v in qs.r_at.items()},
        "best_rank": qs.best_rank,
        "best_score": qs.best_score,
        "num_rows": qs.num_rows,
    }


def score_submission_dir(
    submissions_dir: str | Path,
    gt: str | Path | Mapping[str, Any],
) -> dict[str, Any]:
    """Score every ``*.csv`` in ``submissions_dir`` against the GT JSON (dict result).

    ``gt`` is a path to the ground-truth JSON (module docstring) or an
    already-loaded mapping. CSVs whose stem has no GT entry are reported with
    ``status="unscored"`` — a partially-labelled query set must never crash the
    harness. GT keys with no CSV are listed under ``gt_without_submission``.

    ``macro`` is the official-style headline: ``final`` and each R@k averaged
    over EVERY GT entry, so missing, malformed or unscorable submissions count
    as 0.0 in the denominator (exactly like skipping a query on the server).
    ``macro_scored`` averages over the successfully scored queries only, as a
    diagnostic. Raises ``FileNotFoundError`` when ``submissions_dir`` does not
    exist — a typo must not read as "your run scored 0.0".
    """
    submissions_dir = Path(submissions_dir)
    if not submissions_dir.is_dir():
        raise FileNotFoundError(f"submissions dir not found: {submissions_dir}")
    gt_map: Mapping[str, Any]
    if isinstance(gt, Mapping):
        gt_map = gt
    else:
        # Same loud canonicalization as score_run — a raw read here used to
        # skip the no-usable-window/no-events validation entirely.
        gt_map = load_ground_truth(Path(gt))
    csv_paths = sorted(submissions_dir.glob("*.csv"))
    if not csv_paths and gt_map:
        log.warning("No *.csv in %s but GT has %d entries.", submissions_dir, len(gt_map))
    per_query: dict[str, dict[str, Any]] = {}
    scored: list[dict[str, Any]] = []
    for csv_path in csv_paths:
        stem = csv_path.stem
        entry = gt_map.get(stem)
        if not isinstance(entry, Mapping):
            per_query[stem] = {"status": "unscored", "reason": "no ground-truth entry"}
            continue
        task = _task_for(stem, entry)
        if task is None:
            per_query[stem] = {"status": "unscored",
                               "reason": f"unknown task {entry.get('task')!r}"}
            continue
        if task == TASK_QA and not _entry_answers(entry):
            per_query[stem] = {"status": "unscored", "reason": "gt missing answer"}
            continue
        try:
            res = score_submission_csv(csv_path, entry)
        except Exception as e:  # noqa: BLE001 — one malformed CSV must not sink the report
            log.warning("Failed to score %s: %s", csv_path, e)
            per_query[stem] = {"status": "unscored", "reason": f"error: {e}"}
            continue
        res.update(status="scored", task=task)
        per_query[stem] = res
        scored.append(res)
    n, n_gt = len(scored), len(gt_map)

    def _macro(denom: int) -> dict[str, Any]:
        return {
            "final": (sum(r["final"] for r in scored) / denom) if denom else 0.0,
            "r_at_k": {
                str(k): (sum(r["r_at_k"][str(k)] for r in scored) / denom) if denom else 0.0
                for k in K_VALUES
            },
        }

    report = {
        "per_query": per_query,
        "macro": _macro(n_gt),          # official-style: every GT query in the denominator
        "macro_scored": _macro(n),      # diagnostic: scored queries only
        "num_gt": n_gt,
        "num_scored": n,
        "num_unscored": len(per_query) - n,
        "gt_without_submission": sorted(k for k in gt_map if k not in per_query),
    }
    log.info(
        "Scored %d/%d GT queries -> macro final %.4f (scored-only %.4f)",
        n, n_gt, report["macro"]["final"], report["macro_scored"]["final"],
    )
    return report
