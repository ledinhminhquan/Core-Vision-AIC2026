"""Batch query runner — turns an organiser query pack into submission CSVs.

Query packs are folders of ``.txt`` files (one query each). The task is read
from the filename, matching the organiser convention (…-kis.txt, …-qa.txt,
…-trake.txt; ``avs`` also accepted). BOTH observed organiser layouts are
understood (verified against the real AIC-2025 finals packs):

    query-p1-1-kis.txt    one line: the scene description
    query-p1-2-qa.txt     line 1: scene description, line 2: the question —
                          OR the 2025 single-line form
                          "<mô tả cảnh> … Hỏi <câu hỏi>?" (split automatically)
    query-p1-3-trake.txt  one line per event, in order — OR the 2025 form
                          with a context/header line followed by
                          "E1: …" … "Ek: …" prefixed events (header dropped,
                          prefixes stripped; see :func:`parse_trake_events`)

Each query yields ``{stem}.csv`` in the output folder, ready for Codabench.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Sequence

from cvf.config import Settings, VqaCfg
from cvf.constants import MAX_QA_ANSWER_CHARS
from cvf.search.engine import SearchEngine, SearchResult
from cvf.search.vqa import VqaAssistant
from cvf.submission.packager import infer_task  # single source for task-from-filename
from cvf.submission.writer import write_kis, write_qa, write_trake

log = logging.getLogger(__name__)

# Placeholder for QA rows when no VQA answer is available: scores 0 exactly
# like an empty string, but can never block packaging or trip format checks.
QA_FALLBACK_ANSWER = "không rõ"

# "E1:" / "e2." / "E3)" / "E4 -" event prefixes of the organiser TRAKE files.
_EVENT_RE = re.compile(r"^\s*[Ee](\d{1,2})\s*[:.)\-]\s*")
# The question of a single-line organiser QA file starts at the LAST standalone
# "Hỏi" / "Câu hỏi" (2025 packs: "<mô tả> … Hỏi xã này có tên là gì?").
# CAPITAL-initial only (round-4 fix): the organiser marker always starts a
# sentence, while lowercase "hỏi" is an ordinary verb inside descriptions
# ("dừng lại hỏi đường cảnh sát…") — a case-insensitive match truncated the
# retrieval text at such verbs whenever no real marker followed.
_QA_SPLIT_RE = re.compile(r"(?<!\w)(Hỏi|HỎI|Câu hỏi|Câu Hỏi|CÂU HỎI)\b\s*[:,]?\s*")

__all__ = [
    "QA_FALLBACK_ANSWER", "infer_task", "parse_query_lines", "parse_trake_events",
    "split_qa_line", "group_candidates", "compute_qa_answers", "run_query_file",
    "run_query_folder",
]


def parse_trake_events(lines: list[str], *, prepend_context: bool = False) -> list[str]:
    """Normalize a TRAKE query file's lines into the ordered event list.

    The REAL organiser packs (verified on the AIC-2025 finals sets) write one
    context/header line followed by ``E1:``…``Ek:``-prefixed events::

        Đoạn video múa lân …, tìm các sự kiện sau:
        E1: Lân quay vòng trên cột …
        E2: Khoảnh khắc 4 chân chạm đất …

    Feeding every line to the engine would make the header a phantom event #1
    and shift EVERY submitted frame one position against the ground truth
    (TRAKE would score ≈0). Rule: when ≥2 lines carry an ``En``-prefix, ONLY
    those lines are events (header dropped, prefixes stripped); otherwise the
    lines are already one-event-per-line (``queries/example`` format) and are
    returned unchanged.

    Args:
        lines: non-empty stripped lines of the query file, in order.
        prepend_context: when True (``temporal.event_context: prepend``), the
            dropped header is prepended to every event text — the header often
            carries video-level context ("một con lân màu vàng đen trắng")
            that helps video pooling; off by default (A/B-able via config).
    """
    prefixed = [(i, _EVENT_RE.sub("", ln).strip()) for i, ln in enumerate(lines)
                if _EVENT_RE.match(ln)]
    if len(prefixed) < 2:
        return lines
    events = [text for _i, text in prefixed if text]
    if prepend_context:
        first_event_line = prefixed[0][0]
        header = " ".join(lines[:first_event_line]).rstrip(" :.").strip()
        if header:
            events = [f"{header}. {ev}" for ev in events]
    return events


def split_qa_line(line: str) -> tuple[str, str]:
    """Split a single-line organiser QA query into (description, question).

    The 2025 packs put both on one line: ``"<mô tả cảnh> … Hỏi <câu hỏi>?"``.
    Splitting keeps the interrogative tail out of the dense-search text (it is
    visual noise for CLIP-style encoders) while handing the full question to
    the VQA model. The LAST standalone CAPITAL-initial "Hỏi"/"Câu hỏi" wins:
    lowercase "hỏi" is an ordinary verb ("học hỏi", "hỏi đường") and never
    splits (round-4 fix — the verb used to truncate marker-less descriptions).
    When no marker is found the whole line serves as both (previous behaviour).
    """
    matches = list(_QA_SPLIT_RE.finditer(line))
    if matches:
        m = matches[-1]
        description = line[: m.start()].rstrip(" .,;:").strip()
        question = line[m.start():].strip()
        if description and question:
            return description, question
    return line, line


def parse_query_lines(task: str, lines: list[str]) -> tuple[str, str | None]:
    """Normalise a KIS/AVS/QA query file into ``(retrieval_text, question)``.

    Real 2025 packs are messier than one-line-per-file (round-3 finding):

    * KIS/AVS can span SEVERAL paragraphs — the discriminative detail often
      lives in paragraph 2+ (e.g. finals ``query-p1-11-kis.txt``), so ALL
      lines are joined for retrieval (BM25/OCR/metadata get the full text;
      the Gemini enhancement condenses it for CLIP-style towers).
    * QA can be 1 line (marker form), 2 lines (description + question) or
      ≥3 lines (multi-paragraph description + the question on the LAST line
      — finals ``query-p2-3-qa.txt``). The question is the last line when it
      looks interrogative; otherwise the marker split runs on the full text.

    ``question`` is None for non-QA tasks. TRAKE is handled separately by
    :func:`parse_trake_events`.
    """
    if task != "qa":
        return " ".join(lines), None
    if len(lines) == 1:
        description, question = split_qa_line(lines[0])
        return description, question
    tail = lines[-1]
    if "?" in tail or _QA_SPLIT_RE.search(tail):
        return " ".join(lines[:-1]), tail
    # No interrogative last line — fall back to the marker split on the
    # whole text (some packs wrap a single logical line).
    return split_qa_line(" ".join(lines))


def _time_of(result: SearchResult) -> float | None:
    """Seconds-into-video for a result (duck-typed for test stubs)."""
    ref = getattr(result, "ref", None)
    pts = getattr(ref, "pts_time", None) if ref is not None else getattr(result, "pts_time", None)
    return float(pts) if pts is not None else None


def group_candidates(results: Sequence[SearchResult], gap_s: float = 10.0,
                     max_frame_gap: int = 250) -> list[list[int]]:
    """Bucket a ranked result list into shot-ish groups.

    Two candidates share a group iff they come from the same video and are
    close in time: walking the video's candidates in frame order, a new group
    starts whenever the time gap exceeds ``gap_s`` seconds OR the original
    frame gap exceeds ``max_frame_gap`` frames (≈10s at 25fps when pts is
    unavailable).

    Returns groups of ROW INDICES into ``results``; groups are ordered by
    their best (lowest) rank, members within a group likewise.
    """
    by_video: dict[str, list[int]] = {}
    for i, r in enumerate(results):
        by_video.setdefault(r.video_id, []).append(i)

    groups: list[list[int]] = []
    for idxs in by_video.values():
        ordered = sorted(idxs, key=lambda i: int(results[i].frame_idx))
        current = [ordered[0]]
        for prev, cur in zip(ordered, ordered[1:]):
            t_prev, t_cur = _time_of(results[prev]), _time_of(results[cur])
            time_split = t_prev is not None and t_cur is not None and (t_cur - t_prev) > gap_s
            frame_split = int(results[cur].frame_idx) - int(results[prev].frame_idx) > max_frame_gap
            if time_split or frame_split:
                groups.append(current)
                current = [cur]
            else:
                current.append(cur)
        groups.append(current)

    for g in groups:
        g.sort()  # members best-rank first
    groups.sort(key=lambda g: g[0])  # groups best-rank first
    return groups


def compute_qa_answers(results: Sequence[SearchResult], question: str,
                       vqa: VqaAssistant | None, settings: Settings | None = None) -> list[str]:
    """Per-row QA answers: one VQA call per top candidate group.

    VQA R-Score needs the *right row* to carry the right answer, so instead
    of duplicating one global answer everywhere we bucket the ranking into
    shot-ish groups (``group_candidates``), VQA the top
    ``settings.vqa.answers_per_query`` groups once each (bounded by
    ``settings.vqa.max_calls_per_query``), and stamp each group's answer on
    that group's rows. Rows in unanswered groups inherit the best group's
    answer as fallback. Without a provider (``vqa is None``), or when every
    VQA call fails, rows carry :data:`QA_FALLBACK_ANSWER` instead of "" — it
    scores 0 either way but can never block packaging.
    """
    answers = [""] * len(results)
    if not results:
        return answers
    if vqa is None:
        return [QA_FALLBACK_ANSWER] * len(results)
    cfg = settings.vqa if settings is not None else VqaCfg()
    budget = max(0, min(int(cfg.answers_per_query), int(cfg.max_calls_per_query)))
    groups = group_candidates(results)
    fallback = ""
    for group in groups[:budget]:
        best = group[0]
        ans = ""
        try:
            r = results[best]
            suggestions = vqa.suggest(question, [(r.global_id, r.ref.path)])
            if suggestions:
                ans = str(suggestions[0].answer)[:MAX_QA_ANSWER_CHARS]
        except Exception as e:  # noqa: BLE001 — one failed group must not sink the query
            log.warning("VQA failed for group at rank %d: %s", best + 1, e)
        for i in group:
            answers[i] = ans
        if not fallback and ans:
            fallback = ans
    if fallback:
        answers = [a or fallback for a in answers]
    return [a or QA_FALLBACK_ANSWER for a in answers]


def run_query_file(engine: SearchEngine, path: Path, out_dir: Path,
                   vqa: VqaAssistant | None = None,
                   top1_times: dict[str, list[float]] | None = None) -> Path | None:
    """Run one query file → ``{stem}.csv`` (None for an empty query file).

    When ``top1_times`` is given, the top-1 candidate's pts times are stored
    under the CSV stem (seconds; one per TRAKE event, a single element for
    KIS/QA/AVS) — the DRES client submits millisecond timestamps, not frames.
    """
    task = infer_task(path.name)
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8-sig").splitlines() if ln.strip()]
    if not lines:
        log.warning("Empty query file: %s", path.name)
        return None
    out_path = out_dir / f"{path.stem}.csv"

    def record(times: Sequence[float] | None) -> None:
        if top1_times is not None and times:
            top1_times[path.stem] = [float(t) for t in times]

    if task == "trake":
        # Organiser 2025 format: context/header line + "E1:…" events — the
        # header must NOT become a phantom event (see parse_trake_events).
        temporal_cfg = getattr(getattr(engine, "settings", None), "temporal", None)
        prepend = getattr(temporal_cfg, "event_context", "none") == "prepend"
        events = parse_trake_events(lines, prepend_context=prepend)
        candidates = engine.search_trake(events)
        if candidates:
            record(getattr(candidates[0], "pts_times", None))
        return write_trake(out_path, [(c.video_id, c.frame_idxs) for c in candidates])

    # Round-3 fix: multi-paragraph KIS/AVS joins ALL lines; QA question = last
    # interrogative line (real finals files have ≥3 lines). See parse_query_lines.
    retrieval_text, question = parse_query_lines(task, lines)
    if task == "avs":
        results = engine.search_avs(retrieval_text)
    else:
        results = engine.search_text(retrieval_text)

    if results:
        t = _time_of(results[0])
        record([t] if t is not None else None)
    if task == "qa":
        answers = compute_qa_answers(results, question, vqa, getattr(engine, "settings", None))
        return write_qa(out_path, [(r.video_id, r.frame_idx, a) for r, a in zip(results, answers)])
    return write_kis(out_path, [(r.video_id, r.frame_idx) for r in results])


def run_query_folder(settings: Settings, query_dir: Path, out_dir: Path,
                     with_vqa: bool = True) -> list[Path]:
    engine = SearchEngine(settings)
    vqa = None
    if with_vqa:
        try:
            vqa = VqaAssistant(settings)
        except Exception as e:  # noqa: BLE001
            log.warning("VQA assistant unavailable: %s", e)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for qf in sorted(query_dir.glob("*.txt")):
        try:
            p = run_query_file(engine, qf, out_dir, vqa)
        except Exception as e:  # noqa: BLE001 — one bad query must not stop the pack
            log.error("Query %s failed: %s", qf.name, e)
            continue
        if p:
            written.append(p)
            log.info("%s → %s", qf.name, p.name)
    return written
