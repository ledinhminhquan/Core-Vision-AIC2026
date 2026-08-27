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

import csv
import logging
import re
import unicodedata
from pathlib import Path
from typing import Sequence

from cvp.config import Settings, VqaCfg
from cvp.constants import MAX_QA_ANSWER_CHARS
from cvp.search.engine import SearchEngine, SearchResult
from cvp.search.vqa import VqaAssistant
from cvp.submission.packager import infer_task  # single source for task-from-filename
from cvp.submission.writer import write_kis, write_qa, write_trake

log = logging.getLogger(__name__)


def strip_invisible(text: str) -> str:
    """Remove Unicode format (Cf) characters — zero-width spaces, stray BOMs,
    RTL marks. Organiser text copy-pasted from PDF/Word/chat routinely carries
    them, and they are invisible in every editor while breaking prefix regexes
    (``\\s`` does not match Cf). Round-9 chaos-lens fix."""
    return "".join(c for c in text if unicodedata.category(c) != "Cf")


def load_query_lines(path: Path) -> list[str]:
    """THE round-time query-file loader — the single source used by
    run_query_file, scripts/51 (warm-cache) and scripts/23 (signal dumps).

    Gemini cache keys and tuning signals must be BYTE-IDENTICAL to what the
    engine sees at round time (round-2 C1); round-10 found the two scripts had
    drifted from the round-9 strip_invisible fix — sharing one loader makes
    divergence impossible.
    """
    lines = [strip_invisible(ln).strip()
             for ln in Path(path).read_text(encoding="utf-8-sig").splitlines()]
    return [ln for ln in lines if ln]

# Placeholder for QA rows when no VQA answer is available: scores 0 exactly
# like an empty string, but can never block packaging or trip format checks.
QA_FALLBACK_ANSWER = "không rõ"

# "E1:" / "e2." / "E3)" / "E4 -" / "E1 <text>" event prefixes of the organiser
# TRAKE files. Round-39 (LIVE round-1 loss, 21/08/2026): the real
# query-p1-16-trake.txt wrote "E1 Khoảnh khắc…" with NO separator — the old
# separator-required regex matched nothing, the header became phantom event #1,
# every row carried 4 frames against 3 events, and BTC rejected the whole file
# ("Expected 3 frame IDs, got 4" ×100). Bare whitespace now counts as the
# separator; an event line that is just "E1" still yields empty text and is
# dropped by the caller.
_EVENT_RE = re.compile(r"^\s*[Ee](\d{1,2})\s*(?:[:.)\-]\s*|\s+)")
# The question of a single-line organiser QA file starts at the LAST standalone
# "Hỏi" / "Câu hỏi" (2025 packs: "<mô tả> … Hỏi xã này có tên là gì?").
# CAPITAL-initial only (round-4 fix): the organiser marker always starts a
# sentence, while lowercase "hỏi" is an ordinary verb inside descriptions
# ("dừng lại hỏi đường cảnh sát…") — a case-insensitive match truncated the
# retrieval text at such verbs whenever no real marker followed.
_QA_SPLIT_RE = re.compile(r"(?<!\w)(Hỏi|HỎI|Câu hỏi|Câu Hỏi|CÂU HỎI)\b\s*[:,]?\s*")
# Imperative question openers of real organiser QA files (round-5 fix M-R5-1):
# finals ``query-p2-3-qa.txt`` ends "Hảy cho biết…" — no "?" and the organiser's
# own typo ``Hảy`` for ``Hãy`` — so the last line must also count as a question
# when it starts like a command. Kept CAPITAL-initial for the same reason as
# ``_QA_SPLIT_RE`` (sentence starts only, never mid-sentence verbs).
_IMPERATIVE_Q_RE = re.compile(r"^\s*(Hãy|Hảy|Cho biết|Đếm|Kể tên)\b")
# Sentence boundary used by the marker-less single-line QA split (L-R5-2).
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

__all__ = [
    "QA_FALLBACK_ANSWER", "infer_task", "parse_query_lines", "parse_trake_events",
    "split_trake_query", "split_qa_line", "group_candidates", "compute_qa_answers",
    "run_query_file", "run_query_folder", "ranking_confidence", "rrf_merge_results",
    "maybe_retry_low_confidence",
]


def split_trake_query(lines: list[str]) -> tuple[str, list[str]]:
    """(header, events) of a TRAKE query file; header is "" for the plain
    one-event-per-line format.

    The single splitting logic behind :func:`parse_trake_events` — also used
    directly by the ``temporal.pool_context: prepend`` path, which needs the
    header SEPARATELY (video pooling searches "<header>. <event>" while the
    DP alignment keeps the bare event texts).
    """
    # Invisible Cf chars (zero-width space, stray mid-file BOM, RTL marks) ride
    # along when organiser text is copy-pasted from PDF/Word/chat. Python's \s
    # does NOT match them, so an 'E1:' line with a leading U+200B would fail
    # the prefix match and the event would silently VANISH — the CSV stays
    # structurally valid and the query scores ~0 (round-9 chaos lens).
    lines = [strip_invisible(ln) for ln in lines]
    prefixed = [(i, _EVENT_RE.sub("", ln).strip()) for i, ln in enumerate(lines)
                if _EVENT_RE.match(ln)]
    if len(prefixed) < 2:
        return "", lines
    events = [text for _i, text in prefixed if text]
    header = " ".join(lines[: prefixed[0][0]]).rstrip(" :.").strip()
    return header, events


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
    header, events = split_trake_query(lines)
    if prepend_context and header:
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
    Marker-less lines fall back to a sentence-boundary split when the LAST
    sentence looks like the question (ends "?" or opens Hãy/Hảy/Cho biết/…,
    round-5 fix L-R5-2); otherwise the whole line serves as both.
    """
    matches = list(_QA_SPLIT_RE.finditer(line))
    if matches:
        m = matches[-1]
        description = line[: m.start()].rstrip(" .,;:").strip()
        question = line[m.start():].strip()
        if description and question:
            return description, question
    # Marker-less form (round-5 fix L-R5-2): 4/8 real single-line QA files end
    # in a bare question sentence ("… là gì?", "… hãy cho biết …?"). Split at
    # the LAST sentence boundary when the final sentence looks interrogative
    # (ends with "?" or opens like a command); otherwise keep passthrough.
    sentences = _SENTENCE_SPLIT_RE.split(line.strip())
    if len(sentences) >= 2:
        tail = sentences[-1].strip()
        if tail.endswith("?") or _IMPERATIVE_Q_RE.match(tail):
            description = " ".join(s.strip() for s in sentences[:-1]).strip()
            if description and tail:
                return description, tail
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
    if "?" in tail or _QA_SPLIT_RE.search(tail) or _IMPERATIVE_Q_RE.match(tail):
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


def ranking_confidence(scores: Sequence[float]) -> float:
    """How separated the top of a ranking is from its bulk, in [0, 1].

    ``(s1 − median) / |s1|``: → ~1 when the top score towers over a flat tail,
    → ~0 when the whole list is one indistinguishable plateau (classic sign the
    query text failed to discriminate). Short or degenerate lists count as
    LOW confidence — with <10 candidates a retry can only help. The input is
    sorted internally, so callers may pass rankings whose display order was
    reshuffled by a reranker without updating ``.score`` (review finding C20).
    """
    s = sorted((float(x) for x in scores), reverse=True)
    if len(s) < 10:
        return 0.0
    top, mid = s[0], s[len(s) // 2]
    # Gap relative to the score MAGNITUDE (fused scores are ≥0): a plateau of
    # near-identical values scores ~0 regardless of where its median sits —
    # a (top−med)/(top−min) form would miss exactly that failure mode.
    return max(0.0, min(1.0, (top - mid) / (abs(top) + 1e-12)))


def rrf_merge_results(rankings: Sequence[Sequence[SearchResult]],
                      k: int = 60, limit: int | None = None) -> list[SearchResult]:
    """Reciprocal-rank-fuse several SearchResult lists (first list is primary).

    Duplicate global_ids keep the FIRST list's result object (its signals are
    the ones the operator/tooling has already seen).
    """
    fused: dict[int, float] = {}
    keep: dict[int, SearchResult] = {}
    for ranking in rankings:
        for rank, r in enumerate(ranking):
            gid = r.ref.global_id if hasattr(r, "ref") else r.global_id
            fused[gid] = fused.get(gid, 0.0) + 1.0 / (k + rank + 1)
            keep.setdefault(gid, r)
    order = sorted(fused, key=lambda g: -fused[g])
    out = [keep[g] for g in order]
    return out[:limit] if limit else out


def maybe_retry_low_confidence(engine: SearchEngine, retrieval_text: str,
                               results: list[SearchResult]) -> list[SearchResult]:
    """Auto-track upgrade: reformulate-and-merge when the ranking looks flat.

    Off by default (``search.low_confidence_retry``). When the confidence of
    the initial ranking is below the threshold, re-search the processor's
    cached enhanced/expansion texts VERBATIM via ``engine.search_prepared``
    (no extra Gemini QUERY calls: reading the cache is free and the alts skip
    the query processor entirely) and RRF-merge the rankings, primary first.
    Cost when triggered: up to 3 extra dense searches — the alts also SKIP the
    optional cross/VLM rerank stack (round-10: running it per alt multiplied
    reranker latency and Gemini quota 4×; the RRF rank-merge gains nothing
    from per-alt head reordering). Every failure path returns the original
    ranking.
    """
    settings = getattr(engine, "settings", None)
    cfg = getattr(settings, "search", None)
    if cfg is None or not getattr(cfg, "low_confidence_retry", False) or not results:
        return results
    try:
        conf = ranking_confidence([r.score for r in results])
        if conf >= cfg.low_confidence_threshold:
            return results
        processed = engine.query_processor.process(retrieval_text)
        alt_texts = [t for t in ([processed.enhanced] + list(processed.expansions))
                     if t and t.strip() and t.strip() != retrieval_text.strip()]
        if not alt_texts:
            return results
        rankings: list[list[SearchResult]] = [results]
        # search_prepared bypasses the query processor (the alts ARE its own
        # cached outputs — re-processing would fire fresh Gemini calls and
        # search an enhancement-of-an-enhancement). Stub engines without the
        # method fall back to plain search_text.
        search = getattr(engine, "search_prepared", None)
        for alt in alt_texts[:3]:
            if search is not None:
                alt_results = search(alt, skip_rerank=True)
            else:  # stub engines without the method fall back to plain search
                alt_results = engine.search_text(alt)
            if alt_results:
                rankings.append(alt_results)
        if len(rankings) == 1:
            return results
        merged = rrf_merge_results(rankings, limit=len(results))
        log.info("Low-confidence retry (conf=%.2f): merged %d reformulations",
                 conf, len(rankings) - 1)
        return merged
    except Exception as e:  # noqa: BLE001 — retry is an upgrade, never a risk
        log.warning("Low-confidence retry failed (%s) — keeping original ranking", e)
        return results


def _group_strip(results: Sequence[SearchResult], group: list[int],
                 max_frames: int) -> list[str]:
    """Up to ``max_frames`` image paths spanning one candidate group in
    TEMPORAL order (first / evenly spaced / last by keyframe ordinal) — the
    strip a multi-frame VQA call reads as consecutive evidence."""
    ordered = sorted(group, key=lambda i: getattr(results[i].ref, "n", 0))
    k = max(1, min(int(max_frames), len(ordered)))
    if k == 1:
        picks = [group[0]]
    else:
        idxs = {round(j * (len(ordered) - 1) / (k - 1)) for j in range(k)}
        picks = [ordered[i] for i in sorted(idxs)]
    return [getattr(results[i].ref, "path", "") for i in picks]


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
            # Buổi 4: đây là Q&A chứ không phải VQA — câu hỏi có thể dựa trên
            # ÂM THANH. Đưa thoại ASR quanh khoảnh khắc ứng viên vào prompt.
            ctx = ""
            if settings is not None:
                try:
                    from cvp.search.vqa import asr_context
                    _ref = results[best].ref
                    ctx = asr_context(settings, str(_ref.video_id),
                                      float(getattr(_ref, "pts_time", 0.0)))
                except Exception:  # noqa: BLE001 — context is best-effort
                    ctx = ""
            # Multi-frame strip first (one call sees the whole group — fixes
            # the 2025 "math in video" QA where text spans several frames);
            # single-frame `suggest` remains the compatibility/stub fallback.
            if hasattr(vqa, "answer_group"):
                strip = _group_strip(results, group,
                                     getattr(cfg, "frames_per_answer", 1))
                try:
                    ans = str(vqa.answer_group(question, strip, context=ctx)
                              or "")[:MAX_QA_ANSWER_CHARS]
                except TypeError:  # stub/legacy vqa without the context kwarg
                    ans = str(vqa.answer_group(question, strip) or "")[:MAX_QA_ANSWER_CHARS]
            if not ans:
                r = results[best]
                try:
                    suggestions = vqa.suggest(question, [(r.global_id, r.ref.path)],
                                              context=ctx)
                except TypeError:
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
    lines = load_query_lines(path)
    if not lines:
        log.warning("Empty query file: %s", path.name)
        return None
    out_path = out_dir / f"{path.stem}.csv"

    expected_top: tuple[str, int] | None = None   # (video, first frame) the times belong to

    def record(times: Sequence[float] | None,
               expected: tuple[str, int] | None = None) -> None:
        nonlocal expected_top
        if top1_times is not None and times:
            top1_times[path.stem] = [float(t) for t in times]
            expected_top = expected

    def _reconcile_times(written: Path | None) -> Path | None:
        """Drop recorded DRES timestamps when the writer skipped/reordered the
        top candidate (review C20): row 1 of the CSV must be the SAME
        (video, frame) the pts times were captured from, or the auto-submit
        would pair one candidate's frame with another's milliseconds."""
        if (written is None or top1_times is None
                or path.stem not in top1_times or expected_top is None):
            return written
        try:
            with open(written, "r", encoding="utf-8", newline="") as f:
                first = next((row for row in csv.reader(f) if row), None)
            if (first is None or first[0].strip() != expected_top[0]
                    or int(float(first[1])) != int(expected_top[1])):
                top1_times.pop(path.stem, None)
        except (OSError, ValueError, IndexError):
            top1_times.pop(path.stem, None)
        return written

    if task == "trake":
        # Organiser 2025 format: context/header line + "E1:…" events — the
        # header must NOT become a phantom event (see parse_trake_events).
        temporal_cfg = getattr(getattr(engine, "settings", None), "temporal", None)
        prepend = getattr(temporal_cfg, "event_context", "none") == "prepend"
        events = parse_trake_events(lines, prepend_context=prepend)
        if getattr(temporal_cfg, "pool_context", "none") == "prepend":
            # The header steers the video-POOLING stage only (see TemporalCfg).
            # Gated call: stub engines without the kwarg keep working on the
            # default config, exactly like before this knob existed.
            header, _events = split_trake_query(lines)
            candidates = engine.search_trake(events, context=header or None)
        else:
            candidates = engine.search_trake(events)
        if not candidates:
            # A 0-byte CSV would fail validation and veto packaging of the
            # WHOLE pack — return None so callers route this one query into
            # their "no submission produced" bucket instead.
            log.error("Query %s: engine returned ZERO candidates — no CSV written "
                      "(this query scores 0 unless re-run).", path.name)
            return None
        record(getattr(candidates[0], "pts_times", None),
               (candidates[0].video_id, int(candidates[0].frame_idxs[0]))
               if getattr(candidates[0], "frame_idxs", None) else None)
        return _reconcile_times(
            write_trake(out_path, [(c.video_id, c.frame_idxs) for c in candidates]))

    # Round-3 fix: multi-paragraph KIS/AVS joins ALL lines; QA question = last
    # interrogative line (real finals files have ≥3 lines). See parse_query_lines.
    retrieval_text, question = parse_query_lines(task, lines)
    if task == "avs":
        results = engine.search_avs(retrieval_text)
    else:
        results = engine.search_text(retrieval_text)
        results = maybe_retry_low_confidence(engine, retrieval_text, results)

    if not results:
        log.error("Query %s: engine returned ZERO results — no CSV written "
                  "(this query scores 0 unless re-run).", path.name)
        return None
    t = _time_of(results[0])
    record([t] if t is not None else None,
           (results[0].video_id, int(results[0].frame_idx)))
    if task == "qa":
        answers = compute_qa_answers(results, question, vqa, getattr(engine, "settings", None))
        return _reconcile_times(write_qa(
            out_path, [(r.video_id, r.frame_idx, a) for r, a in zip(results, answers)]))
    return _reconcile_times(
        write_kis(out_path, [(r.video_id, r.frame_idx) for r in results]))


def run_query_folder(settings: Settings, query_dir: Path, out_dir: Path,
                     with_vqa: bool = True) -> list[Path]:
    # A typo'd path exiting 0 with "Wrote 0 CSVs" misdirects the operator
    # toward CSV problems mid-round (round-6) — fail loud instead.
    if not query_dir.is_dir():
        raise FileNotFoundError(f"query dir not found: {query_dir}")
    qfiles = sorted(query_dir.glob("*.txt"))
    if not qfiles:
        log.error("No *.txt query files directly in %s — wrong folder, or did the "
                  "pack unzip into a SUBFOLDER?", query_dir)
    engine = SearchEngine(settings)
    vqa = None
    if with_vqa:
        try:
            vqa = VqaAssistant(settings)
        except Exception as e:  # noqa: BLE001
            log.warning("VQA assistant unavailable: %s", e)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for qf in qfiles:
        try:
            p = run_query_file(engine, qf, out_dir, vqa)
        except Exception as e:  # noqa: BLE001 — one bad query must not stop the pack
            log.error("Query %s failed: %s", qf.name, e)
            continue
        if p:
            written.append(p)
            log.info("%s → %s", qf.name, p.name)
    return written
