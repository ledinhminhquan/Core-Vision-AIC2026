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
import inspect
import logging
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from cvp.config import Settings, VqaCfg
from cvp.constants import MAX_QA_ANSWER_CHARS, MAX_SUBMISSION_ROWS
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
    "QA_FALLBACK_ANSWER", "QaGroupStat", "infer_task", "parse_query_lines",
    "parse_trake_events", "split_trake_query", "split_qa_line", "group_candidates",
    "compute_qa_answers", "compute_qa_answers_with_stats", "plan_consistency_rerank",
    "run_query_file", "run_query_folder", "ranking_confidence", "rrf_merge_results",
    "maybe_retry_low_confidence", "kis_events", "head_diversify",
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


def _cue_kwargs(fn, cue_text: str | None) -> dict[str, str]:
    """Round-88: pass ``cue_text`` only to engines whose search accepts it —
    stub engines in tests and older callers keep their signatures."""
    if not cue_text:
        return {}
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return {}
    return {"cue_text": cue_text} if "cue_text" in params else {}


def maybe_retry_low_confidence(engine: SearchEngine, retrieval_text: str,
                               results: list[SearchResult],
                               cue_text: str | None = None) -> list[SearchResult]:
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
                # round-88: same per-query cue as the primary search, so the
                # RRF alts fuse with the same weights (audit r88).
                alt_results = search(alt, skip_rerank=True, **_cue_kwargs(search, cue_text))
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


@dataclass
class QaGroupStat:
    """Ballot statistics of ONE answered candidate group (batch QA path).

    ``rows`` are indices into the ``results`` list (best rank first, exactly
    the group produced by :func:`group_candidates`); ``votes_for``/``total``
    describe the majority vote that produced ``answer`` (1/1 when the answer
    came from a single legacy ``answer_group``/``suggest`` call, 0/0 when the
    group produced nothing).
    """

    rows: list[int]
    answer: str
    votes_for: int = 0
    total_votes: int = 0

    @property
    def agree(self) -> float:
        """Winning-class share of the ballots, [0, 1] (0.0 without ballots)."""
        return self.votes_for / self.total_votes if self.total_votes else 0.0


def _round_robin_extras(n_groups: int, per_group_max: int, budget: int) -> dict[int, int]:
    """Fair distribution of ``budget`` extra strips: round-robin over groups
    (rank order), at most ``per_group_max`` each — the top moment must not
    starve the other answered moments of their cross-check strips."""
    out = {i: 0 for i in range(n_groups)}
    remaining = max(0, int(budget))
    for _ in range(max(0, int(per_group_max))):
        for i in range(n_groups):
            if remaining <= 0:
                return out
            if out[i] < per_group_max:
                out[i] += 1
                remaining -= 1
    return out


def _neighbor_strips(results: Sequence[SearchResult], group: list[int],
                     max_frames: int, n_extra: int) -> list[list[str]]:
    """Up to ``n_extra`` strips of same-video frames NEAR one candidate group.

    Candidates are rows of ``results`` itself (same video, outside the group),
    nearest-to-the-group first by original-video frame index — no catalog or
    disk access, so the batch path stays offline-testable. Fewer strips than
    requested (sparse video) is reported at INFO, never an error.
    """
    if n_extra <= 0:
        return []
    k = max(1, int(max_frames))
    in_group = set(group)
    video = results[group[0]].video_id

    def _pos(i: int) -> int:
        try:
            return int(results[i].frame_idx)
        except (TypeError, ValueError):
            return 0

    lo = min(_pos(i) for i in group)
    hi = max(_pos(i) for i in group)
    center = (lo + hi) / 2.0
    cand = [i for i in range(len(results))
            if i not in in_group and getattr(results[i], "video_id", None) == video]
    cand.sort(key=lambda i: (abs(_pos(i) - center), i))
    strips: list[list[str]] = []
    for j in range(int(n_extra)):
        chunk = sorted(cand[j * k:(j + 1) * k], key=_pos)
        paths = [p for p in (getattr(results[i].ref, "path", "") for i in chunk) if p]
        if not paths:
            break
        strips.append(paths)
    if len(strips) < n_extra:
        log.info("QA neighbor strips: dựng được %d/%d strip lân cận (video thiếu "
                 "ứng viên quanh nhóm).", len(strips), n_extra)
    return strips


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

    Thin wrapper over :func:`compute_qa_answers_with_stats` (same calls, same
    answers — the stats are simply discarded).
    """
    answers, _stats = compute_qa_answers_with_stats(results, question, vqa, settings)
    return answers


def compute_qa_answers_with_stats(
    results: Sequence[SearchResult], question: str,
    vqa: VqaAssistant | None, settings: Settings | None = None,
) -> tuple[list[str], list[QaGroupStat]]:
    """:func:`compute_qa_answers` + per-group ballot stats (round-73 upgrades).

    With every QA knob at its default this makes EXACTLY the calls the legacy
    function made (one ``answer_group`` per budgeted group, ``suggest`` as the
    fallback). Two knobs change the plan:

    * ``vqa.answer_neighbor_frames`` > 0 — each answered group also asks up to
      N strips of NEIGHBOUR frames (same video, nearest the group) and pools
      ALL raw ballots (``VqaAssistant.answer_group_votes``) into one majority,
      canonicalized when ``vqa.answer_canonicalize`` is on. Extra strips spend
      ``max_calls_per_query``: with the default 5/5 budget there is no room —
      a LOUD warning says so instead of silently doing nothing.
    * ``vqa.consistency_rerank`` — needs real ballot counts, so the ballot
      path is used (when the assistant exposes ``answer_group_votes``) even
      with ``answer_neighbor_frames=0``.

    When the ballot path yields nothing the legacy ``answer_group`` call runs
    as the rescue (it owns the local-model degradation) — on a total Gemini
    outage that retries the strip once more before falling back, an accepted
    cost on an already-broken path.
    """
    answers = [""] * len(results)
    if not results:
        return answers, []
    if vqa is None:
        return [QA_FALLBACK_ANSWER] * len(results), []
    cfg = settings.vqa if settings is not None else VqaCfg()
    budget = max(0, min(int(cfg.answers_per_query), int(cfg.max_calls_per_query)))
    groups = group_candidates(results)
    neighbor_n = max(0, int(getattr(cfg, "answer_neighbor_frames", 0) or 0))
    want_ballots = neighbor_n > 0 or bool(getattr(cfg, "consistency_rerank", False))
    pool_ballots = want_ballots and hasattr(vqa, "answer_group_votes")
    canon = bool(getattr(cfg, "answer_canonicalize", False))
    n_primary = min(budget, len(groups))
    extra_budget = max(0, int(cfg.max_calls_per_query) - n_primary) if neighbor_n else 0
    if neighbor_n and pool_ballots and extra_budget <= 0 and n_primary:
        log.warning(
            "vqa.answer_neighbor_frames=%d nhưng max_calls_per_query=%d đã cạn sau "
            "%d strip chính — knob KHÔNG có tác dụng; tăng vqa.max_calls_per_query "
            "(vd %d).", neighbor_n, int(cfg.max_calls_per_query), n_primary,
            n_primary * (1 + neighbor_n))
    extra_per_group = _round_robin_extras(n_primary, neighbor_n, extra_budget)
    stats: list[QaGroupStat] = []
    fallback = ""

    def _answer_one_group(g_idx: int, group: list[int]) -> tuple[str, int, int]:
        """All model calls for ONE candidate group — pure function of (g_idx,
        group) so groups can run in parallel (round-82 ``vqa.parallel_calls``).
        Returns (answer, votes_for, total_votes); a failed group yields ""."""
        best = group[0]
        ans = ""
        votes_for = total_votes = 0
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
                if pool_ballots:
                    strips = [strip] + _neighbor_strips(
                        results, group, getattr(cfg, "frames_per_answer", 1),
                        extra_per_group.get(g_idx, 0))
                    ballots: list[str] = []
                    for s in strips:
                        try:
                            ballots.extend(vqa.answer_group_votes(question, s, context=ctx))
                        except TypeError:  # stub without the context kwarg
                            ballots.extend(vqa.answer_group_votes(question, s))
                    if ballots:
                        from cvp.search.answer_norm import majority_vote

                        vr = majority_vote(ballots, canonicalize=canon)
                        ans = str(vr.answer or "")[:MAX_QA_ANSWER_CHARS]
                        votes_for, total_votes = vr.votes_for, vr.total
                if not ans:
                    try:
                        ans = str(vqa.answer_group(question, strip, context=ctx)
                                  or "")[:MAX_QA_ANSWER_CHARS]
                    except TypeError:  # stub/legacy vqa without the context kwarg
                        ans = str(vqa.answer_group(question, strip) or "")[:MAX_QA_ANSWER_CHARS]
                    if ans and not total_votes:
                        votes_for = total_votes = 1
            if not ans:
                r = results[best]
                try:
                    suggestions = vqa.suggest(question, [(r.global_id, r.ref.path)],
                                              context=ctx)
                except TypeError:
                    suggestions = vqa.suggest(question, [(r.global_id, r.ref.path)])
                if suggestions:
                    ans = str(suggestions[0].answer)[:MAX_QA_ANSWER_CHARS]
                    if ans:
                        votes_for = total_votes = 1
        except Exception as e:  # noqa: BLE001 — one failed group must not sink the query
            log.warning("VQA failed for group at rank %d: %s", best + 1, e)
        return ans, votes_for, total_votes

    todo = list(enumerate(groups[:budget]))
    n_par = max(1, int(getattr(cfg, "parallel_calls", 1) or 1))
    if n_par > 1 and len(todo) > 1:
        # Round-82: các nhóm độc lập nhau — chạy song song, nhưng KẾT QUẢ
        # được áp theo ĐÚNG thứ tự nhóm cũ (stamp answers, stats, fallback)
        # nên đầu ra bit-identical với đường tuần tự, chỉ nhanh hơn.
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=min(n_par, len(todo))) as _ex:
            outcomes = list(_ex.map(lambda t: _answer_one_group(*t), todo))
    else:
        outcomes = [_answer_one_group(g_idx, group) for g_idx, group in todo]
    for (g_idx, group), (ans, votes_for, total_votes) in zip(todo, outcomes):
        for i in group:
            answers[i] = ans
        stats.append(QaGroupStat(rows=list(group), answer=ans,
                                 votes_for=votes_for, total_votes=total_votes))
        if not fallback and ans:
            fallback = ans
    if fallback:
        answers = [a or fallback for a in answers]
    return [a or QA_FALLBACK_ANSWER for a in answers], stats


# Consistency-rerank thresholds (knob ``vqa.consistency_rerank``): the top
# group must LACK a real majority, and a promoted group must be UNANIMOUS over
# at least 2 ballots. Module constants (with this docstring) instead of extra
# config knobs — one bool is enough to A/B the feature.
_RERANK_TOP_MAX_AGREE = 0.5
_RERANK_STABLE_MIN_VOTES = 2


def plan_consistency_rerank(stats: Sequence[QaGroupStat], n_rows: int) -> list[int] | None:
    """Row order promoting the first UNANIMOUS group when the top is unstable.

    Returns the full new order (indices into the original ``results``) or
    None when nothing should move. Rules — deliberately conservative, the
    reranker must never demote a top group that has a real majority:

    * top group keeps its place when its ballots reach a majority
      (``agree >= 0.5``) — including the 1/1 legacy-call case;
    * the promoted group needs ``answer`` non-empty, not the fallback
      placeholder, and a unanimous vote over ≥ 2 ballots;
    * only the FIRST (best-ranked) such group is promoted, as one block, in
      front of everything else; every other row keeps its relative order.
    """
    if n_rows <= 0 or len(stats) < 2:
        return None
    top = stats[0]
    if top.total_votes and top.agree >= _RERANK_TOP_MAX_AGREE:
        return None
    stable = next(
        (s for s in stats[1:]
         if s.answer and s.total_votes >= _RERANK_STABLE_MIN_VOTES
         and s.agree >= 1.0
         and s.answer.strip().casefold() != QA_FALLBACK_ANSWER),
        None,
    )
    if stable is None:
        return None
    promoted = [i for i in stable.rows if 0 <= i < n_rows]
    if not promoted:
        return None
    promoted_set = set(promoted)
    return promoted + [i for i in range(n_rows) if i not in promoted_set]


_KIS_SEQUENCE_CUES = ("bắt đầu", "kết thúc", "sau đó", "tiếp theo", "cuối cùng",
                      "lần lượt", "trước khi", "sau khi", "chuyển sang", "chuyển cảnh")


def kis_events(lines: Sequence[str]) -> list[str]:
    """Round-84: ordered scene sentences of a MULTI-SCENE KIS query, else [].

    A KIS query is "đa cảnh" when it has ≥3 substantive sentences, or ≥2 with
    an explicit sequence cue ("bắt đầu… kết thúc…", "sau đó"). Sentences shorter
    than 6 words are dropped (fragments, "Video về…" lead-ins). Capped at 6.
    """
    text = " ".join(str(ln).strip() for ln in lines if str(ln).strip())
    sents = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if len(s.split()) >= 6]
    low = text.casefold()
    if len(sents) >= 3 or (len(sents) == 2 and any(c in low for c in _KIS_SEQUENCE_CUES)):
        return sents[:6]
    return []


def _maybe_kis_multi_event(engine: SearchEngine, lines: Sequence[str],
                           results: list[SearchResult], *,
                           force: bool = False) -> list[SearchResult]:
    """``search.kis_multi_event`` (round-84): fuse the single-text KIS ranking
    with DANTE-aligned event chains of the same query.

    Off (default) returns ``results`` untouched. On: the query's scene
    sentences become TRAKE events; every (video, frame) of the top chains
    forms a second ranking; both are RRF-fused (k=60). Chain frames absent
    from ``results`` are materialised through the catalog (ordinal → global
    id, verified against video/frame — a mismatch is skipped, never guessed).
    """
    cfg = getattr(getattr(engine, "settings", None), "search", None)
    if not (force or getattr(cfg, "kis_multi_event", False)) or not results:
        return results
    events = kis_events(lines)
    if len(events) < 2:
        return results
    try:
        chains = engine.search_trake(events)
    except Exception as e:  # noqa: BLE001 — fusion is an add-on, never a veto
        log.warning("kis_multi_event: search_trake failed (%s) — giữ ranking đơn-câu", e)
        return results
    if not chains:
        log.info("kis_multi_event: %d sự kiện nhưng không có chuỗi — giữ ranking đơn-câu",
                 len(events))
        return results
    catalog = getattr(engine, "catalog", None)
    chain_keys: list[tuple[str, int]] = []
    chain_res: dict[tuple[str, int], SearchResult] = {}
    gid_maps: dict[str, dict[int, int]] = {}
    for c in chains[:8]:
        ns = list(getattr(c, "ns", []) or [])
        if catalog is not None and c.video_id not in gid_maps:
            # Round-84b: ``ns`` là SỐ HIỆU keyframe ``n`` của BTC (từ 1, có thể
            # nhảy cóc) — không phải thứ tự 0-based, nên start+n lệch. Ánh xạ
            # n → global_id qua chính manifest của video.
            try:
                df = catalog.load()
                sub = df[df["video_id"] == c.video_id]
                gid_maps[c.video_id] = {int(n): int(g) for n, g
                                        in zip(sub["n"], sub["global_id"])}
            except Exception as e:  # noqa: BLE001
                log.warning("kis_multi_event: không đọc được manifest %s (%s)",
                            c.video_id, e)
                gid_maps[c.video_id] = {}
        for j, f in enumerate(c.frame_idxs):
            key = (c.video_id, int(f))
            if key in chain_res:
                continue
            ref = None
            if catalog is not None and j < len(ns):
                gid = gid_maps.get(c.video_id, {}).get(int(ns[j]))
                if gid is not None:
                    try:
                        cand = catalog.ref(int(gid))
                        if cand.video_id == c.video_id and int(cand.frame_idx) == int(f):
                            ref = cand
                    except Exception:  # noqa: BLE001 — verify-or-skip
                        ref = None
            if ref is None:
                continue
            chain_keys.append(key)
            chain_res[key] = SearchResult(ref=ref, score=float(c.score),
                                          signals={"kis_event": float(c.score)})
    if not chain_keys:
        # Fail LOUD (audit law): bench ABX 02/09 ran this path silently for
        # every query and measured baseline while the knob looked "on".
        log.warning("kis_multi_event: %d chuỗi nhưng KHÔNG dựng được frame nào qua "
                    "catalog (ns↔global_id lệch?) — giữ ranking đơn-câu. KNOB VÔ HIỆU.",
                    len(chains))
        return results
    k = 60.0
    fused: dict[tuple[str, int], float] = {}
    pool: dict[tuple[str, int], SearchResult] = {}
    for rank, r in enumerate(results):
        key = (r.video_id, int(r.frame_idx))
        fused[key] = fused.get(key, 0.0) + 1.0 / (k + rank + 1)
        pool.setdefault(key, r)
    for rank, key in enumerate(chain_keys):
        fused[key] = fused.get(key, 0.0) + 1.0 / (k + rank + 1)
        pool.setdefault(key, chain_res[key])
    order = sorted(fused, key=lambda kk: -fused[kk])
    log.info("kis_multi_event: %d sự kiện, %d chuỗi → trộn %d frame chuỗi vào ranking "
             "(top-1 %s).", len(events), len(chains), len(chain_keys),
             "đổi" if order[0] != (results[0].video_id, int(results[0].frame_idx)) else "giữ")
    return [pool[kk] for kk in order][:max(len(results), 100)]


_HEAD_DIVERSITY_WINDOWS = ((5, 2), (20, 5))   # (window, max rows per video)


def head_diversify(rows: list[tuple]) -> list[tuple]:
    """Round-86: BEST-EFFORT per-video caps inside the head windows (top-5 ≤2,
    top-20 ≤5): excess rows are demoted to just after the window, order
    otherwise preserved, nothing lost. Best-effort = when too few other
    videos exist the window fills short and the demoted rows flow straight
    back (audit r86) — a hard cap would have to DROP rows. Row 1 never moves.
    Not combined with vqa.consistency_rerank (would split its promoted
    block); that knob is off in every battle pack. Pure."""
    out = list(rows)
    for window, cap in _HEAD_DIVERSITY_WINDOWS:
        if len(out) <= window:
            continue
        head, tail = [], []
        seen: dict[str, int] = {}
        for r in out:
            if len(head) < window:
                v = str(r[0])
                if seen.get(v, 0) < cap:
                    seen[v] = seen.get(v, 0) + 1
                    head.append(r)
                    continue
            tail.append(r)
        out = head + tail
    return out


def _maybe_head_diversity(settings: Settings | None, task: str,
                          rows: list[tuple]) -> list[tuple]:
    """``search.head_diversity`` (round-86): off = the SAME list object."""
    if task not in ("kis", "qa"):
        return rows
    if not getattr(getattr(settings, "search", None), "head_diversity", False):
        return rows
    out = head_diversify(rows)
    if out[:20] != rows[:20]:
        log.info("head_diversity: đầu bảng đổi — top-5 videos %s",
                 sorted({str(r[0]) for r in out[:5]}))
    return out


def _maybe_answer_variants(settings: Settings | None, rows: list[tuple]) -> list[tuple]:
    """``vqa.answer_variant_rows`` (round-77): dual-format số↔chữ insurance.

    BTC chấm answer theo (near-)exact text — "sáu" vs GT "6" là 0 điểm dù
    moment đúng. Với mỗi answer PHÂN BIỆT trong 30 dòng đầu có dạng số quy đổi
    được, thêm một dòng (video, frame, answer-định-dạng-kia); các dòng thêm
    THAY THẾ đúng chừng đó dòng cuối bảng (đầu bảng bất khả xâm phạm — chỉ
    hy sinh vé số đuôi). Mặc định off = trả về CHÍNH ``rows``.
    """
    if not getattr(getattr(settings, "vqa", None), "answer_variant_rows", False):
        return rows
    if not rows:
        return rows
    from cvp.search.answer_norm import answer_variants

    seen: set[str] = set()
    variants: list[tuple] = []
    for r in rows[:30]:
        if len(r) < 3:
            continue
        key = str(r[2]).strip().casefold()
        if not key or key in seen:
            continue
        seen.add(key)
        for alt in answer_variants(str(r[2])):      # round-82: nhiều dạng/đáp án
            variants.append((r[0], r[1], alt))
        if len(variants) >= 10:     # đuôi chỉ hy sinh tối đa 10 vé số
            variants = variants[:10]
            break
    if not variants:
        return rows
    keep = max(1, MAX_SUBMISSION_ROWS - len(variants))
    out = list(rows[:keep]) + variants
    log.info("answer_variant_rows: thêm %d dòng song-định-dạng (thế chỗ %d dòng "
             "đuôi ngoài top-%d).", len(variants), max(0, len(rows) - keep), keep)
    return out


def _maybe_diversify_rows(engine: SearchEngine, task: str, rows: list[tuple]) -> list[tuple]:
    """Apply ``search.row_strategy`` to KIS/QA rows (Nhiệm vụ D, round-74).

    ``legacy`` (default) returns ``rows`` untouched — bit-identical CSVs. The
    head is preserved verbatim by :func:`cvp.pipeline.row_budget.diversify_tail`,
    so row 1 (and the recorded DRES timestamp) can never change. AVS is
    excluded on purpose: its MMR pass already diversifies across videos, and
    same-video neighbour variants would undo exactly that.
    """
    cfg = getattr(getattr(engine, "settings", None), "search", None)
    if task not in ("kis", "qa") or getattr(cfg, "row_strategy", "legacy") != "diversify_tail":
        return rows
    from cvp.pipeline.row_budget import catalog_grid_fn, diversify_tail

    catalog = getattr(engine, "catalog", None)
    if catalog is None:
        # Round-75 (tổng kiểm F): knob bật mà engine không có catalog = không
        # có lưới keyframe nào để dựng variant — phải NÓI TO thay vì âm thầm
        # chạy như legacy (bài học 4 bản vá audit đợt 1).
        log.warning("search.row_strategy=diversify_tail nhưng engine KHÔNG có "
                    "catalog — không dựng được variant, ranking giữ nguyên như legacy.")
    return diversify_tail(
        rows,
        grid_fn=catalog_grid_fn(catalog),
        head_keep=int(getattr(cfg, "row_strategy_head", 30)),
        budget=MAX_SUBMISSION_ROWS,
        variants_per_anchor=int(getattr(cfg, "row_strategy_variants", 4)),
    )


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
        # Round-88: cue text = description + question — for QA the on-screen-
        # text / speech cue usually lives in the QUESTION line, which
        # parse_query_lines strips out of the retrieval text (audit r88).
        _cue = " ".join(t for t in (retrieval_text, question) if t)
        results = engine.search_text(retrieval_text, **_cue_kwargs(engine.search_text, _cue))
        results = maybe_retry_low_confidence(engine, retrieval_text, results, cue_text=_cue)
        if task == "kis":
            results = _maybe_kis_multi_event(engine, lines, results)
        elif task == "qa" and getattr(getattr(getattr(engine, "settings", None),
                                              "search", None), "qa_multi_event", False):
            # Round-86: mô tả QA (KHÔNG gồm câu hỏi) cũng là chuỗi cảnh —
            # dùng retrieval_text đã tách, đúng ở mọi nhánh của parse_query_lines
            # (audit: lines[:-1] rò câu hỏi vào sự kiện ở nhánh fallback).
            results = _maybe_kis_multi_event(engine, [retrieval_text], results, force=True)

    if not results:
        log.error("Query %s: engine returned ZERO results — no CSV written "
                  "(this query scores 0 unless re-run).", path.name)
        return None
    t = _time_of(results[0])
    record([t] if t is not None else None,
           (results[0].video_id, int(results[0].frame_idx)))
    if task == "qa":
        settings_ = getattr(engine, "settings", None)
        answers, stats = compute_qa_answers_with_stats(results, question, vqa, settings_)
        if getattr(getattr(settings_, "vqa", None), "consistency_rerank", False):
            order = plan_consistency_rerank(stats, len(results))
            if order:
                results = [results[i] for i in order]
                answers = [answers[i] for i in order]
                log.info("QA consistency rerank (%s): nhóm đồng thuận %s lên đầu — "
                         "nhóm top bất nhất.", path.name, results[0].video_id)
                # Re-record: the DRES timestamp must belong to the NEW top row
                # (same contract _reconcile_times enforces, done properly here).
                t = _time_of(results[0])
                record([t] if t is not None else None,
                       (results[0].video_id, int(results[0].frame_idx)))
        qa_rows = _maybe_head_diversity(
            settings_, "qa", [(r.video_id, r.frame_idx, a) for r, a in zip(results, answers)])
        qa_rows = _maybe_diversify_rows(engine, "qa", qa_rows)
        qa_rows = _maybe_answer_variants(settings_, qa_rows)
        return _reconcile_times(write_qa(out_path, qa_rows))
    kis_rows = _maybe_head_diversity(
        getattr(engine, "settings", None), task, [(r.video_id, r.frame_idx) for r in results])
    kis_rows = _maybe_diversify_rows(engine, task, kis_rows)
    return _reconcile_times(write_kis(out_path, kis_rows))


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
