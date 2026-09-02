"""Round-86: "can this get any better?" — four cheap, evidence-backed ideas.

1. QA answer HEDGE in the RRF merge: merge_rows keyed QA rows on (video,
   frame) only, so when two runs disagree on the answer for the same frame
   (A100-vs-G4 measured only 2/7 QA answers agreeing) one answer was silently
   dropped. qa_keep_all_answers=True keeps both rows — the metric takes the
   max over rows, so the hedge can only cost tail slots. nb03's MERGE_PACKS
   path turns it on; scripts/64 parity stays (default off).
2. search.head_diversity: cap rows per video inside the head (top-5 <= 2,
   top-20 <= 5), demoting the excess to right after the window — neighbour
   boosts cluster one video in the head; if that video is wrong every top-5
   ticket is wasted.
3. search.qa_multi_event: DANTE scene alignment for the QA moment search
   (descriptions are often multi-scene too).
4. nb04 BENCH_PACK "ABKD" = ABK + {head diversity, QA multi-event, TRAKE pool
   widening per_event_topk 300 / max_videos 60}. All bench-gated.
"""
from __future__ import annotations

import json
from pathlib import Path

from cvp.config import Settings
from cvp.pipeline.attempts import merge_rows, rrf_merge_runs
from cvp.pipeline.run_queries import head_diversify

REPO = Path(__file__).resolve().parents[1]


def test_r86_merge_hedge_appends_alternative_answers_in_the_tail():
    a = [["L01_V001", "100", "Con hến"], ["L01_V002", "200", "x"]]
    b = [["L01_V001", "100", "Con heo"], ["L01_V003", "300", "y"]]
    legacy = merge_rows([a, b], [1.0, 1.0], 60, "qa")
    assert [r[2] for r in legacy if r[0] == "L01_V001"] == ["Con hến"]   # one answer
    hedged = merge_rows([a, b], [1.0, 1.0], 60, "qa", qa_keep_all_answers=True)
    assert hedged[:len(legacy)] == legacy                    # head untouched
    assert hedged[-1] == ["L01_V001", "100", "Con heo"]      # alt rides the tail
    # KIS/TRAKE unaffected by the flag
    k = [["L01_V001", "100"]]
    assert merge_rows([k, k], [1.0, 1.0], 60, "kis", qa_keep_all_answers=True) == k
    runs = [{"query-p9-1-qa": a}, {"query-p9-1-qa": b}]
    assert len(rrf_merge_runs(runs, qa_keep_all_answers=True)["query-p9-1-qa"]) == 4


def test_r86_merge_hedge_ignores_fallback_and_normalized_duplicates():
    a = [["L01_V001", "100", "Con hến"]]
    b_fb = [["L01_V001", "100", "không rõ"]]          # dead answer never hedges
    assert merge_rows([a, b_fb], [1.0, 1.0], 60, "qa", qa_keep_all_answers=True) == a
    b_dup = [["L01_V001", "100", "Con hến."]]         # equal under official normalizer
    assert merge_rows([a, b_dup], [1.0, 1.0], 60, "qa", qa_keep_all_answers=True) == a


def test_r86_merge_hedge_is_bounded_to_the_tail():
    a = [[f"L01_V{i:03d}", str(i), "ans-a"] for i in range(1, 101)]
    b = [[f"L01_V{i:03d}", str(i), "ans-b"] for i in range(1, 101)]
    out = merge_rows([a, b], [1.0, 1.0], 60, "qa", qa_keep_all_answers=True)
    assert len(out) == 100
    assert out[:90] == a[:90]                                # 90 real frames kept
    assert all(r[2] == "ans-b" for r in out[90:])            # ≤10 hedge rows, tail only


def test_r86_head_diversify_caps_per_video_and_demotes():
    rows = [("A", 1), ("A", 2), ("A", 3), ("B", 4), ("C", 5), ("A", 6), ("D", 7)]
    out = head_diversify(rows)
    assert out[:5] == [("A", 1), ("A", 2), ("B", 4), ("C", 5), ("A", 6)] or \
           out[:5] == [("A", 1), ("A", 2), ("B", 4), ("C", 5), ("D", 7)]
    assert sorted(out) == sorted(rows)          # nothing lost
    assert head_diversify([("A", 1), ("A", 2)]) == [("A", 1), ("A", 2)]   # short = same


def test_r86_head_diversity_wiring_off_is_identity():
    from cvp.pipeline.run_queries import _maybe_head_diversity
    s = Settings()
    rows = [("A", 1), ("A", 2), ("A", 3), ("A", 4), ("A", 5), ("B", 6)]
    assert _maybe_head_diversity(s, "kis", rows) is rows
    assert _maybe_head_diversity(s, "trake", rows) is rows
    s.search.head_diversity = True
    out = _maybe_head_diversity(s, "kis", rows)
    assert ("B", 6) in out[:5]


def test_r86_defaults_and_pack_d():
    s = Settings()
    assert s.search.head_diversity is False and s.search.qa_multi_event is False
    nb = json.loads((REPO / "notebooks" / "04_lab_artifacts.ipynb")
                    .read_text(encoding="utf-8"))
    src = "".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    assert 'assert BENCH_PACK in ("off", "A", "AB", "ABK", "ABKD", "ABX")' in src
    for k in ("CVP_SEARCH__HEAD_DIVERSITY", "CVP_SEARCH__QA_MULTI_EVENT",
              "CVP_TEMPORAL__PER_EVENT_TOPK", "CVP_TEMPORAL__MAX_VIDEOS"):
        assert k in src, k
    nb3 = json.loads((REPO / "notebooks" / "03_test_system.ipynb")
                     .read_text(encoding="utf-8"))
    pack = next("".join(c["source"]) for c in nb3["cells"]
                if c["cell_type"] == "code" and "MERGE_PACKS = " in "".join(c["source"]))
    assert "qa_keep_all_answers=True" in pack
