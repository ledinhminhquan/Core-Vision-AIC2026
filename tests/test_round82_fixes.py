"""Round-82: the clock offensive + the evidence-based attempt-2 redesign.

Battle night 28/08: 27/30 in 142' (QA ~2/3 of it), no time for attempt 2.
Offline validation on the two bench runs then showed (a) the adaptive
confidence metric has NO predictive power (Pearson +0.10 / -0.04 vs true
per-query score; 0 queries flagged weak at the default threshold) and (b)
the hard failures are SYSTEMATIC — both runs fail the same queries — so a
same-stack retry cannot fix them. Hence:

1. vqa.parallel_calls — QA candidate groups are answered in parallel; results
   are applied in the original group order, so output is bit-identical.
2. nb03 SHARD_INDEX/SHARD_TOTAL — one pack across N Colab sessions (3xA100 +
   1xG4); shards push CSVs to Drive, one REZIP_ONLY session gathers + zips.
3. nb03 LINEUP battle|diverse — attempt 2 = FULL re-run with a DIFFERENT
   lineup (3-lane incl. qwen_embed), then MERGE_PACKS RRF-merges packs.
4. answer_variants — several exact-text forms per QA answer ("7 cái." -> "7",
   "bảy cái.") on the tail rows.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from cvp.config import Settings
from cvp.search.answer_norm import answer_variants

REPO = Path(__file__).resolve().parents[1]


class _Res:
    def __init__(self, vid, frame, gid):
        self.video_id, self.frame_idx, self.global_id = vid, frame, gid
        self.ref = type("R", (), {"video_id": vid, "path": f"{vid}_{frame}.jpg",
                                  "pts_time": frame / 25.0})()


class _SlowVqa:
    """Answers depend only on the strip's first frame; sleeps to expose order."""
    def __init__(self):
        self.calls = 0

    def answer_group(self, question, strip, context=""):
        self.calls += 1
        time.sleep(0.02 * (5 - len(strip) % 5))   # uneven latencies
        return f"ans-{Path(strip[0]).stem}"

    def suggest(self, question, items, context=""):
        return []


def _results():
    # 6 candidate groups: distinct videos, far-apart frames
    return [_Res(f"L01_V{i:03d}", 1000 * i, i) for i in range(1, 7)]


def test_r82_parallel_groups_are_bit_identical_to_sequential():
    from cvp.pipeline.run_queries import compute_qa_answers_with_stats
    s1 = Settings()
    s1.vqa.parallel_calls = 1
    s4 = Settings()
    s4.vqa.parallel_calls = 4
    res = _results()
    a1, st1 = compute_qa_answers_with_stats(res, "q?", _SlowVqa(), s1)
    a4, st4 = compute_qa_answers_with_stats(res, "q?", _SlowVqa(), s4)
    assert a1 == a4
    assert [(x.rows, x.answer, x.votes_for, x.total_votes) for x in st1] == \
           [(x.rows, x.answer, x.votes_for, x.total_votes) for x in st4]


def test_r82_parallel_calls_default_is_one():
    assert Settings().vqa.parallel_calls == 1


def test_r82_answer_variants_exact_text_forms():
    assert answer_variants("7 cái.") == ["bảy cái.", "7"]
    assert answer_variants("bảy cái") == ["7 cái", "7"]
    assert answer_variants("6") == ["sáu"]
    assert answer_variants("Cá hanh") == []
    assert answer_variants("không rõ") == []


def test_r82_variant_rows_take_multiple_forms():
    from cvp.pipeline.run_queries import _maybe_answer_variants
    s = Settings()
    s.vqa.answer_variant_rows = True
    rows = [(f"L01_V{i:03d}", i, "7 cái.") for i in range(1, 101)]
    out = _maybe_answer_variants(s, rows)
    tail = {r[2] for r in out[-2:]}
    assert tail == {"bảy cái.", "7"} and len(out) == 100
    assert out[:30] == rows[:30]


def test_r82_nb03_toggles_present_and_defaults_legacy():
    nb = json.loads((REPO / "notebooks" / "03_test_system.ipynb")
                    .read_text(encoding="utf-8"))
    cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    engine = next(s for s in cells if "engine = SearchEngine(settings)" in s)
    pack = next(s for s in cells if "RUN_PACK" in s and "SHARD_TOTAL" in s)
    assert 'LINEUP = "battle"' in engine
    assert "QA_PARALLEL = 4" in engine
    assert '"finetuned", "metaclip2", "qwen_embed"' in engine     # diverse lineup
    assert "SHARD_INDEX = 0" in pack and "SHARD_TOTAL = 1" in pack
    assert "MERGE_PACKS = []" in pack
    assert "rrf_merge_runs" in pack and "write_merged" in pack
    # shard mode never syncs a partial zip; the leader gathers from Drive
    assert "KHÔNG nộp zip của shard lẻ" in pack
    assert 'gom từ Drive' in pack


# ── audit r82 hardening pins ────────────────────────────────────────────────

def test_r82_leading_number_needs_whitespace():
    # audit: backtracking produced nonsense ('2024'→'202', '25'→'2')
    assert answer_variants("2024") == []
    assert answer_variants("1000") == []
    assert "2" not in answer_variants("25")
    assert answer_variants("500g") == []          # glued unit stays put


def test_r82_vqa_assistant_serializes_local_model():
    import threading
    from cvp.search.vqa import VqaAssistant
    assert isinstance(VqaAssistant._LOCK, type(threading.RLock()))
    src = (REPO / "src" / "cvp" / "search" / "vqa.py").read_text(encoding="utf-8")
    assert "with self._LOCK:      # one GPU model, one caller at a time" in src
    assert src.count("with self._LOCK:\n                if self._gemini_client is None:") == 2


def test_r82_qwen_embed_loads_resiliently():
    src = (REPO / "src" / "cvp" / "models" / "qwen_embed.py").read_text(encoding="utf-8")
    assert src.count("resilient_from_pretrained(") >= 2


def test_r82_nb03_shard_hardening():
    nb = json.loads((REPO / "notebooks" / "03_test_system.ipynb")
                    .read_text(encoding="utf-8"))
    pack = next("".join(c["source"]) for c in nb["cells"]
                if c["cell_type"] == "code" and "SHARD_TOTAL" in "".join(c["source"]))
    assert "THIẾU {len(_missing)}" in pack            # coverage gate (blocker)
    assert "if _c.stem not in _expected" in pack      # stale-CSV filter
    assert '_merged"' in pack and "_zip_dir" in pack  # merge → new dir, idempotent
    assert "SINH ĐÔI" in pack                         # Drive twin-folder gate
    assert "_push_loop" in pack and "kéo {_n_pull}" in pack   # push/pull resume
    assert "ỔN ĐỊNH" in pack                          # stable listing before slicing
