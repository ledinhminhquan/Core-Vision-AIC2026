"""Round-71: final pre-campaign audit fixes for nb04/nb02 (6 findings).

1. L3's signal-dump subprocess inherited L2's full reranker stack (Qwen-8B +
   Gemini VLM 48×3) for signals captured BEFORE reranking — now explicitly
   disabled, like L4 always did.
2. The engine "degrades gracefully" to a single lane when a member fails to
   load — a multi-hour bench could silently measure the wrong line-up. L2 now
   builds the engine first, asserts both ensemble members loaded, and hands
   that engine to run_auto via engine_factory (no double model load).
3. Bench #2 used to clobber bench #1's record — bench_full.json/lab_full now
   rotate to -prev before each write, preserving the before/after comparison.
4. The tuned-weights stat and the -prev backup stat get a tuning/ metadata
   nudge, and a loud warning prints when the bench falls back to default
   weights.
5. The best_weights-prev safe is write-once — a rerun of L3 can no longer
   overwrite the original battle weights with the freshly tuned set.
6. nb02's two-live-sessions guard now applies regardless of hash equality,
   and a 10-minute pointer heartbeat thread keeps updated_utc fresh for the
   whole training run (stopped via _HB_STOP before the final status write).
"""
from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BUILDER = (REPO / "notebooks" / "_build_notebooks.py").read_text(encoding="utf-8")


def _frag(start: str, end: str) -> str:
    return BUILDER.split(start)[1].split(end)[0]


def test_r71_tune_dump_disables_rerankers():
    frag = _frag("LAB_TUNE = r", "LAB_METACLIP")
    assert '"CVP_SEARCH__RERANKER"] = "none"' in frag
    assert '"CVP_SEARCH__VLM_RERANK"] = "false"' in frag
    # before the dump subprocess launches
    assert frag.index('"CVP_SEARCH__RERANKER"] = "none"') < frag.index(
        "23_dump_signals.py")


def test_r71_bench_asserts_both_lanes_and_reuses_the_engine():
    frag = _frag("LAB_BENCH_FULL = r", "LAB_TUNE")
    assert 'engine.member_names == ["finetuned", "metaclip2"]' in frag
    assert "engine_factory=lambda _s: engine" in frag
    assert frag.index("member_names ==") < frag.index("run_auto(")


def test_r71_bench_record_rotates_not_clobbers():
    frag = _frag("LAB_BENCH_FULL = r", "LAB_TUNE")
    assert "bench_full-prev.json" in frag and "lab_full-prev" in frag
    assert frag.index("bench_full-prev.json") < frag.index(
        '(_drv_lab / "bench_full.json").write_text(')


def test_r71_tuning_stats_are_nudged_and_default_fallback_is_loud():
    bench = _frag("LAB_BENCH_FULL = r", "LAB_TUNE")
    tune = _frag("LAB_TUNE = r", "LAB_METACLIP")
    assert "list(_tw.parent.iterdir())" in bench
    assert "trọng số " in bench and "MẶC ĐỊNH" in bench
    assert "list(_tune_out.parent.iterdir())" in tune


def test_r71_prev_safe_is_write_once():
    tune = _frag("LAB_TUNE = r", "LAB_METACLIP")
    assert "if _tune_out.exists() and not _prev.exists():" in tune
    assert "giữ nguyên két sắt" in tune


def test_r71_nb02_guard_any_hash_plus_heartbeat():
    frag = _frag("NB2_RUN_POINTER = r", "NB2_TRAIN")
    assert 'if ptr and ptr.get("status") == "running":' in frag
    assert "def _pointer_heartbeat():" in frag
    assert "_HB_STOP" in frag
    train = _frag("NB2_TRAIN = r", "NB2_DELIVERABLES")
    assert train.count('globals()["_HB_STOP"] = True') == 2  # crashed + finished
