"""Round-72: merge of the Cursor-lab handoff + fixes for its audit findings.

The isolated Cursor lab (copy of 786327e) delivered four config-gated TRAKE
upgrades (caption-step signal, event query variants, pool context, submit
jitter), the adaptive multi-attempt harness (attempts.py + scripts/64), and
the bench diff reporter (scripts/65) with 49 new tests. An 18-agent
adversarial review confirmed default-off behavior is legacy-identical and
found four sharp edges, fixed here:

1. (degrade) caption_signal_weight > 0 with no obtainable caption signal ran
   silently as baseline — an A/B bench would measure baseline and convict
   the caption channel wrongly. trake_search now warns loudly when the knob
   is on but the normalized caption map comes back empty.
2. (degrade) on the in-memory BM25 fallback the caption scorer costs
   k×~30 full-corpus scans per query (minutes each on the dense store).
   TextSignals.persisted_field_ready() gates the scorer: engine disables it
   loudly unless the persisted text_index[caption] is ready.
3. (minor) jitter variants on partially-mapped videos mixed real frames with
   n-1 fallback estimates — such videos now keep their base chains only.
4. (minor) scripts/64 run refuses --out pointing at the attempt-1 dir
   (plan.run_dir) so the merge's primary run can never be clobbered.
"""
from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ENGINE = (REPO / "src" / "cvp" / "search" / "engine.py").read_text(encoding="utf-8")
TEMPORAL = (REPO / "src" / "cvp" / "search" / "temporal.py").read_text(encoding="utf-8")
SIGNALS = (REPO / "src" / "cvp" / "search" / "text_signals.py").read_text(encoding="utf-8")
SCRIPT64 = (REPO / "scripts" / "64_adaptive_attempts.py").read_text(encoding="utf-8")


def test_r72_caption_scorer_requires_persisted_index():
    assert "def persisted_field_ready" in SIGNALS
    assert 'persisted_field_ready("caption")' in ENGINE
    # the gate wraps the ONLY construction site (the import has no paren)
    assert ENGINE.count("caption_scorer_from_signals(") == 1
    assert "TẮT tín hiệu caption" in ENGINE


def test_r72_empty_caption_signal_is_loud():
    assert "KHÔNG thu được tín hiệu" in TEMPORAL
    assert "Y HỆT baseline" in TEMPORAL


def test_r72_jitter_skips_partially_mapped_videos():
    i_guard = TEMPORAL.index('"has_map" in rows_df.columns')
    i_call = TEMPORAL.index("jitter_frame_variants(rows, fidx)")
    assert i_guard < i_call                    # guard sits before the call


def test_r72_attempt_run_refuses_clobbering_attempt1():
    assert 'plan.get("run_dir")' in SCRIPT64
    assert "Dùng một thư mục MỚI cho lượt 2." in SCRIPT64


def test_r72_cursor_handoff_files_present():
    for p in ("src/cvp/pipeline/attempts.py", "scripts/64_adaptive_attempts.py",
              "scripts/65_bench_diff.py", "tests/test_trake_upgrades.py",
              "tests/test_adaptive_attempts.py", "tests/test_bench_diff.py",
              "report/cursor_lab_report.md"):
        assert (REPO / p).is_file(), p
