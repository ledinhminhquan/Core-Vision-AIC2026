"""Round-75: the AB knob pack enters the battle config (nb03) — bench-gated.

Evidence: bench#2 off=0.6478 → bench#3 A=0.6739 → bench#4 AB=0.6826.
Consistent-across-runs effects of pack A: q22-qa 0→0.8 (the moment-drift
near-miss recovered), q18-trake 0.5→0.65, q4-trake 0→0.05, q20/q25-kis +0.2
each, q24-kis −0.2 (diversify tail eviction, accepted trade). The AB−A delta
(+0.0087) sits inside run variance (q12/q13 flips with untouched-by-B KIS
paths, 23 Gemini-pro 504s that run), so pack B rides along as free format
insurance for the real 2026 pack — measured harmless (q22 held 0.8, zero
budget-exhausted warnings).

Pinned here: nb03's engine cell hard-sets all 9 knobs, and the nb04 bench
defaults to BENCH_PACK="AB" so the bench keeps measuring the battle line-up
(the round-70 mirror law).
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

KNOBS = {
    "CVP_SEARCH__NEIGHBOR_CONSISTENCY_BOOST": "0.15",
    "CVP_SEARCH__ROW_STRATEGY": "diversify_tail",
    "CVP_TEMPORAL__SUBMIT_STRATEGY": "jitter",
    "CVP_TEMPORAL__POOL_CONTEXT": "prepend",
    "CVP_TEMPORAL__EVENT_QUERY_VARIANTS": "all",
    "CVP_TEMPORAL__CAPTION_SIGNAL_WEIGHT": "0.2",
    "CVP_VQA__ANSWER_CANONICALIZE": "true",
    "CVP_VQA__ANSWER_NEIGHBOR_FRAMES": "1",
    "CVP_VQA__MAX_CALLS_PER_QUERY": "10",
}


def _cells(name: str) -> list[str]:
    nb = json.loads((REPO / "notebooks" / name).read_text(encoding="utf-8"))
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def test_r75_battle_engine_cell_sets_the_full_ab_pack():
    cell = next(s for s in _cells("03_test_system.ipynb") if "engine ready in" in s)
    for k, v in KNOBS.items():
        assert f'"{k}": "{v}"' in cell, (k, v)
    # the pack lands BEFORE the engine is built, so it reaches load_settings
    assert cell.index("Gói knob AB (round-75)") < cell.index("engine = SearchEngine(settings)")


def test_r75_bench_defaults_to_the_battle_pack():
    cell = next(s for s in _cells("04_lab_artifacts.ipynb") if "BENCH_PACK" in s)
    assert 'BENCH_PACK = "AB"' in cell
    # the values benched are the values fielded — bench pack A == battle values
    for k, v in KNOBS.items():
        assert f'"{k}": "{v}"' in cell, (k, v)


def test_r75_code_defaults_stay_legacy():
    # adoption lives in notebook env, NOT in code defaults — tests, scripts and
    # any bare load_settings() run legacy-identical (settings.yaml untouched).
    import yaml
    cfg = yaml.safe_load((REPO / "configs" / "settings.yaml").read_text(encoding="utf-8"))
    assert cfg["search"]["neighbor_consistency_boost"] == 0.0
    assert cfg["search"]["row_strategy"] == "legacy"
    assert cfg["temporal"]["submit_strategy"] == "legacy"
    assert cfg["vqa"]["answer_canonicalize"] is False
