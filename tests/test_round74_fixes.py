"""Round-74: BENCH_PACK A/B toggle in the nb04 bench cell.

The 66/67 diagnostics on the real bench#2 run (28/08) gave the campaign its
targets: TRAKE headroom to the grid ceiling averages 0.417/query with zero
benefit from current deep rows (jitter/diversify territory), the pool misses
the GT video on 1/3 TRAKE queries (pool_context territory), and all three QA
zeros are moment-drift or content-miss — NOT format — so the QA knobs are
insurance for the real pack, not a fix for this one.

The bench cell now takes ONE declared toggle, BENCH_PACK:
  "off" = bench#2-identical baseline; "A" = ranking/rows knobs (consistency
  boost, diversify_tail, 4 TRAKE knobs) — no API-semantics change; "AB" = A +
  the QA voting pack (canonicalize, neighbor strips ×2 calls, budget 10).
Knob envs are POPPED before applying so a re-run of the cell can never inherit
the previous run's pack, and bench_full.json records {bench_pack, bench_knobs}
so every saved bench self-identifies.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BUILD = (REPO / "notebooks" / "_build_notebooks.py").read_text(encoding="utf-8")
NB04 = json.loads((REPO / "notebooks" / "04_lab_artifacts.ipynb")
                  .read_text(encoding="utf-8"))
NB04_SRC = "".join("".join(c["source"]) for c in NB04["cells"]
                   if c["cell_type"] == "code")


def test_r74_bench_pack_default_off_and_validated():
    # round-75 flipped the declared default to "AB" (the adopted battle pack —
    # mirror law); the toggle itself plus its validation stay pinned here.
    assert 'BENCH_PACK = "AB"' in NB04_SRC
    # round-84 added the max-effort pack "ABX" to the allowed set
    assert 'assert BENCH_PACK in ("off", "A", "AB", "ABK", "ABX")' in NB04_SRC


def test_r74_pack_a_is_ranking_only_and_pack_b_is_qa():
    a = re.search(r"_PACK_A = \{(.*?)\}", NB04_SRC, re.S).group(1)
    b = re.search(r"_PACK_B = \{(.*?)\}", NB04_SRC, re.S).group(1)
    for k in ("CVP_SEARCH__NEIGHBOR_CONSISTENCY_BOOST", "CVP_SEARCH__ROW_STRATEGY",
              "CVP_TEMPORAL__SUBMIT_STRATEGY", "CVP_TEMPORAL__POOL_CONTEXT",
              "CVP_TEMPORAL__EVENT_QUERY_VARIANTS",
              "CVP_TEMPORAL__CAPTION_SIGNAL_WEIGHT"):
        assert k in a, k
    assert "CVP_VQA__" not in a          # A never touches API semantics
    for k in ("CVP_VQA__ANSWER_CANONICALIZE", "CVP_VQA__ANSWER_NEIGHBOR_FRAMES",
              "CVP_VQA__MAX_CALLS_PER_QUERY"):
        assert k in b, k


def test_r74_knob_envs_are_popped_before_apply():
    i_pop = NB04_SRC.index("os.environ.pop(_k, None)")
    i_set = NB04_SRC.index("os.environ[_k] = _v")
    assert i_pop < i_set                 # re-runs can't inherit the last pack


def test_r74_saved_bench_self_identifies():
    assert '"bench_pack": BENCH_PACK' in NB04_SRC
    assert '"bench_knobs": _knobs' in NB04_SRC
    assert "pack={BENCH_PACK}" in NB04_SRC


def test_r74_l3_and_l4_scrub_the_pack_envs():
    # kernel envs sống dai qua cell — tune và lane-A/B phải tự dọn pack keys
    # (audit round-74: L4 chạy sau L2 pack A sẽ đo lane với knob bật).
    cells = ["".join(c["source"]) for c in NB04["cells"] if c["cell_type"] == "code"]
    l3 = next(s for s in cells if "RUN_TUNE = " in s)
    l4 = next(s for s in cells if "RUN_METACLIP = " in s)
    for cell in (l3, l4):
        for k in ("CVP_SEARCH__NEIGHBOR_CONSISTENCY_BOOST",
                  "CVP_TEMPORAL__SUBMIT_STRATEGY", "CVP_VQA__ANSWER_CANONICALIZE"):
            assert k in cell, k
        assert "os.environ.pop(_k, None)" in cell


def test_r74_knob_names_exist_in_config():
    cfg = (REPO / "src" / "cvp" / "config.py").read_text(encoding="utf-8")
    for field in ("neighbor_consistency_boost", "row_strategy", "submit_strategy",
                  "pool_context", "event_query_variants", "caption_signal_weight",
                  "answer_canonicalize", "answer_neighbor_frames"):
        assert field in cfg, field
