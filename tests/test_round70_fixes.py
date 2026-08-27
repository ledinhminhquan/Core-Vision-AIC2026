"""Round-70: the Lab must measure the REAL battle line-up.

Pre-run review of nb04 against the upgraded artifacts found the bench cell
still frozen in the pre-tournament era: it benched the single `finetuned`
lane with DEFAULT fusion weights, while the round-49 battle config is the
finetuned+metaclip2 60/40 ensemble with tuned+50%-shrinkage weights. Fixes:

1. LAB_BENCH_FULL mirrors nb03's engine env exactly — ensemble members and
   weights, plus the identical tuned-weights loader (same `.get("best")`
   structure, same shrinkage baseline) — so bench numbers are the battle's
   numbers.
2. LAB_TUNE dumps signals on the same battle retrieval lane (ensemble), and
   backs up the live best_weights.json to best_weights-prev.json before the
   tuner overwrites it — one-swap rollback, matching the deliverables/
   previous pattern.
(23_dump_signals re-dumps every query fresh — verified no resume-skip, so no
stale-signal trap after the artifact upgrades.)
"""
from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BUILDER = (REPO / "notebooks" / "_build_notebooks.py").read_text(encoding="utf-8")


def _frag(start: str, end: str) -> str:
    return BUILDER.split(start)[1].split(end)[0]


def test_r70_bench_mirrors_the_battle_lineup():
    frag = _frag("LAB_BENCH_FULL = r", "LAB_TUNE")
    assert '"CVP_EMBEDDING__MODEL"] = "ensemble"' in frag
    assert '\'["finetuned", "metaclip2"]\'' in frag
    assert '"[0.6, 0.4]"' in frag
    # the tuned-weights loader matches nb03's structure and shrinkage exactly
    assert '.get("best", {}).get("weights")' in frag
    assert '_base = {"visual": 1.0, "ocr": 0.35' in frag
    assert "0.5 * float(_w.get(k, v)) + 0.5 * v" in frag
    # the old single-lane bench is gone
    assert '"CVP_EMBEDDING__MODEL"] = "finetuned"' not in frag


def test_r70_tuner_uses_battle_lane_and_keeps_a_rollback():
    frag = _frag("LAB_TUNE = r", "LAB_METACLIP")
    assert '"CVP_EMBEDDING__MODEL"] = "ensemble"' in frag
    assert "best_weights-prev.json" in frag
    # the backup happens BEFORE the tuner run writes the new file
    assert frag.index("best_weights-prev.json") < frag.index("21_tune_weights.py")


def test_r70_generated_nb04_carries_it():
    import json
    nb = json.loads((REPO / "notebooks" / "04_lab_artifacts.ipynb")
                    .read_text(encoding="utf-8"))
    body = "\n".join("".join(c["source"]) for c in nb["cells"]
                     if c["cell_type"] == "code")
    assert body.count('"CVP_EMBEDDING__MODEL"] = "ensemble"') >= 2
    assert "best_weights-prev.json" in body
    assert "y hệt trận" in body
