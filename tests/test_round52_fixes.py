"""Round-52: Drive duplicate-folder trap on parallel shard mkdir.

Live 05 run (24/08): three shards launched together; Google Drive created TWO
same-named `asr-large-partial` folders (Drive allows duplicate names — a
concurrent mkdir race), shard 1 bound to one twin and shards 2+3 to the other.
Every shard finished its 291/291 videos, yet no session ever counted 873/873,
so FINALIZE never fired (shard 1 stayed at 291 for over an hour). Two defenses:
(1) only shard 0 CREATES the shared store — later shards WAIT to see it;
(2) a start-of-cell net merges any "<name> (1)" twin into the canonical store,
    so rerunning the build cell on ONE session self-heals and finalizes.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BUILDER = (REPO / "notebooks" / "_build_notebooks.py").read_text(encoding="utf-8")

_SWEEPS = (("NB5_SWEEP = r", "NB6_TITLE"), ("NB6_SWEEP = r", "def main"))


def _frag(start: str, end: str) -> str:
    return BUILDER.split(start)[1].split(end)[0]


def _code_sources(nb_name: str):
    nb = json.loads((REPO / "notebooks" / nb_name).read_text(encoding="utf-8"))
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def test_r52_creation_gate_only_shard0_creates():
    for start, end in _SWEEPS:
        frag = _frag(start, end)
        assert "SHARD_INDEX == 0 or _partial.exists()" in frag
        assert "đợi ca 1 tạo kho chung" in frag
        # the gate must run BEFORE this shard's own mkdir of the store
        assert frag.index("SHARD_INDEX == 0 or") < frag.index("_partial.mkdir")


def test_r52_twin_store_merge_before_seed():
    for start, end in _SWEEPS:
        frag = _frag(start, end)
        assert 'glob(_partial.name + " (*")' in frag
        assert "Gộp kho trùng tên" in frag
        # merge must precede the resume seed so the staging copy gets the UNION
        assert frag.index("Gộp kho trùng tên") < frag.index(
            "copytree(_partial, _job_local")


def test_r52_baked_shards_keep_gate_literal():
    """Shard baking rewrites the `SHARD_INDEX = 0` assignment only — the
    `SHARD_INDEX == 0` creation-gate comparison must survive verbatim in every
    generated file (shards b/c must NOT create the store themselves)."""
    for fam, job in (("05", "asr_large"), ("06", "caption_dense")):
        for i, letter in enumerate("abc"):
            src = "\n".join(_code_sources(f"{fam}{letter}_{job}_shard{i + 1}.ipynb"))
            assert f"SHARD_INDEX = {i}" in src
            assert "SHARD_INDEX == 0 or _partial.exists()" in src
            if i:
                assert "SHARD_INDEX = 0" not in src
