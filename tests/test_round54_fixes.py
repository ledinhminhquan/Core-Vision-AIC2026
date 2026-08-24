"""Round-54: count the partial store by VIDEO NAME, and prune stray files.

After the hand-merge of the duplicate `asr-large-partial` twins in Drive web,
the surviving store held 874 items for 873 videos — one stray (a same-name
"xxx (1).json" duplicate from the move, or a file uploaded into the wrong
folder). The old finalize gate counted FILES (`len(glob) >= len(_vids)`), so a
stray inflates the count and could green-light finalize while an actual video
is missing; a stray json copied into staging could also feed the store or
text-index a phantom video id. Two changes in both sweep templates:
(1) `_prune_staging()` drops anything that is not `<video>.json` for a known
    video (called after both staging seeds, printing what it dropped);
(2) the completion gate compares STEMS against the video list (`_missing`),
    printing examples when short.
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


def test_r54_prune_defined_and_called_after_both_seeds():
    for start, end in _SWEEPS:
        frag = _frag(start, end)
        assert "def _prune_staging():" in frag
        assert '_p.suffix == ".json" and _p.stem in _vidset' in frag
        # def + one call after the resume seed + one inside FINALIZE's re-seed
        assert frag.count("_prune_staging()") == 3


def test_r54_gate_counts_video_stems_not_files():
    for start, end in _SWEEPS:
        frag = _frag(start, end)
        assert "_missing = [_v for _v in _vids if _v not in _done]" in frag
        assert "if not _missing:" in frag
        assert 'len(list(_partial.glob("*.json"))) >= len(_vids)' not in frag
        assert "còn thiếu" in frag


def test_r54_generated_shards_carry_the_guards():
    for fam, job in (("05", "asr_large"), ("06", "caption_dense")):
        for i, letter in enumerate("abc"):
            src = "\n".join(_code_sources(f"{fam}{letter}_{job}_shard{i + 1}.ipynb"))
            assert "def _prune_staging():" in src
            assert "if not _missing:" in src
