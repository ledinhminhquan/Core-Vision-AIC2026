"""Round-55: stage the catalog to local before launching the aux build job.

Live 05a run (24/08): with round-53's VM-local staging root, the aux script
crashed at startup — `FileNotFoundError: Catalog missing:
/content/artifacts/catalog/manifest.parquet` — because 03_build_aux_indexes.py
unconditionally does `catalog.load()` and the catalog only exists on Drive.
Both sweep templates now copy `artifacts/catalog` from Drive into the local
staging root (with a clear assert if Drive has no catalog) BEFORE the build
subprocess launches. The same run also proved round-54's guards live: the
stray `L30_V051 (1).json` merge-duplicate was pruned and named in the output.
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


def test_r55_catalog_staged_before_the_job():
    for start, end in _SWEEPS:
        frag = _frag(start, end)
        assert 'shutil.copytree(_drv_cat, _la / "catalog", dirs_exist_ok=True)' in frag
        assert "Thiếu artifacts/catalog/manifest.parquet" in frag
        # staging must happen after the local repoint and before the subprocess
        repoint = frag.index('CVP_PATHS__ARTIFACTS_ROOT"] = str(_la)')
        stage = frag.index("_drv_cat, _la / ")
        assert repoint < stage < frag.index("03_build_aux_indexes.py")


def test_r55_generated_shards_stage_catalog():
    for fam, job in (("05", "asr_large"), ("06", "caption_dense")):
        for i, letter in enumerate("abc"):
            src = "\n".join(_code_sources(f"{fam}{letter}_{job}_shard{i + 1}.ipynb"))
            assert '_drv_cat = PROJECT / "artifacts" / "catalog"' in src
