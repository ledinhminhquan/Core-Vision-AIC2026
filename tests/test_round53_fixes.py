"""Round-53: shard staging must live on the VM's LOCAL disk, never on Drive.

Live 05 run (24/08, discovered via Drive Activity): cell 4 leaves
CVP_PATHS__ARTIFACTS_ROOT pointing at Drive and the sweep cell derived its
"local" staging from that env var — so `_job_local` WAS `Drive artifacts/asr`.
Consequences observed live: the restored medium ASR store was rmtree'd to the
Drive trash, three sessions racing rmtree+copytree spawned duplicate same-named
`asr` staging folders, the job's end-of-run text-index build overwrote the
battle `text_index` on Drive with a partial-coverage one, and FINALIZE would
have crashed (rename moves `asr` away, then copytree reads the same path).
The 06 family was one Run all away from rmtree'ing the real captions store.

Fix: the sweep cell repoints _la to /content/artifacts (true local) and
exports CVP_PATHS__ARTIFACTS_ROOT before launching the build subprocess —
Drive is only ever touched through the partial-store syncer and FINALIZE.
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


def test_r53_staging_root_is_vm_local():
    for start, end in _SWEEPS:
        frag = _frag(start, end)
        assert '_la = Path("/content/artifacts")' in frag
        assert 'os.environ["CVP_PATHS__ARTIFACTS_ROOT"] = str(_la)' in frag
        # the Drive-rooted derivation must be gone entirely
        assert '_la = Path(os.environ["CVP_PATHS__ARTIFACTS_ROOT"])' not in frag


def test_r53_repoint_happens_before_the_build_job():
    """The subprocess inherits os.environ — the export must precede launch,
    and the staging rmtree must come after the repoint (so it can only ever
    delete the LOCAL staging copy, never a real Drive store)."""
    for start, end in _SWEEPS:
        frag = _frag(start, end)
        repoint = frag.index('CVP_PATHS__ARTIFACTS_ROOT"] = str(_la)')
        assert repoint < frag.index("03_build_aux_indexes.py")
        # r57 replaced the rmtree+copytree seed with a fill-missing loop; the
        # invariant stands: staging is created only after the local repoint.
        assert repoint < frag.index("_job_local.mkdir")


def test_r53_generated_shards_carry_local_staging():
    for fam, job in (("05", "asr_large"), ("06", "caption_dense")):
        for i, letter in enumerate("abc"):
            src = "\n".join(_code_sources(f"{fam}{letter}_{job}_shard{i + 1}.ipynb"))
            assert '_la = Path("/content/artifacts")' in src
            assert '_la = Path(os.environ["CVP_PATHS__ARTIFACTS_ROOT"])' not in src
