"""Round-56: FINALIZE hardening — the day's Drive failure modes, армored.

The 24/08 live runs surfaced four ways FINALIZE could still fail or silently
degrade the battle artifacts:
1. orphan lock — a finalize that crashes (or a killed VM) left _finalize.lock
   behind forever, and every future session skipped finalize;
2. short staging seed — copytree(partial → local) over FUSE delivered 863/873
   live; finalize would have rebuilt the battle text index missing videos;
3. short aux staging — the same listing lag can under-copy the ocr/captions
   (or asr) stores pulled in for the BM25 rebuild;
4. short upload — the final copytree(local → Drive) can drop files unnoticed.

Both sweep templates now: distinguish done/running/crashed via a
_finalize.done marker + 2h lock freshness (stale locks are taken over);
release the lock on any exception; retry the staging seed and recompute any
still-missing videos in place (returning them to the shared store); verify
aux-staging counts with retries and abort loudly instead of building a
degraded index; and count-verify the Drive upload with a top-up copy.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BUILDER = (REPO / "notebooks" / "_build_notebooks.py").read_text(encoding="utf-8")

_SWEEPS = {"05": ("NB5_SWEEP = r", "NB6_TITLE"), "06": ("NB6_SWEEP = r", "def main")}


def _frag(fam: str) -> str:
    start, end = _SWEEPS[fam]
    return BUILDER.split(start)[1].split(end)[0]


def _code_sources(nb_name: str):
    nb = json.loads((REPO / "notebooks" / nb_name).read_text(encoding="utf-8"))
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def test_r56_lock_knows_done_running_and_crashed():
    for fam in _SWEEPS:
        frag = _frag(fam)
        assert '_done_mark = _partial / "_finalize.done"' in frag
        assert "_lock_fresh" in frag and "6 * 3600" in frag  # r57 raised 2h→6h
        assert "tiếp quản" in frag                       # stale-lock takeover
        assert "_lock.unlink(missing_ok=True)" in frag   # released on crash
        assert "except BaseException:" in frag
        # the done marker is written before the crash-guard ends
        assert frag.index("_done_mark.write_text") < frag.index("except BaseException:")


def test_r56_staging_seed_retries_and_recomputes():
    for fam in _SWEEPS:
        frag = _frag(fam)
        assert "kéo lại {_try}/3" in frag
        assert "tự chạy bù" in frag and '"--videos", *_still' in frag
        assert "trả bản bù về kho chung" in frag


def test_r56_recompute_uses_the_family_job():
    assert '"--asr",\n                     "--videos", *_still' in _frag("05")
    frag06 = _frag("06")
    # the top-up captions run must keep stride 1 (same as the main job)
    assert frag06.count('"--caption-stride", "1"') == 2


def test_r56_aux_staging_and_upload_are_count_verified():
    for fam in _SWEEPS:
        frag = _frag(fam)
        assert "_got >= _need" in frag
        assert "Kéo kho {_aux} về máy mãi vẫn thiếu" in frag
        assert "_n_dst < _n_src" in frag                 # upload top-up copy


def test_r56_generated_shards_carry_the_hardening():
    for fam, job in (("05", "asr_large"), ("06", "caption_dense")):
        for i, letter in enumerate("abc"):
            src = "\n".join(_code_sources(f"{fam}{letter}_{job}_shard{i + 1}.ipynb"))
            assert "_finalize.done" in src
            assert "except BaseException:" in src
