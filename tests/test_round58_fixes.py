"""Round-58: two-way partial-store sync + the 06d "sweeper" notebook.

With a 4th parallel session available, a naive 4th worker walking the list
would duplicate work: the a/b/c shards seed their staging ONCE at cell start,
so anything another session finishes later gets recomputed when a shard
reaches it. Two pieces fix that:
1. the 10-minute syncer now syncs BOTH ways — after pushing local results it
   pulls store files it does not have yet, so every session learns what the
   others finished within ~10 minutes and its job subprocess skips those
   videos (the resume check reads staging per video, at the moment it gets
   there);
2. 06d_caption_dense_sweeper.ipynb — an optional 4th session that walks the
   video list in REVERSE, taking batches of 12 not-yet-done videos and
   refreshing its view of the shared store between batches (so overlap with
   the forward shards is at most the in-flight tail). It never creates the
   shared store, shares the same finalize lock/marker, and is generated from
   NB6_SWEEP by assert-guarded surgery in main().
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


def test_r58_syncer_pulls_as_well_as_pushes():
    for fam in _SWEEPS:
        frag = _frag(fam)
        syncer = frag.split("def _syncer():")[1].split("threading.Thread")[0]
        assert "KÉO chiều về" in syncer
        assert "đẩy {_n} / kéo {_p_new}" in syncer


def test_r58_sweeper_notebook_generated_correctly():
    srcs = _code_sources("06d_caption_dense_sweeper.ipynb")
    assert len(srcs) == 7                          # same cell layout as 06a/b/c
    body = "\n".join(srcs)
    assert "SHARD_INDEX = 3" in body               # gate: never creates the store
    assert "KHÔNG BAO GIỜ tạo kho chung" in body
    assert "reversed(_vids)" in body               # walks the list backwards
    assert '"--videos", *_batch' in body           # batched job invocations
    assert '"--videos", *_my' not in body          # the single big run is gone
    assert "_todo[:12]" in body                    # small batches → fresh view
    # the sweeper keeps the full hardened finalize (shared lock + done marker)
    assert "_finalize.done" in body
    assert "_touch_lock()" in body


def test_r58_sweeper_stride_matches_the_family():
    body = "\n".join(_code_sources("06d_caption_dense_sweeper.ipynb"))
    # batch run + finalize makeup run — both at stride 1, like 06a/b/c
    assert body.count('"--caption-stride", "1"') == 2


def test_r58_generated_shards_carry_the_pull():
    for fam, job in (("05", "asr_large"), ("06", "caption_dense")):
        src = "\n".join(_code_sources(f"{fam}a_{job}_shard1.ipynb"))
        assert "KÉO chiều về" in src
