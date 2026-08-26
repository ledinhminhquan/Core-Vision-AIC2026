"""Round-61: fixes for the final-verification audit of the round-60 state.

Wheel availability was CONFIRMED live (cu126 and cu129 official indexes both
carry paddlepaddle_gpu 3.2.0/3.2.1/3.2.2 cp312 linux wheels; the paddleocr<4
float was verified compatible). Four residual findings, all fixed:

1. (degrade) a single transient smoke failure (model-download hiccup) used to
   permanently downgrade a healthy GPU session to CPU paddle — the GPU smoke
   now retries once (30s apart) before the CPU fallback.
2. (degrade) the finalize backfill never re-checked that the recomputed
   videos actually landed in staging — with round-60's skip-write, a store
   missing videos could be swapped in and _finalize.done stamped forever.
   The backfill now re-verifies and ABORTS (lock released) if still short.
3. (degrade) round-13 + round-60 wedge: one video with locally corrupt
   keyframes became "first processed" on every rerun and was misdiagnosed as
   systemic failure forever — the round-13 guard (ocr.py AND captioner.py)
   now anchors on the first video of the todo LIST.
4. (minor) the GPU matmul probe swallowed stderr — it now prints the tail on
   failure so cu-index/driver problems are diagnosable from the log.
Plus, from the Drive-touch whitelist sweep: the twin-merge now only accepts
Drive's own " (N)" duplicate suffix — a hand-made folder like
"ocr-v2-partial (backup)" is no longer merged-then-deleted.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BUILDER = (REPO / "notebooks" / "_build_notebooks.py").read_text(encoding="utf-8")
OCR_SRC = (REPO / "src" / "cvp" / "auxindex" / "ocr.py").read_text(encoding="utf-8")
CAP_SRC = (REPO / "src" / "cvp" / "auxindex" / "captioner.py").read_text(encoding="utf-8")

_SWEEPS = {"05": ("NB5_SWEEP = r", "NB6_TITLE"), "06": ("NB6_SWEEP = r", "NB6D_TITLE")}


def _frag(fam: str) -> str:
    start, end = _SWEEPS[fam]
    return BUILDER.split(start)[1].split(end)[0]


def _body(nb_name: str) -> str:
    nb = json.loads((REPO / "notebooks" / nb_name).read_text(encoding="utf-8"))
    return "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")


def test_r61_gpu_smoke_retries_before_cpu_downgrade():
    body = _body("07a_ocr_paddle_shard1.ipynb")
    assert "thử lại sau 30s" in body
    assert "for _try in (1, 2):" in body
    # probe diagnostics are printed, not swallowed
    assert "probe GPU lỗi" in body
    assert "_probe.stderr" in body


def test_r61_backfill_recheck_blocks_short_finalize():
    for fam in _SWEEPS:
        frag = _frag(fam)
        assert "_still2" in frag
        assert "không chốt kho thiếu" in frag
    # inherited by the surgery families
    for nb in ("07a_ocr_paddle_shard1.ipynb", "06d_caption_dense_sweeper.ipynb"):
        assert "không chốt kho thiếu" in _body(nb)


def test_r61_round13_guard_anchored_on_first_of_list():
    assert "if vid == todo_videos[0] and len(grp) > 0" in OCR_SRC
    assert "if vid == todo_videos[0] and len(wanted) > 0" in CAP_SRC
    assert "if done == 0 and len(grp) > 0" not in OCR_SRC
    assert "if done == 0 and len(wanted) > 0" not in CAP_SRC


def test_r61_twin_merge_only_accepts_drive_numeric_suffix():
    for fam in _SWEEPS:
        frag = _frag(fam)
        assert '_suf[2:-1].isdigit()' in frag
    assert '_suf[2:-1].isdigit()' in _body("07a_ocr_paddle_shard1.ipynb")
