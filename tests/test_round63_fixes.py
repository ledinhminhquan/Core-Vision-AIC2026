"""Round-63: never mkdir the project on a metadata-lazy mount.

Live 07 launch, diagnosed with an in-session probe: My Drive held TWO
`AIC2025` folders — the real one and an empty twin containing only an
artifacts skeleton. Each fresh Colab session bound the name to one of the
twins at random, so 07a/07c saw `AIC2025 = ['artifacts']` (data
FileNotFoundError) while 07b saw everything, reproducibly across VM swaps.
The twin factory was CELL_MOUNT's unconditional
`mkdir(parents=True, exist_ok=True)`: on a mount whose MyDrive listing had
not yet surfaced AIC2025, mkdir CREATED a second one (Drive allows duplicate
names). The mount cell now waits up to 3 minutes for the existing project to
become visible (with listing nudges) and only creates it after that grace
period — i.e. only on a genuine first-time setup — with a loud warning.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BUILDER = (REPO / "notebooks" / "_build_notebooks.py").read_text(encoding="utf-8")


def _mount_frag() -> str:
    return BUILDER.split("CELL_MOUNT = r")[1].split("CELL_REPO_DEPS")[0]


def test_r63_mkdir_waits_for_existing_project():
    frag = _mount_frag()
    assert "while not PROJECT.exists() and time.time() - _t0p < 180:" in frag
    assert "TUYỆT ĐỐI không tự tạo vội" in frag
    # the wait must come BEFORE the mkdir loop
    assert frag.index("while not PROJECT.exists()") < frag.index(
        "for p in (DATA_DIR, ARTIFACTS):")
    # round-66 upgraded the escape hatch: creation requires the explicit
    # FIRST_TIME_SETUP flag; otherwise the cell raises with the fresh-VM remedy
    assert "FIRST_TIME_SETUP" in frag
    assert "hỏng metadata Drive" in frag


def test_r63_every_generated_notebook_carries_the_guard():
    for nb_name in ("07a_ocr_paddle_shard1.ipynb", "05a_asr_large_shard1.ipynb",
                    "06d_caption_dense_sweeper.ipynb", "03_test_system.ipynb"):
        nb = json.loads((REPO / "notebooks" / nb_name).read_text(encoding="utf-8"))
        body = "\n".join("".join(c["source"]) for c in nb["cells"]
                         if c["cell_type"] == "code")
        assert "TUYỆT ĐỐI không tự tạo vội" in body
