"""Round-66: last three findings from the r65 verification audit.

The core --no-deps strategy was CONFIRMED against real wheel metadata
(cp313 wheels exist on cu126/cu129; torch cu128's nccl 2.28.9 carries
ncclCommShrink that paddle's 2.25 pin lacked — the exact crash mechanics;
same site-packages/nvidia lib paths). Three residual fixes:

1. the torch health check no longer false-aborts CPU-only runtimes — the
   cuda assertion only applies when a GPU was detected (_sm != (0, 0));
2. the paddleocr install (a ~170-package dependency tree) now checks its
   return code and retries once before failing loudly — instead of passing
   silently and wasting the whole smoke/CPU-fallback ladder;
3. the twin-project mkdir window is CLOSED: creating the project folder now
   requires the explicit FIRST_TIME_SETUP=True flag in cell 1 — a
   metadata-dead VM raises with the fresh-VM remedy instead of ever creating
   a twin AIC2025.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BUILDER = (REPO / "notebooks" / "_build_notebooks.py").read_text(encoding="utf-8")


def _body(nb_name: str) -> str:
    nb = json.loads((REPO / "notebooks" / nb_name).read_text(encoding="utf-8"))
    return "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")


BODY = _body("07a_ocr_paddle_shard1.ipynb")


def test_r66_torch_check_tolerates_cpu_runtimes():
    assert '_sm != (0, 0)' in BODY
    assert "import torch; print(torch.zeros(2).sum().item())" in BODY
    assert "đừng chẩn oan" in BODY


def test_r66_paddleocr_install_checked_and_retried():
    assert "cài paddleocr lỗi (lần {_try})" in BODY
    assert "Cài paddleocr thất bại sau 2 lần" in BODY


def test_r66_project_creation_requires_explicit_flag():
    frag = BUILDER.split("CELL_PARAMS = r")[1].split("CELL_MOUNT")[0]
    assert "FIRST_TIME_SETUP = False" in frag
    mount = BUILDER.split("CELL_MOUNT = r")[1].split("CELL_REPO_DEPS")[0]
    assert "if not FIRST_TIME_SETUP:" in mount
    assert mount.index("raise RuntimeError") < mount.index(
        "for p in (DATA_DIR, ARTIFACTS):")
    # every generated notebook carries both the flag and the guard
    for nb in ("07a_ocr_paddle_shard1.ipynb", "03_test_system.ipynb",
               "06d_caption_dense_sweeper.ipynb"):
        body = _body(nb)
        assert "FIRST_TIME_SETUP = False" in body
        assert "if not FIRST_TIME_SETUP:" in body
