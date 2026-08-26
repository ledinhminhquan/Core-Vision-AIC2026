"""Round-65: paddle GPU may never touch Colab torch's CUDA stack.

Live 07a (with round-64's surfaced logs): `pip install paddlepaddle-gpu`
pulled OLDER nvidia-* wheels over the exact ones Colab's torch links against
— every NEW python process died importing torch
("libtorch_cuda.so: undefined symbol: ncclCommShrink"), which killed the
smoke test on BOTH the GPU and CPU paths (paddleocr → paddlex → modelscope →
import torch). The repo's iron rule "never touch Colab torch" now covers the
INDIRECT route:

- paddlepaddle-gpu installs with --no-deps (it shares torch's newer nvidia-*
  wheels — forward-compatible), with its harmless pure-python deps added
  explicitly;
- immediately after the installs, a fresh-subprocess torch health check
  (cuda tensor op) aborts loudly with the fresh-VM remedy if torch was
  damaged anyway — no more silent poisoned sessions.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _body(nb_name: str) -> str:
    nb = json.loads((REPO / "notebooks" / nb_name).read_text(encoding="utf-8"))
    return "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")


BODY = _body("07a_ocr_paddle_shard1.ipynb")


def test_r65_paddle_gpu_installs_with_no_deps():
    assert '"--no-deps",\n            "paddlepaddle-gpu==3.2.*"' in BODY
    # the harmless pure-python deps are added explicitly
    assert '"decorator", "astor", "opt_einsum"' in BODY


def test_r65_torch_health_check_gates_the_sweep():
    assert "import torch; assert torch.cuda.is_available()" in BODY
    assert "torch của Colab đã bị bộ cài Paddle làm hỏng" in BODY
    # the check must run BEFORE the paddle GPU probe / smoke
    assert BODY.index("torch nguyên vẹn sau khi cài Paddle") < BODY.index(
        "paddle.set_device('gpu')")


def test_r65_all_three_shards_carry_it():
    for i, letter in enumerate("abc"):
        body = _body(f"07{letter}_ocr_paddle_shard{i + 1}.ipynb")
        assert '"--no-deps",\n            "paddlepaddle-gpu==3.2.*"' in body
        assert "torch nguyên vẹn sau khi cài Paddle" in body
