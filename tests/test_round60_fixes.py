"""Round-60: fixes for the 5 confirmed findings of the 07-family audit.

1. (blocker) PyPI's `paddlepaddle-gpu` is frozen at 2.6.2/CUDA-10.2 — 3.x GPU
   wheels live only on Paddle's own index. The install now pins
   `paddlepaddle-gpu==3.2.*` with `--extra-index-url` chosen by GPU arch
   (Blackwell sm_120 → cu129, else cu126), pins paddleocr <4, and skips the
   Drive-backed pip cache (--no-cache-dir).
2. (blocker) `is_compiled_with_cuda()` is a compile-time flag that passes on
   unusable wheels — the GPU gate now executes a REAL matmul on the GPU, and
   a smoke failure on the GPU install retries the pinned CPU path
   (`paddlepaddle==3.2.*`, dodging the 3.3 oneDNN/PIR regression) before
   giving up.
3. (degrade) a mid-run sticky engine death used to write processed_count:0
   jsons that the name-based finalize gates count as done — ocr_all_keyframes
   now SKIPS writing all-fail videos and aborts after 3 consecutive ones.
4. (degrade) the 3.x PaddleOCR constructor now disables the document
   preprocessing defaults (orientation/unwarp/textline) that are useless for
   TV keyframes and multiply per-frame latency.
5. (minor) the smoke test finds a keyframe via local dir OR the Drive dir
   (next() instead of sorting 177k paths), so the Drive-direct fallback path
   still passes the gate.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OCR_SRC = (REPO / "src" / "cvp" / "auxindex" / "ocr.py").read_text(encoding="utf-8")


def _body(nb_name: str) -> str:
    nb = json.loads((REPO / "notebooks" / nb_name).read_text(encoding="utf-8"))
    return "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")


BODY = _body("07a_ocr_paddle_shard1.ipynb")


def test_r60_gpu_wheel_comes_from_the_official_index_pinned():
    assert '"paddlepaddle-gpu==3.2.*"' in BODY
    assert "paddlepaddle.org.cn/packages/stable/cu126/" in BODY
    assert "paddlepaddle.org.cn/packages/stable/cu129/" in BODY   # Blackwell lane
    assert "get_device_capability" in BODY                        # arch-aware pick
    assert '"--extra-index-url"' in BODY                          # deps still on PyPI
    assert '"--no-cache-dir"' in BODY                             # skip Drive pip cache
    assert '"paddleocr>=3.0,<4"' in BODY


def test_r60_gpu_gate_runs_a_real_op_and_cpu_retry_is_pinned():
    assert "paddle.set_device('gpu')" in BODY
    assert "x = paddle.ones([64, 64])" in BODY
    assert "is_compiled_with_cuda" not in BODY                    # compile-flag gate gone
    assert '"paddlepaddle==3.2.*"' in BODY                        # CPU fallback pin
    # smoke runs on the GPU install AND again after the CPU fallback
    assert BODY.count("_smoke_rc()") >= 3                         # def + 2 call sites
    assert "THẤT BẠI cả GPU lẫn CPU" in BODY


def test_r60_smoke_keyframe_discovery_supports_drive_fallback():
    assert 'PROJECT / "data" / "keyframes"' in BODY
    assert "next(iter(_kd.glob(" in BODY                          # no 177k sort


def test_r60_all_fail_videos_are_not_written_and_abort_early():
    assert "consecutive_failures" in OCR_SRC
    assert "not writing" in OCR_SRC
    assert "3 consecutive videos" in OCR_SRC
    # the skip guard must sit BEFORE the store write
    assert OCR_SRC.index("consecutive_failures += 1") < OCR_SRC.index(
        'atomic_write_json(out_path, {"n_to_text"')


def test_r60_paddle3_constructor_disables_doc_preprocessing():
    assert "use_doc_orientation_classify=False" in OCR_SRC
    assert "use_doc_unwarping=False" in OCR_SRC
    assert "use_textline_orientation=False" in OCR_SRC
