"""Round-59: the 07 OCR family — PaddleOCR re-read of every keyframe.

The weight tuner wanted OCR crushed toward zero (0.35 → 0.0124 raw), the
classic signature of a NOISY signal — EasyOCR's Vietnamese diacritics are the
suspect. The repo already carried a defensive _PaddleEngine (2.x/3.x APIs) and
a `settings.ocr.engine` switch; round-59 adds the zero-edit shard notebooks
07a/b/c generated from NB6_SWEEP by assert-guarded surgery, inheriting the
full round-52..58 armor. New Drive names (the ONLY ones this family may
create): `ocr-v2-partial` (shared store), `ocr-easyocr-backup` (old store
after finalize). The sweep installs paddlepaddle-gpu (CPU fallback with a
loud warning) and runs a one-keyframe smoke test BEFORE burning GPU hours.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BUILDER = (REPO / "notebooks" / "_build_notebooks.py").read_text(encoding="utf-8")
OCR_SRC = (REPO / "src" / "cvp" / "auxindex" / "ocr.py").read_text(encoding="utf-8")
CFG_SRC = (REPO / "src" / "cvp" / "config.py").read_text(encoding="utf-8")


def _code_sources(nb_name: str):
    nb = json.loads((REPO / "notebooks" / nb_name).read_text(encoding="utf-8"))
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def test_r59_engine_switch_exists_in_src():
    assert "class _PaddleEngine" in OCR_SRC
    assert 'if engine == "paddle":' in OCR_SRC
    assert 'engine: str = "easyocr"' in CFG_SRC   # battle default untouched


def test_r59_builder_generates_the_07_family():
    assert 'write_nb(f"07{_letter}_ocr_paddle_shard{_i + 1}.ipynb"' in BUILDER
    assert "NB7 surgery mất mốc" in BUILDER       # assert-guarded surgery


def test_r59_generated_shards_are_correct():
    for i, letter in enumerate("abc"):
        srcs = _code_sources(f"07{letter}_ocr_paddle_shard{i + 1}.ipynb")
        assert len(srcs) == 7                     # same layout as 05/06 shards
        body = "\n".join(srcs)
        assert f"SHARD_INDEX = {i}" in body
        assert "SHARD_TOTAL = 3" in body
        # the family touches ONLY its own Drive names
        assert "ocr-v2-partial" in body
        assert "ocr-easyocr-backup" in body
        assert "captions-dense-partial" not in body
        assert "captions-stride4-backup" not in body
        # job wiring: paddle engine, --ocr, no caption flags anywhere
        assert 'os.environ["CVP_OCR__ENGINE"] = "paddle"' in body
        assert '"--ocr",' in body
        assert '"--captions"' not in body and "--caption-stride" not in body
        assert '_job_local = _la / "ocr"' in body
        assert 'if _aux == "ocr" or not _src.is_dir():' in body
        assert 'for _d in ("ocr", "text_index"):' in body
        # smoke test gates the sweep, with a CPU fallback that uninstalls first
        assert "SMOKE OCR" in body
        assert "PaddleOCR smoke test THẤT BẠI" in body
        assert '"uninstall", "-q", "-y"' in body
        # full round-52..58 armor inherited
        assert "_finalize.done" in body and "_touch_lock()" in body
        assert "KÉO chiều về" in body and "def _prune_staging():" in body
