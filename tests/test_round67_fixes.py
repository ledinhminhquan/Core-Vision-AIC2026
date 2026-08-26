"""Round-67: paddleocr 3.x rejects unknown kwargs with ValueError.

Live 07a (with round-64's surfaced logs): PaddleOCR 3.7's constructor raises
`ValueError: Unknown argument: show_log` — not TypeError — so _PaddleEngine's
2.x-first `except TypeError` chain never reached its 3.x branch and every
init died on both the GPU and CPU paths. The constructor now tries the 3.x
signature FIRST (the 07 notebooks pin paddleocr>=3.0,<4) and catches BOTH
TypeError and ValueError at every fallback step. GPU choice is irrelevant to
this failure (pure-python argument validation).
"""
from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OCR_SRC = (REPO / "src" / "cvp" / "auxindex" / "ocr.py").read_text(encoding="utf-8")


def test_r67_paddle_ctor_tries_3x_first_and_catches_valueerror():
    # 3.x attempt must come before the 2.x show_log attempt
    assert OCR_SRC.index("use_doc_orientation_classify=False") < OCR_SRC.index(
        'show_log=False')
    # both fallback hops catch ValueError as well as TypeError
    assert OCR_SRC.count("except (TypeError, ValueError)") == 2
    # the old TypeError-only hop is gone from the constructor chain
    ctor = OCR_SRC.split("class _PaddleEngine")[1].split("def read")[0]
    assert "except TypeError:" not in ctor
