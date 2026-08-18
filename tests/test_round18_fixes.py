"""Round-18 — training-session-4 audit (18/08/2026).

Session 4 revealed: prelim QA is Q&A, NOT VQA — questions may depend on the
AUDIO. The VQA path now injects the ASR transcript around the candidate
moment into every prompt (Gemini strip, Gemini single, local Vintern), and
the batch runner + app thread it through. Prompts are domain-neutral (B1 is
cooking/education, not news) and query enhancement expands named entities
into visual descriptions.
"""

from __future__ import annotations

from pathlib import Path

from cvp.utils.io import atomic_write_json

REPO = Path(__file__).resolve().parents[1]


def test_asr_context_picks_overlapping_segments(corpus):
    from cvp.search.vqa import asr_context

    atomic_write_json(corpus.paths.art("asr") / "L21_V001.json", {"segments": [
        {"start": 0.0, "end": 5.0, "text": "mở đầu bản tin"},
        {"start": 100.0, "end": 110.0, "text": "cho hai muỗng đường vào chảo"},
        {"start": 500.0, "end": 505.0, "text": "kết thúc"},
    ]})
    ctx = asr_context(corpus, "L21_V001", pts_time=105.0, window_s=20.0)
    assert "hai muỗng đường" in ctx
    assert "mở đầu" not in ctx and "kết thúc" not in ctx
    assert asr_context(corpus, "L21_V404", 10.0) == ""      # no artifact → ""


def test_with_context_prefixes_transcript():
    from cvp.search.vqa import _VQA_PROMPT, _with_context

    base = _VQA_PROMPT.format(question="Ai đang nói?")
    assert _with_context(base, "") == base                   # no context → unchanged
    out = _with_context(base, "xin chào quý vị")
    assert out.index("xin chào quý vị") < out.index("Ai đang nói?")
    assert "âm thanh" in out                                 # labeled as ASR speech


def test_vqa_public_api_accepts_context():
    import inspect

    from cvp.search.vqa import VqaAssistant

    for name in ("answer_group", "suggest", "_ask_gemini", "_ask_local",
                 "_ask_gemini_strip"):
        assert "context" in inspect.signature(
            getattr(VqaAssistant, name)).parameters, name


def test_prompts_are_domain_neutral():
    # B1 = cooking/education, B2 = traffic/sports — "tin tức" assumptions gone
    for rel in ("src/cvp/search/vqa.py", "src/cvp/models/query_processor.py"):
        src = (REPO / rel).read_text(encoding="utf-8")
        for marker in ('"Bạn đang xem một khung hình từ video tin tức',
                       "Vietnamese TV news"):
            assert marker not in src, rel


def test_enhancement_prompt_expands_named_entities():
    src = (REPO / "src" / "cvp" / "models" / "query_processor.py").read_text(encoding="utf-8")
    assert "VISUALLY observable" in src                      # Donald-Trump case


def test_run_queries_threads_asr_context_with_stub_fallback():
    src = (REPO / "src" / "cvp" / "pipeline" / "run_queries.py").read_text(encoding="utf-8")
    assert "asr_context" in src
    assert src.count("except TypeError") >= 2                # legacy/stub vqa safe


def test_docs_record_own_submission_system():
    pc = (REPO / "docs" / "PROJECT_CONTEXT.md").read_text(encoding="utf-8")
    assert "HỆ THỐNG RIÊNG" in pc and "TỐI THỨ SÁU" in pc
    pb = (REPO / "docs" / "COMPETITION_PLAYBOOK.md").read_text(encoding="utf-8")
    assert "CHECKLIST TUẦN THI ĐẦU" in pb and "Ctrl+G" in pb
