"""Round-45: hidden-spend controls — live billing forensics (₫227K on 20/08)
showed default 'medium' thinking (billed as output, ~75% of the bill) and
default 1120-token image accounting dominated cost. The cheap call families
(listwise scoring, suggests, enhancement) now ride the Lite tier with
thinking floored and media_resolution LOW; the QA answer path deliberately
keeps Pro at full depth — those 4-5 queries are where spend buys score.
"""

from types import SimpleNamespace

import pytest

from cvp.config import Settings
from cvp.models.query_processor import economical_config
from cvp.search.vqa import generate_with_fallback


def test_round45_defaults():
    s = Settings()
    assert s.query.gemini_model == "gemini-3.5-flash-lite"
    assert s.search.vlm_rerank_model == "gemini-3.5-flash-lite"
    assert s.vqa.answer_model == "gemini-3.1-pro-preview"   # QA keeps the Pro brain


def test_economical_level_respects_37_floor():
    cfg = economical_config("gemini-3.7-flash")
    if cfg is None:
        pytest.skip("google-genai without thinking_level on this machine")
    assert str(getattr(cfg, "thinking_level", "")).endswith("low")
    lite = economical_config("gemini-3.5-flash-lite")
    assert str(getattr(lite, "thinking_level", "")).endswith("minimal")


def test_generate_with_fallback_rescues_rejected_config(monkeypatch):
    calls = []

    class _Client:
        class models:  # noqa: N801 — mimic google-genai surface
            @staticmethod
            def generate_content(model, contents, config=None):
                calls.append((model, config is not None))
                if config is not None:
                    raise RuntimeError("400 unsupported thinking_level")
                return SimpleNamespace(text="ok")

    import cvp.models.query_processor as QP
    monkeypatch.setattr(QP, "economical_config", lambda m: object())
    out = generate_with_fallback(_Client(), ["m1"], "x", economical=True)
    assert out == "ok"
    # economical attempt failed → SAME model retried plain, chain not consumed
    assert calls == [("m1", True), ("m1", False)]


def test_vlm_scorer_uses_its_own_cheap_model():
    src = open("src/cvp/search/vlm_rerank.py", encoding="utf-8").read()
    assert "vlm_rerank_model" in src
    assert "economical=True" in src
    vqa_src = open("src/cvp/search/vqa.py", encoding="utf-8").read()
    strip = vqa_src.split("_ask_gemini_strip")[1].split("def ")[0]
    assert "economical" not in strip      # QA answers stay full-depth Pro


def test_r46_client_budget_covers_the_pro_answer_wall(monkeypatch):
    """Live 504: the genai client HTTP deadline was built from the 45s generic
    wall, so the Pro answer path's 90s budget never applied."""
    captured = {}

    class _FakeGenai:
        class Client:
            def __init__(self, api_key=None, http_options=None):
                captured["timeout_ms"] = (http_options or {}).get("timeout")

    import sys
    monkeypatch.setitem(sys.modules, "google", type(sys)("google"))
    monkeypatch.setitem(sys.modules, "google.genai", _FakeGenai)
    sys.modules["google"].genai = _FakeGenai
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    from cvp.models import gemini_keys as GK
    from cvp.search.vqa import make_gemini_client
    GK.reset_for_tests()
    s = Settings()
    c = make_gemini_client(s)                     # round-90: a key POOL, inner client lazy
    assert c._http["timeout"] >= int(s.vqa.answer_timeout_s * 1000)
    c._inner()                                    # building it passes the same deadline
    assert captured["timeout_ms"] >= int(s.vqa.answer_timeout_s * 1000)
