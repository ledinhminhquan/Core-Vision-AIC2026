"""Round-89: Gemini 3.8 Flash (GA 02/09/2026) — verified facts, wired safely.

Verified on ai.google.dev / the DeepMind model card (03/09/2026): model id
``gemini-3.8-flash``, stable, inputs text/image/video/audio/PDF, 1M context,
64K output, thinking levels low/medium/high — "minimal is not supported and
returns an error"; promo price $0.75 / $3.75 per 1M through 31/12/2026 (same
as 3.7 Flash); "based on Gemini 3.7 Flash"; knowledge cutoff March 2026.

1. economical_config sent "minimal" to every model that is not 3.7 — with
   3.8 (or a "-latest" alias that now resolves to 3.7/3.8) that is a wasted
   400 followed by the config-less rescue = default thinking at output rates.
   Now: "minimal" only for the 3.5 family, "low" elsewhere.
2. gemini-3.8-flash leads the shared rescue chain (query.gemini_model_fallbacks).
3. nb09 gains two single-change arms (ABK+G38R: VLM rerank model; ABK+G38QA:
   QA answer model) and re-measures ABK when the Drive baseline is from
   another session (the old ABK becomes an extra noise draw).
"""
from __future__ import annotations

import json
from pathlib import Path

from cvp.config import Settings
from cvp.models.query_processor import economical_config, thinking_level_for

REPO = Path(__file__).resolve().parents[1]


def test_r89_thinking_level_minimal_only_on_the_35_family():
    assert thinking_level_for("gemini-3.5-flash-lite") == "minimal"
    assert thinking_level_for("gemini-3.5-flash") == "minimal"
    assert thinking_level_for("gemini-3.7-flash") == "low"
    assert thinking_level_for("gemini-3.8-flash") == "low"      # "minimal ... returns an error"
    assert thinking_level_for("gemini-flash-latest") == "low"   # alias hot-swaps to the newest
    cfg = economical_config("gemini-3.8-flash")                 # None only without the SDK
    if cfg is not None:
        assert str(getattr(cfg, "thinking_level", "")).lower().endswith("low")


def test_r89_38_flash_leads_the_rescue_chain():
    from cvp.search.vqa import gemini_model_chain
    s = Settings()
    assert s.query.gemini_model_fallbacks[:2] == ["gemini-3.8-flash", "gemini-3.7-flash"]
    # every call site walks the same chain: primary first, then the fallbacks
    assert gemini_model_chain(s, "gemini-3.5-flash-lite")[:2] == [
        "gemini-3.5-flash-lite", "gemini-3.8-flash"]
    # battle primaries unchanged — a model swap enters nb03 only via the bench
    assert s.search.vlm_rerank_model == "gemini-3.5-flash-lite"
    assert s.vqa.answer_model == "gemini-3.1-pro-preview"
    yaml = (REPO / "configs" / "settings.yaml").read_text(encoding="utf-8")
    assert "gemini_model_fallbacks: [gemini-3.8-flash, gemini-3.7-flash," in yaml


def test_r89_campaign_arms_and_session_baseline():
    nb = json.loads((REPO / "notebooks" / "09_campaign.ipynb").read_text(encoding="utf-8"))
    camp = next("".join(c["source"]) for c in nb["cells"]
                if c["cell_type"] == "code" and "_ARM_ORDER" in "".join(c["source"]))
    assert '"ABK+G38R": {"CVP_SEARCH__VLM_RERANK_MODEL": "gemini-3.8-flash"}' in camp
    assert '"ABK+G38QA": {"CVP_VQA__ANSWER_MODEL": "gemini-3.8-flash"}' in camp
    # the base env pins the battle models so a G38 arm cannot leak into the next
    assert 'os.environ["CVP_SEARCH__VLM_RERANK_MODEL"] = "gemini-3.5-flash-lite"' in camp
    assert 'os.environ["CVP_VQA__ANSWER_MODEL"] = "gemini-3.1-pro-preview"' in camp
    assert '"CVP_SEARCH__VLM_RERANK_MODEL", "CVP_VQA__ANSWER_MODEL"' in camp   # reset per arm
    assert 'settings.vqa.answer_model == "gemini-3.8-flash"' in camp          # knob took effect
    # a baseline from another session is rotated to ABK-prev*.json and re-measured
    assert '_camp / "ABK-prev.json"' in camp and 'glob("ABK-prev*.json")' in camp
    assert "_abk_raw.get(\"session\") != SESSION and _pending" in camp   # raw file, not _done()
    # verdict compares each arm with the ABK draw of ITS OWN session
    assert "_abk_by_session" in camp and '"≠phiên"' in camp and "campaign_summary-{SESSION}" in camp
    # audit r89: stable session per kernel, rotate-on-save, model preflight + fallback gate
    assert 'SESSION = globals().get("SESSION") or uuid' in camp
    assert "_reabk" in camp and "_abk_p.rename(_rot)" in camp
    assert "không bench 70 phút với " in camp and "thử lại {_try + 2}/6" in camp   # r92 retry
    assert '"model_fallbacks": dict(_storm.fallback)' in camp and "rớt model" in camp
    md = next("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown")
    assert "ABK-prev*.json" in md and "`ABK+G38R`" in md and "`ABK+G38QA`" in md


def _storm_counter_class():
    import logging
    import re
    nb = json.loads((REPO / "notebooks" / "09_campaign.ipynb").read_text(encoding="utf-8"))
    camp = next("".join(c["source"]) for c in nb["cells"]
                if c["cell_type"] == "code" and "_ARM_ORDER" in "".join(c["source"]))
    src = camp[camp.index("class _StormCounter"):camp.index("\n\n\nif RUN_CAMPAIGN")]
    ns = {"logging": logging, "re": re}
    exec(src, ns)                                # noqa: S102 — the notebook's own class
    return ns["_StormCounter"]


def test_r89_storm_counter_counts_model_fallbacks_per_model():
    import logging
    SC = _storm_counter_class()
    h = SC()

    def rec(name, msg):
        return logging.LogRecord(name, logging.WARNING, "", 0, msg, None, None)

    # the plain attempt failing = the chain moved past this model (vqa.py + query_processor.py)
    h.emit(rec("cvp.search.vqa", "Gemini model 'gemini-3.8-flash' failed (503 UNAVAILABLE) — trying next"))
    h.emit(rec("cvp.models.query_processor",
               "Gemini model 'gemini-3.5-flash-lite' failed (timeout) — trying next fallback"))
    # the economical rescue retries the SAME model — not a fallback
    h.emit(rec("cvp.search.vqa",
               "Gemini model 'gemini-3.8-flash' (economical) failed (400 thinking) — trying next"))
    assert h.fallback == {"gemini-3.8-flash": 1, "gemini-3.5-flash-lite": 1}
    assert h.storm.get("503") == 1 and h.storm.get("UNAVAILABLE") == 1
    assert h.fatal == {}                         # storms are never structural


def test_r89_config_rejected_only_for_request_shape_errors():
    from cvp.models.query_processor import config_rejected
    assert config_rejected(RuntimeError("400 INVALID_ARGUMENT: thinking_level minimal is not supported"))
    assert config_rejected(RuntimeError("media_resolution not supported"))
    assert not config_rejected(RuntimeError("503 UNAVAILABLE high demand"))
    assert not config_rejected(RuntimeError("429 RESOURCE_EXHAUSTED"))
    assert not config_rejected(TimeoutError("call exceeded 45.0s wall clock"))


def test_r89_transient_error_moves_to_next_model_not_a_full_price_retry(monkeypatch):
    from cvp.models import query_processor as QP
    from cvp.search import vqa as V
    monkeypatch.setattr(QP, "economical_config", lambda m: object())   # SDK-independent

    class _Models:
        def __init__(self):
            self.calls = []

        def generate_content(self, model, contents, config=None):
            self.calls.append((model, config is not None))
            if model == "gemini-3.8-flash":
                raise RuntimeError("503 UNAVAILABLE")
            return type("R", (), {"text": "ok"})()

    class _Client:
        def __init__(self):
            self.models = _Models()

    c = _Client()
    out = V.generate_with_fallback(c, ["gemini-3.8-flash", "gemini-3.7-flash"], ["x"],
                                   economical=True)
    assert out == "ok"
    # 3.8 tried ONCE (economical), no config-less retry; 3.7 answered
    assert c.models.calls[0] == ("gemini-3.8-flash", True)
    assert c.models.calls[1][0] == "gemini-3.7-flash"
    assert len([m for m, _ in c.models.calls if m == "gemini-3.8-flash"]) == 1
