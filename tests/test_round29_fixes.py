"""Round-29: the Gemini deadline floor + retired-model fallback.

Live nb03 gemini A/B (Aug 20): every Gemini call failed BEFORE running —
the API now rejects client deadlines under 10s (400 INVALID_ARGUMENT
"Minimum allowed deadline is 10s") and gemini-2.5-flash 404s for new users
("no longer available"). The chain silently degraded to free Google
Translate, so the "gemini" runs never exercised Gemini enhancement.
"""

from pathlib import Path

from cvp.config import Settings

REPO = Path(__file__).resolve().parents[1]


def test_gemini_deadline_meets_api_floor():
    s = Settings()
    assert s.query.timeout_s >= 10.0, (
        "Gemini rejects deadlines under 10s — every call would 400 before running"
    )


def test_no_retired_model_in_fallback_chain():
    s = Settings()
    chain = [s.query.gemini_model, *s.query.gemini_model_fallbacks]
    assert "gemini-2.5-flash" not in chain          # 404s for new users (live)
    assert chain[-1].endswith("-latest")            # rolling alias can't retire


def test_settings_yaml_matches_the_fix():
    """The live runs load configs/settings.yaml, which pinned BOTH the 8s
    deadline and the retired model — fixing only the code defaults would
    leave production broken."""
    yaml_text = (REPO / "configs" / "settings.yaml").read_text(encoding="utf-8")
    assert "gemini-2.5-flash" not in yaml_text
    assert "timeout_s: 8.0" not in yaml_text
    assert "gemini-flash-latest" in yaml_text


def test_nb03_battle_defaults_are_finetuned_gemini():
    """Round-1 lane decision (A/B of all 5 configs, Aug 20): finetuned +
    provider=gemini. The shipped notebook must default to the battle config —
    competition night is not the time to hand-edit knobs."""
    src = (REPO / "notebooks" / "_build_notebooks.py").read_text(encoding="utf-8")
    frag = src.split("NB3_ENGINE = r")[1].split("NB3_QUERIES")[0]
    assert 'ENGINE_MODEL   = "finetuned"' in frag
    assert 'QUERY_PROVIDER = "gemini"' in frag
