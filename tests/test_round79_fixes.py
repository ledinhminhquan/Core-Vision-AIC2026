"""Round-79: pluggable local-VQA fallback backend (Gemini-storm insurance).

vqa.local_backend = "vintern" (default, byte-identical legacy path) or
"hf_auto": any chat-template VLM via AutoModelForImageTextToText — built for
the Qwen3.5 generation (201 languages) the 29/08 research verified as the
best local stand-in when Gemini 504-storms. The id is REQUIRED config
(vqa.local_hf_id) — no unverified default id is baked in; empty id fails
loud instead of downloading a guess.
"""
from __future__ import annotations

import pytest

from cvp.config import Settings


def test_r79_defaults_keep_the_legacy_path():
    s = Settings()
    assert s.vqa.local_backend == "vintern"
    assert s.vqa.local_hf_id == ""


def test_r79_router_picks_backend(monkeypatch):
    from cvp.search.vqa import VqaAssistant
    va = VqaAssistant.__new__(VqaAssistant)          # no heavy init
    va.cfg = Settings().vqa
    calls = []
    monkeypatch.setattr(VqaAssistant, "_ask_local_vintern",
                        lambda self, *a, **k: calls.append("vintern") or "v")
    monkeypatch.setattr(VqaAssistant, "_ask_local_hf_auto",
                        lambda self, *a, **k: calls.append("hf") or "h")
    assert va._ask_local("x.jpg", "q?") == "v"
    va.cfg.local_backend = "hf_auto"
    assert va._ask_local("x.jpg", "q?") == "h"
    assert calls == ["vintern", "hf"]


def test_r79_hf_auto_without_id_fails_loud():
    from cvp.search.vqa import VqaAssistant
    va = VqaAssistant.__new__(VqaAssistant)
    va.cfg = Settings().vqa
    va.cfg.local_backend = "hf_auto"
    va._local = None
    with pytest.raises(RuntimeError, match="local_hf_id"):
        va._ask_local_hf_auto("x.jpg", "q?")
