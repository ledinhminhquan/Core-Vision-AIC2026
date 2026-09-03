"""Round-90: what the two real nb09 sessions of 03/09 taught.

Session 3b24e654 (round-88 cell, 380'): gemini-3.1-pro-preview returned
429 "You exceeded your current quota" 253 times after the first two arms —
QA scored 0 on every arm for the rest of the day. Session 850660ee
(round-89 cell): the metaclip2 lane failed to load ("[Errno 5] Input/output
error" reading the HF cache on Drive), the lane assert fired (good), but the
loop then skipped LOADING the finished arms of the other session and the
summary lost its baseline.

1. cvp.models.gemini_keys — a pool GEMINI_API_KEY(_2.._5); quota-exhausted
   429 rotates the whole process to the next key and retries once per key.
   Both Gemini client sites (vqa.make_gemini_client, QueryProcessor) build
   through it; a single key behaves exactly as before.
2. metaclip2 loads through resilient_from_pretrained like siglip2/qwen
   (torn / EIO Drive cache → re-download to local disk and retry).
3. nb09: finished arms are loaded before the "no baseline" skip; the engine
   build is retried on a lane-short ensemble; when the in-session ABK fails
   the old ABK stays the table's baseline; GEMINI_API_KEY_2/3 exported from
   Colab secrets.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from cvp.models import gemini_keys as GK

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _fresh_pool(monkeypatch):
    for n in GK.KEY_ENV_NAMES:
        monkeypatch.delenv(n, raising=False)
    GK.reset_for_tests()
    yield
    GK.reset_for_tests()


class _Inner:
    """Fake genai.Client: the behaviour per key is scripted by the test."""
    def __init__(self, key, script):
        self.key = key
        self.script = script
        self.models = self

    def generate_content(self, **kwargs):
        outcome = self.script[self.key]
        if isinstance(outcome, Exception):
            raise outcome
        return type("R", (), {"text": f"{outcome}@{self.key}"})()


def _client(script, http=None):
    built = []

    def factory(key, http_options):
        built.append(key)
        return _Inner(key, script)

    c = GK.RotatingGeminiClient(http_options=http, client_factory=factory)
    return c, built


def test_r90_pool_order_and_dedup(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k1")
    monkeypatch.setenv("GOOGLE_API_KEY", "k1")          # same key twice → once
    monkeypatch.setenv("GEMINI_API_KEY_3", "k3")
    monkeypatch.setenv("GEMINI_API_KEY_2", "k2")
    assert GK.api_keys() == ["k1", "k2", "k3"]
    assert GK.current_key() == "k1" and GK.key_position() == (1, 3)


def test_r90_quota_429_rotates_process_wide_and_retries(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k1")
    monkeypatch.setenv("GEMINI_API_KEY_2", "k2")
    quota = RuntimeError("429 RESOURCE_EXHAUSTED. You exceeded your current quota")
    c, built = _client({"k1": quota, "k2": "ok"})
    assert c.models.generate_content(model="m", contents="x").text == "ok@k2"
    assert built == ["k1", "k2"] and GK.key_position() == (2, 2)
    # a SECOND client in the same process starts straight on the new key
    c2, built2 = _client({"k1": quota, "k2": "ok"})
    assert c2.models.generate_content(model="m", contents="x").text == "ok@k2"
    assert built2 == ["k2"]


def test_r90_single_key_behaves_as_before(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k1")
    quota = RuntimeError("429 RESOURCE_EXHAUSTED")
    c, built = _client({"k1": quota})
    with pytest.raises(RuntimeError, match="429"):
        c.models.generate_content(model="m", contents="x")
    assert built == ["k1"]                                 # no rotation possible


def test_r90_non_quota_errors_do_not_rotate(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k1")
    monkeypatch.setenv("GEMINI_API_KEY_2", "k2")
    c, built = _client({"k1": RuntimeError("503 UNAVAILABLE"), "k2": "ok"})
    with pytest.raises(RuntimeError, match="503"):
        c.models.generate_content(model="m", contents="x")
    assert GK.key_position() == (1, 2) and built == ["k1"]


def test_r90_all_keys_exhausted_raises_the_last_error(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k1")
    monkeypatch.setenv("GEMINI_API_KEY_2", "k2")
    c, built = _client({"k1": RuntimeError("429 quota"), "k2": RuntimeError("429 quota too")})
    with pytest.raises(RuntimeError, match="quota too"):
        c.models.generate_content(model="m", contents="x")
    assert built == ["k1", "k2"]


def test_r90_stale_caller_retries_on_the_already_rotated_key(monkeypatch):
    # thread A rotated k1→k2 while thread B's call on k1 was in flight: B must
    # retry on k2, not rotate again (skipping k2) nor give up
    monkeypatch.setenv("GEMINI_API_KEY", "k1")
    monkeypatch.setenv("GEMINI_API_KEY_2", "k2")
    monkeypatch.setenv("GEMINI_API_KEY_3", "k3")
    quota = RuntimeError("429 RESOURCE_EXHAUSTED quota")
    script = {"k1": quota, "k2": "ok", "k3": "ok"}
    built = []

    def factory(key, http_options):
        built.append(key)
        inner = _Inner(key, script)
        if key == "k1":                           # simulate the concurrent rotation
            orig = inner.generate_content

            def racing(**kw):
                GK.rotate("thread A")
                return orig(**kw)
            inner.generate_content = racing
        return inner

    c = GK.RotatingGeminiClient(client_factory=factory)
    assert c.models.generate_content(model="m", contents="x").text == "ok@k2"
    assert built == ["k1", "k2"] and GK.key_position() == (2, 3)   # k3 untouched


def test_r90_build_client_requires_a_key_and_sets_the_http_deadline(monkeypatch):
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        GK.build_client(30.0)
    monkeypatch.setenv("GOOGLE_API_KEY", "k1")
    c = GK.build_client(90.0)
    assert isinstance(c, GK.RotatingGeminiClient) and c._http == {"timeout": 90000}


def test_r90_both_call_sites_build_through_the_pool():
    vqa = (REPO / "src" / "cvp" / "search" / "vqa.py").read_text(encoding="utf-8")
    qp = (REPO / "src" / "cvp" / "models" / "query_processor.py").read_text(encoding="utf-8")
    assert "build_client(" in vqa and "build_client(" in qp
    assert "genai.Client(" not in vqa and "genai.Client(" not in qp   # no stray single-key client


def test_r90_metaclip2_loads_resiliently():
    src = (REPO / "src" / "cvp" / "models" / "metaclip2.py").read_text(encoding="utf-8")
    assert src.count("resilient_from_pretrained(") >= 2


def test_r90_notebooks_export_the_extra_keys_and_nb09_survives_a_failed_baseline():
    nb = json.loads((REPO / "notebooks" / "09_campaign.ipynb").read_text(encoding="utf-8"))
    code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    env_cell = next(s for s in code if "userdata.get(_sec)" in s)
    assert '"GEMINI_API_KEY_2"' in env_cell and '"GEMINI_API_KEY_3"' in env_cell
    camp = next(s for s in code if "_ARM_ORDER" in s)
    # finished arms are loaded BEFORE the no-baseline skip
    assert camp.index("_results[_arm] = _prev") < camp.index("cánh bench không có baseline")
    assert "thử lại sau 20s" in camp                       # lane-short engine build retried
    assert "dùng ABK phiên" in camp                        # old ABK stays the table baseline
    nb3 = json.loads((REPO / "notebooks" / "03_test_system.ipynb").read_text(encoding="utf-8"))
    env3 = next("".join(c["source"]) for c in nb3["cells"]
                if c["cell_type"] == "code" and "userdata.get(_sec)" in "".join(c["source"]))
    assert '"GEMINI_API_KEY_2"' in env3
