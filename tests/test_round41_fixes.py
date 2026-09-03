"""Round-41: model-tier upgrades from the 22/08/2026 web research.

- gemini-3.5-pro does NOT exist publicly (closed Vertex preview) — the QA
  answer path rides gemini-3.1-pro-preview (thinks longer, own wall cap),
  degrading Pro → 3.7-flash → 3.5-flash → flash-latest.
- Everything Flash moves to gemini-3.7-flash (newer than 3.5-flash, half its
  price through 2026); battle-proven 3.5-flash stays first fallback.
- Cross-encoder: Qwen3-VL-Reranker-8B (MMEB-v2 image retrieval 80.7 vs the
  2B's 73.8; ~18GB BF16 fits the A100).
"""

from pathlib import Path

from cvp.config import Settings

REPO = Path(__file__).resolve().parents[1]


def test_round41_model_defaults():
    s = Settings()
    assert s.query.gemini_model == "gemini-3.5-flash-lite"
    assert s.query.gemini_model_fallbacks[0] == "gemini-3.8-flash"   # round-89 (GA 02/09/2026)
    assert s.query.gemini_model_fallbacks[-1].endswith("-latest")
    assert s.vqa.gemini_model == "gemini-3.7-flash"
    assert s.vqa.answer_model == "gemini-3.1-pro-preview"
    assert s.vqa.answer_timeout_s >= 60
    assert s.search.qwen_reranker_id == "Qwen/Qwen3-VL-Reranker-8B"


def test_qa_answer_path_uses_pro_with_flash_inserted(monkeypatch):
    from cvp.search import vqa as V

    s = Settings()
    a = V.VqaAssistant(s)
    seen = {}

    def fake_generate(client, models, contents, timeout_s=None):
        seen["models"] = list(models)
        seen["timeout"] = timeout_s
        return "đáp án"

    monkeypatch.setattr(V, "generate_with_fallback", fake_generate)
    monkeypatch.setattr(V, "make_gemini_client", lambda st: object())
    monkeypatch.setattr(V, "load_rgb", lambda p: __import__("PIL.Image", fromlist=["new"]).new("RGB", (8, 8)))
    out = a._ask_gemini_strip(["/x.jpg"], "hỏi?")
    assert out == "đáp án"
    assert seen["models"][0] == "gemini-3.1-pro-preview"
    assert seen["models"][1] == "gemini-3.7-flash"   # VQA Flash = FIRST rescue (round-89)
    assert seen["models"][2] == "gemini-3.8-flash"   # 3.8 rides behind it
    assert seen["timeout"] >= 90


def test_yaml_mirrors_round41():
    y = (REPO / "configs" / "settings.yaml").read_text(encoding="utf-8")
    assert "gemini-3.5-flash-lite" in y
    assert "answer_model: gemini-3.1-pro-preview" in y
    assert "Qwen/Qwen3-VL-Reranker-8B" in y
