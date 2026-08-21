"""Round-40: test-time compute — self-consistency voting for the LLM stages.

Two same-config live runs scored 9.4 vs 9.0, and QA q3 flip-flopped
'300 kg' ↔ '30 kg' — single-sample Gemini calls are dice rolls. The VLM
reranker now averages N independent score vectors and strip-VQA keeps the
majority of N answers; both default to 1 vote (exact old behaviour) and the
battle notebook turns them up to 3.
"""

from types import SimpleNamespace

import pytest

from cvp.config import Settings
from cvp.search import vlm_rerank as VR
from cvp.search.vqa import VqaAssistant


def _results(n):
    return [SimpleNamespace(ref=SimpleNamespace(path=f"/img/{i}.jpg"),
                            signals={}, score=1.0 - i * 0.01)
            for i in range(n)]


def test_vlm_votes_average_the_score_vectors(monkeypatch):
    s = Settings()
    s.search.vlm_rerank_provider = "gemini"
    s.search.vlm_rerank_topk = 3
    s.search.vlm_rerank_votes = 3
    # vote 1 loves frame 0, votes 2+3 love frame 2 — the MEAN must win, and a
    # crashed 4th call must count as a lost vote, not a lost query.
    votes = iter([[9.0, 1.0, 5.0], [1.0, 2.0, 9.0], [2.0, 3.0, 8.0]])
    monkeypatch.setattr(VR, "_gemini_scores", lambda q, p, st: next(votes))
    rows = _results(4)
    out = VR.vlm_rerank(rows, "q", s)
    assert out[0] is rows[2]                      # mean 7.33 tops mean 4.0
    assert out[3] is rows[3]                      # tail untouched
    assert out[0].signals["vlm"] == pytest.approx((5 + 9 + 8) / 3)


def test_vlm_single_vote_failure_keeps_original_order(monkeypatch):
    s = Settings()
    s.search.vlm_rerank_provider = "gemini"
    s.search.vlm_rerank_votes = 2
    def _boom(q, p, st):
        raise RuntimeError("api down")
    monkeypatch.setattr(VR, "_gemini_scores", _boom)
    rows = _results(3)
    assert VR.vlm_rerank(rows, "q", s) == rows    # fail-open unchanged


def test_vqa_majority_vote(monkeypatch):
    s = Settings()
    s.vqa.provider = "gemini"
    s.vqa.self_consistency = 3
    vqa = VqaAssistant(s)
    answers = iter(["300 kg", "30 kg", "30 KG "])
    monkeypatch.setattr(vqa, "_ask_gemini_strip",
                        lambda paths, q, ctx: next(answers))
    got = vqa.answer_group("cân nặng?", ["/img/a.jpg"])
    assert got.strip().casefold() == "30 kg"      # 2/3 majority, case-blind


def test_vqa_one_vote_is_old_behaviour(monkeypatch):
    s = Settings()
    s.vqa.provider = "gemini"
    assert s.vqa.self_consistency == 1            # default = exact old cost
    vqa = VqaAssistant(s)
    calls = []
    monkeypatch.setattr(vqa, "_ask_gemini_strip",
                        lambda paths, q, ctx: calls.append(1) or "đáp án")
    assert vqa.answer_group("q?", ["/img/a.jpg"]) == "đáp án"
    assert len(calls) == 1


def test_notebook_battle_knobs():
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / "notebooks" /
           "_build_notebooks.py").read_text(encoding="utf-8")
    frag = src.split("NB3_ENGINE = r")[1].split("NB3_QUERIES")[0]
    assert "CVP_SEARCH__VLM_RERANK_VOTES" in frag
    assert "CVP_VQA__SELF_CONSISTENCY" in frag
