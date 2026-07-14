"""Multi-frame VQA strip (2026 upgrade — fixes the 'math in video' QA type).

CPU-only: stub assistants, no network. Pins: temporal strip selection, the
answer_group-first / suggest-fallback wiring in compute_qa_answers, and the
offline no-provider behaviour of VqaAssistant.answer_group.
"""

from __future__ import annotations

from types import SimpleNamespace

from cvp.config import Settings
from cvp.pipeline.run_queries import _group_strip, compute_qa_answers
from cvp.search.vqa import VqaAssistant


def _result(gid: int, vid: str = "L01_V001", n: int | None = None):
    n = n if n is not None else gid + 1
    ref = SimpleNamespace(global_id=gid, video_id=vid, n=n, frame_idx=100 * n,
                          pts_time=float(n), path=f"/fake/{vid}/{n:03d}.jpg")
    return SimpleNamespace(ref=ref, score=1.0 - 0.01 * gid, signals={},
                           video_id=vid, frame_idx=ref.frame_idx, global_id=gid)


def test_group_strip_temporal_order_and_cap():
    # Group rows arrive in RANK order (n = 7, 2, 9, 4) — the strip must come
    # back in TEMPORAL order, ≤ max_frames, always spanning first…last.
    results = [_result(0, n=7), _result(1, n=2), _result(2, n=9), _result(3, n=4)]
    strip = _group_strip(results, [0, 1, 2, 3], max_frames=3)
    assert strip[0].endswith("002.jpg") and strip[-1].endswith("009.jpg")
    assert len(strip) <= 3
    ns = [int(p[-7:-4]) for p in strip]
    assert ns == sorted(ns)


def test_group_strip_single_frame_uses_best_ranked():
    results = [_result(0, n=5), _result(1, n=1)]
    assert _group_strip(results, [0, 1], max_frames=1) == [results[0].ref.path]


class _StripVqa:
    """Stub with the new multi-frame API."""

    def __init__(self, answer="ba chiếc thuyền"):
        self.answer = answer
        self.strips: list[list[str]] = []

    def answer_group(self, question, image_paths):
        self.strips.append(list(image_paths))
        return self.answer

    def suggest(self, question, frames):  # must NOT be reached when strip works
        raise AssertionError("suggest called despite answer_group success")


def test_compute_qa_answers_prefers_answer_group():
    settings = Settings()
    results = [_result(0), _result(1), _result(2, vid="L02_V002")]
    vqa = _StripVqa()
    answers = compute_qa_answers(results, "có mấy chiếc thuyền?", vqa, settings)
    assert answers == ["ba chiếc thuyền"] * 3
    assert vqa.strips and all(len(s) <= settings.vqa.frames_per_answer
                              for s in vqa.strips)


class _EmptyStripVqa:
    """answer_group yields nothing → the single-frame suggest fallback runs."""

    def answer_group(self, question, image_paths):
        return ""

    def suggest(self, question, frames):
        return [SimpleNamespace(global_id=frames[0][0], answer="màu đỏ",
                                provider="stub")]


def test_compute_qa_answers_falls_back_to_suggest():
    answers = compute_qa_answers([_result(0)], "màu gì?", _EmptyStripVqa(), Settings())
    assert answers == ["màu đỏ"]


def test_real_assistant_answer_group_offline_returns_empty():
    settings = Settings()
    settings.vqa.provider = "none"
    vqa = VqaAssistant(settings)
    assert vqa.answer_group("q?", ["/fake/a.jpg", "/fake/b.jpg"]) == ""
    assert vqa.answer_group("q?", []) == ""
