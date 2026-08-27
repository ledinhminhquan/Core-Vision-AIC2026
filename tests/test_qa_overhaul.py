"""QA overhaul đợt 2 (Nhiệm vụ A): answer_norm + 3 knob config-gated.

CPU-only, KHÔNG API: mọi lời gọi model đều mock/stub. Mỗi knob có cặp test
(off = bit-identical hành vi cũ, on = xử đúng ca fixture của lớp lỗi nó nhắm):

* ``vqa.answer_canonicalize`` — F1/F2/F3 + A5 (vote vỡ vì biến thể định dạng);
* ``vqa.answer_neighbor_frames`` — A2/A3 (frame lệch nhẹ, vote chéo strip);
* ``vqa.consistency_rerank`` — A1-họ hàng (nhóm top bất nhất, nhóm dưới đồng
  thuận tuyệt đối được ưu tiên khối dòng).
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

from cvp.config import Settings
from cvp.eval.official import normalize_answer
from cvp.pipeline.run_queries import (
    compute_qa_answers,
    compute_qa_answers_with_stats,
    plan_consistency_rerank,
    run_query_file,
)
from cvp.search.answer_norm import (
    canonical_key,
    fold_numbers_vi,
    fold_units,
    looks_decorated,
    majority_vote,
    strip_decoration,
)
from cvp.search.vqa import VqaAssistant


# ── answer_norm: canonicalization ────────────────────────────────────────────


def test_tier0_is_the_official_normalizer():
    for s in ('  Màu Xanh. ', '"x"', 'ba trăm kg'):
        assert canonical_key(s, tier=0) == normalize_answer(s)


def test_strip_decoration_preamble_markdown_quotes():
    assert strip_decoration("Đáp án: 27") == "27"
    assert strip_decoration("Trả lời là **màu đỏ**.") == "màu đỏ"
    assert strip_decoration('"Khung trời mơ ước"') == "Khung trời mơ ước"
    assert strip_decoration("«Nhớ ai»") == "Nhớ ai"
    # không bao giờ nuốt sạch input không rỗng
    assert strip_decoration('"..."') != ""


def test_leading_quote_gap_of_official_scorer_is_closed_at_tier1():
    # F3: normalize_answer chỉ rstrip → ngoặc MỞ đầu chuỗi sống sót.
    ours, gt = '"Khung trời mơ ước"', "Khung trời mơ ước"
    assert normalize_answer(ours) != normalize_answer(gt)          # lỗ hổng thật
    assert canonical_key(ours, 1) == canonical_key(gt, 1)          # tier 1 vá


def test_fold_numbers_vi_composites_and_safety():
    assert fold_numbers_vi("sáu") == "6"
    assert fold_numbers_vi("hai mươi bảy") == "27"
    assert fold_numbers_vi("mười lăm") == "15"
    assert fold_numbers_vi("bốn mươi tư") == "44"
    assert fold_numbers_vi("một trăm linh năm") == "105"
    assert fold_numbers_vi("một trăm hai mươi ba") == "123"
    # an toàn: "không" và các từ continuation-only không bị fold bậy
    assert fold_numbers_vi("không rõ") == "không rõ"
    assert fold_numbers_vi("tư vấn") == "tư vấn"
    assert fold_numbers_vi("thứ tư") == "thứ tư"   # "tư" chỉ fold sau mươi/mười


def test_fold_units_spacing_synonyms_percent():
    assert fold_units("300kg") == "300 kg"
    assert fold_units("88 %") == "88%"
    assert fold_units("200 gam") == "200 g"
    assert canonical_key("ba trăm ki-lô-gam", 2) == canonical_key("300 kg", 2)
    assert canonical_key("88 phần trăm", 2) == canonical_key("88%", 2)


def test_canonical_numbers_decimal_thousands_punct():
    assert canonical_key("sáu", 2) == canonical_key("6", 2)
    assert canonical_key("06", 2) == canonical_key("6", 2)
    assert canonical_key("3,5", 2) == canonical_key("3.5", 2)
    assert canonical_key("1.000 người", 2) == canonical_key("1000 người", 2)
    assert canonical_key("màu đỏ, trắng", 2) == canonical_key("màu đỏ trắng", 2)
    assert canonical_key("TP. HCM", 2) == canonical_key("TP HCM", 2)


def test_tier3_accent_fold_is_diagnostic_only():
    assert canonical_key("màu xanh", 2) != canonical_key("mau xanh", 2)
    assert canonical_key("màu xanh", 3) == canonical_key("mau xanh", 3)


def test_canonical_key_idempotent_and_never_empty():
    for s in ('Đáp án: "300kg"', "hai mươi bảy", "...", "không rõ"):
        for tier in (1, 2, 3):
            k = canonical_key(s, tier)
            assert k != ""
            assert canonical_key(k, tier) == k


def test_looks_decorated_flags_only_decorated():
    assert looks_decorated("Đáp án: 27")
    assert looks_decorated('"Nhớ ai"')
    assert not looks_decorated("màu xanh")
    assert not looks_decorated("300 kg")


def test_majority_vote_legacy_vs_canonical():
    votes = ["hai", "2", "ba", "ba"]
    off = majority_vote(votes, canonicalize=False)
    assert (off.answer, off.votes_for, off.total) == ("ba", 2, 4)   # chuỗi thô: "ba" thắng
    on = majority_vote(votes, canonicalize=True)
    assert on.answer == "hai"                    # lớp {hai,2} = 2 phiếu, hòa "ba" → lớp gặp trước
    assert on.votes_for == 2 and on.total == 4 and abs(on.agree - 0.5) < 1e-9


def test_majority_vote_representative_is_modal_raw():
    on = majority_vote(["2", "hai", "2"], canonicalize=True)
    assert on.answer == "2" and on.votes_for == 3   # cả 3 cùng lớp; raw phổ biến nhất
    assert majority_vote([], canonicalize=True).total == 0


# ── knob 1: vqa.answer_canonicalize trong VqaAssistant.answer_group ─────────


def _scripted_assistant(votes: list[str], canonicalize: bool) -> VqaAssistant:
    settings = Settings()
    settings.vqa.self_consistency = len(votes)
    settings.vqa.answer_canonicalize = canonicalize
    vqa = VqaAssistant(settings)
    seq = list(votes)
    vqa._ask_gemini_strip = lambda paths, q, ctx="": seq.pop(0)  # mock — KHÔNG API
    return vqa


def test_answer_group_off_keeps_legacy_raw_vote():
    vqa = _scripted_assistant(["hai", "2", "ba", "ba"], canonicalize=False)
    assert vqa.answer_group("mấy?", ["/f/a.jpg"]) == "ba"


def test_answer_group_on_pools_equivalent_ballots():
    vqa = _scripted_assistant(["hai", "2", "ba", "ba"], canonicalize=True)
    assert vqa.answer_group("mấy?", ["/f/a.jpg"]) == "hai"


def test_answer_group_votes_exposes_raw_ballots():
    vqa = _scripted_assistant(["300 kg", "300kg", "30 kg"], canonicalize=True)
    assert vqa.answer_group_votes("nặng?", ["/f/a.jpg"]) == ["300 kg", "300kg", "30 kg"]
    # và majority canonical gộp đúng biến thể đơn vị (round-40 flip-flop case)
    vqa2 = _scripted_assistant(["300 kg", "300kg", "30 kg"], canonicalize=True)
    assert vqa2.answer_group("nặng?", ["/f/a.jpg"]) == "300 kg"


# ── fixtures chung cho compute_qa_answers ────────────────────────────────────


def _result(gid: int, vid: str, frame: int, t: float):
    ref = SimpleNamespace(global_id=gid, video_id=vid, n=gid + 1, frame_idx=frame,
                          pts_time=t, path=f"/fake/{vid}/{frame:06d}.jpg")
    return SimpleNamespace(ref=ref, score=1.0 - 0.01 * gid, signals={},
                           video_id=vid, frame_idx=frame, global_id=gid)


def _two_moment_results():
    """Video A: nhóm 1 (rank 1-2, frame 100/200) + nhóm 2 xa (rank 3, frame 5000)."""
    return [
        _result(0, "L01_V001", 100, 4.0),
        _result(1, "L01_V001", 200, 8.0),
        _result(2, "L01_V001", 5000, 200.0),
    ]


class _BallotVqa:
    """Stub trả phiếu THEO STRIP (mỗi lời gọi 1 kịch bản); đếm mọi lời gọi."""

    def __init__(self, ballots_per_call: list[list[str]]):
        self.script = list(ballots_per_call)
        self.vote_calls: list[list[str]] = []
        self.answer_group_calls = 0

    def answer_group_votes(self, question, image_paths, context=""):
        self.vote_calls.append(list(image_paths))
        return list(self.script.pop(0)) if self.script else []

    def answer_group(self, question, image_paths, context=""):
        self.answer_group_calls += 1
        return ""

    def suggest(self, question, frames, context=""):
        return []


def _settings(**vqa_overrides) -> Settings:
    s = Settings()
    for k, v in vqa_overrides.items():
        setattr(s.vqa, k, v)
    return s


# ── knob 2: vqa.answer_neighbor_frames ───────────────────────────────────────


def test_neighbor_off_never_touches_ballot_api():
    vqa = _BallotVqa([])

    class _Legacy(_BallotVqa):
        def answer_group(self, question, image_paths, context=""):
            self.answer_group_calls += 1
            return "đáp án cũ"

    legacy = _Legacy([])
    answers = compute_qa_answers(_two_moment_results(), "q?", legacy, _settings())
    assert answers == ["đáp án cũ"] * 3
    assert legacy.answer_group_calls == 2          # 2 nhóm trong budget, 1 call/nhóm
    assert legacy.vote_calls == []                 # knob off → API phiếu KHÔNG được đụng
    assert vqa.vote_calls == []


def test_neighbor_pools_ballots_across_strips():
    # Nhóm 1 (frame 100-200) hỏi strip chính → phiếu "gai"; strip lân cận
    # (frame 5000 cùng video) → 2 phiếu "đúng" → vote chéo chọn "đúng".
    vqa = _BallotVqa([["gai"], ["đúng", "đúng"]])
    settings = _settings(answer_neighbor_frames=1, answers_per_query=1,
                         max_calls_per_query=3)
    answers, stats = compute_qa_answers_with_stats(
        _two_moment_results(), "q?", vqa, settings)
    assert answers == ["đúng"] * 3                 # nhóm 1 stamp + nhóm 2 thừa kế
    assert len(vqa.vote_calls) == 2                # strip chính + 1 strip lân cận
    assert vqa.vote_calls[1] == ["/fake/L01_V001/005000.jpg"]
    assert vqa.answer_group_calls == 0             # có phiếu → không gọi đường cũ
    assert stats[0].votes_for == 2 and stats[0].total_votes == 3


def test_neighbor_with_canonicalize_merges_format_variants():
    vqa = _BallotVqa([["sáu"], ["6", "06"]])
    settings = _settings(answer_neighbor_frames=1, answers_per_query=1,
                         max_calls_per_query=3, answer_canonicalize=True)
    answers, stats = compute_qa_answers_with_stats(
        _two_moment_results(), "đếm?", vqa, settings)
    assert stats[0].votes_for == 3 and stats[0].total_votes == 3   # 1 lớp duy nhất
    assert answers[0] in ("sáu", "6", "06")


def test_neighbor_without_budget_warns_loud_and_does_nothing(caplog):
    vqa = _BallotVqa([["x"], ["x"]])
    settings = _settings(answer_neighbor_frames=2, answers_per_query=5,
                         max_calls_per_query=2)    # budget cạn ngay sau 2 strip chính
    with caplog.at_level(logging.WARNING):
        answers, _ = compute_qa_answers_with_stats(
            _two_moment_results(), "q?", vqa, settings)
    assert len(vqa.vote_calls) == 2                # chỉ 2 strip chính, KHÔNG strip thêm
    assert any("KHÔNG có tác dụng" in r.message for r in caplog.records)
    assert answers[0] == "x"


def test_neighbor_stub_without_ballot_api_falls_back_to_answer_group():
    class _OldStub:
        def __init__(self):
            self.calls = 0

        def answer_group(self, question, image_paths, context=""):
            self.calls += 1
            return "cũ vẫn chạy"

    stub = _OldStub()
    settings = _settings(answer_neighbor_frames=2, max_calls_per_query=9)
    answers = compute_qa_answers(_two_moment_results(), "q?", stub, settings)
    assert answers == ["cũ vẫn chạy"] * 3
    assert stub.calls == 2                         # đường cũ, 1 call/nhóm — không nhân


# ── knob 3: vqa.consistency_rerank ───────────────────────────────────────────


def _rerank_results():
    """Nhóm 1 = video A (row 0-1), nhóm 2 = video B (row 2-3)."""
    return [
        _result(0, "L01_V001", 100, 4.0),
        _result(1, "L01_V001", 200, 8.0),
        _result(2, "L02_V002", 300, 12.0),
        _result(3, "L02_V002", 400, 16.0),
    ]


def test_plan_promotes_unanimous_group_when_top_disagrees():
    vqa = _BallotVqa([["a", "b", "c"], ["x", "x", "x"]])
    settings = _settings(consistency_rerank=True)
    answers, stats = compute_qa_answers_with_stats(_rerank_results(), "q?", vqa, settings)
    assert stats[0].agree < 0.5 and stats[1].agree == 1.0
    order = plan_consistency_rerank(stats, 4)
    assert order == [2, 3, 0, 1]                   # khối nhóm B lên đầu, còn lại giữ nguyên
    assert answers[2] == "x"


def test_plan_keeps_top_with_real_majority():
    vqa = _BallotVqa([["a", "a", "c"], ["x", "x", "x"]])
    settings = _settings(consistency_rerank=True)
    _answers, stats = compute_qa_answers_with_stats(_rerank_results(), "q?", vqa, settings)
    assert plan_consistency_rerank(stats, 4) is None   # top 2/3 — không đụng


def test_plan_ignores_fallback_and_single_vote_groups():
    from cvp.pipeline.run_queries import QA_FALLBACK_ANSWER, QaGroupStat

    stats = [
        QaGroupStat(rows=[0, 1], answer="a", votes_for=1, total_votes=3),
        QaGroupStat(rows=[2], answer=QA_FALLBACK_ANSWER, votes_for=3, total_votes=3),
        QaGroupStat(rows=[3], answer="x", votes_for=1, total_votes=1),
    ]
    assert plan_consistency_rerank(stats, 4) is None   # fallback & 1-phiếu không được thăng


class _StubEngine:
    def __init__(self, results, settings):
        self.settings = settings
        self._results = results

    def search_text(self, query, topk=None, display_k=None):
        return list(self._results)


def test_run_query_file_rerank_on_reorders_csv_and_times(tmp_path):
    qf = tmp_path / "query-p1-9-qa.txt"
    qf.write_text("Mô tả cảnh. Hỏi cái gì đó?\n", encoding="utf-8")
    settings = _settings(consistency_rerank=True)
    engine = _StubEngine(_rerank_results(), settings)
    vqa = _BallotVqa([["a", "b", "c"], ["x", "x", "x"]])
    times: dict[str, list[float]] = {}
    out = run_query_file(engine, qf, tmp_path, vqa=vqa, top1_times=times)
    rows = [r.split(",") for r in out.read_text(encoding="utf-8").strip().splitlines()]
    assert rows[0][0] == "L02_V002" and rows[0][2] == "x"   # khối đồng thuận lên đầu
    assert rows[2][0] == "L01_V001"
    assert times["query-p1-9-qa"] == [12.0]                 # timestamp theo dòng top MỚI


def test_run_query_file_rerank_off_is_bit_identical(tmp_path):
    qf = tmp_path / "query-p1-9-qa.txt"
    qf.write_text("Mô tả cảnh. Hỏi cái gì đó?\n", encoding="utf-8")
    settings = _settings()                                  # knob off
    engine = _StubEngine(_rerank_results(), settings)
    vqa = _BallotVqa([["a", "b", "c"], ["x", "x", "x"]])
    times: dict[str, list[float]] = {}
    out = run_query_file(engine, qf, tmp_path, vqa=vqa, top1_times=times)
    rows = [r.split(",") for r in out.read_text(encoding="utf-8").strip().splitlines()]
    assert [r[0] for r in rows] == ["L01_V001", "L01_V001", "L02_V002", "L02_V002"]
    assert times["query-p1-9-qa"] == [4.0]                  # thứ tự cũ, times cũ
