"""Organiser query-pack parsing (fix M1, adversarial review 2026-07-08).

The REAL AIC-2025 finals packs write TRAKE files as one context/header line
followed by "E1:"…"Ek:" prefixed events, and QA files as a SINGLE line with
the question embedded ("… Hỏi <câu hỏi>?"). Treating every line as an event
(the old behaviour) made the header a phantom event #1 and shifted every
submitted frame against the ground truth. These tests pin the normalisers
(:func:`parse_trake_events`, :func:`split_qa_line`) and the ``run_query_file``
wiring, on both organiser layouts AND the plain one-event-per-line layout.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from cvf.config import Settings
from cvf.pipeline.run_queries import (
    parse_query_lines,
    parse_trake_events,
    run_query_file,
    split_qa_line,
)

REPO = Path(__file__).resolve().parents[1]

ORGANISER_TRAKE_LINES = [
    "Đoạn video múa lân một con lân màu vàng đen trắng, tìm các sự kiện sau:",
    "E1: Lân quay vòng trên cột bằng 2 chân trước rồi tiếp đất.",
    "E2: Khoảnh khắc 4 chân hoàn toàn chạm đất đầu tiên.",
    "E3: Khoảnh khắc đầu tiên 2 người biểu diễn cúi chào ban giám khảo.",
    "E4: Khoảnh khắc đầu tiên con rồng cử động đầu.",
]


# ── parse_trake_events ───────────────────────────────────────────────────────


def test_trake_organiser_format_drops_header_and_prefixes():
    events = parse_trake_events(ORGANISER_TRAKE_LINES)
    assert len(events) == 4                          # the header is NOT an event
    assert events[0].startswith("Lân quay vòng")     # "E1:" prefix stripped
    assert all("E1" not in ev and "múa lân" not in ev for ev in events)


def test_trake_plain_format_unchanged():
    plain = ["vận động viên chạy đà", "vận động viên giậm nhảy", "vận động viên tiếp cát"]
    assert parse_trake_events(plain) == plain


def test_trake_prefix_style_variants():
    lines = ["bối cảnh chung:", "e1. sự kiện một", "E2) sự kiện hai", "E3 - sự kiện ba"]
    assert parse_trake_events(lines) == ["sự kiện một", "sự kiện hai", "sự kiện ba"]


def test_trake_single_prefixed_line_is_not_reinterpreted():
    # <2 E-prefixed lines → ambiguous → keep the lines as-is (plain layout).
    lines = ["E1: chỉ một sự kiện", "một dòng thường"]
    assert parse_trake_events(lines) == lines


def test_trake_duplicated_event_number_typo_still_parses():
    # Real packs contain organiser typos (two "E2:" lines) — both are kept, in order.
    lines = ["Một cảnh khám xét.", "E1: cảnh sát vẫy tay.", "E2: chỉ tay vào vị trí.",
             "E2: khom người xuống kiểm tra."]
    events = parse_trake_events(lines)
    assert len(events) == 3
    assert events[-1].startswith("khom người")


def test_trake_prepend_context_merges_header_into_each_event():
    events = parse_trake_events(ORGANISER_TRAKE_LINES, prepend_context=True)
    assert len(events) == 4
    assert all(ev.startswith("Đoạn video múa lân") for ev in events)
    assert "tìm các sự kiện sau" in events[0]  # header kept verbatim (minus trailing ':')
    assert events[0].endswith("tiếp đất.")


# ── split_qa_line ────────────────────────────────────────────────────────────


def test_qa_single_line_splits_at_hoi():
    line = ("Đoạn video về một chương trình từ thiện của một câu lạc bộ. "
            "Hỏi xã này có tên là gì? (tại thời điểm đó)")
    desc, question = split_qa_line(line)
    assert "Hỏi" not in desc and desc.startswith("Đoạn video")
    assert question.startswith("Hỏi xã này")


def test_qa_split_uses_last_marker_so_hoc_hoi_cannot_false_trigger():
    line = "Các em nhỏ thể hiện tinh thần học hỏi trong lớp. Hỏi lớp học có mấy em?"
    desc, question = split_qa_line(line)
    assert desc.endswith("trong lớp")            # split at the REAL question, not "học hỏi"
    assert question == "Hỏi lớp học có mấy em?"


def test_qa_cau_hoi_variant_and_passthrough():
    desc, question = split_qa_line("Một bản tin thời sự. Câu hỏi: phát thanh viên mặc áo màu gì?")
    assert desc == "Một bản tin thời sự"
    assert question.lower().startswith("câu hỏi")
    # no marker → the whole line serves as both (previous behaviour)
    plain = "biển báo công trình xây dựng bên đường"
    assert split_qa_line(plain) == (plain, plain)


# ── run_query_file wiring (stub engine — no models, no index) ────────────────


class _StubEngine:
    def __init__(self, results=None):
        self.settings = Settings()
        self.trake_events: list[str] | None = None
        self.text_query: str | None = None
        self._results = results or []

    def search_trake(self, events, max_results=100):
        self.trake_events = list(events)
        frames = [100 * (i + 1) for i in range(len(events))]
        return [SimpleNamespace(video_id="L01_V001", frame_idxs=frames,
                                pts_times=[float(f) / 25.0 for f in frames], score=1.0)]

    def search_text(self, query, topk=None, display_k=None):
        self.text_query = query
        return list(self._results)

    def search_avs(self, query, **kw):
        self.text_query = query
        return list(self._results)


def test_run_query_file_trake_organiser_format_emits_k_frames(tmp_path):
    qf = tmp_path / "query-p1-16-trake.txt"
    qf.write_text("\n".join(ORGANISER_TRAKE_LINES), encoding="utf-8")
    engine = _StubEngine()
    out = run_query_file(engine, qf, tmp_path)
    assert engine.trake_events is not None and len(engine.trake_events) == 4
    row = out.read_text(encoding="utf-8").strip()
    assert row == "L01_V001,100,200,300,400"       # 4 frames — NOT 5 (no phantom event)


def test_run_query_file_qa_single_line_searches_description_only(tmp_path):
    qf = tmp_path / "query-p1-15-qa.txt"
    qf.write_text(
        "Đoạn video trao quà tại một xã thuộc tỉnh Khánh Hòa. Hỏi xã này có tên là gì?\n",
        encoding="utf-8",
    )
    hit = SimpleNamespace(video_id="L05_V005", frame_idx=888, global_id=7,
                          ref=SimpleNamespace(path="x.jpg", pts_time=35.5))
    engine = _StubEngine(results=[hit])
    questions: list[str] = []

    class _StubVqa:
        def suggest(self, question, frames):
            questions.append(question)
            return [SimpleNamespace(global_id=7, answer="xã Cam Hải Đông", provider="stub")]

    out = run_query_file(engine, qf, tmp_path, vqa=_StubVqa())
    assert "Hỏi" not in engine.text_query          # dense search sees the description only
    assert questions and questions[0].startswith("Hỏi xã này")
    assert out.read_text(encoding="utf-8").strip() == "L05_V005,888,xã Cam Hải Đông"


def test_repo_example_organiser_fixtures_parse():
    """The committed real-format examples stay parseable end-to-end."""
    trake = REPO / "queries" / "example" / "query-0-4-trake-organiser.txt"
    qa = REPO / "queries" / "example" / "query-0-5-qa-organiser.txt"
    lines = [ln.strip() for ln in trake.read_text(encoding="utf-8").splitlines() if ln.strip()]
    events = parse_trake_events(lines)
    assert len(events) == 4 and events[0].startswith("Vận động viên bắt đầu")
    desc, question = split_qa_line(qa.read_text(encoding="utf-8").strip())
    assert question.startswith("Hỏi") and "Hỏi" not in desc


# ── parse_query_lines (round-3 fixes H-R3-1 QA≥3 dòng · H-R3-2 KIS đa đoạn) ──


def test_qa_three_line_file_takes_last_line_as_question():
    # Shape of the REAL finals file query-p2-3-qa.txt: 2 description paragraphs
    # + the actual question on the LAST line (the old code sent line 2 to VQA).
    lines = [
        "Đoạn clip về một gian trưng bày văn hóa - du lịch. Ở giữa là bản đồ Việt Nam.",
        "Phía trên là quốc kỳ Việt Nam treo chính giữa.",
        "Hãy cho biết đây là khu du lịch quốc gia tại địa điểm nào của Việt Nam?",
    ]
    description, question = parse_query_lines("qa", lines)
    assert question == lines[-1]
    assert "quốc kỳ Việt Nam" in description          # paragraph 2 kept for retrieval
    assert "khu du lịch quốc gia tại địa điểm nào" not in description


def test_qa_two_line_file_unchanged_regression():
    lines = ["Mô tả cảnh.", "Câu hỏi là gì?"]
    assert parse_query_lines("qa", lines) == ("Mô tả cảnh.", "Câu hỏi là gì?")


def test_qa_multiline_without_interrogative_tail_falls_back_to_marker_split():
    lines = [
        "Đoạn video về một chương trình từ thiện tại Khánh Hòa.",
        "Hỏi xã này có tên là gì (tại thời điểm đó)",  # no '?', but marker present
    ]
    description, question = parse_query_lines("qa", lines)
    assert question.startswith("Hỏi xã này")
    assert "Hỏi xã" not in description and "từ thiện" in description


def test_kis_multi_paragraph_joins_all_lines():
    # Shape of the REAL finals file query-p1-11-kis.txt: the discriminative
    # detail (shadow portrait) lives ENTIRELY in paragraph 2.
    lines = [
        "Trong đoạn clip có một chàng trai đội mũ lưỡi trai đen sắp xếp nhiều mảnh bìa cắt rời.",
        "Nhờ ánh sáng chiếu từ một phía, các mảnh bìa đổ bóng lên tường, tạo thành hình "
        "chân dung một người đàn ông mặc vest.",
    ]
    text, question = parse_query_lines("kis", lines)
    assert question is None
    assert "chân dung" in text and "mũ lưỡi trai" in text


def test_avs_multi_paragraph_joins_all_lines():
    text, question = parse_query_lines("avs", ["đoạn một.", "đoạn hai."])
    assert text == "đoạn một. đoạn hai." and question is None


def test_run_query_file_kis_multi_paragraph_searches_full_text(tmp_path):
    qf = tmp_path / "query-p1-11-kis.txt"
    qf.write_text("đoạn mở đầu chung chung.\n\nchi tiết phân biệt nằm ở đoạn hai.\n",
                  encoding="utf-8")
    engine = _StubEngine()
    run_query_file(engine, qf, tmp_path)
    assert "chi tiết phân biệt" in engine.text_query   # paragraph 2 not dropped


def test_run_query_file_qa_three_lines_vqa_gets_real_question(tmp_path):
    qf = tmp_path / "query-p2-3-qa.txt"
    qf.write_text(
        "Đoạn clip về một gian trưng bày văn hóa - du lịch.\n"
        "Phía trên là quốc kỳ Việt Nam treo chính giữa.\n"
        "Hãy cho biết đây là khu du lịch quốc gia tại địa điểm nào?\n",
        encoding="utf-8",
    )
    hit = SimpleNamespace(video_id="L05_V005", frame_idx=888, global_id=7,
                          ref=SimpleNamespace(path="x.jpg", pts_time=35.5))
    engine = _StubEngine(results=[hit])
    questions: list[str] = []

    class _StubVqa:
        def suggest(self, question, frames):
            questions.append(question)
            return [SimpleNamespace(global_id=7, answer="Măng Đen", provider="stub")]

    run_query_file(engine, qf, tmp_path, vqa=_StubVqa())
    assert questions and questions[0].startswith("Hãy cho biết")   # the REAL question
    assert "quốc kỳ Việt Nam" in engine.text_query                 # paragraph 2 searched


def test_repo_example_round3_fixtures_parse():
    """Round-3 committed fixtures: multi-paragraph KIS + 3-line QA."""
    kis = REPO / "queries" / "example" / "query-0-6-kis-multipara.txt"
    qa = REPO / "queries" / "example" / "query-0-7-qa-multiline.txt"
    kis_lines = [ln.strip() for ln in kis.read_text(encoding="utf-8").splitlines() if ln.strip()]
    text, _ = parse_query_lines("kis", kis_lines)
    assert all(part in text for part in (kis_lines[0][:20], kis_lines[-1][:20]))
    qa_lines = [ln.strip() for ln in qa.read_text(encoding="utf-8").splitlines() if ln.strip()]
    desc, question = parse_query_lines("qa", qa_lines)
    assert question == qa_lines[-1] and qa_lines[0][:20] in desc
