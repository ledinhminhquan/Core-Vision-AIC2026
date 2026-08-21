"""Round-39: the separator-less TRAKE prefix that cost query-p1-16 at round 1.

BTC's LIVE sơ-tuyển-đợt-1 pack wrote event lines as ``E1 Khoảnh khắc…`` —
no ":" / "." / ")" / "-" after the number, unlike every 2025 pack. The old
``_EVENT_RE`` matched nothing, ``parse_trake_events`` fell back to
"every line is an event", the header line became phantom event #1, all 100
rows carried 4 frames against the 3 real events, and the organiser grader
rejected the file outright ("Expected 3 frame IDs, got 4"). One full query
scored 0 while the format looked locally valid.
"""

from cvp.pipeline.run_queries import parse_trake_events

# The EXACT live file content (query-p1-16-trake.txt, sơ tuyển đợt 1).
LIVE_LINES = [
    "Đoạn video bắt đầu bằng ảnh cận đầu một con lân trắng, mũi đỏ, "
    "bên cạnh lá cờ trắng viền đỏ.",
    "E1 Khoảnh khắc đầu tiên xuất hiện đầy đủ hai con rồng vàng đang xoay vòng.",
    "E2 Khoảnh khắc đầu tiên con lân hoàn tất cú xoay người trên các thanh trụ "
    "(thời điểm đâu tiên các chân của lân đặt trên trụ).",
    "E3 Khoảnh khắc đầu tiên dùi chạm vào kẻng đồng múa lân.",
]


def test_live_round1_separator_less_prefixes():
    events = parse_trake_events(LIVE_LINES)
    assert len(events) == 3                       # header dropped, NOT event #1
    assert events[0].startswith("Khoảnh khắc đầu tiên xuất hiện")
    assert all(not e.startswith(("E1", "E2", "E3")) for e in events)


def test_2025_separator_forms_still_parse():
    lines = ["Mô tả chung:", "E1: chạy đà", "e2. giậm nhảy", "E3) bay",
             "E4 - tiếp đất"]
    assert parse_trake_events(lines) == ["chạy đà", "giậm nhảy", "bay", "tiếp đất"]


def test_bare_marker_line_is_dropped_and_no_prefix_files_unchanged():
    # a line that is ONLY "E2" yields empty text → dropped, order preserved
    assert parse_trake_events(["header", "E1 mở màn", "E2", "E3 kết thúc"]) == \
        ["mở màn", "kết thúc"]
    # one-event-per-line files without any markers pass through untouched
    plain = ["vận động viên chạy đà", "vận động viên giậm nhảy"]
    assert parse_trake_events(plain) == plain
