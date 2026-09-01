"""Round-83: the organiser rules (thể lệ AIC26, read 01/09) pinned into the flow.

Facts that change strategy: max 3 submissions per pack and the LAST one
counts (not the best); the public leaderboard scores ~50% of the queries and
ONLY KIS (QA/TRAKE score on private only); QA answers are compared as exact
strings, edge whitespace NOT trimmed; TRAKE frame count must equal the event
count; zip must contain a submission/ folder. The writer already strips edge
whitespace and quotes per the CSV rules; cell 9b now prints the last-counts
reminder with every zip.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_r83_last_submission_counts_reminder_in_nb03():
    nb = json.loads((REPO / "notebooks" / "03_test_system.ipynb")
                    .read_text(encoding="utf-8"))
    pack = next("".join(c["source"]) for c in nb["cells"]
                if c["cell_type"] == "code" and "SHARD_TOTAL" in "".join(c["source"]))
    assert "lần nộp CUỐI CÙNG mới tính điểm" in pack
    assert "Public chỉ chấm ~50% câu và chỉ KIS" in pack


def test_r83_writer_obeys_the_csv_rules():
    from cvp.submission.writer import sanitize_answer
    assert sanitize_answer("  7 cái. ") == "7 cái."            # no edge whitespace
    assert sanitize_answer("A, B") == '"A, B"'                  # comma → quoted
    assert sanitize_answer('nói "hi"') == '"nói ""hi"""'        # quote → doubled
