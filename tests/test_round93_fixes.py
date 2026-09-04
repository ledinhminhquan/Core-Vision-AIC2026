"""Round-93: battle-night consolidation (04/09/2026, sơ tuyển at 19:30 VN).

1. nb03 QA_PARALLEL = 1 — gemini-3.1-pro-preview is capped at 25 requests per
   minute on the battle key; 4 shards x 2 threads = 8 concurrent Pro calls
   would trip that cap and push QA onto Flash while quota remains.
2. Attempt 2 is a RESCUE, not a re-run: nb03 RESCUE_QA (with RESUME_PACK)
   deletes the QA CSVs whose rows are all the fallback answer so run_auto
   re-answers exactly those queries; everything else is kept, so the rescued
   pack can never score below attempt 1. The 03/09 bench refuted the old
   "diverse lineup + RRF merge" plan (MERGE2 -0.015 vs ABK).
"""
from __future__ import annotations

import json
from pathlib import Path

from cvp.pipeline.attempts import is_non_answer, qa_fallback_only, rescue_fallback_qa

REPO = Path(__file__).resolve().parents[1]


def test_r93_qa_fallback_only():
    assert qa_fallback_only([["L01_V001", "10", "không rõ"], ["L01_V002", "20", "Không rõ "]])
    assert qa_fallback_only([["L01_V001", "10", ""], ["L01_V002", "20"]])       # no answers at all
    assert qa_fallback_only([]) is True
    assert not qa_fallback_only([["L01_V001", "10", "không rõ"], ["L01_V002", "20", "7"]])
    assert not qa_fallback_only([["L01_V001", "10", "Con hến"]])


def test_r94_refusal_phrases_are_non_answers():
    # seen in the 04/09 rehearsal when Pro was out of quota and Flash answered
    assert is_non_answer("Không có thông tin trong dữ liệu được cung cấp.")
    assert is_non_answer("Không có thông tin về X.")
    assert is_non_answer("Không xác định được")
    assert is_non_answer("Không thể xác định từ hình ảnh")
    assert is_non_answer("Chưa rõ")
    assert is_non_answer("Unknown") and is_non_answer("N/A") and is_non_answer("không rõ")
    # legitimate answers stay answers
    assert not is_non_answer("Không")                 # yes/no question
    assert not is_non_answer("Không Gian Xanh")       # a proper name
    assert not is_non_answer("Cá hồi") and not is_non_answer("7 cái") and not is_non_answer("2")
    rows = [["L01_V001", "10", "Không có thông tin trong dữ liệu được cung cấp."],
            ["L01_V002", "20", "Không có thông tin về X."]]
    assert qa_fallback_only(rows)
    assert not qa_fallback_only(rows + [["L01_V003", "30", "Cá bống mú"]])


def test_r93_rescue_deletes_only_all_fallback_qa_csvs(tmp_path):
    (tmp_path / "query-p2-1-qa.csv").write_text("L01_V001,10,không rõ\nL01_V002,20,không rõ\n",
                                                encoding="utf-8")
    (tmp_path / "query-p2-2-qa.csv").write_text("L01_V001,10,không rõ\nL01_V003,30,7\n",
                                                encoding="utf-8")
    (tmp_path / "query-p2-3-kis.csv").write_text("L01_V001,10\n", encoding="utf-8")
    (tmp_path / "query-p2-4-trake.csv").write_text("L01_V001,10,20,30\n", encoding="utf-8")
    removed = rescue_fallback_qa(tmp_path)
    assert removed == ["query-p2-1-qa"]
    assert not (tmp_path / "query-p2-1-qa.csv").exists()
    assert (tmp_path / "query-p2-2-qa.csv").exists()          # has a real answer → kept
    assert (tmp_path / "query-p2-3-kis.csv").exists() and (tmp_path / "query-p2-4-trake.csv").exists()
    assert rescue_fallback_qa(tmp_path) == []                  # idempotent


def test_r93_nb03_battle_knobs():
    nb = json.loads((REPO / "notebooks" / "03_test_system.ipynb").read_text(encoding="utf-8"))
    cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    engine = next(s for s in cells if "engine = SearchEngine(settings)" in s)
    pack = next(s for s in cells if "RUN_PACK" in s and "SHARD_TOTAL" in s)
    assert "QA_PARALLEL = 1" in engine and "25 request/" in engine
    assert "RESCUE_QA = False" in pack                          # off by default: attempt 1 untouched
    assert "if RESCUE_QA and RESUME_PACK:" in pack and "rescue_fallback_qa(_out)" in pack
    # the rescue runs AFTER the cross-VM pull and BEFORE run_auto
    assert pack.index("kéo {_n_pull}") < pack.index("rescue_fallback_qa(_out)") < pack.index(
        "rep = run_auto(_qsrc, _out, settings, submit=False, resume=RESUME_PACK")
    # battle line-up unchanged: ABK knobs + 2-lane ensemble + Pro QA
    assert '"CVP_SEARCH__KIS_MULTI_EVENT": "true"' in engine and 'LINEUP = "battle"' in engine
