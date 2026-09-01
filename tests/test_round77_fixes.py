"""Round-77: the đợt-3 backlog, self-built (Cursor usage ran out) + the two
burned-in lessons from battle night 28/08.

1. run_auto(resume=True) keeps existing non-empty query CSVs and only runs
   the missing ones — the VM died mid-pack that night and a full re-run was
   the only option, with the submission window closing.
2. submission.pack_deadline_min: past the mark, the remaining queries run in
   SPRINT mode (QA votes 1, no neighbor strips, no VLM rerank) — the 504
   storm stretched a 60' pack to 142'.
3. vqa.answer_variant_rows: dual-format số↔chữ rows ("sáu"↔"6") replacing
   tail rows — exact-text grading insurance.
4. vqa.exact_transcription: QA prompts demand verbatim transcription (q19
   lost GT by ONE word because the model paraphrased the poem).
All default-off; legacy paths untouched.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from cvp.config import Settings
from cvp.search.answer_norm import answer_format_variant, num_to_words_vi

REPO = Path(__file__).resolve().parents[1]


def test_r77_num_to_words_vi():
    for n, want in [(0, "không"), (6, "sáu"), (10, "mười"), (15, "mười lăm"),
                    (21, "hai mươi mốt"), (24, "hai mươi tư"), (25, "hai mươi lăm"),
                    (100, "một trăm"), (105, "một trăm linh năm"),
                    (230, "hai trăm ba mươi"), (999, "chín trăm chín mươi chín")]:
        assert num_to_words_vi(n) == want, n
    with pytest.raises(ValueError):
        num_to_words_vi(1000)


def test_r77_answer_format_variant_both_directions():
    assert answer_format_variant("6") == "sáu"
    assert answer_format_variant("sáu") == "6"
    assert answer_format_variant("27 cái") == "hai mươi bảy cái"
    assert answer_format_variant("hai mươi bảy cái") == "27 cái"
    # protected / no-op cases
    assert answer_format_variant("không rõ") is None
    assert answer_format_variant("Cá hanh") is None
    assert answer_format_variant("500g") is None      # digit glued to a unit


def test_r77_variant_rows_off_is_identity_on_replaces_tail():
    from cvp.pipeline.run_queries import _maybe_answer_variants
    s = Settings()
    rows = [(f"L01_V{i:03d}", i, "sáu") for i in range(1, 101)]
    assert _maybe_answer_variants(s, rows) is rows          # off = same object
    s.vqa.answer_variant_rows = True
    out = _maybe_answer_variants(s, rows)
    assert len(out) == 100
    assert out[:30] == rows[:30]                            # head untouched
    assert out[-1] == ("L01_V001", 1, "6")                  # variant rides the tail
    assert sum(1 for r in out if r[2] == "6") == 1          # dedup by answer


def test_r77_exact_transcription_prompt_gated():
    from cvp.search.vqa import _exact_suffix
    off = type("C", (), {"exact_transcription": False})()
    on = type("C", (), {"exact_transcription": True})()
    assert _exact_suffix(off) == ""
    assert "CHÍNH XÁC" in _exact_suffix(on)


def test_r77_resume_keeps_existing_csvs(tmp_path, monkeypatch):
    from cvp.pipeline import auto_agent
    qdir = tmp_path / "q"; qdir.mkdir()
    out = tmp_path / "out"; out.mkdir()
    (qdir / "query-p9-1-kis.txt").write_text("mô tả một", encoding="utf-8")
    (qdir / "query-p9-2-kis.txt").write_text("mô tả hai", encoding="utf-8")
    (out / "query-p9-1-kis.csv").write_text("L01_V001,5\n", encoding="utf-8")
    ran = []

    def fake_run_query_file(engine, qf, out_dir, vqa, top1_times=None):
        ran.append(qf.stem)
        p = out_dir / f"{qf.stem}.csv"
        p.write_text("L01_V002,7\n", encoding="utf-8")
        return p
    import cvp.pipeline.run_queries as rq
    monkeypatch.setattr(rq, "run_query_file", fake_run_query_file)
    s = Settings()
    s.vqa.provider = "none"
    rep = auto_agent.run_auto(qdir, out, s, submit=False,
                              engine_factory=lambda _s: object(), resume=True)
    assert ran == ["query-p9-2-kis"]                       # 1 was kept, 2 ran
    assert sorted(p.name for p in rep.written) == [
        "query-p9-1-kis.csv", "query-p9-2-kis.csv"]


def test_r77_sprint_mode_flips_settings_after_deadline(tmp_path, monkeypatch):
    from cvp.pipeline import auto_agent
    qdir = tmp_path / "q"; qdir.mkdir()
    out = tmp_path / "out"
    for i in (1, 2):
        (qdir / f"query-p9-{i}-kis.txt").write_text("x", encoding="utf-8")
    seen = []

    def fake_run_query_file(engine, qf, out_dir, vqa, top1_times=None):
        seen.append((qf.stem, s.vqa.self_consistency, s.search.vlm_rerank))
        p = out_dir / f"{qf.stem}.csv"
        p.write_text("L01_V001,5\n", encoding="utf-8")
        return p
    import cvp.pipeline.run_queries as rq
    monkeypatch.setattr(rq, "run_query_file", fake_run_query_file)
    s = Settings()
    s.vqa.provider = "none"
    s.vqa.self_consistency = 3
    s.search.vlm_rerank = True
    s.submission.pack_deadline_min = 1e-9                   # already past
    auto_agent.run_auto(qdir, out, s, submit=False,
                        engine_factory=lambda _s: object())
    # first query already ran in sprint mode (deadline hit before it)
    assert seen[0][1] == 1 and seen[0][2] is False
    # ...and the shared settings object is RESTORED once the pack ends, so a
    # later pack in the same kernel never silently inherits sprint mode.
    assert s.vqa.self_consistency == 3
    assert s.search.vlm_rerank is True


def test_r77_resume_three_shields(tmp_path):
    from cvp.pipeline.auto_agent import _keep_resumed_csv
    import os
    q = tmp_path / "query-p9-1-kis.txt"; q.write_text("x", encoding="utf-8")
    p = tmp_path / "query-p9-1-kis.csv"
    assert not _keep_resumed_csv(p, q)                     # missing
    p.write_text("", encoding="utf-8")
    assert not _keep_resumed_csv(p, q)                     # empty
    p.write_text("L01_V001,5\n", encoding="utf-8")
    assert _keep_resumed_csv(p, q)                         # valid + fresh
    # stale-pack shield: CSV older than the query file → re-run
    os.utime(p, (1, 1))
    assert not _keep_resumed_csv(p, q)
    # corrupt shield: validate errors → re-run instead of vetoing the pack zip
    os.utime(p, None)
    p.write_text("NOT_A_VIDEO_ID,5\n", encoding="utf-8")
    assert not _keep_resumed_csv(p, q)


def test_r77_sprint_restore_survives_interrupt(tmp_path, monkeypatch):
    from cvp.pipeline import auto_agent
    qdir = tmp_path / "q"; qdir.mkdir()
    (qdir / "query-p9-1-kis.txt").write_text("x", encoding="utf-8")

    def boom(engine, qf, out_dir, vqa, top1_times=None):
        raise KeyboardInterrupt        # battle-night Stop button mid-sprint
    import cvp.pipeline.run_queries as rq
    monkeypatch.setattr(rq, "run_query_file", boom)
    s = Settings()
    s.vqa.provider = "none"
    s.vqa.self_consistency = 3
    s.search.vlm_rerank = True
    s.submission.pack_deadline_min = 1e-9
    with pytest.raises(KeyboardInterrupt):
        auto_agent.run_auto(qdir, tmp_path / "o", s, submit=False,
                            engine_factory=lambda _s: object())
    assert s.vqa.self_consistency == 3     # finally-restore held the line
    assert s.search.vlm_rerank is True


def test_r77_notebook_toggles_and_wiring():
    nb = json.loads((REPO / "notebooks" / "03_test_system.ipynb")
                    .read_text(encoding="utf-8"))
    cell = next("".join(c["source"]) for c in nb["cells"]
                if c["cell_type"] == "code" and "RUN_PACK" in "".join(c["source"]))
    assert "RESUME_PACK = True" in cell
    assert "PACK_DEADLINE_MIN = 0" in cell
    assert "resume=RESUME_PACK" in cell
    # audit r77: gán VÔ ĐIỀU KIỆN — không còn `if PACK_DEADLINE_MIN:` sticky
    assert "settings.submission.pack_deadline_min = float(PACK_DEADLINE_MIN)" in cell
    assert "if PACK_DEADLINE_MIN:" not in cell
