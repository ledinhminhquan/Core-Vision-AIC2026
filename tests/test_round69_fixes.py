"""Round-69: fixes for the nb02-retrain audit (5 confirmed findings).

1. (blocker) stale export bar: lit_trainer seeds best_r5 from
   export_dir/export_meta.json — the OLD tower's R@5 measured on the OLD
   sparse val (an easier pool). The fresh dense-val run could never beat it,
   would export nothing and early-stop in minutes. On identity change the
   notebook now archives export_dir alongside the run dir.
2. (degrade) the archive rename was silently skipped when the archive name
   already existed (A/B toggles) or the pointer was torn — _archive_unique
   auto-numbers collisions, unreadable pointers still archive (unknown
   identity = untrusted), and the message prints only after a real rename.
3. (degrade) a fresh-but-different-hash 'running' pointer updated <30 min ago
   now aborts: two live nb02 sessions must never share a run dir.
4. (degrade) the deliverables rotation is now idempotent — when latest
   already equals the export (same export_meta bytes) nothing rotates, so a
   second Run all cannot wipe the deliverables/previous rollback.
5. (degrade) the anti-twin gate now covers PROJECT's children: data/ and
   artifacts/ are only created under FIRST_TIME_SETUP; otherwise the mount
   cell waits for them (2 min, with children nudges) and aborts with the
   fresh-VM remedy.
"""
from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BUILDER = (REPO / "notebooks" / "_build_notebooks.py").read_text(encoding="utf-8")


def _frag(start: str, end: str) -> str:
    return BUILDER.split(start)[1].split(end)[0]


def test_r69_export_dir_archived_with_the_run():
    frag = _frag("NB2_RUN_POINTER = r", "NB2_TRAIN")
    assert "_archive_unique(cfg.export_dir" in frag
    assert "xà ngang" in frag                      # the stale-bar rationale


def test_r69_archive_never_silently_skips():
    frag = _frag("NB2_RUN_POINTER = r", "NB2_TRAIN")
    assert "while _dst.exists():" in frag          # auto-numbered collisions
    assert "_has_old = POINTER.exists() or any(" in frag
    assert "(ptr is None or ptr.get(" in frag      # torn pointer still archives
    # the announcement prints AFTER the rename returns
    assert frag.index("Path(_src).rename(_dst)") < frag.index("đã cất run cũ")


def test_r69_live_second_session_guard():
    frag = _frag("NB2_RUN_POINTER = r", "NB2_TRAIN")
    assert '"running"' in frag and "_age < 1800" in frag
    assert "MỘT phiên nb02" in frag


def test_r69_deliverables_rotation_is_idempotent():
    frag = _frag("NB2_DELIVERABLES = r", "NB2_EVAL")
    assert "_meta_bytes(dest) == _meta_bytes(export_dir)" in frag
    assert "không xoay vòng" in frag


def test_r69_anti_twin_gate_covers_children():
    mount = BUILDER.split("CELL_MOUNT = r")[1].split("CELL_REPO_DEPS")[0]
    assert "if p.exists() or FIRST_TIME_SETUP:" in mount
    assert "chưa thấy {p.name}/ trong dự án" in mount
    assert "list(PROJECT.iterdir())" in mount      # children metadata nudge
