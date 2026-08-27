"""Round-68: make nb02 exploit the upgraded artifacts safely.

The dense-caption store (177K pairs, 3.5× the training supervision) flows
into nb02's train_data automatically via the existing `_stale_caps` rebuild.
Three gaps closed so the retrain is real, clean, and reversible:

1. the run identity now includes a DATASET FINGERPRINT (sizes of the
   train_data parquet/npy files) — without it, a rerun after the caption
   upgrade would hash-match the old finished run and "resume" it in two
   minutes, training nothing on the new data;
2. on an identity change the old run dir is auto-ARCHIVED
   (runs/vi_siglip2-<oldhash>) instead of asking the user to hand-delete —
   zero-edit UX, nothing destroyed, no same-name Drive collisions;
3. the deliverables mirror backs up the live tower to deliverables/previous
   before overwriting latest — the battle lane has a one-swap rollback if
   the offline bench rejects the new tower.
"""
from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BUILDER = (REPO / "notebooks" / "_build_notebooks.py").read_text(encoding="utf-8")


def _frag(start: str, end: str) -> str:
    return BUILDER.split(start)[1].split(end)[0]


def test_r68_run_identity_includes_dataset_fingerprint():
    frag = _frag("NB2_RUN_POINTER = r", "NB2_TRAIN")
    assert 'd["dataset_fingerprint"] = DATA_FP' in frag
    assert '_td.glob("*.parquet")' in frag and '_td.glob("*.npy")' in frag
    # fingerprint must be computed BEFORE the hash definition uses it
    assert frag.index("DATA_FP = ") < frag.index("def cfg_hash")


def test_r68_identity_change_archives_the_old_run():
    frag = _frag("NB2_RUN_POINTER = r", "NB2_TRAIN")
    # round-69 hardened the archive: unique-suffix names via _archive_unique
    assert 'f"vi_siglip2-{_old_tag}"' in frag
    assert "cất run cũ" in frag
    assert "def _archive_unique(" in frag
    # never delete anything, never require manual cleanup
    assert "rmtree" not in frag and "unlink" not in frag
    assert "Muốn train sạch từ đầu: xoá" not in frag   # the old manual advice is gone


def test_r68_deliverables_mirror_keeps_a_rollback():
    frag = _frag("NB2_DELIVERABLES = r", "NB2_EVAL")
    assert '"deliverables" / "previous"' in frag
    # the backup happens BEFORE latest is overwritten
    assert frag.index("shutil.copytree(dest, prev)") < frag.index(
        "shutil.copytree(export_dir, dest")
