"""Round-14 — live-run-5 lesson: Drive FUSE CRASHED mid-copytree of 177k JPGs
([Errno 107] Transport endpoint is not connected) after 3h+. Cell 6 now
materializes local data FROM THE SOURCE ZIPS (few large sequential reads),
resumable per zip, with automatic remount; long GPU/aux stages re-check the
mount between lanes/stages.
"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = (REPO / "notebooks" / "_build_notebooks.py").read_text(encoding="utf-8")
CELL6 = SRC.split("Materialize data")[1].split("INTEGRITY + SELF-HEAL")[0]


def test_cell6_is_zip_first_not_copytree_over_fuse():
    # the killer path — a bulk copytree of the Drive keyframes tree — is gone
    assert "shutil.copytree(src, tmp_dst)" not in CELL6
    # zips are copied local first, then extracted locally
    assert "shutil.copyfile(zp, lz)" in CELL6 and "z.extractall(tmp_root)" in CELL6


def test_cell6_remount_guard_and_retries():
    # round-19: the HARDENED _ensure_drive lives in the MOUNT cell (cell 2)
    # and is shared by every later cell — cell 6 only calls it.
    mount = SRC.split("Mount Drive + folder layout")[1].split("Get repo")[0]
    assert "def _ensure_drive" in mount and "force_remount=bool(_try)" in mount
    # dead-daemon recovery: lazy-unmount the corpse, clear LOCAL leftovers
    # ONLY when nothing is mounted, then mount again
    assert "fusermount" in mount and "os.path.ismount" in mount
    assert "Mountpoint must not already contain files" in mount  # names run 8
    assert "def _ensure_drive" not in CELL6 and "_ensure_drive()" in CELL6
    # both phases retry through FUSE hiccups instead of dying on attempt 1
    assert CELL6.count("for _attempt in (1, 2, 3):") == 2
    assert "Transport endpoint" in SRC  # the failure is named for future readers


def test_cell6_resume_granularity_and_stamps():
    # per-zip local markers + per-family session stamp → same-session re-runs
    # skip instantly; a crash mid-zip re-extracts only that zip
    assert '.unzipped-{_sub}-{zp.stem}' in CELL6
    assert '.materialized-{_sub}' in CELL6


def test_cell6_zip_family_matches_cell5_priority():
    # map-keyframes must be tested BEFORE the generic keyframes prefix in BOTH
    # routers, or one cell extracts a zip the other refuses to
    body = CELL6.split("def _zip_family")[1].split("def _walk_wrapper")[0]
    assert body.index('"map-keyframes"') < body.index('"keyframes"')
    assert '"clip-features-32"' in body and '"objects"' in body


def test_cell6_disk_space_guard():
    assert "disk_usage" in CELL6 and "COPY_KEYFRAMES_LOCAL=False" in CELL6


def test_cell6_verify_findings_closed():
    # HIGH: session stamp must NOT hide zips uploaded after materialization
    assert "zip mới sau lần materialize trước" in CELL6
    # HIGH: file-only family with no source zip → dst must exist before phase 2
    p2 = CELL6.split("PHA 2")[1]
    assert p2.index("dst.mkdir(parents=True, exist_ok=True)") < p2.index("for _attempt")
    # MED: every FUSE-touching syscall of phase 1 lives INSIDE the retry —
    # the per-zip stat/disk check follows the attempt-loop _ensure_drive()
    p1 = CELL6.split("PHA 1")[1].split("PHA 2")[0]
    assert p1.index("for _attempt") < p1.index("zp.stat()")
    # MED: phase-2 srcD.exists() is also inside the guarded retry
    assert p2.index("_ensure_drive()") < p2.index("srcD.exists()")


def test_long_stages_recheck_mount():
    # cell 8 (per embed lane) and cell 9 (per aux stage) both re-check the
    # mount so a FUSE death in one multi-hour stage can't poison the next
    assert SRC.count('globals().get("_ensure_drive")') == 2
