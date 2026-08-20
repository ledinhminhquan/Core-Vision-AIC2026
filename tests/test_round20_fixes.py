"""Round-20 — live-run-9: on a fresh VM, DriveFS can list data/ as EMPTY for
minutes after a successful mount (lazy metadata sync). Cell 2's mkdir
exist_ok=True then MASKS the symptom and cell 7 dies with a baffling
"Keyframes folder not found". The mount cell now gates on data/ actually
listing content (3-minute poll — each listdir nudges DriveFS to fetch) and
stops LOUDLY with the three real causes (lazy sync / wrong Google account /
nothing uploaded yet) instead of letting later cells run against a void.
"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = (REPO / "notebooks" / "_build_notebooks.py").read_text(encoding="utf-8")
MOUNT = SRC.split("Mount Drive + folder layout")[1].split("Get repo")[0]


def test_mount_cell_gates_on_data_presence():
    assert "DATA-PRESENCE GATE" in MOUNT
    # polls (nudging DriveFS) instead of failing on the first empty listing
    assert "any(DATA_DIR.iterdir())" in MOUNT and "time.sleep(10)" in MOUNT
    # the gate sits AFTER the mkdir that would otherwise mask the void
    assert MOUNT.index("exist_ok=True") < MOUNT.index("DATA-PRESENCE GATE")


def test_gate_failure_names_all_three_causes():
    assert "NHẦM tài khoản" in MOUNT          # wrong Google account
    assert "sync quá chậm" in MOUNT           # DriveFS lag → fresh VM
    assert "chưa upload dữ liệu" in MOUNT     # first run, nothing uploaded


# ── R28 · lazy subdir stats must never flip the materialize gate ─────────────
def test_materialize_gate_polls_and_accepts_zips():
    """Live nb03 run 5: DriveFS listed data/ but stat'd data/keyframes as
    absent → the shared materialize cell silently fell to Drive-direct and the
    objects cell died at catalog build. The gate now nudge-polls (listdir
    forces metadata) and accepts Keyframes*.zip as proof."""
    src = (REPO / "notebooks" / "_build_notebooks.py").read_text(encoding="utf-8")
    assert "def _kf_visible" in src
    assert 'list(DATA_DIR.iterdir())' in src              # the metadata nudge
    assert "Keyframes*.zip" in src or "key-frames" in src # zips count as proof
    assert "COPY_KEYFRAMES_LOCAL and _kf_ok" in src       # gate uses the poll


def test_objects_cell_polls_before_rebuilding():
    src = (REPO / "notebooks" / "_build_notebooks.py").read_text(encoding="utf-8")
    frag = src.split("NB3_OBJECTS = r")[1].split("NB3_ARTIFACTS_LOCAL")[0]
    assert "nudge" in frag and "_pq.exists()" in frag
