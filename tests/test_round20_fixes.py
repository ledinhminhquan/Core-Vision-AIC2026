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
