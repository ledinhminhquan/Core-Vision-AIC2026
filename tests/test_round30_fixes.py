"""Round-30: nb03 becomes the round-day cockpit (trial round, Aug 20).

Two changes ahead of the first live submission:
- The UI cell's verify-R23 env scrub is REVERSED: since round-29 the engine
  cell exports the battle config (finetuned + gemini + local artifacts), so
  popping CVP_EMBEDDING__MODEL / CVP_QUERY__PROVIDER silently downgraded the
  competition UI to zero-shot siglip2 reading Drive.
- A one-button organiser-pack cell (RUN_PACK) runs queries/<pack>/*.txt
  through run_auto with the warm cell-5 engine → CSVs → strict validation →
  submission.zip → synced to Drive.
"""

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = (REPO / "notebooks" / "_build_notebooks.py").read_text(encoding="utf-8")


def _cell(name: str) -> str:
    return SRC.split(f"{name} = r")[1].split("'''")[1]


def test_ui_cell_inherits_battle_env():
    ui = _cell("NB3_UI")
    assert '_env.pop("CVP_QUERY__PROVIDER"' not in ui
    assert '_env.pop("CVP_EMBEDDING__MODEL"' not in ui
    assert "env=_env" in ui                      # still passes an explicit env


def test_run_pack_cell_is_wired():
    cell = _cell("NB3_RUN_PACK")
    assert "run_auto" in cell
    assert "engine_factory=lambda _s: engine" in cell   # no second engine build
    assert "submit=False" in cell                       # BTC web upload, not DRES
    assert "rep.failed" in cell                         # dropped queries surfaced
    assert "dirs_exist_ok=True" in cell                 # Drive sync
    # The cell must sit in the built notebook, between package and GT scoring.
    order = SRC.split('write_nb("03_test_system.ipynb"')[1].split("])")[0]
    cells = [ln.strip() for ln in order.splitlines() if "code(" in ln]
    assert cells.index("code(NB3_RUN_PACK),") == cells.index("code(NB3_PACKAGE),") + 1
