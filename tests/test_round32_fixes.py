"""Round-32: team access to the competition UI (5-person online session).

Colab's kernel-port proxy window authenticates as the notebook owner — no
teammate can open it. The UI cell gains SHARE_URL: a cloudflared quick tunnel
publishes ONLY the Streamlit interface on a random trycloudflare.com URL
(no Drive/source/notebook exposure; dies with the session). Teammates curate
in parallel — st.session_state keeps each browser's baskets independent.
"""

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_ui_cell_offers_team_tunnel():
    src = (REPO / "notebooks" / "_build_notebooks.py").read_text(encoding="utf-8")
    frag = src.split("NB3_UI = r")[1].split("def main")[0]
    assert "SHARE_URL" in frag
    assert "trycloudflare" in frag
    assert "cloudflared-linux-amd64" in frag
    assert "serve_kernel_port_as_window" in frag     # owner window still there
    # the tunnel must never replace the battle-env inheritance
    assert "_env = dict(os.environ)" in frag
    assert "_env.pop" not in frag


def test_run_pack_survives_missing_pack_dir():
    """Round-34: on round night Run all fires BEFORE the organisers publish
    the pack — a crash here would cut Run all and the UI cell would never
    launch. Missing/empty pack dir must print-and-continue, not raise."""
    src = (REPO / "notebooks" / "_build_notebooks.py").read_text(encoding="utf-8")
    frag = src.split("NB3_RUN_PACK = r")[1].split("NB3_SCORE_GT")[0]
    assert "Run all vẫn đi tiếp" in frag
    assert 'any(_qdir.glob("*.txt"))' in frag
    assert "elif RUN_PACK:" in frag
