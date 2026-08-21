"""Round-36: the fast team web UI (SPA + shared-account auth + pack export).

Streamlit's full-script rerun felt laggy to the operators; the new layer
serves a vanilla-JS SPA from the SAME process as the CPU-tested JSON service,
gated behind ONE shared team login, with direct-to-pack CSV export.
"""

import csv
from pathlib import Path
from types import SimpleNamespace

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from cvp.config import Settings  # noqa: E402
from cvp.web.server import create_team_app  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def _ref(gid, video="L21_V001", n=None, frame=None):
    return SimpleNamespace(global_id=gid, video_id=video, n=n or gid,
                           frame_idx=frame or gid * 25, pts_time=float(gid),
                           path=f"/nope/{gid}.jpg")


class _Catalog:
    def ref(self, gid):
        if int(gid) > 500:
            raise KeyError(gid)
        return _ref(int(gid))

    def refs(self, gids):
        return [self.ref(g) for g in gids]

    def __len__(self):
        return 100


class _Engine:
    catalog = _Catalog()
    member_names = ["finetuned"]
    settings = None


@pytest.fixture()
def client(tmp_path):
    s = Settings()
    s.paths.artifacts_root = tmp_path
    app = create_team_app(_Engine(), s, "aic2026-222", "secret-pass")
    return TestClient(app)


def _login(client):
    r = client.post("/login", json={"username": "aic2026-222",
                                    "password": "secret-pass"})
    assert r.status_code == 200


def test_everything_is_locked_without_login(client):
    for path in ("/whoami", "/keyframe/1", "/context/1"):
        assert client.get(path).status_code == 401, path
    assert client.post("/export", json={"task": "kis", "stem": "x-kis",
                                        "pins": [1]}).status_code == 401
    # the SPA shell and login stay open — that's the whole point of a login page
    assert client.get("/").status_code == 200
    assert client.get("/health").status_code == 200


def test_wrong_password_rejected(client):
    r = client.post("/login", json={"username": "aic2026-222", "password": "nope"})
    assert r.status_code == 401


def test_empty_team_password_refused():
    with pytest.raises(ValueError):
        create_team_app(_Engine(), Settings(), "u", "")


def test_export_pins_first_dedup_and_task_guard(client, tmp_path):
    _login(client)
    # wrong-task stem refused (QA export over a -kis stem)
    r = client.post("/export", json={"task": "qa", "pack": "p1",
                                     "stem": "query-p1-9-kis", "pins": [1],
                                     "results": [2], "answer": "x"})
    assert r.status_code == 422
    # pins lead, machine tail deduped
    r = client.post("/export", json={"task": "kis", "pack": "p1",
                                     "stem": "query-p1-9-kis",
                                     "pins": [7, 3], "results": [1, 3, 7, 2]})
    assert r.status_code == 200, r.text
    out = tmp_path / "submissions" / "p1" / "query-p1-9-kis.csv"
    rows = [tuple(x) for x in csv.reader(out.open(encoding="utf-8")) if x]
    assert [int(r[1]) for r in rows] == [7 * 25, 3 * 25, 1 * 25, 2 * 25]


def test_qa_export_requires_answer(client):
    _login(client)
    r = client.post("/export", json={"task": "qa", "pack": "p1",
                                     "stem": "query-p1-15-qa",
                                     "pins": [1], "results": [], "answer": "  "})
    assert r.status_code == 422
    assert "ĐÁP ÁN" in r.json()["detail"]


def test_stem_path_traversal_refused(client):
    _login(client)
    r = client.post("/export", json={"task": "kis", "pack": "../evil",
                                     "stem": "query-p1-9-kis", "pins": [1]})
    assert r.status_code == 422


def test_notebook_cell_and_watchdog_wired():
    src = (REPO / "notebooks" / "_build_notebooks.py").read_text(encoding="utf-8")
    frag = src.split("NB3_FASTUI = r")[1].split("NB3_KEEPALIVE")[0]
    assert "create_team_app(engine, settings" in frag   # reuses the warm engine
    assert "secrets.token_urlsafe" in frag              # empty pass → generated
    assert "8600" in frag
    keep = src.split("NB3_KEEPALIVE = r")[1].split("def main")[0]
    assert "_tun2" in keep and "LINK WEB NHANH MỚI" in keep
    order = src.split('write_nb("03_test_system.ipynb"')[1].split("])")[0]
    cells = [ln.strip() for ln in order.splitlines() if "code(" in ln]
    assert cells[-2:] == ["code(NB3_FASTUI),", "code(NB3_KEEPALIVE),"]


def test_spa_ships_with_the_package():
    html = (REPO / "src" / "cvp" / "web" / "static" / "index.html").read_text(
        encoding="utf-8")
    for marker in ("/login", "/search/text", "/search/qa", "/trake2",
                   "/export", "/keyframe/", "REZIP_ONLY"):
        assert marker in html, marker


def test_r37_vlm_reach_extended_and_trake_depth():
    """Round-37, mined from the 19.8/23 reference submission: ground truth sat
    at ranks 25/38/79 in three KIS queries — outside the VLM reranker's old
    top-24 window; one TRAKE truth sat at rank 91 — outside the SPA's old
    50-row fetch."""
    src = (REPO / "notebooks" / "_build_notebooks.py").read_text(encoding="utf-8")
    frag = src.split("NB3_ENGINE = r")[1].split("NB3_QUERIES")[0]
    assert "VLM_RERANK_TOPK = 48" in frag
    assert "CVP_SEARCH__VLM_RERANK_TOPK" in frag
    html = (REPO / "src" / "cvp" / "web" / "static" / "index.html").read_text(
        encoding="utf-8")
    assert "max_results: 100" in html


def test_r38_avs_tab_in_fast_ui():
    """Round-38: AVS tab added to the SPA (the /search/avs endpoint already
    existed); KIS-C stays Streamlit-only by design (stateful dialogue)."""
    html = (REPO / "src" / "cvp" / "web" / "static" / "index.html").read_text(
        encoding="utf-8")
    assert "/search/avs" in html
    assert "m-avs" in html and "searchAvs" in html
    assert "limit: 100" in html
