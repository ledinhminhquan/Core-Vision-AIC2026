"""Round-42: the 24/7 practice web + private thumbnail store.

Thumbnails of organiser keyframes are competition-licensed data — they live
in PRIVATE HF datasets and are served only BEHIND the team login (never a
public CDN). The same /thumb store also makes the Colab battle web ~10x
lighter over the tunnel (8KB webp vs 60-150KB JPEG).
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from cvp.config import Settings  # noqa: E402
from cvp.web.server import create_team_app  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


class _Catalog:
    def ref(self, gid):
        return SimpleNamespace(global_id=int(gid), video_id="L21_V001",
                               n=int(gid), frame_idx=int(gid) * 25,
                               pts_time=float(gid), path=f"/nope/{gid}.jpg")

    def __len__(self):
        return 10


class _Engine:
    catalog = _Catalog()
    member_names = ["finetuned"]
    settings = None


def _client(tmp_path, monkeypatch, with_thumbs):
    if with_thumbs:
        d = tmp_path / "thumbs" / "L21_V001"
        d.mkdir(parents=True)
        (d / "7.webp").write_bytes(b"RIFFfakewebp")
        monkeypatch.setenv("CVP_WEB__THUMBS_DIR", str(tmp_path / "thumbs"))
    else:
        monkeypatch.delenv("CVP_WEB__THUMBS_DIR", raising=False)
    s = Settings()
    s.paths.artifacts_root = tmp_path
    c = TestClient(create_team_app(_Engine(), s, "u", "p"))
    c.post("/login", json={"username": "u", "password": "p"})
    return c


def test_thumb_endpoint_serves_and_is_auth_gated(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch, with_thumbs=True)
    assert c.get("/whoami").json()["thumbs"] is True
    r = c.get("/thumb/L21_V001/7.webp")
    assert r.status_code == 200
    assert "immutable" in r.headers["cache-control"]
    assert c.get("/thumb/L21_V001/8.webp").status_code == 404
    assert c.get("/thumb/../secrets/7.webp").status_code in (404, 422)
    # no cookie → 401 even for images
    from fastapi.testclient import TestClient as TC
    s = Settings()
    s.paths.artifacts_root = tmp_path
    anon = TC(create_team_app(_Engine(), s, "u", "p"))
    assert anon.get("/thumb/L21_V001/7.webp").status_code == 401


def test_without_store_whoami_says_no(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch, with_thumbs=False)
    assert c.get("/whoami").json()["thumbs"] is False
    assert c.get("/thumb/L21_V001/7.webp").status_code == 404


def test_spa_uses_thumbs_with_keyframe_fallback():
    html = (REPO / "src" / "cvp" / "web" / "static" / "index.html").read_text(
        encoding="utf-8")
    assert "function imgTag" in html
    assert "/thumb/${video}/${n}.webp" in html
    assert "this.src='/keyframe/${gid}'" in html    # graceful fallback


def test_deploy_bundle_is_private_by_design():
    dk = (REPO / "deploy" / "hf-space" / "Dockerfile").read_text(encoding="utf-8")
    assert "--mount=type=secret,id=GITHUB_TOKEN" in dk   # token never in a layer
    assert "7860" in dk
    ap = (REPO / "deploy" / "hf-space" / "app.py").read_text(encoding="utf-8")
    assert 'os.environ["TEAM_PASS"]' in ap               # login is mandatory
    assert 'env("CVP_SEARCH__RERANKER", "none")' in ap   # CPU practice profile
    for script in ("60_make_thumbs.py", "61_push_practice_artifacts.py"):
        t = (REPO / "scripts" / script).read_text(encoding="utf-8")
        assert "private=True" in t                       # competition data stays private
    nb = (REPO / "notebooks" / "_build_notebooks.py").read_text(encoding="utf-8")
    assert '"thumbs"' in nb and "CVP_WEB__THUMBS_DIR" in nb
