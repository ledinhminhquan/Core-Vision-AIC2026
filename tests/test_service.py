"""HTTP service endpoints exercised in-process with a stub engine (no models).

The service is the machine-callable face for the 2026 automatic track; these
tests pin the response schemas and error paths so any future protocol adapter
can rely on them. Skipped cleanly when fastapi is not installed.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from cvp.config import Settings  # noqa: E402
from cvp.service.app import create_app  # noqa: E402


def _ref(gid: int, vid: str = "L01_V001", n: int | None = None):
    n = n or gid + 1
    return SimpleNamespace(global_id=gid, video_id=vid, n=n,
                           frame_idx=100 * n, pts_time=float(4 * n),
                           path=f"/fake/{vid}/{n:03d}.jpg", fps=25.0, has_map=True)


def _result(gid: int, score: float, vid: str = "L01_V001"):
    ref = _ref(gid, vid)
    return SimpleNamespace(ref=ref, score=score, signals={"visual": score},
                           video_id=ref.video_id, frame_idx=ref.frame_idx,
                           global_id=ref.global_id)


class _StubEngine:
    member_names = ["siglip2", "openclip"]

    def __init__(self):
        self.settings = Settings()
        self.catalog = SimpleNamespace(
            __len__=lambda s: 42,
            ref=lambda gid: _ref(int(gid)) if int(gid) < 42 else (_ for _ in ()).throw(KeyError(gid)),
        )
        # catalog len via SimpleNamespace needs a real object; replace below.

    def search_text(self, query, topk=None, display_k=None):
        return [_result(0, 0.9), _result(1, 0.8), _result(2, 0.7, vid="L02_V002")]

    def search_avs(self, query, limit=100):
        return [_result(0, 0.9), _result(2, 0.6, vid="L02_V002")]

    def search_trake(self, events, max_results=100):
        frames = [100 * (i + 1) for i in range(len(events))]
        return [SimpleNamespace(video_id="L03_V003", frame_idxs=frames,
                                pts_times=[f / 25.0 for f in frames], score=0.5)]

    def nearest(self, global_id, k=60):
        if int(global_id) >= 42:
            raise KeyError(global_id)
        return [_result(int(global_id), 1.0)]

    def search_image(self, image, display_k=None):
        self.last_image_size = image.size
        return [_result(7, 0.99)]


class _Catalog:
    def __len__(self):
        return 42

    def ref(self, gid):
        gid = int(gid)
        if gid >= 42:
            raise KeyError(gid)
        return _ref(gid)


@pytest.fixture()
def client():
    engine = _StubEngine()
    engine.catalog = _Catalog()
    settings = Settings()
    settings.vqa.provider = "none"          # deterministic offline QA fallback
    app = create_app(engine=engine, settings=settings)
    with TestClient(app) as c:
        yield c


def test_health_reports_corpus_and_members(client):
    body = client.get("/health").json()
    assert body == {"status": "ok", "keyframes": 42,
                    "members": ["siglip2", "openclip"]}


def test_search_text_schema(client):
    r = client.post("/search/text", json={"query": "một người áo đỏ"})
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 3
    top = body["results"][0]
    assert top["video_id"] == "L01_V001" and top["frame_idx"] == 100
    assert top["image_url"] == "/keyframe/0"
    assert top["signals"]["visual"] == pytest.approx(0.9)
    assert top["answer"] is None


def test_search_qa_rows_carry_fallback_answer_offline(client):
    r = client.post("/search/qa", json={"query": "mô tả", "question": "màu gì?"})
    assert r.status_code == 200
    answers = [row["answer"] for row in r.json()["results"]]
    # provider=none → deterministic fallback answer on every row (never null).
    assert answers and all(a == "không rõ" for a in answers)


def test_search_qa_answers_flag_off_keeps_null(client):
    r = client.post("/search/qa", json={"query": "mô tả", "answers": False})
    assert all(row["answer"] is None for row in r.json()["results"])


def test_search_trake_schema(client):
    r = client.post("/search/trake", json={"events": ["E1", "E2", "E3"]})
    body = r.json()
    assert body["events"] == ["E1", "E2", "E3"] and body["count"] == 1
    item = body["results"][0]
    assert item["frame_sequence"] == [100, 200, 300]
    assert len(item["pts_times"]) == 3


def test_search_avs_schema_and_limit_validation(client):
    assert client.post("/search/avs", json={"query": "q"}).json()["count"] == 2
    assert client.post("/search/avs", json={"query": "q", "limit": 500}).status_code == 422


def test_search_image_accepts_b64_and_data_url(client):
    import base64
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (4, 4), (200, 30, 30)).save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    for payload in (b64, f"data:image/png;base64,{b64}"):
        r = client.post("/search/image", json={"image_b64": payload})
        assert r.status_code == 200
        assert r.json()["results"][0]["global_id"] == 7


def test_search_image_rejects_garbage(client):
    assert client.post("/search/image",
                       json={"image_b64": "not-base64!!"}).status_code == 422


def test_nearest_ok_and_404(client):
    assert client.get("/nearest/1").json()["count"] == 1
    assert client.get("/nearest/999").status_code == 404


def test_keyframe_404_on_unknown_gid(client):
    assert client.get("/keyframe/999").status_code == 404


def test_engine_not_ready_returns_503():
    # Client used WITHOUT entering the lifespan context: the lazy engine load
    # never runs, so every search endpoint must answer 503, not crash.
    app = create_app(engine=None, settings=Settings())
    c = TestClient(app, raise_server_exceptions=False)
    assert c.post("/search/text", json={"query": "q"}).status_code == 503
    assert c.get("/health").json()["status"] == "loading"