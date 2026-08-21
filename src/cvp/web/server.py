"""Team web app: cvp.service endpoints + shared-account auth + pack export.

    create_team_app(engine, settings, username, password) -> FastAPI app

Everything rides on the EXISTING, CPU-tested service app (search/text, qa,
trake, nearest, keyframe …). This wrapper adds exactly what a competition
operator needs on top:

* ``POST /login``      one shared team account → HttpOnly auth cookie; every
                       other endpoint (images included — <img> sends cookies)
                       returns 401 without it. The tunnel URL alone is useless.
* ``GET  /``           the SPA (static/index.html) — the only anonymous page.
* ``POST /export``     write a curated ranking STRAIGHT into the organiser
                       pack as ``submissions/<pack>/<stem>.csv`` — pins first,
                       machine tail after, wrong-task stems refused (mirror of
                       the Streamlit round-31 guard).
* ``GET  /context/{gid}`` temporal neighbours of a keyframe (verify-before-pin).
* ``POST /trake2``     TRAKE with per-event global_ids so the SPA can render
                       the frame strips (the service's TrakeItem carries none).

NOTE: no ``from __future__ import annotations`` — FastAPI needs real
annotations to resolve body models (same rule as cvp.service.app).
"""

import hashlib
import hmac
import logging
import re
from pathlib import Path

from pydantic import BaseModel, Field

from cvp.config import Settings

log = logging.getLogger(__name__)

_STEM_RE = re.compile(r"^[A-Za-z0-9._-]+$")
# Anonymous surface: the SPA shell, the login call, and docs-free plumbing.
_OPEN_PATHS = {"/", "/login", "/favicon.ico", "/health"}


class LoginBody(BaseModel):
    username: str
    password: str


class SequenceBody(BaseModel):
    video_id: str
    frames: list[int] = Field(..., min_length=1)


class ExportBody(BaseModel):
    task: str                                  # kis | qa | trake | avs
    pack: str = ""                             # subfolder of submissions/
    stem: str                                  # organiser stem, e.g. query-p1-9-kis
    pins: list[int] = Field(default_factory=list)      # global_ids, rank order
    results: list[int] = Field(default_factory=list)   # machine tail, in order
    answer: str = ""                                   # QA only
    sequences: list[SequenceBody] = Field(default_factory=list)  # TRAKE only


class TrakeBody(BaseModel):
    events: list[str] = Field(..., min_length=2)
    max_results: int = Field(default=50, ge=1, le=100)


def _auth_token(username: str, password: str) -> str:
    return hashlib.sha256(f"{username}:{password}:cvp-web-r36".encode()).hexdigest()


def create_team_app(engine, settings: Settings, username: str, password: str):
    from fastapi import HTTPException, Request
    from fastapi.responses import FileResponse, JSONResponse

    from cvp.service.app import create_app

    if not password:
        raise ValueError("TEAM password must be non-empty — an empty password "
                         "would let anyone with the tunnel URL in.")

    app = create_app(engine=engine, settings=settings)
    token = _auth_token(username, password)
    static_dir = Path(__file__).parent / "static"

    @app.middleware("http")
    async def _require_team_cookie(request: Request, call_next):
        if request.url.path not in _OPEN_PATHS:
            got = request.cookies.get("cvp_auth", "")
            if not hmac.compare_digest(got, token):
                return JSONResponse({"detail": "Chưa đăng nhập tài khoản đội."},
                                    status_code=401)
        return await call_next(request)

    @app.post("/login")
    def login(body: LoginBody):
        ok = hmac.compare_digest(body.username.strip(), username) and \
            hmac.compare_digest(body.password, password)
        if not ok:
            raise HTTPException(401, "Sai tài khoản hoặc mật khẩu đội.")
        resp = JSONResponse({"ok": True})
        resp.set_cookie("cvp_auth", token, httponly=True, samesite="lax",
                        max_age=24 * 3600)
        return resp

    @app.get("/")
    def index():
        return FileResponse(str(static_dir / "index.html"), media_type="text/html")

    import os as _os

    _raw_thumbs = _os.environ.get("CVP_WEB__THUMBS_DIR", "")
    # Path("") is the CWD — an empty env var must mean "no store", not "serve
    # the working directory" (caught by the round-42 test suite).
    thumbs_dir = Path(_raw_thumbs) if _raw_thumbs else None
    has_thumbs = thumbs_dir is not None and thumbs_dir.is_dir()
    if has_thumbs:
        log.info("Thumbnail store: %s", thumbs_dir)

    @app.get("/whoami")
    def whoami():
        # Auth-gated no-op: the SPA probes it on load to skip the login form
        # while the day-old cookie is still valid. Round-42: also tells the
        # SPA whether ~8KB webp thumbnails are available (10x faster grids).
        return {"ok": True, "username": username, "thumbs": has_thumbs}

    @app.get("/thumb/{video_id}/{n}.webp")
    def thumb(video_id: str, n: int):
        # Auth-gated like /keyframe — competition imagery never leaves the
        # login wall. Immutable cache: a keyframe's thumb never changes.
        if not has_thumbs or not _STEM_RE.match(video_id):
            raise HTTPException(404, "no thumbnail store")
        p = thumbs_dir / video_id / f"{int(n)}.webp"
        if not p.is_file():
            raise HTTPException(404, "thumb missing")
        return FileResponse(str(p), media_type="image/webp",
                            headers={"Cache-Control": "public, max-age=604800, immutable"})

    @app.get("/context/{global_id}")
    def context(global_id: int, window: int = 6):
        e = app_engine()
        try:
            ref = e.catalog.ref(int(global_id))
        except Exception:  # noqa: BLE001 — any lookup failure is a 404
            raise HTTPException(404, f"unknown global_id {global_id}") from None
        rows = e.catalog.video_rows(ref.video_id)
        near = rows[(rows["n"] >= ref.n - window) & (rows["n"] <= ref.n + window)]
        return {"video_id": ref.video_id, "center": int(global_id), "frames": [
            {"gid": int(r.global_id), "frame": int(r.frame_idx), "n": int(r.n),
             "t": round(float(r.pts_time), 2)}
            for r in near.sort_values("n").itertuples()
        ]}

    @app.post("/trake2")
    def trake2(body: TrakeBody):
        e = app_engine()
        cands = e.search_trake(body.events, max_results=body.max_results)
        out = []
        gid_maps: dict[str, dict] = {}
        for c in cands:
            m = gid_maps.get(c.video_id)
            if m is None:
                rows = e.catalog.video_rows(c.video_id)
                m = dict(zip((int(n) for n in rows["n"]),
                             (int(g) for g in rows["global_id"])))
                gid_maps[c.video_id] = m
            out.append({
                "video_id": c.video_id, "score": round(float(c.score), 4),
                "frames": [{"gid": m.get(int(n), -1), "n": int(n),
                            "frame": int(f), "t": round(float(t), 2)}
                           for n, f, t in zip(c.ns, c.frame_idxs, c.pts_times)],
            })
        return {"count": len(out), "results": out}

    @app.post("/export")
    def export(body: ExportBody):
        from cvp.submission.packager import infer_task
        from cvp.submission.writer import write_kis, write_qa, write_trake

        task = body.task.strip().lower()
        stem = body.stem.strip().removesuffix(".csv")
        pack = body.pack.strip().strip("/")
        if not stem or not _STEM_RE.match(stem) or (pack and not _STEM_RE.match(pack)):
            raise HTTPException(422, "Stem/Pack chỉ gồm chữ, số, '.', '_', '-'.")
        file_task = infer_task(f"{stem}.csv")
        if file_task != task:
            raise HTTPException(
                422, f"Stem '{stem}' là bài {file_task.upper()} nhưng bạn đang "
                     f"export tab {task.upper()} — chặn ghi nhầm task.")

        e = app_engine()
        out_dir = settings.paths.art("submissions")
        out_dir = out_dir / pack if pack else out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{stem}.csv"

        if task == "trake":
            if not body.sequences:
                raise HTTPException(422, "TRAKE cần ít nhất một chuỗi (sequences).")
            write_trake(path, [(s.video_id, s.frames) for s in body.sequences])
            n_rows = len(body.sequences)
        else:
            seen: set[int] = set()
            ordered = [g for g in [*body.pins, *body.results]
                       if not (g in seen or seen.add(g))]
            if not ordered:
                raise HTTPException(422, "Không có dòng nào để ghi — search "
                                         "rồi ghim, sau đó export lại.")
            refs = e.catalog.refs(ordered)
            if task == "qa":
                if not body.answer.strip():
                    # An all-blank QA export scores 0 and burns a submission.
                    raise HTTPException(422, "QA thiếu ĐÁP ÁN — nhập đáp án "
                                             "(≤100 ký tự) rồi export lại.")
                write_qa(path, [(r.video_id, r.frame_idx, body.answer)
                                for r in refs])
            else:  # kis / avs share the 2-column row shape
                write_kis(path, [(r.video_id, r.frame_idx) for r in refs])
            n_rows = min(len(ordered), 100)
        log.info("Web export: %s (%d pins) → %s", task, len(body.pins), path)
        return {"path": str(path), "rows": n_rows, "pins": len(body.pins)}

    def app_engine():
        # The wrapped service keeps its engine in a closure; ours is the same
        # object we were constructed with (pre-built in the notebook kernel).
        if engine is None:
            raise HTTPException(503, "Engine not ready")
        return engine

    return app
