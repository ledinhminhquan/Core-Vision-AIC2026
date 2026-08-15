"""Round-7 review regressions — auditing the round-6 diff plus the auxiliary
surfaces (report/ kit, dev pack paths).

  R7-1  organiser keyframes/map csvs survive even a GLOBAL --overwrite unless
        --force-organiser; overwrite is scopable with --video
  R7-2  the .selfmade marker is content-bound (sha256) — a stale marker no
        longer vouches for the organiser csv that replaced ours
  R7-3  HTTP client timeout == wall cap (24s HTTP no longer undercuts the
        30s image-call floor)
  R7-4  _call_with_timeout uses a DAEMON thread — a hung Gemini read cannot
        keep a finished process alive at interpreter exit
  R7-5  run_auto fails loud on a typo'd query dir (before the engine build)
  R7-6  QA 'Suggest answers' is provenance-gated like the grid
  R7-7  report/ + docs point at the real dev pack (queries/dev-2025-finals)
        and carry the true test count
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import TimeoutError as FuturesTimeoutError
from pathlib import Path

import pytest

from cvp.config import Settings

REPO = Path(__file__).resolve().parents[1]

ORGANISER_CSV = "n,pts_time,fps,frame_idx\n1,0.0,30.0,0\n2,3.0,30.0,90\n"


# ── R7-1 · global overwrite must not destroy organiser data ──────────────────
def test_overwrite_alone_refuses_organiser_map_csv(tmp_path):
    from cvp.data.extraction import extract_video

    kf, mp = tmp_path / "keyframes", tmp_path / "map-keyframes"
    mp.mkdir(parents=True)
    (mp / "L21_V001.csv").write_text(ORGANISER_CSV, encoding="utf-8")
    n = extract_video(tmp_path / "L21_V001.mp4", kf, mp, overwrite=True)
    assert n == 0
    assert (mp / "L21_V001.csv").read_text(encoding="utf-8") == ORGANISER_CSV


def test_overwrite_alone_refuses_organiser_jpgs(tmp_path):
    from cvp.data.extraction import extract_video

    kf, mp = tmp_path / "keyframes", tmp_path / "map-keyframes"
    vdir = kf / "L21_V001"
    vdir.mkdir(parents=True)
    (vdir / "001.jpg").write_bytes(b"organiser-jpg")
    n = extract_video(tmp_path / "L21_V001.mp4", kf, mp, overwrite=True)
    assert n == 0
    assert (vdir / "001.jpg").read_bytes() == b"organiser-jpg"


def test_force_organiser_passes_the_guards(tmp_path):
    from cvp.data.extraction import extract_video

    kf, mp = tmp_path / "keyframes", tmp_path / "map-keyframes"
    mp.mkdir(parents=True)
    (mp / "L21_V001.csv").write_text(ORGANISER_CSV, encoding="utf-8")
    # Explicit double opt-in reaches actual extraction (which then fails on the
    # nonexistent video / absent cv2 — proof the refusal did NOT fire).
    with pytest.raises((RuntimeError, ModuleNotFoundError)):
        extract_video(tmp_path / "L21_V001.mp4", kf, mp,
                      overwrite=True, force_organiser=True)


def test_extract_missing_only_filter(tmp_path):
    from cvp.data import extraction

    s = Settings()
    s.paths.data_root = tmp_path
    vdir = tmp_path / s.paths.videos_dir
    vdir.mkdir(parents=True)
    (vdir / "L21_V001.mp4").write_bytes(b"fake")
    # only= filters to ids that do not exist → nothing processed, loud log.
    assert extraction.extract_missing(s, only=["NOPE_V999"]) == 0


def test_scripts01_has_video_and_force_flags():
    src = (REPO / "scripts" / "01_extract_keyframes.py").read_text(encoding="utf-8")
    assert "--video" in src and "--force-organiser" in src
    assert "only=args.video" in src and "force_organiser=args.force_organiser" in src


# ── R7-2 · content-bound marker ──────────────────────────────────────────────
def test_stale_selfmade_marker_no_longer_vouches(tmp_path):
    from cvp.data.extraction import extract_video

    kf, mp = tmp_path / "keyframes", tmp_path / "map-keyframes"
    mp.mkdir(parents=True)
    (mp / "K01_V001.csv").write_text(ORGANISER_CSV, encoding="utf-8")   # official
    marker = mp / "K01_V001.csv.selfmade"
    marker.write_text("deadbeef" * 8, encoding="utf-8")                 # stale hash
    n = extract_video(tmp_path / "K01_V001.mp4", kf, mp, overwrite=False)
    assert n == 0                                                        # refused
    assert (mp / "K01_V001.csv").read_text(encoding="utf-8") == ORGANISER_CSV
    assert not marker.exists()                                           # hygiene


# ── R7-3 · HTTP timeout aligned with the wall cap ────────────────────────────
def test_http_client_timeout_uses_wall_cap():
    src = (REPO / "src" / "cvp" / "search" / "vqa.py").read_text(encoding="utf-8")
    assert "int(gemini_wall_timeout(settings) * 1000)" in src
    assert "timeout_s * 1000 * 3" not in src


# ── R7-4 · daemon-thread timeout ─────────────────────────────────────────────
def test_call_with_timeout_daemon_and_semantics():
    from cvp.models.query_processor import _call_with_timeout

    seen: list[threading.Thread] = []

    def _hang():
        seen.append(threading.current_thread())
        time.sleep(5)

    with pytest.raises(FuturesTimeoutError):
        _call_with_timeout(_hang, 0.2)
    assert seen and seen[0].daemon        # abandoned thread cannot block exit
    assert _call_with_timeout(lambda: 42, 2.0) == 42

    def _boom():
        raise ValueError("inner")

    with pytest.raises(ValueError, match="inner"):
        _call_with_timeout(_boom, 2.0)


# ── R7-5 · automatic track fails loud on a bad query dir ─────────────────────
def test_run_auto_raises_on_missing_query_dir(tmp_path):
    from cvp.pipeline.auto_agent import run_auto

    with pytest.raises(FileNotFoundError, match="query dir"):
        run_auto(tmp_path / "nope", tmp_path / "out", Settings(),
                 engine_factory=lambda s: pytest.fail("engine must not be built"))


# ── R7-6 · QA suggest provenance gate (source pin) ───────────────────────────
def test_app_qa_suggest_is_provenance_gated():
    src = (REPO / "app" / "streamlit_app.py").read_text(encoding="utf-8")
    assert '_qa_results = _grid_results_for("qa")' in src
    assert "do_vqa and _qa_results" in src


# ── R7-7 · report/docs truth ─────────────────────────────────────────────────
def test_report_counts_and_paths_are_current():
    tex = (REPO / "report" / "main.tex").read_text(encoding="utf-8")
    assert "400 test" not in tex and "514 test" in tex
    assert "thiếu sạch" not in tex                      # stale no-map-keyframes claim
    assert "queries/dev-2025-finals" in tex


def test_no_stale_queries_dev_paths_anywhere():
    import re

    offenders = []
    for base in ("docs", "scripts", "report"):
        for p in (REPO / base).rglob("*"):
            if p.suffix in (".py", ".md", ".tex") and "research" not in p.parts:
                if re.search(r"queries/dev(?!-2025)", p.read_text(encoding="utf-8", errors="ignore")):
                    offenders.append(str(p))
    assert not offenders, offenders
