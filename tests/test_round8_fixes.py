"""Round-8 review regressions — auditing the round-7 doc sweep and the
overwrite summary reporting.

  R8-1  the dump-signals → tune-weights command pairs agree on the signals dir
        (the round-7 sweep rewrote only the query-dir half in 4 docs)
  R8-2  extract_missing counts forced RE-extractions (catalog rebuild keys off
        the return) and reports guard refusals distinctly
"""

from __future__ import annotations

import re
from pathlib import Path

from cvp.config import Settings

REPO = Path(__file__).resolve().parents[1]


# ── R8-1 · doc command pairs are internally consistent ───────────────────────
def test_docs_signal_dump_pairs_are_consistent():
    # These docs use scripts/23's DEFAULT out-dir (derived from the query-dir
    # name) — their scripts/21 line must read the same derived dir.
    for rel in ("HUONG_DAN.md", "docs/DRIVE_SETUP.md",
                "docs/PROJECT_CONTEXT.md", "docs/PROJECT_PLAN.md"):
        t = (REPO / rel).read_text(encoding="utf-8")
        assert not re.search(r"signal_dumps/dev(?!-2025)", t), rel
    # EVALUATION.md + the two script docstrings pass --out-dir explicitly and
    # stay a consistent pair on signal_dumps/dev — both halves must survive.
    ev = (REPO / "docs" / "EVALUATION.md").read_text(encoding="utf-8")
    assert "--out-dir ./artifacts/signal_dumps/dev" in ev
    assert "--signals-dir ./artifacts/signal_dumps/dev" in ev


# ── R8-2 · overwrite summary: re-extractions counted, refusals distinct ──────
def _seed_video(tmp_path, s: Settings, vid: str = "L21_V001"):
    (tmp_path / s.paths.videos_dir).mkdir(parents=True, exist_ok=True)
    (tmp_path / s.paths.videos_dir / f"{vid}.mp4").write_bytes(b"fake")
    vdir = tmp_path / s.paths.keyframes_dir / vid
    vdir.mkdir(parents=True, exist_ok=True)
    (vdir / "001.jpg").write_bytes(b"jpg")
    mp = tmp_path / s.paths.map_keyframes_dir
    mp.mkdir(parents=True, exist_ok=True)
    (mp / f"{vid}.csv").write_text("n,pts_time,fps,frame_idx\n1,0.0,25.0,0\n", encoding="utf-8")


def test_extract_missing_counts_forced_reextraction(tmp_path, monkeypatch):
    from cvp.data import extraction

    s = Settings()
    s.paths.data_root = tmp_path
    _seed_video(tmp_path, s)

    def _fake_extract(vp, kf_dir, mp_dir, **kw):
        # A real re-extraction REWRITES the map csv — that's how the counter
        # tells it apart from the untouched skip path (round-9 mtime check).
        (mp_dir / "L21_V001.csv").write_text(
            "n,pts_time,fps,frame_idx\n1,0.0,25.0,0\n2,1.0,25.0,25\n", encoding="utf-8")
        return 210

    monkeypatch.setattr(extraction, "extract_video", _fake_extract)
    assert extraction.extract_missing(s, overwrite=True, only=["L21_V001"]) == 1


def test_extract_missing_reports_refusal_as_zero(tmp_path, monkeypatch):
    from cvp.data import extraction

    s = Settings()
    s.paths.data_root = tmp_path
    _seed_video(tmp_path, s)
    monkeypatch.setattr(extraction, "extract_video",
                        lambda *a, **k: 0)            # guard refusal
    assert extraction.extract_missing(s, overwrite=True, only=["L21_V001"]) == 0


def test_extract_missing_skip_path_still_uncounted(tmp_path, monkeypatch):
    from cvp.data import extraction

    s = Settings()
    s.paths.data_root = tmp_path
    _seed_video(tmp_path, s)
    # NOT overwrite: complete pair short-circuits with its existing count —
    # must not be counted as fresh work.
    monkeypatch.setattr(extraction, "extract_video",
                        lambda *a, **k: 1)
    assert extraction.extract_missing(s, overwrite=False) == 0
