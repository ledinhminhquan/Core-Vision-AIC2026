"""Round-9 review regressions — the operator-chaos and clean-install lenses.

  R9-1  invisible Unicode Cf chars on query lines can no longer shift/drop
        TRAKE events (zero-width space, stray mid-file BOM, RTL marks)
  R9-2  packager writes manifest/ledger/history BEFORE the zip lands — a crash
        can never leave a fresh zip whose audit trail describes the old one
  R9-3  a corrupt signal dump names ITSELF in the error
  R9-4  faiss-dependent fixtures skip cleanly without the extra; dev extra is
        self-sufficient
  R9-5  extract_missing counts the no-overwrite mismatch repair too (map csv
        mtime detection), never the untouched skip path
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from cvp.config import Settings

REPO = Path(__file__).resolve().parents[1]


# ── R9-1 · invisible chars in query files ────────────────────────────────────
def test_strip_invisible_removes_cf_chars():
    from cvp.pipeline.run_queries import strip_invisible

    assert strip_invisible("​E1: a") == "E1: a"
    assert strip_invisible("E﻿2: b") == "E2: b"
    assert strip_invisible("bình thường") == "bình thường"


def test_trake_events_survive_zero_width_space():
    from cvp.pipeline.run_queries import parse_trake_events

    lines = ["Đoạn video múa lân, tìm các sự kiện sau:",
             "E1: lân quay vòng trên cột",
             "​E2: bốn chân chạm đất",       # zero-width space prefix
             "E﻿3: lân đứng dậy"]            # stray BOM inside the marker
    events = parse_trake_events(lines)
    assert events == ["lân quay vòng trên cột", "bốn chân chạm đất", "lân đứng dậy"]


def test_run_query_file_strips_invisible_lines(tmp_path):
    from cvp.pipeline.run_queries import run_query_file

    class _Eng:
        settings = Settings()

        def search_trake(self, events, **kw):
            self.seen = events
            return []

    qf = tmp_path / "query-1-trake.txt"
    qf.write_text("Bối cảnh\nE1: chạy đà\n​E2: giậm nhảy\n", encoding="utf-8")
    eng = _Eng()
    run_query_file(eng, qf, tmp_path, None)
    assert eng.seen == ["chạy đà", "giậm nhảy"]


# ── R9-2 · packager audit-before-install ordering ────────────────────────────
def test_packager_crash_never_leaves_zip_newer_than_manifest(tmp_path, monkeypatch):
    import zipfile

    from cvp.submission import packager

    sub = tmp_path / "subs"
    sub.mkdir()
    csv_p = sub / "query-p1-1-kis.csv"
    csv_p.write_text("L21_V001,505\n", encoding="utf-8")
    zip_p = sub / "submission.zip"
    assert not packager.has_errors(
        packager.package_codabench(sub, zip_p, package_name="submission", files=[csv_p]))
    v1_bytes = zip_p.read_bytes()

    csv_p.write_text("L21_V001,900\n", encoding="utf-8")
    monkeypatch.setattr(packager, "atomic_write_json",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("kill")))
    with pytest.raises(RuntimeError):
        packager.package_codabench(sub, zip_p, package_name="submission", files=[csv_p])
    # The OLD zip must still be in place — audit records can be newer than the
    # visible zip after a crash, but never the other way around.
    assert zip_p.read_bytes() == v1_bytes
    monkeypatch.undo()
    assert not packager.has_errors(
        packager.package_codabench(sub, zip_p, package_name="submission", files=[csv_p]))
    manifest = json.loads((sub / "MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["zip_sha256"] == packager._sha256(zip_p)
    with zipfile.ZipFile(zip_p) as zf:
        # normalize: Path.write_text on Windows translates \n to \r\n
        assert zf.read("submission/query-p1-1-kis.csv").replace(b"\r\n", b"\n") == b"L21_V001,900\n"


# ── R9-3 · corrupt signal dump names itself ──────────────────────────────────
def test_tune_weights_names_corrupt_dump(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "tune_weights_for_round9_tests", REPO / "scripts" / "21_tune_weights.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    (tmp_path / "ok.json").write_text('{"q1": {"visual": {"0": 1.0}}}', encoding="utf-8")
    (tmp_path / "torn.json").write_text('{"q2": {"visual"', encoding="utf-8")
    with pytest.raises(ValueError, match="torn.json"):
        mod.load_signals(tmp_path)


# ── R9-4 · faiss skip + dev extra self-sufficiency ───────────────────────────
def test_conftest_guards_faiss_and_dev_extra_carries_it():
    conftest = (REPO / "tests" / "conftest.py").read_text(encoding="utf-8")
    assert 'pytest.importorskip("faiss"' in conftest
    pyproject = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    dev_line = next(ln for ln in pyproject.splitlines() if ln.startswith("dev = "))
    assert "faiss-cpu" in dev_line


# ── R9-5 · mismatch repair counted; skip path still not ──────────────────────
def test_extract_missing_counts_no_overwrite_mismatch_repair(tmp_path, monkeypatch):
    from cvp.data import extraction

    s = Settings()
    s.paths.data_root = tmp_path
    (tmp_path / s.paths.videos_dir).mkdir(parents=True)
    (tmp_path / s.paths.videos_dir / "K01_V001.mp4").write_bytes(b"fake")
    vdir = tmp_path / s.paths.keyframes_dir / "K01_V001"
    vdir.mkdir(parents=True)
    (vdir / "001.jpg").write_bytes(b"jpg")
    mp = tmp_path / s.paths.map_keyframes_dir
    mp.mkdir(parents=True)
    (mp / "K01_V001.csv").write_text("n,pts_time,fps,frame_idx\n1,0.0,25.0,0\n"
                                     "2,1.0,25.0,25\n", encoding="utf-8")   # 1 jpg vs 2 rows

    def _fake_extract(vp, kf_dir, mp_dir, **kw):
        (mp_dir / "K01_V001.csv").write_text(
            "n,pts_time,fps,frame_idx\n1,0.0,25.0,0\n", encoding="utf-8")
        return 1

    monkeypatch.setattr(extraction, "extract_video", _fake_extract)
    # NO overwrite: the mismatched self-made pair gets repaired — must count
    # so callers rebuild the catalog off the changed map.
    assert extraction.extract_missing(s, overwrite=False) == 1
