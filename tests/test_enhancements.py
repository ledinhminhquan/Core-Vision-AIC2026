"""Enhancement round: ensemble image search, episodic log, `cvp eval`.

CPU-only, stub-based — no models, no network, no index files.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np

from cvp.config import Settings
from cvp.models.agent import EpisodicLog
from cvp.search.engine import SearchEngine


# ── ensemble image search (KIS-V path) ───────────────────────────────────────
def _ref(gid: int):
    return SimpleNamespace(global_id=gid, video_id="L01_V001", n=gid + 1,
                           frame_idx=100 * (gid + 1), pts_time=float(gid),
                           path=f"/fake/{gid:03d}.jpg", fps=25.0, has_map=True)


class _Model:
    def __init__(self, key):
        self.key = key

    def encode_image(self, images):
        return np.ones((1, 4), dtype=np.float32)


class _Store:
    def __init__(self, hits):  # hits: {gid: score}
        self._hits = hits

    def search(self, vec, k):
        gids = list(self._hits)
        scores = [self._hits[g] for g in gids]
        return np.asarray([scores], dtype=np.float32), np.asarray([gids])


class _BoomModel(_Model):
    def encode_image(self, images):
        raise RuntimeError("no weights on this machine")


def _image_engine(members, weights):
    eng = object.__new__(SearchEngine)
    eng.settings = Settings()
    eng.members = members
    eng.member_weights = weights
    eng.catalog = SimpleNamespace(ref=lambda gid: _ref(int(gid)))
    return eng


def test_search_image_fuses_all_members():
    # gid 3 ranks near the top of BOTH lanes; each lane also has its own
    # exclusive favourite + a weak tail row (the tail anchors min-max) →
    # after normalise + weighted sum the cross-lane agreement must win.
    a = _Store({1: 0.9, 3: 0.85, 7: 0.1})
    b = _Store({2: 0.8, 3: 0.75, 8: 0.1})
    eng = _image_engine([(_Model("a"), a), (_Model("b"), b)], [0.5, 0.5])
    results = SearchEngine.search_image(eng, image=object(), display_k=10)
    gids = [r.ref.global_id for r in results]
    assert gids[0] == 3                      # cross-lane agreement wins
    assert set(gids) == {1, 2, 3, 7, 8}
    assert all(r.signals["visual"] >= 0 for r in results)


def test_search_image_skips_failing_member():
    good = _Store({5: 0.9})
    eng = _image_engine([(_BoomModel("dead"), good), (_Model("ok"), good)],
                        [0.6, 0.4])
    results = SearchEngine.search_image(eng, image=object(), display_k=5)
    assert [r.ref.global_id for r in results] == [5]


def test_search_image_all_members_fail_returns_empty():
    eng = _image_engine([(_BoomModel("dead"), _Store({1: 0.5}))], [1.0])
    assert SearchEngine.search_image(eng, image=object()) == []


# ── episodic log (organiser buổi-3 recipe) ───────────────────────────────────
def test_episodic_log_append_read_and_summary(tmp_path):
    s = Settings()
    s.paths.artifacts_root = tmp_path
    elog = EpisodicLog(s, session="t1")
    elog.append("kisc_turn", hint="áo đỏ", merged_query="red shirt market")
    elog.append("basket_add", video="L01_V001", frame=505)
    events = elog.events()
    assert [e["kind"] for e in events] == ["kisc_turn", "basket_add"]
    assert events[0]["hint"] == "áo đỏ"
    summary = elog.recent_summary()
    assert "kisc_turn" in summary and "L01_V001" in summary


def test_episodic_log_skips_corrupt_lines(tmp_path):
    s = Settings()
    s.paths.artifacts_root = tmp_path
    elog = EpisodicLog(s, session="t2")
    elog.append("a", x=1)
    with open(elog.path, "a", encoding="utf-8") as f:
        f.write("{not json}\n")
    elog.append("b", y=2)
    assert [e["kind"] for e in elog.events()] == ["a", "b"]


def test_episodic_log_append_never_raises(tmp_path):
    s = Settings()
    s.paths.artifacts_root = tmp_path / "nope"
    elog = EpisodicLog(s, session="t3")
    elog.path = tmp_path  # a DIRECTORY — open() will fail
    elog.append("x")      # must swallow, not raise


# ── cvp eval subcommand ──────────────────────────────────────────────────────
def test_cli_eval_scores_folder(tmp_path, capsys):
    from cvp.cli import main

    sub = tmp_path / "subs"
    sub.mkdir()
    (sub / "query-1-kis.csv").write_text("L01_V001, 505\n", encoding="utf-8")
    gt = tmp_path / "gt.json"
    gt.write_text(json.dumps({
        "query-1-kis": {"task": "kis", "video_id": "L01_V001", "range": [500, 510]},
    }), encoding="utf-8")
    rc = main(["eval", "--submission-dir", str(sub), "--gt", str(gt)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "mean final = 1.0000" in out


def test_cli_version_runs(capsys):
    from cvp.cli import main

    assert main(["version"]) == 0
    assert "Perfect V1" in capsys.readouterr().out


def test_cli_eval_reports_missing_submissions(tmp_path, capsys):
    from cvp.cli import main

    sub = tmp_path / "subs"
    sub.mkdir()
    gt = tmp_path / "gt.json"
    gt.write_text(json.dumps({
        "query-9-kis": {"task": "kis", "video_id": "L01_V001", "range": [1, 2]},
    }), encoding="utf-8")
    rc = main(["eval", "--submission-dir", str(sub), "--gt", str(gt)])
    out = capsys.readouterr().out
    assert rc == 0 and "MISSING submission" in out
    assert "mean final = 0.0000" in out


def test_warm_cache_script_help_is_light():
    # scripts/51 must follow the lazy-import convention (usable torch-less).
    import ast
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / "scripts" / "51_warm_cache.py")
    tree = ast.parse(src.read_text(encoding="utf-8"))
    top = {n.module if isinstance(n, ast.ImportFrom) else n.names[0].name
           for n in ast.walk(tree)
           if isinstance(n, (ast.Import, ast.ImportFrom)) and n.col_offset == 0}
    heavy = {m for m in top if m and (m.startswith("cvp.models") or m in ("torch", "numpy"))}
    assert not heavy, f"module-level heavy imports break torch-less --help: {heavy}"
