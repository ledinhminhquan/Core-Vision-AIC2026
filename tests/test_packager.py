"""Packager validation/zip, QA answer grouping, DRES client, auto-agent routing.

Pure CPU, no network: urllib is monkeypatched, the engine is a stub injected
via ``engine_factory``.
"""

from __future__ import annotations

import hashlib
import io
import json
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from cvf.config import Settings
from cvf.submission.dres_client import DresClient, SubmitResult
from cvf.submission.packager import (
    ValidationIssue,
    has_errors,
    infer_task,
    package_codabench,
    validate_submission_dir,
)

# ── helpers ──────────────────────────────────────────────────────────────────


def _errors(issues: list[ValidationIssue]) -> list[ValidationIssue]:
    return [i for i in issues if i.severity == "error"]


def _write(dirpath: Path, name: str, text: str) -> Path:
    dirpath.mkdir(parents=True, exist_ok=True)
    p = dirpath / name
    p.write_text(text, encoding="utf-8", newline="\n")
    return p


def _valid_pack(dirpath: Path) -> Path:
    _write(dirpath, "query-p1-1-kis.csv", "L21_V001,100\nL21_V002,250\n")
    _write(dirpath, "query-p1-2-qa.csv", 'L21_V001,100,hai\nL21_V002,250,"xã Cam, Hải"\n')
    _write(dirpath, "query-p1-3-trake.csv", "L21_V001,10,20,30\n")
    _write(dirpath, "query-p1-4-avs.csv", "K01_V001,5\n")
    return dirpath


class _Ref:
    def __init__(self, pts_time: float, path: str = "frame.jpg"):
        self.pts_time = pts_time
        self.path = path


class _Res:
    """SearchResult stand-in (duck-typed: video_id/frame_idx/global_id/ref)."""

    def __init__(self, video_id: str, frame_idx: int, global_id: int, pts_time: float | None = None):
        self.video_id = video_id
        self.frame_idx = frame_idx
        self.global_id = global_id
        self.ref = _Ref(pts_time if pts_time is not None else frame_idx / 25.0)


class _StubVqa:
    """Maps global_id → answer; records which frames were asked."""

    def __init__(self, mapping: dict[int, str]):
        self.mapping = mapping
        self.calls: list[list[int]] = []

    def suggest(self, question: str, frames: list[tuple[int, str]]):
        self.calls.append([gid for gid, _ in frames])
        gid = frames[0][0]
        ans = self.mapping.get(gid, "")
        if not ans:
            return []
        return [SimpleNamespace(global_id=gid, answer=ans, provider="stub")]


# ── task inference ───────────────────────────────────────────────────────────


def test_infer_task_priority():
    assert infer_task("query-p1-3-trake.csv") == "trake"
    assert infer_task("query-p1-4-avs.csv") == "avs"
    assert infer_task("something-qa-kis.csv") == "qa"  # qa outranks kis
    assert infer_task("query-p1-1-kis.csv") == "kis"
    assert infer_task("mystery.csv") == "kis"  # default


# ── validation ───────────────────────────────────────────────────────────────


def test_validate_clean_pack_has_no_errors(tmp_path):
    issues = validate_submission_dir(_valid_pack(tmp_path / "sub"))
    assert not _errors(issues)


def test_validate_rejects_header_row(tmp_path):
    d = tmp_path / "sub"
    _write(d, "h-kis.csv", "video_id,frame_idx\nL21_V001,100\n")
    errs = _errors(validate_submission_dir(d))
    assert any("header" in e.message for e in errs)


def test_validate_rejects_row_overflow(tmp_path):
    d = tmp_path / "sub"
    _write(d, "big-kis.csv", "\n".join(f"L21_V001,{i}" for i in range(101)) + "\n")
    errs = _errors(validate_submission_dir(d))
    assert any("101 rows" in e.message for e in errs)


def test_validate_rejects_bad_video_id(tmp_path):
    d = tmp_path / "sub"
    _write(d, "bad-kis.csv", "L21V001,100\n")
    errs = _errors(validate_submission_dir(d))
    assert any("video id" in e.message for e in errs)


def test_validate_rejects_bad_frame_idx(tmp_path):
    d = tmp_path / "sub"
    _write(d, "neg-kis.csv", "L21_V001,-5\nL21_V002,x9\nL21_V003,1.5\n")
    errs = _errors(validate_submission_dir(d))
    assert len(errs) == 3
    assert all("integer" in e.message for e in errs)


def test_validate_rejects_wrong_column_count(tmp_path):
    d = tmp_path / "sub"
    _write(d, "cols-kis.csv", "L21_V001,100,extra\n")
    errs = _errors(validate_submission_dir(d))
    assert any("2 columns" in e.message for e in errs)


def test_validate_trake_strictly_increasing(tmp_path):
    d = tmp_path / "sub"
    _write(d, "t-trake.csv", "L21_V001,10,10,30\nL21_V002,5,6,7\n")
    errs = _errors(validate_submission_dir(d))
    assert len(errs) == 1
    assert errs[0].line == 1
    assert "strictly increasing" in errs[0].message


def test_validate_qa_answer_rules(tmp_path):
    d = tmp_path / "sub"
    _write(d, "e-qa.csv", "L21_V001,100,\n")
    _write(d, "long-qa.csv", f"L21_V001,100,{'a' * 101}\n")
    _write(d, "quoted-qa.csv", 'L21_V001,100,"xã Cam, Hải Đông"\n')
    issues = validate_submission_dir(d)
    errs = _errors(issues)
    files = {e.file for e in errs}
    # empty answer = scoring loss, NOT a format violation → warning, not error
    assert "e-qa.csv" not in files
    empty = [i for i in issues if i.file == "e-qa.csv" and "empty" in i.message]
    assert empty and all(i.severity == "warning" for i in empty)
    assert "long-qa.csv" in files and any("100 chars" in e.message for e in errs)
    assert "quoted-qa.csv" not in files  # quoted comma answer is legal CSV


def test_package_proceeds_with_empty_qa_answer(tmp_path):
    # A valid submission may carry an empty QA answer — packaging must not refuse.
    d = tmp_path / "sub"
    _write(d, "e-qa.csv", "L21_V001,100,\n")
    out_zip = tmp_path / "out" / "submission.zip"
    issues = package_codabench(d, out_zip)
    assert not has_errors(issues)
    assert any(i.severity == "warning" and "empty" in i.message for i in issues)
    assert out_zip.exists()


def test_validate_rejects_non_utf8_and_empty(tmp_path):
    d = tmp_path / "sub"
    d.mkdir()
    (d / "bin-kis.csv").write_bytes(b"L21_V001,100\xff\xfe\n")
    (d / "empty-kis.csv").write_bytes(b"")
    errs = _errors(validate_submission_dir(d))
    assert any("UTF-8" in e.message for e in errs)
    assert any("empty" in e.message for e in errs)


def test_validate_rejects_blank_interior_line(tmp_path):
    d = tmp_path / "sub"
    _write(d, "gap-kis.csv", "L21_V001,100\n\nL21_V002,200\n")
    errs = _errors(validate_submission_dir(d))
    assert any("blank line" in e.message for e in errs)


def test_validate_empty_dir_and_missing_dir(tmp_path):
    (tmp_path / "empty").mkdir()
    assert has_errors(validate_submission_dir(tmp_path / "empty"))
    assert has_errors(validate_submission_dir(tmp_path / "nope"))


# ── packaging ────────────────────────────────────────────────────────────────


def test_package_refuses_on_errors(tmp_path):
    d = tmp_path / "sub"
    _write(d, "bad-kis.csv", "notavid,100\n")
    out_zip = tmp_path / "out" / "submission.zip"
    issues = package_codabench(d, out_zip)
    assert has_errors(issues)
    assert not out_zip.exists()
    assert not (out_zip.parent / "MANIFEST.json").exists()


def test_package_zip_layout_and_manifest(tmp_path):
    d = _valid_pack(tmp_path / "sub")
    out_zip = tmp_path / "out" / "submission.zip"
    issues = package_codabench(d, out_zip, package_name="mypack")
    assert not has_errors(issues)
    assert out_zip.exists()

    with zipfile.ZipFile(out_zip) as zf:
        names = sorted(zf.namelist())
    assert names == [
        "mypack/query-p1-1-kis.csv",
        "mypack/query-p1-2-qa.csv",
        "mypack/query-p1-3-trake.csv",
        "mypack/query-p1-4-avs.csv",
    ]

    manifest = json.loads((out_zip.parent / "MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["package_name"] == "mypack"
    assert manifest["created_utc"].endswith("Z")
    assert manifest["zip_sha256"] == hashlib.sha256(out_zip.read_bytes()).hexdigest()
    by_name = {f["name"]: f for f in manifest["files"]}
    kis = by_name["query-p1-1-kis.csv"]
    assert kis["rows"] == 2
    assert kis["task"] == "kis"
    assert kis["sha256"] == hashlib.sha256((d / "query-p1-1-kis.csv").read_bytes()).hexdigest()
    assert by_name["query-p1-3-trake.csv"]["rows"] == 1


def test_package_files_param_excludes_stale_csvs(tmp_path):
    # A stale (and here even invalid) CSV from an earlier run sits in the same
    # folder: whole-dir packaging refuses, files= packages only this run's CSVs.
    d = _valid_pack(tmp_path / "sub")
    _write(d, "stale-kis.csv", "notavid,100\n")
    out_zip = tmp_path / "out" / "submission.zip"
    assert has_errors(package_codabench(d, out_zip))
    assert not out_zip.exists()

    fresh = sorted(p for p in d.glob("*.csv") if p.name != "stale-kis.csv")
    issues = package_codabench(d, out_zip, package_name="mypack", files=fresh)
    assert not has_errors(issues)
    with zipfile.ZipFile(out_zip) as zf:
        names = zf.namelist()
    assert "mypack/stale-kis.csv" not in names
    assert sorted(names) == [f"mypack/{p.name}" for p in fresh]
    manifest = json.loads((out_zip.parent / "MANIFEST.json").read_text(encoding="utf-8"))
    assert "stale-kis.csv" not in {f["name"] for f in manifest["files"]}
    # an empty explicit file list is refused outright
    assert has_errors(package_codabench(d, tmp_path / "out" / "s2.zip", files=[]))


# ── QA grouping / per-row answers ────────────────────────────────────────────


def test_group_candidates_splits_videos_and_shots():
    from cvf.pipeline.run_queries import group_candidates

    results = [
        _Res("L21_V001", 100, 0),      # t = 4.0s
        _Res("L21_V002", 5000, 1),     # other video
        _Res("L21_V001", 150, 2),      # t = 6.0s → same shot as rank 0
        _Res("L21_V001", 9000, 3),     # t = 360s → far away, new group
    ]
    groups = group_candidates(results)
    assert len(groups) >= 2
    assert groups == [[0, 2], [1], [3]]  # best-rank order, shot-mates merged


def test_group_candidates_frame_gap_split_without_pts():
    from cvf.pipeline.run_queries import group_candidates

    class _NoPts:
        def __init__(self, video_id, frame_idx, global_id):
            self.video_id, self.frame_idx, self.global_id = video_id, frame_idx, global_id
            self.ref = SimpleNamespace(pts_time=None, path="f.jpg")

    results = [_NoPts("L21_V001", 0, 0), _NoPts("L21_V001", 200, 1), _NoPts("L21_V001", 900, 2)]
    groups = group_candidates(results, max_frame_gap=250)
    assert groups == [[0, 1], [2]]  # 200 apart merges, 700 apart splits


def test_compute_qa_answers_per_group_with_fallback():
    from cvf.pipeline.run_queries import compute_qa_answers

    settings = Settings.model_validate(
        {"vqa": {"provider": "vintern", "answers_per_query": 2, "max_calls_per_query": 2}}
    )
    results = [
        _Res("L21_V001", 100, 0),
        _Res("L21_V002", 5000, 1),
        _Res("L21_V001", 150, 2),
        _Res("L21_V001", 9000, 3),   # group beyond the 2-group budget
    ]
    vqa = _StubVqa({0: "hai", 1: "ba", 3: "bốn"})
    answers = compute_qa_answers(results, "có mấy người?", vqa, settings)
    assert answers == ["hai", "ba", "hai", "hai"]  # row 3 inherits best group's answer
    assert len(vqa.calls) == 2  # call budget respected
    assert vqa.calls == [[0], [1]]  # one call per group, best frame only


def test_compute_qa_answers_no_provider_uses_placeholder():
    from cvf.pipeline.run_queries import QA_FALLBACK_ANSWER, compute_qa_answers

    results = [_Res("L21_V001", 100, 0)]
    assert compute_qa_answers(results, "q?", None, None) == [QA_FALLBACK_ANSWER]
    assert compute_qa_answers([], "q?", None, None) == []


def test_compute_qa_answers_placeholder_when_all_vqa_calls_fail():
    # Empty answers can never block packaging: no provider or all VQA calls
    # failing must yield the placeholder, not "".
    from cvf.pipeline.run_queries import QA_FALLBACK_ANSWER, compute_qa_answers

    assert QA_FALLBACK_ANSWER == "không rõ"
    results = [_Res("L21_V001", 100, 0), _Res("L21_V002", 5000, 1)]
    vqa = _StubVqa({})  # provider present but every suggestion comes back empty
    assert compute_qa_answers(results, "q?", vqa, None) == [QA_FALLBACK_ANSWER] * 2


# ── DRES client ──────────────────────────────────────────────────────────────


class _FakeResp:
    def __init__(self, body: bytes, status: int = 200):
        self._body = body
        self.status = status

    def read(self) -> bytes:
        return self._body

    def getcode(self) -> int:
        return self.status

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _patch_urlopen(monkeypatch, captured: list, body: bytes = b'{"sessionId": "s-1", "status": true}'):
    def fake(req, timeout=None):
        captured.append((req, timeout))
        return _FakeResp(body)

    monkeypatch.setattr(urllib.request, "urlopen", fake)


def test_dres_login_and_kis_payload(monkeypatch):
    captured: list = []
    _patch_urlopen(monkeypatch, captured)
    client = DresClient("http://dres.test/", timeout=3.0)

    res = client.login("user1", "pw1")
    assert res.ok and res.status == 200
    assert client.session_id == "s-1"
    req, timeout = captured[0]
    assert req.full_url == "http://dres.test/api/v2/login"
    assert timeout == 3.0
    assert json.loads(req.data) == {"username": "user1", "password": "pw1"}
    assert req.headers.get("Content-type") == "application/json"

    res = client.submit_kis("L21_V001", frame_idx=4567, fps=25.0)
    assert res.ok
    req, _ = captured[1]
    assert req.full_url == "http://dres.test/api/v2/submit?session=s-1"
    payload = json.loads(req.data)
    # frame 4567 @ 25 fps → round(4567/25*1000) = 182680 ms; NO frame field
    assert payload == {"answerSets": [{"answers": [
        {"mediaItemName": "L21_V001", "start": 182680, "end": 182680}
    ]}]}


def test_dres_kis_time_ms_and_evaluation_id(monkeypatch):
    captured: list = []
    _patch_urlopen(monkeypatch, captured)
    client = DresClient("http://dres.test", evaluation_id="eval7", session_id="tok")
    client.submit_kis("L21_V002", time_ms=90500)
    req, _ = captured[0]
    assert req.full_url == "http://dres.test/api/v2/submit/eval7?session=tok"
    answer = json.loads(req.data)["answerSets"][0]["answers"][0]
    assert answer == {"mediaItemName": "L21_V002", "start": 90500, "end": 90500}


def test_dres_qa_and_trake_payloads(monkeypatch):
    captured: list = []
    _patch_urlopen(monkeypatch, captured)
    client = DresClient("http://dres.test", session_id="s")

    client.submit_qa("L21_V001", 100, "hai người", time_ms=4000)
    answer = json.loads(captured[0][0].data)["answerSets"][0]["answers"][0]
    assert answer == {"mediaItemName": "L21_V001", "text": "hai người",
                      "start": 4000, "end": 4000}

    # frames + fps → per-event milliseconds, one answer per event in ONE set
    client.submit_trake("L21_V003", [10, 20, 30], fps=25.0)
    answer_sets = json.loads(captured[1][0].data)["answerSets"]
    assert len(answer_sets) == 1
    answers = answer_sets[0]["answers"]
    assert [a["start"] for a in answers] == [400, 800, 1200]
    assert [a["end"] for a in answers] == [400, 800, 1200]
    assert all(a["mediaItemName"] == "L21_V003" for a in answers)

    # explicit per-event timestamps are preferred over frames
    client.submit_trake("L21_V003", [10, 20], times_ms=[450, 950])
    answers = json.loads(captured[2][0].data)["answerSets"][0]["answers"]
    assert [(a["start"], a["end"]) for a in answers] == [(450, 450), (950, 950)]


def test_dres_answers_never_carry_frame_field(monkeypatch, caplog):
    # DRES v2 ApiClientAnswer allows ONLY {text, mediaItemName,
    # mediaItemCollectionName, start, end} (additionalProperties: false).
    captured: list = []
    _patch_urlopen(monkeypatch, captured)
    client = DresClient("http://dres.test", session_id="s")

    with caplog.at_level("WARNING", logger="cvf.submission.dres_client"):
        client.submit_kis("L21_V001", frame_idx=4567)          # no time_ms, no fps
        client.submit_qa("L21_V002", 100, "hai")               # no time_ms, no fps
        client.submit_trake("L21_V003", [10, 20])              # no times_ms, no fps
    assert sum("no frame field" in r.message for r in caplog.records) == 3

    payloads = [json.loads(req.data) for req, _ in captured]
    all_answers = [a for p in payloads for s in p["answerSets"] for a in s["answers"]]
    assert all_answers and all("frame" not in a for a in all_answers)
    assert payloads[0]["answerSets"][0]["answers"][0] == {"mediaItemName": "L21_V001"}
    assert payloads[1]["answerSets"][0]["answers"][0] == {"mediaItemName": "L21_V002", "text": "hai"}
    assert payloads[2]["answerSets"][0]["answers"] == [{"mediaItemName": "L21_V003"}] * 2


def test_dres_env_credentials_and_missing(monkeypatch):
    captured: list = []
    _patch_urlopen(monkeypatch, captured)
    monkeypatch.setenv("DRES_USER", "envu")
    monkeypatch.setenv("DRES_PASSWORD", "envp")
    client = DresClient("http://dres.test")
    assert client.login().ok
    assert json.loads(captured[0][0].data) == {"username": "envu", "password": "envp"}

    monkeypatch.delenv("DRES_USER")
    monkeypatch.delenv("DRES_PASSWORD")
    res = DresClient("http://dres.test").login()
    assert not res.ok and res.status == 0
    assert "credentials" in res.message
    assert len(captured) == 1  # no HTTP call without credentials


def test_dres_transport_and_http_errors(monkeypatch):
    def refuse(req, timeout=None):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(urllib.request, "urlopen", refuse)
    res = DresClient("http://dres.test", session_id="s").submit_kis("L21_V001", frame_idx=1)
    assert res == SubmitResult(False, 0, res.message)
    assert "connection refused" in res.message

    def http_412(req, timeout=None):
        raise urllib.error.HTTPError(
            req.full_url, 412, "Precondition Failed", None,
            io.BytesIO(b'{"description": "Duplicate submission", "status": false}'),
        )

    monkeypatch.setattr(urllib.request, "urlopen", http_412)
    res = DresClient("http://dres.test", session_id="s").submit_kis("L21_V001", frame_idx=1)
    assert not res.ok and res.status == 412
    assert "Duplicate submission" in res.message


# ── auto agent ───────────────────────────────────────────────────────────────


class _StubEngine:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.calls: list[tuple] = []

    @staticmethod
    def _results():
        return [_Res("L21_V001", 100, 0), _Res("L21_V002", 4000, 7)]

    def search_text(self, query, **kw):
        self.calls.append(("search_text", query))
        return self._results()

    def search_avs(self, query, **kw):
        self.calls.append(("search_avs", query))
        return self._results()

    def search_trake(self, events, **kw):
        self.calls.append(("search_trake", tuple(events)))
        return [SimpleNamespace(video_id="L21_V001", frame_idxs=[10, 20, 30],
                                pts_times=[0.4, 0.8, 1.2])]


def _query_pack(tmp_path: Path) -> Path:
    q = tmp_path / "queries"
    q.mkdir(exist_ok=True)
    (q / "query-p3-1-kis.txt").write_text("người đàn ông áo đỏ\n", encoding="utf-8")
    (q / "query-p3-2-qa.txt").write_text("cảnh chợ nổi\ncó mấy chiếc thuyền?\n", encoding="utf-8")
    (q / "query-p3-3-trake.txt").write_text("chạy đà\nnhảy lên\ntiếp đất\n", encoding="utf-8")
    (q / "query-p3-4-avs.txt").write_text("cảnh mưa lớn ngập đường\n", encoding="utf-8")
    return q


def _auto_settings(extra: dict | None = None) -> Settings:
    raw: dict = {"vqa": {"provider": "none"}}
    raw.update(extra or {})
    return Settings.model_validate(raw)


def test_auto_agent_imports_without_engine_stack():
    import importlib

    import cvf.pipeline.auto_agent as mod

    importlib.reload(mod)  # module import must stay torch/engine-free


def test_auto_agent_routing_and_packaging(tmp_path):
    from cvf.pipeline.auto_agent import run_auto

    settings = _auto_settings()
    engine_holder: list[_StubEngine] = []

    def factory(s: Settings) -> _StubEngine:
        engine_holder.append(_StubEngine(s))
        return engine_holder[-1]

    report = run_auto(
        _query_pack(tmp_path), tmp_path / "out", settings,
        engine_factory=factory, vqa=_StubVqa({0: "hai"}),
    )

    engine = engine_holder[0]
    called = [c[0] for c in engine.calls]
    assert called.count("search_text") == 2       # kis + qa description
    assert ("search_trake", ("chạy đà", "nhảy lên", "tiếp đất")) in engine.calls
    assert ("search_avs", "cảnh mưa lớn ngập đường") in engine.calls

    assert [p.name for p in report.written] == [
        "query-p3-1-kis.csv", "query-p3-2-qa.csv", "query-p3-3-trake.csv", "query-p3-4-avs.csv",
    ]
    qa_lines = (tmp_path / "out" / "query-p3-2-qa.csv").read_text(encoding="utf-8").strip().splitlines()
    assert qa_lines[0] == "L21_V001,100,hai"
    assert qa_lines[1] == "L21_V002,4000,hai"     # fallback answer on the other group

    assert report.ok
    assert report.zip_path is not None and report.zip_path.exists()
    with zipfile.ZipFile(report.zip_path) as zf:
        assert f"{settings.submission.package_name}/query-p3-1-kis.csv" in zf.namelist()
    assert (tmp_path / "out" / "MANIFEST.json").exists()
    assert report.submitted == []                 # no --submit


def test_auto_agent_qa_without_provider_uses_placeholder(tmp_path):
    from cvf.pipeline.auto_agent import run_auto
    from cvf.pipeline.run_queries import QA_FALLBACK_ANSWER

    report = run_auto(
        _query_pack(tmp_path), tmp_path / "out", _auto_settings(),
        engine_factory=lambda s: _StubEngine(s),  # vqa=None → placeholder QA answers
    )
    qa_lines = (tmp_path / "out" / "query-p3-2-qa.csv").read_text(encoding="utf-8").strip().splitlines()
    assert all(ln.endswith(f",{QA_FALLBACK_ANSWER}") for ln in qa_lines)
    assert not has_errors(report.issues)          # placeholder never blocks packaging
    assert report.zip_path is not None and report.zip_path.exists()


def test_auto_agent_ignores_stale_csvs_in_out_dir(tmp_path):
    from cvf.pipeline.auto_agent import run_auto

    out = tmp_path / "out"
    out.mkdir()
    (out / "stale-kis.csv").write_text("notavid,100\n", encoding="utf-8")  # earlier run
    report = run_auto(
        _query_pack(tmp_path), out, _auto_settings(),
        engine_factory=lambda s: _StubEngine(s), vqa=_StubVqa({0: "hai"}),
    )
    assert not has_errors(report.issues)          # stale file must not block
    assert report.zip_path is not None
    with zipfile.ZipFile(report.zip_path) as zf:
        assert not any(name.endswith("stale-kis.csv") for name in zf.namelist())


class _StubClient:
    def __init__(self):
        self.calls: list[tuple] = []

    def login(self, *a, **k):
        self.calls.append(("login",))
        return SubmitResult(True, 200, "ok")

    def submit_kis(self, video_id, frame_idx=None, time_ms=None, fps=None):
        self.calls.append(("kis", video_id, frame_idx, time_ms))
        return SubmitResult(True, 200, "correct")

    def submit_qa(self, video_id, frame_idx, answer, time_ms=None, fps=None):
        self.calls.append(("qa", video_id, frame_idx, answer, time_ms))
        return SubmitResult(True, 200, "correct")

    def submit_trake(self, video_id, frames=(), times_ms=None, fps=None):
        self.calls.append(("trake", video_id, list(frames),
                           list(times_ms) if times_ms is not None else None))
        return SubmitResult(True, 200, "correct")


def test_auto_agent_submit_top1_respects_gate(tmp_path):
    from cvf.pipeline.auto_agent import run_auto

    # Gate closed: auto_submit stays false → nothing is pushed.
    gated = run_auto(
        _query_pack(tmp_path), tmp_path / "out1",
        _auto_settings({"submission": {"dres_base_url": "http://dres.test"}}),
        submit=True, engine_factory=lambda s: _StubEngine(s), vqa=_StubVqa({0: "hai"}),
        client_factory=lambda s: pytest.fail("client must not be built when gate is closed"),
    )
    assert gated.submitted == []

    # Gate open: top-1 of each CSV goes out, routed per task.
    client = _StubClient()
    report = run_auto(
        _query_pack(tmp_path), tmp_path / "out2",
        _auto_settings({"submission": {"dres_base_url": "http://dres.test", "auto_submit": True}}),
        submit=True, engine_factory=lambda s: _StubEngine(s), vqa=_StubVqa({0: "hai"}),
        client_factory=lambda s: client,
    )
    assert len(report.submitted) == 4
    assert all(res.ok for _, res in report.submitted)
    assert ("login",) in client.calls
    # time_ms comes from the SearchResult's ref.pts_time (frame/25 s → ms),
    # never a bare frame index — DRES v2 answers have no frame field.
    assert ("kis", "L21_V001", 100, 4000) in client.calls     # KIS + AVS both top-1 rows
    assert ("qa", "L21_V001", 100, "hai", 4000) in client.calls
    assert ("trake", "L21_V001", [10, 20, 30], [400, 800, 1200]) in client.calls


def test_auto_agent_submit_skipped_on_validation_errors(tmp_path, monkeypatch, caplog):
    import cvf.pipeline.auto_agent as aa

    monkeypatch.setattr(aa, "validate_file", lambda p, strict=True: [
        ValidationIssue(file=p.name, line=0, severity="error", message="forced error")
    ])
    with caplog.at_level("ERROR", logger="cvf.pipeline.auto_agent"):
        report = aa.run_auto(
            _query_pack(tmp_path), tmp_path / "out",
            _auto_settings({"submission": {"dres_base_url": "http://dres.test",
                                           "auto_submit": True}}),
            submit=True, engine_factory=lambda s: _StubEngine(s), vqa=_StubVqa({0: "hai"}),
            client_factory=lambda s: pytest.fail("client must not be built on errors"),
        )
    assert has_errors(report.issues)
    assert report.submitted == []
    assert any("NOT submitting" in r.message for r in caplog.records)


def test_auto_agent_submit_aborts_when_login_fails(tmp_path, caplog):
    from cvf.pipeline.auto_agent import run_auto

    class _FailLoginClient(_StubClient):
        def login(self, *a, **k):
            self.calls.append(("login",))
            return SubmitResult(False, 401, "bad credentials")

    client = _FailLoginClient()
    with caplog.at_level("ERROR", logger="cvf.pipeline.auto_agent"):
        report = run_auto(
            _query_pack(tmp_path), tmp_path / "out",
            _auto_settings({"submission": {"dres_base_url": "http://dres.test",
                                           "auto_submit": True}}),
            submit=True, engine_factory=lambda s: _StubEngine(s), vqa=_StubVqa({0: "hai"}),
            client_factory=lambda s: client,
        )
    assert client.calls == [("login",)]           # nothing submitted after the failure
    assert report.submitted == []
    assert any("login failed" in r.message for r in caplog.records)
