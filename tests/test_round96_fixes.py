"""Round-96: lessons of sơ tuyển 3 (04/09/2026, public 5.0 with 14/36 queries submitted).

Google's Gemini API stormed for 8 hours (503 "high demand" / 504 on 80-90 % of
calls). Every failed call still walked the whole 5-model fallback chain with a
45-90 s wall clock per model, so a KIS query took 4-7 minutes for three EMPTY
VLM-rerank votes and 22/36 queries never got submitted. Five VMs creating the
same Drive folder produced THREE twin folders, so the gather saw 22 of 36 CSVs.

1. cvp.models.gemini_health — per-model circuit breaker shared by every Gemini
   call site (query enhancement, VQA strips, VLM rerank): open after
   consecutive transient failures, skip without waiting, half-open probe.
2. Local VLM plan B: one shared hf_auto model (cvp.models.local_vlm) answers
   whole strips and scores frames for the rerank when Gemini yields nothing.
3. QA gets on-screen OCR text next to the ASR transcript (vqa.ocr_context).
4. run_auto can order KIS/AVS → TRAKE → QA (submission.query_order).
5. Rescue mode "head": re-answer when the TOP rows are non-answers.
6. deadline.governor_minutes turns a close time into PACK_DEADLINE_MIN.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import pytest

from cvp.config import Settings
from cvp.models.gemini_health import HEALTH, AllModelsOpen, GeminiHealth, classify
from cvp.pipeline.attempts import qa_head_fallback, rescue_fallback_qa
from cvp.pipeline.auto_agent import order_query_files
from cvp.pipeline.deadline import VN_TZ, governor_minutes, minutes_until, parse_hhmm

REPO = Path(__file__).resolve().parents[1]


# ── 1. circuit breaker ───────────────────────────────────────────────────────

class _Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def _open_breaker(clock=None):
    clock = clock or _Clock()
    h = GeminiHealth(window=8, fail_ratio=0.75, min_consecutive=3, cooldown_s=120.0, clock=clock)
    h.enabled = True
    return h, clock


def test_r96_breaker_stays_closed_while_healthy_and_on_isolated_failures():
    h, _ = _open_breaker()
    for _ in range(20):
        h.record("m", True)
    assert h.state("m") == "closed" and not h.should_skip("m")
    h.record("m", False, "503")
    h.record("m", True)
    h.record("m", False, "504")
    h.record("m", True)
    assert h.state("m") == "closed" and not h.should_skip("m")     # no 3-run of failures


def test_r96_breaker_opens_after_three_consecutive_transient_failures():
    h, clock = _open_breaker()
    for k in ("503", "504", "timeout"):
        h.record("m", False, k)
    assert h.state("m") == "open" and h.should_skip("m")
    assert h.snapshot()["m"]["skipped"] == 1
    clock.t += 119
    assert h.should_skip("m")                                        # still cooling down
    clock.t += 2
    assert h.state("m") == "half-open"
    assert not h.should_skip("m")                                    # exactly ONE probe
    assert h.should_skip("m")                                        # others keep skipping
    h.record("m", True)                                              # probe succeeded
    assert h.state("m") == "closed" and not h.should_skip("m")


def test_r96_breaker_failed_probe_reopens_and_4xx_never_counts():
    h, clock = _open_breaker()
    for _ in range(3):
        h.record("m", False, "503")
    clock.t += 121
    assert not h.should_skip("m")                                    # probe goes through
    h.record("m", False, "503")                                      # ...and fails
    assert h.state("m") == "open" and h.should_skip("m")
    # 400/404 are caller bugs / dead ids, not storms
    for _ in range(10):
        h.record("x", False, "4xx")
    assert h.state("x") == "closed"


def test_r96_breaker_disabled_is_a_no_op():
    h, _ = _open_breaker()
    h.enabled = False
    for _ in range(5):
        h.record("m", False, "503")
    assert h.state("m") == "open"            # ledger still kept for telemetry...
    assert not h.should_skip("m")            # ...but nothing is skipped
    assert not h.all_open(["m"])


def test_r96_all_open_needs_every_model_open_and_no_probe_available():
    h, clock = _open_breaker()
    for m in ("a", "b"):
        for _ in range(3):
            h.record(m, False, "503")
    assert h.all_open(["a", "b"])
    assert not h.all_open(["a", "b", "c"])                          # c never failed
    clock.t += 121
    assert not h.all_open(["a", "b"])                                # probes available


def test_r96_classify_and_settings_configure():
    assert classify(RuntimeError("503 UNAVAILABLE. high demand")) == "503"
    assert classify(RuntimeError("504 DEADLINE_EXCEEDED")) == "504"
    assert classify(RuntimeError("call exceeded 45.0s wall clock")) == "timeout"
    assert classify(RuntimeError("400 INVALID_ARGUMENT")) == "4xx"
    assert classify(RuntimeError("boom")) == "other"
    s = Settings()
    s.breaker.enabled = True
    s.breaker.cooldown_s = 33.0
    h = GeminiHealth().configure(s)
    assert h.enabled is True and h.cooldown_s == 33.0
    assert Settings().breaker.enabled is False                       # default off


def test_r96_load_settings_configures_the_process_breaker(monkeypatch):
    from cvp.config import load_settings
    monkeypatch.setenv("CVP_BREAKER__ENABLED", "true")
    monkeypatch.setenv("CVP_BREAKER__COOLDOWN_S", "45")
    load_settings()
    assert HEALTH.enabled is True and HEALTH.cooldown_s == 45.0
    monkeypatch.setenv("CVP_BREAKER__ENABLED", "false")
    load_settings()
    assert HEALTH.enabled is False
    HEALTH.reset()


class _Models:
    def __init__(self, dead: set[str], log: list[str]):
        self.dead, self.log = dead, log

    def generate_content(self, model, contents, config=None):
        self.log.append(model)
        if model in self.dead:
            raise RuntimeError("503 UNAVAILABLE. This model is currently experiencing high demand")
        return type("R", (), {"text": f"ok:{model}"})()


class _Client:
    def __init__(self, dead, log):
        self.models = _Models(dead, log)


def test_r96_generate_with_fallback_skips_open_models_without_calling_them(monkeypatch):
    from cvp.search import vqa as V
    clock = _Clock()
    h = GeminiHealth(window=8, fail_ratio=0.75, min_consecutive=3, cooldown_s=120.0, clock=clock)
    h.enabled = True
    monkeypatch.setattr(V, "HEALTH", h)
    calls: list[str] = []
    client = _Client({"a"}, calls)
    for _ in range(3):
        assert V.generate_with_fallback(client, ["a", "b"], ["x"]) == "ok:b"
    assert calls.count("a") == 3 and h.state("a") == "open"
    calls.clear()
    assert V.generate_with_fallback(client, ["a", "b"], ["x"]) == "ok:b"
    assert calls == ["b"]                                             # a was skipped, no wait
    # every model open → AllModelsOpen at once, zero network calls
    for _ in range(3):
        h.record("b", False, "503")
    calls.clear()
    with pytest.raises(AllModelsOpen):
        V.generate_with_fallback(client, ["a", "b"], ["x"])
    assert calls == []


def test_r96_generate_with_fallback_unchanged_when_breaker_disabled(monkeypatch):
    from cvp.search import vqa as V
    h = GeminiHealth()
    monkeypatch.setattr(V, "HEALTH", h)
    calls: list[str] = []
    client = _Client({"a"}, calls)
    for _ in range(5):
        assert V.generate_with_fallback(client, ["a", "b"], ["x"]) == "ok:b"
    assert calls.count("a") == 5                                      # always tried first


def test_r96_vlm_rerank_skips_all_votes_when_the_chain_is_open(monkeypatch, caplog):
    from cvp.models import gemini_health as GH
    from cvp.search import vlm_rerank as VR
    h = GeminiHealth(cooldown_s=120.0)
    h.enabled = True
    monkeypatch.setattr(GH, "HEALTH", h)
    s = Settings()
    s.search.vlm_rerank_provider = "gemini"
    s.search.vlm_rerank_votes = 3
    chain = [s.search.vlm_rerank_model] + list(s.query.gemini_model_fallbacks)
    for m in chain:
        for _ in range(3):
            h.record(m, False, "503")
    called = []
    monkeypatch.setattr(VR, "_gemini_scores", lambda q, p, st: called.append(1) or [1.0] * len(p))

    class R:
        def __init__(self, i):
            self.ref = type("Ref", (), {"path": f"/f{i}.jpg"})()
            self.signals = {}

    rows = [R(i) for i in range(5)]
    with caplog.at_level(logging.WARNING):
        out = VR.vlm_rerank(rows, "q", s, topk=5)
    assert out == rows and called == []                               # order kept, no votes fired
    assert any("cầu dao bão" in r.getMessage() for r in caplog.records)


def test_r96_vlm_rerank_local_fallback_when_gemini_has_no_scores(monkeypatch):
    from cvp.search import vlm_rerank as VR
    s = Settings()
    s.search.vlm_rerank_provider = "gemini"
    s.search.vlm_rerank_votes = 1
    s.search.vlm_rerank_local_fallback = True
    s.vqa.local_backend = "hf_auto"
    s.vqa.local_hf_id = "org/verified-vlm"
    monkeypatch.setattr(VR, "_gemini_scores", lambda q, p, st: None)
    monkeypatch.setattr(VR, "_hf_auto_scores", lambda q, p, st: [float(i) for i in range(len(p))])

    class R:
        def __init__(self, i):
            self.ref = type("Ref", (), {"path": f"/f{i}.jpg"})()
            self.signals = {}

    rows = [R(i) for i in range(4)]
    out = VR.vlm_rerank(rows, "q", s, topk=4)
    assert [r.ref.path for r in out] == ["/f3.jpg", "/f2.jpg", "/f1.jpg", "/f0.jpg"]
    s.search.vlm_rerank_local_fallback = False
    assert VR.vlm_rerank([R(i) for i in range(4)], "q", s, topk=4)[0].ref.path == "/f0.jpg"


def test_r96_vlm_rerank_hf_auto_provider_and_pointwise_parser(monkeypatch):
    from cvp.models import local_vlm as LV
    from cvp.search import vlm_rerank as VR
    s = Settings()
    s.search.vlm_rerank_provider = "hf_auto"
    s.vqa.local_backend = "hf_auto"
    s.vqa.local_hf_id = "org/verified-vlm"
    answers = iter(["7", "Điểm: 10", "nonsense", "3.5"])
    monkeypatch.setattr(LV, "generate", lambda mid, imgs, prompt, max_new_tokens=4: next(answers))
    monkeypatch.setattr(VR.Image if hasattr(VR, "Image") else __import__("PIL.Image").Image, "open",
                        lambda p: type("I", (), {"convert": lambda self_, m: self_,
                                                 "thumbnail": lambda self_, sz: None})(),
                        raising=False)
    scores = VR._hf_auto_scores("q", ["a", "b", "c", "d"], s)
    assert scores == [7.0, 10.0, 0.0, 3.5]


# ── 2. local VLM strip + provider "local" ───────────────────────────────────

def test_r96_answer_group_local_provider_reads_the_whole_strip(monkeypatch, tmp_path):
    from cvp.models import local_vlm as LV
    from cvp.search import vqa as V
    s = Settings()
    s.vqa.provider = "local"
    s.vqa.local_backend = "hf_auto"
    s.vqa.local_hf_id = "org/verified-vlm"
    s.vqa.frames_per_answer = 3
    seen = {}

    def fake_generate(mid, imgs, prompt, max_new_tokens=64):
        seen["n"] = len(imgs)
        seen["prompt"] = prompt
        seen["mid"] = mid
        return " 7 cái "

    monkeypatch.setattr(LV, "generate", fake_generate)
    monkeypatch.setattr(V, "load_rgb", lambda p: type("I", (), {"thumbnail": lambda self_, sz: None})())
    a = V.VqaAssistant(s)
    assert a.answer_group("Có mấy cái?", ["/a.jpg", "/b.jpg", "/c.jpg"], context="lời thoại") == "7 cái"
    assert seen["n"] == 3 and seen["mid"] == "org/verified-vlm"
    assert "3 khung hình LIÊN TIẾP" in seen["prompt"] and "lời thoại" in seen["prompt"]
    assert a.answer_group_votes("q", ["/a.jpg"]) == []                 # non-gemini → caller degrades


def test_r96_local_hf_auto_fails_loud_without_an_id():
    from cvp.models import local_vlm as LV
    with pytest.raises(RuntimeError, match="local_hf_id"):
        LV.get_local_vlm("")


# ── 3. OCR context ──────────────────────────────────────────────────────────

def test_r96_ocr_context_reads_neighbouring_keyframes_and_caps(tmp_path):
    from cvp.search.vqa import OCR_MARK, _with_context, ocr_context
    s = Settings()
    s.paths.artifacts_root = tmp_path
    d = tmp_path / "ocr"
    d.mkdir()
    (d / "L01_V001.json").write_text(json.dumps({"n_to_text": {
        "9": "TRƯỜNG TIỂU HỌC  ĐẮK SƠ MEI", "10": "trường tiểu học đắk sơ mei",
        "11": "TRƯỜNG TIỂU HỌC  ĐẮK SƠ MEI", "30": "xa"}}), encoding="utf-8")
    out = ocr_context(s, "L01_V001", [10])
    assert out == "TRƯỜNG TIỂU HỌC ĐẮK SƠ MEI | trường tiểu học đắk sơ mei"   # ±1, deduped, squeezed
    assert ocr_context(s, "L01_V001", [30], pad=0) == "xa"
    assert ocr_context(s, "L01_V002", [10]) == "" and ocr_context(s, "L01_V001", []) == ""
    s.vqa.ocr_context_chars = 10
    assert len(ocr_context(s, "L01_V001", [10])) == 10
    # prompt assembly: plain ASR prompt byte-identical to before; OCR labelled separately
    plain = _with_context("Q?", "xin chào")
    assert plain.startswith("Lời thoại trong đoạn video") and plain.endswith("\nQ?")
    both = _with_context("Q?", f"xin chào{OCR_MARK}ĐẮK SƠ MEI")
    assert "Lời thoại" in both and "Chữ hiển thị trên màn hình" in both and both.endswith("\nQ?")
    only = _with_context("Q?", f"{OCR_MARK.strip()} ĐẮK SƠ MEI")
    assert "Lời thoại" not in only and "ĐẮK SƠ MEI" in only


def test_r96_qa_group_context_carries_ocr_when_enabled(monkeypatch, tmp_path):
    from cvp.pipeline.run_queries import compute_qa_answers_with_stats
    s = Settings()
    s.paths.artifacts_root = tmp_path
    s.vqa.ocr_context = True
    s.vqa.frames_per_answer = 1
    (tmp_path / "ocr").mkdir()
    (tmp_path / "ocr" / "L01_V001.json").write_text(json.dumps({"n_to_text": {"5": "BẢNG TÊN"}}),
                                                     encoding="utf-8")

    class Ref:
        def __init__(self, n):
            self.video_id, self.n, self.frame_idx = "L01_V001", n, n * 25
            self.pts_time, self.path, self.global_id = float(n), f"/{n}.jpg", n

    class Res:
        def __init__(self, n):
            self.ref = Ref(n)
            self.video_id, self.frame_idx, self.global_id = "L01_V001", n * 25, n

    got = {}

    class Vqa:
        def answer_group(self, question, strip, context=""):
            got["ctx"] = context
            return "x"

        def suggest(self, *a, **k):
            return []

    compute_qa_answers_with_stats([Res(5)], "q?", Vqa(), s)
    assert "[OCR] BẢNG TÊN" in got["ctx"]
    s.vqa.ocr_context = False
    compute_qa_answers_with_stats([Res(5)], "q?", Vqa(), s)
    assert "[OCR]" not in got["ctx"]


# ── 4. query order ──────────────────────────────────────────────────────────

def test_r96_order_query_files_kis_first():
    names = ["query-p2-1-kis.txt", "query-p2-10-qa.txt", "query-p2-2-trake.txt",
             "query-p2-3-avs.txt", "query-p2-4-qa.txt", "query-p2-5-kis.txt"]
    files = [Path(n) for n in sorted(names)]
    assert order_query_files(files, "name") == files
    assert [p.name for p in order_query_files(files, "kis_first")] == [
        "query-p2-1-kis.txt", "query-p2-3-avs.txt", "query-p2-5-kis.txt",
        "query-p2-2-trake.txt", "query-p2-10-qa.txt", "query-p2-4-qa.txt"]
    assert Settings().submission.query_order == "name"


def test_r96_run_auto_honours_query_order(tmp_path):
    from cvp.pipeline import auto_agent as AA
    q = tmp_path / "q"
    q.mkdir()
    for n in ("query-p2-1-qa.txt", "query-p2-2-kis.txt", "query-p2-3-trake.txt"):
        (q / n).write_text("x", encoding="utf-8")
    order: list[str] = []

    def fake_run_query_file(engine, qf, out_dir, vqa, top1_times=None):
        order.append(qf.name)
        p = Path(out_dir) / f"{qf.stem}.csv"
        p.write_text("L01_V001,10\n", encoding="utf-8")
        return p

    import cvp.pipeline.run_queries as RQ
    orig = RQ.run_query_file
    RQ.run_query_file = fake_run_query_file
    try:
        s = Settings()
        s.vqa.provider = "none"
        s.submission.query_order = "kis_first"
        AA.run_auto(q, tmp_path / "out", s, engine_factory=lambda _s: object())
    finally:
        RQ.run_query_file = orig
    assert order == ["query-p2-2-kis.txt", "query-p2-3-trake.txt", "query-p2-1-qa.txt"]


# ── 5. rescue head mode ─────────────────────────────────────────────────────

def test_r96_rescue_head_mode(tmp_path):
    head_bad = "\n".join(["L01_V001,10,Không có thông tin"] * 12 + ["L01_V002,20,Cá hồi"]) + "\n"
    (tmp_path / "query-p2-18-qa.csv").write_text(head_bad, encoding="utf-8")
    (tmp_path / "query-p2-5-qa.csv").write_text("L01_V001,10,5\nL01_V002,20,không rõ\n", encoding="utf-8")
    rows = [r.split(",") for r in head_bad.strip().splitlines()]
    assert qa_head_fallback(rows, 10) and not qa_head_fallback(rows, 13)
    assert rescue_fallback_qa(tmp_path, mode="all") == []              # tail has a real answer
    assert rescue_fallback_qa(tmp_path, mode="head", head=10) == ["query-p2-18-qa"]
    assert (tmp_path / "query-p2-5-qa.csv").exists()
    with pytest.raises(ValueError):
        rescue_fallback_qa(tmp_path, mode="top")


# ── 6. close time → governor ────────────────────────────────────────────────

def test_r96_deadline_math():
    now = datetime(2026, 9, 4, 20, 0, tzinfo=VN_TZ)
    assert parse_hhmm("22:30") == (22, 30) and parse_hhmm(" 7h05 ") == (7, 5)
    assert minutes_until("22:30", now) == 150.0
    assert minutes_until("19:30", now) == 0.0                          # already closed
    assert minutes_until("01:00", datetime(2026, 9, 4, 23, 30, tzinfo=VN_TZ)) == 90.0   # past midnight
    assert governor_minutes("22:30", now=now) == round((150 - 25) * 0.45, 1)
    assert governor_minutes("20:10", now=now) == 10.0                  # floor, never 0
    with pytest.raises(ValueError):
        parse_hhmm("22.30")
    with pytest.raises(ValueError):
        parse_hhmm("25:00")


# ── 6b. second lanes: vertex: / hf: prefixes ────────────────────────────────

def test_r96_expand_chain_adds_vertex_twin_and_drops_unconfigured_lanes(monkeypatch):
    from cvp.models.gemini_keys import expand_chain
    monkeypatch.delenv("GEMINI_VERTEX_KEY", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    fb = ["gemini-3.8-flash", "hf:org/m", "vertex:gemini-3.7-flash", "gemini-3.8-flash"]
    assert expand_chain("gemini-3.1-pro-preview", fb) == ["gemini-3.1-pro-preview", "gemini-3.8-flash"]
    monkeypatch.setenv("GEMINI_VERTEX_KEY", "vk")
    monkeypatch.setenv("HF_TOKEN", "hk")
    assert expand_chain("gemini-3.1-pro-preview", fb) == [
        "gemini-3.1-pro-preview", "vertex:gemini-3.1-pro-preview", "gemini-3.8-flash",
        "hf:org/m", "vertex:gemini-3.7-flash"]
    assert expand_chain("hf:org/x", ["gemini-3.8-flash"])[:1] == ["hf:org/x"]   # no vertex twin of hf
    assert expand_chain("vertex:g", [])[0] == "vertex:g"


def test_r96_qa_chain_keeps_vertex_twin_of_pro_ahead_of_the_flash_rescue(monkeypatch):
    from cvp.search import vqa as V
    monkeypatch.setenv("GEMINI_VERTEX_KEY", "vk")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    s = Settings()
    chain = V.gemini_model_chain(s, s.vqa.answer_model)
    assert chain[:2] == ["gemini-3.1-pro-preview", "vertex:gemini-3.1-pro-preview"]
    # reproduce the rescue insertion of _ask_gemini_strip
    primary, rescue = s.vqa.answer_model, s.vqa.gemini_model
    head = [primary] + ([chain[1]] if chain[1] == f"vertex:{primary}" else [])
    chain2 = head + [rescue] + [m for m in chain[len(head):] if m != rescue]
    assert chain2[:3] == ["gemini-3.1-pro-preview", "vertex:gemini-3.1-pro-preview", "gemini-3.7-flash"]
    assert chain2.count("gemini-3.7-flash") == 1


def test_r96_rotating_client_dispatches_prefixes(monkeypatch):
    from cvp.models import gemini_keys as GK
    from cvp.models import hf_router as HR
    monkeypatch.setenv("GEMINI_API_KEY", "k1")
    monkeypatch.setenv("GEMINI_VERTEX_KEY", "vk")
    monkeypatch.setenv("HF_TOKEN", "hk")
    GK.reset_for_tests()
    seen = []

    class M:
        def __init__(self, tag):
            self.tag = tag

        def generate_content(self, **kw):
            seen.append((self.tag, kw["model"]))
            return type("R", (), {"text": f"{self.tag}:{kw['model']}"})()

    class C:
        def __init__(self, tag):
            self.models = M(tag)

    factories = {}
    c = GK.RotatingGeminiClient(
        http_options={"timeout": 30000},
        client_factory=lambda key, http: factories.setdefault("studio", (key, http)) and C("studio"),
        vertex_factory=lambda key, http: factories.setdefault("vertex", (key, http)) and C("vertex"))
    assert c.models.generate_content(model="gemini-3.8-flash", contents="hi").text == "studio:gemini-3.8-flash"
    assert c.models.generate_content(model="vertex:gemini-3.8-flash", contents="hi").text == "vertex:gemini-3.8-flash"
    assert factories["vertex"] == ("vk", {"timeout": 30000}) and factories["studio"][0] == "k1"
    monkeypatch.setattr(HR, "generate", lambda mid, contents, timeout_s: type("R", (), {
        "text": f"hf:{mid}:{timeout_s}"})())
    assert c.models.generate_content(model="hf:org/m:novita", contents=["p"]).text == "hf:hf:org/m:novita:30.0"
    assert [t for t, _ in seen] == ["studio", "vertex"]           # hf never touched a genai client
    monkeypatch.delenv("GEMINI_VERTEX_KEY")
    c2 = GK.RotatingGeminiClient(client_factory=lambda k, h: C("studio"))
    with pytest.raises(RuntimeError, match="GEMINI_VERTEX_KEY"):
        c2.models.generate_content(model="vertex:x", contents="hi")


def test_r96_hf_router_messages_and_guards(monkeypatch):
    from PIL import Image

    from cvp.models import hf_router as HR
    monkeypatch.delenv("HF_TOKEN", raising=False)
    assert not HR.available() and HR.is_hf_id("hf:a/b") and not HR.is_hf_id("gemini-3.8-flash")
    assert HR.strip_prefix("hf:a/b:novita") == "a/b:novita"
    with pytest.raises(RuntimeError, match="HF_TOKEN"):
        HR.generate("hf:a/b", "x")
    img = Image.new("RGB", (1600, 900), "red")
    part = HR.image_part(img, max_side=448)
    assert part["type"] == "image_url" and part["image_url"]["url"].startswith("data:image/jpeg;base64,")
    text, images = HR.split_contents(["prompt", img, img])
    assert text == "prompt" and len(images) == 2
    msgs = HR.build_messages(text, images)
    assert msgs[0]["role"] == "user" and msgs[0]["content"][0] == {"type": "text", "text": "prompt"}
    assert len(msgs[0]["content"]) == 3
    monkeypatch.setenv("HF_TOKEN", "hk")

    class FakeClient:
        def __init__(self, token, timeout):
            self.timeout = timeout

        def chat_completion(self, model, messages, max_tokens, temperature):
            msg = type("Msg", (), {"content": f"  ans from {model} / {len(messages[0]['content'])} parts "})()
            return type("Resp", (), {"choices": [type("Ch", (), {"message": msg})()]})()

    import sys
    import types
    try:
        import huggingface_hub as HH
    except ImportError:  # pragma: no cover — CI without the hub client
        HH = types.ModuleType("huggingface_hub")
        monkeypatch.setitem(sys.modules, "huggingface_hub", HH)
    monkeypatch.setattr(HH, "InferenceClient", FakeClient, raising=False)
    HR._CLIENTS.clear()
    out = HR.generate("hf:org/m:deepinfra", ["q", img], timeout_s=20)
    assert out.text == "ans from org/m:deepinfra / 2 parts" and out.model == "org/m:deepinfra"


def test_r96_build_client_caps_sdk_retries_only_with_the_breaker(monkeypatch):
    from cvp.models import gemini_keys as GK
    from cvp.models.gemini_health import HEALTH
    monkeypatch.setenv("GEMINI_API_KEY", "k1")
    HEALTH.enabled = False
    assert GK.build_client(45.0)._http == {"timeout": 45000}
    HEALTH.enabled = True
    try:
        http = GK.build_client(45.0)._http
        assert http["timeout"] == 45000 and http["retry_options"]["attempts"] == 2
    finally:
        HEALTH.enabled = False


def test_r96_build_genai_degrades_http_options(monkeypatch):
    from cvp.models import gemini_keys as GK
    calls = []

    class FakeGenai:
        class Client:
            def __init__(self, http_options=None, **kw):
                calls.append(http_options)
                if http_options and "retry_options" in http_options:
                    raise ValueError("retry_options: extra fields not permitted")
                self.kw = kw

    import sys
    monkeypatch.setitem(sys.modules, "google", type("G", (), {"genai": FakeGenai})())
    monkeypatch.setitem(sys.modules, "google.genai", FakeGenai)
    c = GK._build_genai({"timeout": 1000, "retry_options": {"attempts": 2}}, api_key="k")
    assert c.kw == {"api_key": "k"} and calls == [{"timeout": 1000, "retry_options": {"attempts": 2}},
                                                  {"timeout": 1000}]


# ── 7. notebook pins ────────────────────────────────────────────────────────

def _nb_cells(name: str) -> list[str]:
    nb = json.loads((REPO / "notebooks" / name).read_text(encoding="utf-8"))
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def test_r96_nb03_battle_knobs():
    cells = _nb_cells("03_test_system.ipynb")
    engine = next(s for s in cells if "engine = SearchEngine(settings)" in s)
    pack = next(s for s in cells if "RUN_PACK" in s and "SHARD_TOTAL" in s)
    assert "STORM_BREAKER = True" in engine and 'os.environ["CVP_BREAKER__ENABLED"]' in engine
    assert 'LOCAL_VLM_ID = ""' in engine                      # plan B off until a benched id
    assert '"CVP_SEARCH__VLM_RERANK_LOCAL_FALLBACK"] = "true"' in engine
    assert 'os.environ["CVP_SUBMISSION__QUERY_ORDER"] = "kis_first"' in engine
    assert 'HF_ROUTER_MODEL = ""' in engine and 'os.environ["CVP_QUERY__GEMINI_MODEL_FALLBACKS"]' in engine
    env_cell = next(s for s in cells if "userdata.get(_sec)" in s)
    assert '"GEMINI_VERTEX_KEY"' in env_cell and '"HF_TOKEN"' in env_cell
    assert 'CLOSE_TIME = ""' in pack and 'RESCUE_QA_MODE = "head"' in pack
    assert "governor_minutes(CLOSE_TIME)" in pack and "KHÔNG có thống đốc thời gian" in pack
    assert "rescue_fallback_qa(_out, mode=RESCUE_QA_MODE)" in pack
    # per-shard Drive folders: shard writes ONLY to <pack>__shard<k>, gom unions <pack>*
    assert '_drv_shard = _drv.parent / f"{_pack_name}__shard{SHARD_INDEX}"' in pack
    assert "_d2 = _drv_shard / _c.name" in pack and "_sh.copy2(_c, _drv_shard / _c.name)" in pack
    assert 'd.name.startswith(f"{_pack_name}__shard")' in pack
    assert "_drv.mkdir(parents=True, exist_ok=True)" in pack     # gom still creates <pack>/ for the zip
    # ...and the shard branch never creates the shared <pack>/ folder any more
    shard_branch = pack[pack.index("if SHARD_TOTAL > 1:"):pack.index("if RESCUE_QA and RESUME_PACK:")]
    assert "_drv.mkdir(" not in shard_branch and "_drv_shard.mkdir(" in shard_branch
    # the battle line-up itself is unchanged
    assert "QA_PARALLEL = 2" in engine and 'LINEUP = "battle"' in engine
    assert "VLM_VOTES = 3" in engine and "QA_VOTES  = 3" in engine


def test_r96_nb09_new_arms_and_gates():
    cells = _nb_cells("09_campaign.ipynb")
    camp = next(s for s in cells if "_ARM_ORDER" in s)
    for arm in ("ABK+BREAKER", "ABK+OCRCTX", "ABK+F6", "ABK+LOCALR", "ABK+LOCALQA"):
        assert f'"{arm}"' in camp, arm
    assert 'LOCAL_VLM_ID = "Qwen/Qwen3-VL-8B-Instruct"' in camp       # verified on HF 05/09/2026
    assert 'LOCAL_VLM_ID_2 = "Qwen/Qwen3.5-9B"' in camp
    assert 'HF_ROUTER_QA_MODEL = "Qwen/Qwen3-VL-235B-A22B-Instruct"' in camp
    assert 'HF_ROUTER_RERANK_MODEL = "zai-org/GLM-5.3-Flash"' in camp
    assert "_LOCAL_ARMS = tuple(_LOCAL_ARM_ID)" in camp and '_HF_ARMS = ("ABK+HFQA", "ABK+HFR")' in camp
    for arm in ("ABK+LOCALR2", "ABK+LOCALQA2", "ABK+HFQA", "ABK+HFR"):
        assert f'"{arm}"' in camp, arm
    assert '"ABK+BREAKER": {"CVP_BREAKER__ENABLED": "true"}' in camp
    assert '"ABK+OCRCTX": {"CVP_VQA__OCR_CONTEXT": "true"}' in camp
    assert '"ABK+F6": {"CVP_VQA__FRAMES_PER_ANSWER": "6"}' in camp
    assert "_HEALTH.reset()" in camp and '"breaker": _HEALTH.snapshot()' in camp
    assert "_storm.local_loads < 1" in camp and "Gemini Pro vẫn được gọi" in camp
    assert "LOCAL_LOAD = \"Loading local hf_auto VLM\"" in camp
    # order: local arms last (they load ~17 GiB more), breaker/ocr/f6 before the G38 arms
    order = camp[camp.index("_ARM_ORDER = ("):camp.index("_LOCAL_ARMS")]
    assert order.index("ABK+BREAKER") < order.index("ABK+G38QA") < order.index("ABK+LOCALR")
