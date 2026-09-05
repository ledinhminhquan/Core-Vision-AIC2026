"""Round-89: DRY-RUN of the nb09 campaign cell — the real cell source executes
here with the heavy parts stubbed (engine, run_auto, scorer, Gemini client,
CUDA, the two tuner scripts), across the situations a real Colab session
hits: a fresh 13-arm run, a same-kernel re-run, a new session with every
arm done, a new session with a pending arm (ABK re-measured, old ABK rotated
to ABK-prev.json), a dead model (preflight fails loud, arm not saved), a
model that keeps falling back (arm not saved), a tuner with no in-sample win
(ABK+TUNED self-skips) and a clear winner (verdict rule fires).

A crash anywhere in the cell — including the summary block that only runs
after 9 hours — fails this test in seconds.
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
STEMS = ["query-p1-1-kis", "query-p1-2-kis", "query-p1-3-qa", "query-p1-4-trake",
         "query-p1-5-kis", "query-p1-6-qa"]
TEXTS = {
    "query-p1-1-kis": "Một người đàn ông mặc áo đỏ đi xe đạp trên cầu.",
    "query-p1-2-kis": "Sân khấu với dòng chữ nổi 3D có nội dung SẮC CỔ.",
    "query-p1-3-qa": "Đoạn phim về một người đang cân cá.\nCon số hiển thị trên cân là bao nhiêu?",
    "query-p1-4-trake": "E1: xe vào. E2: xe dừng. E3: người xuống xe.",
    "query-p1-5-kis": "Cảnh biển buổi sáng, thuyền đánh cá ra khơi.",
    "query-p1-6-qa": "Người dẫn chương trình nói rằng buổi lễ bắt đầu.\nBuổi lễ diễn ra ở đâu?",
}
ARMS = ("TUNE", "ABK", "ABK+TUNED", "ABK+W", "ABK+RRF", "DIVERSE", "ABK+V5",
        "MERGE2", "MERGE3", "MERGE_SIB", "MERGE2_NOHEDGE",
        "ABK+BREAKER", "ABK+OCRCTX", "ABK+F6",            # round-96
        "ABK+G38R", "ABK+G38QA",
        "ABK+HFQA", "ABK+HFR")                            # round-96: HF_TOKEN set in the World
LOCAL_ARMS = ("ABK+LOCALR", "ABK+LOCALQA")                # round-96: only with LOCAL_VLM_ID
LOCAL_ARMS_2 = ("ABK+LOCALR2", "ABK+LOCALQA2")            # round-96: only with LOCAL_VLM_ID_2


def _campaign_src() -> str:
    nb = json.loads((REPO / "notebooks" / "09_campaign.ipynb").read_text(encoding="utf-8"))
    return next("".join(c["source"]) for c in nb["cells"]
                if c["cell_type"] == "code" and "_ARM_ORDER" in "".join(c["source"]))


class World:
    """One Drive + VM sandbox with every stub installed; ``run`` executes the cell."""

    def __init__(self, tmp_path: Path, monkeypatch):
        self.project = tmp_path / "drive"
        self.local = tmp_path / "local"
        self.trial = self.project / "queries" / "p1"
        self.gt = self.project / "queries" / "gt-thunghiem.json"
        self.camp = self.project / "artifacts" / "lab" / "campaign"
        for d in (self.project / "artifacts" / "tuning", self.project / "artifacts" / "lab",
                  self.trial, self.local):
            d.mkdir(parents=True, exist_ok=True)
        for stem, t in TEXTS.items():
            (self.trial / f"{stem}.txt").write_text(t, encoding="utf-8")
        gt = {}
        for i, stem in enumerate(STEMS):
            vid = f"L01_V{i + 1:03d}"
            if stem.endswith("kis"):
                gt[stem] = {"task": "kis", "video_id": vid, "frame_start": 90, "frame_end": 110}
            elif stem.endswith("qa"):
                gt[stem] = {"task": "qa", "video_id": vid, "range": [90, 110], "answers": ["7"]}
            else:
                gt[stem] = {"task": "trake", "video_id": vid,
                            "moments": [[10, 20], [30, 40], [50, 60]]}
        self.gt.write_text(json.dumps(gt), encoding="utf-8")
        (self.project / "artifacts" / "tuning" / "best_weights.json").write_text(json.dumps(
            {"best": {"weights": {"visual": 1.0, "ocr": 0.6, "asr": 0.2, "caption": 0.2,
                                  "metadata": 0.1, "object": 0.3}}}), encoding="utf-8")
        (self.project / "artifacts" / "lab" / "bench_full.json").write_text(json.dumps(
            {"bench_pack": "ABK", "mean_final": 0.61,
             "per_query": {s: {"task": s.rsplit("-", 1)[1], "final": 0.6} for s in STEMS}}),
            encoding="utf-8")
        for k in list(os.environ):
            if k.startswith("CVP_"):
                monkeypatch.delenv(k)
        monkeypatch.setenv("CVP_PATHS__ARTIFACTS_ROOT", str(self.local))
        monkeypatch.setenv("GEMINI_API_KEY", "dry-run")
        monkeypatch.setenv("HF_TOKEN", "dry-run")       # round-96: HF arms run by default
        monkeypatch.delenv("GEMINI_VERTEX_KEY", raising=False)
        # scenario knobs
        self.dead_models: set[str] = set()
        self.fallback_storm: dict[str, int] = {}
        self.tuner_delta = 0.03
        self.score_bias: dict[str, float] = {}
        self.lane_short_times = 0      # first N engine builds drop metaclip2 (DriveFS EIO)
        self.quota_storm = 0           # Pro "exceeded your current quota" warnings per arm
        self.flaky_preflight: dict[str, int] = {}   # model -> initial 503s before it answers
        self.http_profile: dict[str, tuple] = {}    # model -> (n_200, n_503) httpx lines per run
        self.lane_short_forever = False
        # round-96 scenario knobs
        self.local_id = ""                          # LOCAL_VLM_ID value substituted into the cell
        self.local_id_2 = ""                        # LOCAL_VLM_ID_2 (second local model)
        self.local_model_loads = True               # fake run logs "Loading local hf_auto VLM"
        self.local_pro_calls = 0                    # httpx Pro 200s emitted during LOCAL arms
        self.hf_missing_ids: set[str] = set()       # model_info raises 404 for these
        self._install(monkeypatch)

    # ── stubs ──────────────────────────────────────────────────────────
    def _install(self, mp):
        import torch
        mp.setattr(time, "sleep", lambda s: None)
        mp.setattr(torch.cuda, "is_available", lambda: True)
        mp.setattr(torch.cuda, "get_device_name", lambda i=0: "dry-run-gpu")
        mp.setattr(torch.cuda, "mem_get_info", lambda: (30 * 2**30, 40 * 2**30))
        mp.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
        mp.setattr(torch.cuda, "max_memory_allocated", lambda: 7 * 2**30)
        mp.setattr(torch.cuda, "empty_cache", lambda: None)
        world = self

        import cvp.search.engine as E

        class FakeEngine:
            def __init__(self, settings):
                self.settings = settings
                self.member_names = list(settings.embedding.ensemble_members)
                if world.lane_short_forever or world.lane_short_times > 0:
                    world.lane_short_times -= 1
                    self.member_names = [m for m in self.member_names if m != "metaclip2"]

            def search_prepared(self, text, skip_rerank=False, cue_text=None):
                return []

        mp.setattr(E, "SearchEngine", FakeEngine)
        import cvp.search.cross_rerank as CR
        mp.setattr(CR, "_RERANKER", None, raising=False)
        mp.setattr(CR, "_RERANKER_KEY", None, raising=False)
        mp.setattr(CR, "_get_reranker", lambda settings: object())

        import cvp.pipeline.auto_agent as AA

        def fake_run_auto(query_dir, out_dir, settings, submit=False, resume=False,
                          engine_factory=None):
            out = Path(out_dir)
            out.mkdir(parents=True, exist_ok=True)
            vlog = logging.getLogger("cvp.search.vqa")
            for m, n in world.fallback_storm.items():
                for _ in range(n):
                    vlog.warning("Gemini model %r failed (503 UNAVAILABLE) — trying next", m)
            for _ in range(world.quota_storm):
                vlog.warning("Gemini model 'gemini-3.1-pro-preview' failed (429 RESOURCE_EXHAUSTED. "
                             "You exceeded your current quota, please check your plan) — trying next")
            hlog = logging.getLogger("httpx")
            hlog.setLevel(logging.INFO)
            if os.environ.get("CVP_VQA__LOCAL_HF_ID"):          # round-96: a LOCAL arm is running
                llog = logging.getLogger("cvp.models.local_vlm")
                llog.setLevel(logging.INFO)
                if world.local_model_loads:
                    llog.info("Loading local hf_auto VLM %s (plan B khi Gemini bão)",
                              os.environ["CVP_VQA__LOCAL_HF_ID"])
                for _ in range(world.local_pro_calls):
                    hlog.info('HTTP Request: POST https://generativelanguage.googleapis.com/'
                              'v1beta/models/gemini-3.1-pro-preview:generateContent "HTTP/1.1 200 OK"')
            for m, (n_ok, n_bad) in world.http_profile.items():
                for code, n in (("200 OK", n_ok), ("503 Service Unavailable", n_bad)):
                    for _ in range(n):
                        hlog.info('HTTP Request: POST https://generativelanguage.googleapis.com/'
                                  'v1beta/models/%s:generateContent "HTTP/1.1 %s"', m, code)
            for i, stem in enumerate(STEMS):
                vid = f"L01_V{i + 1:03d}"
                if stem.endswith("qa"):
                    rows = [f"{vid},{100 + j},7" for j in range(3)]
                elif stem.endswith("trake"):
                    rows = [f"{vid},{10 + j},{30 + j},{50 + j}" for j in range(3)]
                else:
                    rows = [f"{vid},{100 + j}" for j in range(3)] + [f"L01_V099,{j}" for j in range(2)]
                (out / f"{stem}.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")

            class Rep:
                failed = {}
                written = sorted(out.glob("*.csv"))

            return Rep()

        mp.setattr(AA, "run_auto", fake_run_auto)
        import cvp.eval.official as OF

        def fake_score_run(sub_dir, gt_path):
            bias = sum(b for marker, b in world.score_bias.items()
                       if any(v == marker for v in os.environ.values()))
            pq = {s: {"task": s.rsplit("-", 1)[1],
                      "final": round(min(1.0, 0.5 + 0.05 * i + bias), 3)}
                  for i, s in enumerate(STEMS)}
            mean = round(sum(q["final"] for q in pq.values()) / len(pq), 4)

            class R:
                def to_dict(self_):
                    return {"mean_final": mean, "num_gt": len(STEMS), "num_scored": len(STEMS),
                            "by_task": {"kis": mean, "qa": mean, "trake": mean},
                            "per_query": pq}

            return R()

        mp.setattr(OF, "score_run", fake_score_run)
        import cvp.search.vqa as V

        class FakeModels:
            def generate_content(self_, model, contents, config=None):
                if model in world.dead_models:
                    raise RuntimeError("404 NOT_FOUND: model not found for this key")
                if world.flaky_preflight.get(model, 0) > 0:
                    world.flaky_preflight[model] -= 1
                    raise RuntimeError("503 UNAVAILABLE. This model is currently experiencing high demand")
                return type("Resp", (), {"text": "pong"})()

        class FakeClient:
            models = FakeModels()

        mp.setattr(V, "make_gemini_client", lambda settings: FakeClient())
        # CI is "pure CPU, no network" and has no huggingface_hub — the cell only
        # needs model_info(); provide a stub module when the real one is absent.
        import sys
        import types
        try:
            import huggingface_hub as HH
        except ImportError:  # pragma: no cover — CI without the hub client
            HH = types.ModuleType("huggingface_hub")
            mp.setitem(sys.modules, "huggingface_hub", HH)

        def fake_model_info(model_id, *a, **k):
            if model_id in world.hf_missing_ids:
                raise RuntimeError("404 Client Error. Repository Not Found for url")
            return {"id": model_id}

        mp.setattr(HH, "model_info", fake_model_info, raising=False)

    def _run_script(self, *args):
        a = [str(x) for x in args]
        if "23_dump_signals" in a[0]:
            out = Path(a[a.index("--out-dir") + 1])
            out.mkdir(parents=True, exist_ok=True)
            sig = {}
            for i, s in enumerate(STEMS):
                vid = f"L01_V{i + 1:03d}"
                rows = {f"{vid},{100 + j}": 1.0 - 0.1 * j for j in range(5)}
                rows.update({f"L01_V099,{j}": 0.6 - 0.1 * j for j in range(3)})
                sig[s] = {"visual": rows, "ocr": {k: v * 0.5 for k, v in rows.items()},
                          "asr": {k: v * 0.3 for k, v in rows.items()}}
            (out / "signals.json").write_text(json.dumps(sig), encoding="utf-8")
        elif "21_tune_weights" in a[0]:
            out = Path(a[a.index("--out") + 1])
            best = {"visual": 1.0, "ocr": 0.8, "asr": 0.1, "caption": 0.2, "metadata": 0.1,
                    "object": 0.3}
            out.write_text(json.dumps({
                "best": {"weights": best, "score": 0.7},
                "baseline": {"weights": {}, "score": round(0.7 - self.tuner_delta, 4)},
                "delta": self.tuner_delta, "per_task_delta": {"kis": self.tuner_delta},
                "trials_evaluated": 400, "queries": len(STEMS), "skipped_queries": [],
                "active_signals": ["ocr", "asr"], "scorer": "dry"}), encoding="utf-8")
        else:
            raise AssertionError(f"unexpected _run {a}")

    # ── execute the cell ───────────────────────────────────────────────
    def run(self, g: dict | None = None, arms="all") -> dict:
        g = {} if g is None else g
        g.update({"PROJECT": self.project, "REPO_DIR": REPO, "TRIAL_DIR": self.trial,
                  "GT_PATH": self.gt, "_run": self._run_script})
        src = _campaign_src().replace('ARMS = "all"', f"ARMS = {arms!r}", 1)
        src = src.replace('LOCAL_VLM_ID = "Qwen/Qwen3-VL-8B-Instruct"',
                          f"LOCAL_VLM_ID = {self.local_id!r}", 1)
        src = src.replace('LOCAL_VLM_ID_2 = "Qwen/Qwen3.5-9B"',
                          f"LOCAL_VLM_ID_2 = {self.local_id_2!r}", 1)
        assert f"LOCAL_VLM_ID = {self.local_id!r}" in src and f"LOCAL_VLM_ID_2 = {self.local_id_2!r}" in src
        exec(compile(src, "nb09-campaign-cell", "exec"), g)   # noqa: S102 — the cell itself
        return g

    def payload(self, arm: str) -> dict:
        return json.loads((self.camp / f"{arm}.json").read_text(encoding="utf-8"))


def test_dryrun_fresh_session_runs_all_13_arms_and_the_summary(tmp_path, monkeypatch):
    w = World(tmp_path, monkeypatch)
    g = w.run()
    for arm in ARMS:
        assert (w.camp / f"{arm}.json").exists(), arm
    assert all(any((w.camp / f"{a}_run").glob("query-*.csv")) for a in ARMS if a != "TUNE")
    tune = w.payload("TUNE")
    assert tune["cv"]["folds"] == 16 and tune["report"]["delta"] == 0.03
    assert tune["battle_weights_untouched"] is True
    g38 = w.payload("ABK+G38QA")
    assert g38["declared_model"] == "gemini-3.8-flash" and g38["model_fallbacks"] == {}
    assert g38["env"]["CVP_VQA__ANSWER_MODEL"] == "gemini-3.8-flash"
    assert w.payload("ABK")["env"]["CVP_VQA__ANSWER_MODEL"] == "gemini-3.1-pro-preview"
    assert w.payload("ABK+W")["cue_hits"] == ["query-p1-2-kis", "query-p1-3-qa", "query-p1-6-qa"]
    assert w.payload("DIVERSE")["env"]["CVP_EMBEDDING__ENSEMBLE_MEMBERS"].count("qwen_embed") == 1
    summ = json.loads((w.camp / "campaign_summary.json").read_text(encoding="utf-8"))
    assert set(summ["arms"]) == set(ARMS) and "ABK_prev" in summ["abk_draws"]
    assert (w.camp / f"campaign_summary-{g['SESSION']}.json").exists()
    assert not list(w.camp.glob("ABK-prev*.json"))
    # the tuning file on Drive was only read
    assert json.loads((w.project / "artifacts" / "tuning" / "best_weights.json")
                      .read_text(encoding="utf-8"))["best"]["weights"]["ocr"] == 0.6


def test_dryrun_same_kernel_rerun_skips_everything(tmp_path, monkeypatch):
    w = World(tmp_path, monkeypatch)
    g = w.run()
    stamp = w.payload("ABK")["ended_at"]
    w.run(g)                                             # same globals → same SESSION
    assert w.payload("ABK")["ended_at"] == stamp
    assert not list(w.camp.glob("ABK-prev*.json"))
    assert not list(w.camp.glob("*_run-prev*"))


def test_dryrun_new_session_with_everything_done_measures_nothing(tmp_path, monkeypatch):
    w = World(tmp_path, monkeypatch)
    w.run()
    s1 = w.payload("ABK")["session"]
    g2 = w.run({})                                       # new kernel → new SESSION
    assert g2["SESSION"] != s1
    assert w.payload("ABK")["session"] == s1             # no re-measure without pending arms
    assert not list(w.camp.glob("ABK-prev*.json"))
    summ = json.loads((w.camp / "campaign_summary.json").read_text(encoding="utf-8"))
    assert all("≠phiên" not in " ".join(f) for f in summ["flags"].values())


def test_dryrun_new_session_with_a_pending_arm_remeasures_abk(tmp_path, monkeypatch):
    w = World(tmp_path, monkeypatch)
    w.run()
    s1 = w.payload("ABK")["session"]
    (w.camp / "ABK+G38QA.json").unlink()                 # one arm still to measure
    g2 = w.run({})
    assert w.payload("ABK")["session"] == g2["SESSION"] != s1
    assert w.payload("ABK-prev")["session"] == s1        # old baseline kept as a draw
    assert (w.camp / "ABK_run-prev").is_dir() and (w.camp / "ABK_run").is_dir()
    assert w.payload("ABK+G38QA")["session"] == g2["SESSION"]
    summ = json.loads((w.camp / "campaign_summary.json").read_text(encoding="utf-8"))
    assert "ABK-prev" in summ["abk_draws"]
    assert "≠phiên" not in " ".join(summ["flags"]["ABK+G38QA"])
    assert any(f.startswith("so với ABK phiên") for f in summ["flags"]["ABK+W"])   # session-1 arm


def test_dryrun_dead_model_fails_loud_and_is_not_saved(tmp_path, monkeypatch):
    w = World(tmp_path, monkeypatch)
    w.dead_models = {"gemini-3.8-flash"}
    w.run()
    assert not (w.camp / "ABK+G38R.json").exists()
    assert not (w.camp / "ABK+G38QA.json").exists()
    assert (w.camp / "ABK+V5.json").exists()             # the rest of the campaign went on


def test_dryrun_model_that_keeps_falling_back_is_not_saved(tmp_path, monkeypatch):
    w = World(tmp_path, monkeypatch)
    w.fallback_storm = {"gemini-3.8-flash": 3}           # > len(queries)//4 == 1
    w.run()
    assert not (w.camp / "ABK+G38R.json").exists()
    assert not (w.camp / "ABK+G38QA.json").exists()
    assert w.payload("ABK")["model_fallbacks"] == {"gemini-3.8-flash": 3}   # recorded everywhere


def test_dryrun_tuner_without_in_sample_win_skips_abk_tuned(tmp_path, monkeypatch):
    w = World(tmp_path, monkeypatch)
    w.tuner_delta = -0.01
    w.run()
    p = w.payload("ABK+TUNED")
    assert p["skipped"] is True and "tuner không thắng" in p["reason"]


def test_dryrun_clear_winner_passes_the_verdict_rule(tmp_path, monkeypatch):
    w = World(tmp_path, monkeypatch)
    w.score_bias = {"gemini-3.8-flash": 0.08}            # both G38 arms +0.08 on every query
    w.run()
    summ = json.loads((w.camp / "campaign_summary.json").read_text(encoding="utf-8"))
    assert "ABK+G38R" in summ["wins"] and "ABK+G38QA" in summ["wins"]
    assert "ABK+RRF" not in summ["wins"]


def test_dryrun_transient_lane_drop_is_retried_not_fatal(tmp_path, monkeypatch):
    w = World(tmp_path, monkeypatch)
    w.lane_short_times = 1                               # first build short, second fine
    w.run()
    assert (w.camp / "ABK.json").exists()
    assert w.payload("ABK")["env"]["CVP_EMBEDDING__ENSEMBLE_MEMBERS"].count("metaclip2") == 1


def test_dryrun_failed_baseline_in_new_session_keeps_old_arms_and_baseline(tmp_path, monkeypatch):
    w = World(tmp_path, monkeypatch)
    w.run()
    s1 = w.payload("ABK")["session"]
    (w.camp / "ABK+G38QA.json").unlink()                 # pending arm → re-measure wanted
    w.lane_short_forever = True                          # ...but the engine never loads the lane
    w.run({})
    assert w.payload("ABK")["session"] == s1             # old baseline untouched, not rotated
    assert not list(w.camp.glob("ABK-prev*.json"))
    assert not (w.camp / "ABK+G38QA.json").exists()      # pending arm correctly not measured
    summ = json.loads((w.camp / "campaign_summary.json").read_text(encoding="utf-8"))
    assert summ["base_abk"] == w.payload("ABK")["mean_final"]
    assert "ABK+W" in summ["arms"] and "MERGE2" in summ["arms"]   # session-1 arms still visible
    assert "≠phiên" not in " ".join(summ["flags"].get("ABK+W", []))


def test_dryrun_daily_quota_exhaustion_is_flagged_and_never_wins(tmp_path, monkeypatch):
    w = World(tmp_path, monkeypatch)
    w.score_bias = {"gemini-3.8-flash": 0.08}            # would win on score alone...
    w.quota_storm = 4                                    # ...but Pro quota ran out
    w.run()
    p = w.payload("ABK+G38R")
    assert p["quota_429"] == 4 and p["model_fallbacks"] == {"gemini-3.1-pro-preview": 4}
    summ = json.loads((w.camp / "campaign_summary.json").read_text(encoding="utf-8"))
    assert any(f.startswith("hết quota") for f in summ["flags"]["ABK+G38R"])
    assert "ABK+G38R" not in summ["wins"] and "ABK+G38QA" not in summ["wins"]


def test_dryrun_preflight_survives_a_transient_503_storm(tmp_path, monkeypatch):
    w = World(tmp_path, monkeypatch)
    w.flaky_preflight = {"gemini-3.8-flash": 4}          # 4 x 503 then it answers
    w.run()
    assert (w.camp / "ABK+G38QA.json").exists()          # retried, then measured
    assert w.flaky_preflight["gemini-3.8-flash"] == 0


def test_dryrun_declared_model_share_gate(tmp_path, monkeypatch):
    # ≥90% self-answered → eligible; 75-90% → saved but flagged; <75% → not saved
    w = World(tmp_path, monkeypatch)
    w.score_bias = {"gemini-3.8-flash": 0.08}
    w.http_profile = {"gemini-3.8-flash": (95, 5), "gemini-3.1-pro-preview": (60, 0)}
    w.run()
    p = w.payload("ABK+G38QA")
    assert p["declared_share"] == 0.95 and p["http_by_model"]["gemini-3.8-flash"] == {"200": 95, "503": 5}
    summ = json.loads((w.camp / "campaign_summary.json").read_text(encoding="utf-8"))
    assert "ABK+G38QA" in summ["wins"] and not summ["flags"].get("ABK+G38QA")

    w2 = World(tmp_path / "b", monkeypatch)
    w2.score_bias = {"gemini-3.8-flash": 0.08}
    w2.http_profile = {"gemini-3.8-flash": (80, 20)}
    w2.run()
    assert w2.payload("ABK+G38R")["declared_share"] == 0.8
    summ2 = json.loads((w2.camp / "campaign_summary.json").read_text(encoding="utf-8"))
    assert any(f.startswith("rớt model") for f in summ2["flags"]["ABK+G38R"])
    assert "ABK+G38R" not in summ2["wins"]

    w3 = World(tmp_path / "c", monkeypatch)
    w3.http_profile = {"gemini-3.8-flash": (50, 50)}
    w3.run()
    assert not (w3.camp / "ABK+G38R.json").exists() and not (w3.camp / "ABK+G38QA.json").exists()


# ── round-96 scenarios ─────────────────────────────────────────────────────

def test_dryrun_r96_new_arms_run_and_local_arms_wait_for_an_id(tmp_path, monkeypatch):
    w = World(tmp_path, monkeypatch)
    w.run()
    for arm in ("ABK+BREAKER", "ABK+OCRCTX", "ABK+F6"):
        assert (w.camp / f"{arm}.json").exists(), arm
    for arm in LOCAL_ARMS + LOCAL_ARMS_2:
        assert not (w.camp / f"{arm}.json").exists(), arm      # skipped, NOT saved as done
    b = w.payload("ABK+BREAKER")
    assert b["env"]["CVP_BREAKER__ENABLED"] == "true" and isinstance(b["breaker"], dict)
    assert w.payload("ABK+OCRCTX")["env"]["CVP_VQA__OCR_CONTEXT"] == "true"
    assert w.payload("ABK+F6")["env"]["CVP_VQA__FRAMES_PER_ANSWER"] == "6"
    assert "CVP_BREAKER__ENABLED" not in w.payload("ABK")["env"]   # breaker OFF elsewhere
    assert w.payload("ABK")["local_loads"] == 0
    summ = json.loads((w.camp / "campaign_summary.json").read_text(encoding="utf-8"))
    assert "ABK+BREAKER" in summ["arms"] and "ABK+BREAKER" not in summ["wins"]   # = ABK, no win


def test_dryrun_r96_local_arms_measure_with_a_verified_id(tmp_path, monkeypatch):
    w = World(tmp_path, monkeypatch)
    w.local_id = "org/verified-vlm"
    w.run()
    for arm in LOCAL_ARMS:
        p = w.payload(arm)
        assert p["local_loads"] >= 1 and p["env"]["CVP_VQA__LOCAL_HF_ID"] == "org/verified-vlm"
    assert w.payload("ABK+LOCALR")["env"]["CVP_SEARCH__VLM_RERANK_PROVIDER"] == "hf_auto"
    assert w.payload("ABK+LOCALQA")["env"]["CVP_VQA__PROVIDER"] == "local"
    assert "CVP_VQA__LOCAL_HF_ID" not in w.payload("ABK+G38QA")["env"]    # env cleaned between arms


def test_dryrun_r96_local_arm_not_saved_when_model_never_loads_or_pro_answers(tmp_path, monkeypatch):
    w = World(tmp_path, monkeypatch)
    w.local_id = "org/verified-vlm"
    w.local_model_loads = False
    w.run()
    for arm in LOCAL_ARMS:
        assert not (w.camp / f"{arm}.json").exists(), arm
    assert (w.camp / "ABK+F6.json").exists()                     # campaign went on

    w2 = World(tmp_path / "b", monkeypatch)
    w2.local_id = "org/verified-vlm"
    w2.local_pro_calls = 5                                       # Gemini Pro still answered QA
    w2.run()
    assert (w2.camp / "ABK+LOCALR.json").exists()                # rerank arm may call Pro for QA
    assert not (w2.camp / "ABK+LOCALQA.json").exists()           # QA arm must not


def test_dryrun_r96_unknown_hf_id_fails_the_preflight_only(tmp_path, monkeypatch):
    w = World(tmp_path, monkeypatch)
    w.local_id = "org/does-not-exist"
    w.hf_missing_ids = {"org/does-not-exist"}
    w.run()
    for arm in LOCAL_ARMS:
        assert not (w.camp / f"{arm}.json").exists(), arm
    assert (w.camp / "ABK+G38QA.json").exists()


def test_dryrun_r96_hf_arms_need_the_token_and_pin_their_model(tmp_path, monkeypatch):
    w = World(tmp_path, monkeypatch)
    w.run()
    assert w.payload("ABK+HFQA")["env"]["CVP_VQA__ANSWER_MODEL"] == "hf:Qwen/Qwen3-VL-235B-A22B-Instruct"
    assert w.payload("ABK+HFR")["env"]["CVP_SEARCH__VLM_RERANK_MODEL"] == "hf:zai-org/GLM-5.3-Flash"
    assert w.payload("ABK+HFQA")["declared_model"] == "hf:Qwen/Qwen3-VL-235B-A22B-Instruct"
    assert w.payload("ABK")["env"]["CVP_VQA__ANSWER_MODEL"] == "gemini-3.1-pro-preview"

    w2 = World(tmp_path / "b", monkeypatch)
    monkeypatch.delenv("HF_TOKEN")
    w2.run()
    for arm in ("ABK+HFQA", "ABK+HFR"):
        assert not (w2.camp / f"{arm}.json").exists(), arm      # skipped, not saved as done
    assert (w2.camp / "ABK+F6.json").exists()


def test_dryrun_r96_second_local_id_gets_its_own_arms(tmp_path, monkeypatch):
    w = World(tmp_path, monkeypatch)
    w.local_id = ""
    w.local_id_2 = "org/second-vlm"
    w.run()
    for arm in LOCAL_ARMS:
        assert not (w.camp / f"{arm}.json").exists(), arm
    for arm in LOCAL_ARMS_2:
        assert w.payload(arm)["env"]["CVP_VQA__LOCAL_HF_ID"] == "org/second-vlm", arm
