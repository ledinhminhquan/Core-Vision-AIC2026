"""Adaptive multi-attempt (Nhiệm vụ 2) — cvp.pipeline.attempts + scripts/64.

Everything here is OFFLINE: submission-CSV folders and signal-dump JSONs built
in tmp_path, no engine, no API. The weighted N-run RRF merge must reproduce
``scripts/63_ensemble_runs`` row-for-row on 2 equal-weight runs — 63 is the
battle-tested baseline (round-43) and stays untouched.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from cvp.pipeline.attempts import (
    HIGH_EFFORT_ENV,
    QueryConfidence,
    answer_consensus,
    build_plan,
    diff_top1,
    load_run,
    load_signal_dumps,
    margin_confidence,
    merge_rows,
    plan_env,
    query_confidence,
    render_diff_markdown,
    rrf_merge_runs,
    select_weak,
    stage_weak_queries,
    video_concentration,
    write_merged,
)

REPO = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _write_run(root: Path, stems_rows: dict[str, list[list[str]]]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for stem, rows in stems_rows.items():
        with open(root / f"{stem}.csv", "w", encoding="utf-8", newline="") as f:
            csv.writer(f, lineterminator="\n").writerows(rows)
    return root


def _kis_rows(video: str, n: int, start: int = 100) -> list[list[str]]:
    return [[video, str(start + 50 * i)] for i in range(n)]


# ── confidence components ────────────────────────────────────────────────────


def test_margin_confidence_peaked_vs_flat_vs_absent():
    peaked = {f"V1,{i}": (10.0 if i == 0 else 1.0) for i in range(12)}
    flat = {f"V1,{i}": 3.0 for i in range(12)}
    assert margin_confidence(peaked) == pytest.approx((10.0 - 1.0) / 10.0)
    assert margin_confidence(flat) == 0.0
    assert margin_confidence({f"V1,{i}": 9.0 - i for i in range(5)}) == 0.0  # <10 → low
    assert margin_confidence(None) is None
    assert margin_confidence({}) is None


def test_video_concentration_concentrated_vs_scattered():
    assert video_concentration(_kis_rows("V1", 10)) == 1.0
    alternating = [["V1", "10"], ["V2", "20"]] * 5
    assert video_concentration(alternating) == 0.5
    assert video_concentration([]) == 0.0
    assert video_concentration([["V1", "10"]]) == 1.0
    # head window: rows beyond `head` never count
    rows = _kis_rows("V1", 10) + [["V9", str(i)] for i in range(90)]
    assert video_concentration(rows, head=10) == 1.0


def test_answer_consensus_majority_and_fallback():
    rows = [["V1", "10", "30 kg"]] * 7 + [["V1", "99", "300 kg"]] * 3
    assert answer_consensus(rows) == pytest.approx(0.7)
    # the fallback placeholder counts as NO answer but stays in the denominator
    diluted = [["V1", "10", "30 kg"]] * 6 + [["V1", "99", "không rõ"]] * 4
    assert answer_consensus(diluted) == pytest.approx(0.6)
    assert answer_consensus([["V1", "10", "không rõ"]] * 10) == 0.0
    assert answer_consensus([["V1", "10"]] * 10) == 0.0  # no answer column at all


def test_query_confidence_combines_and_renormalizes():
    # No dumps → margin absent → conf = concentration only (weights renormalize)
    kis = query_confidence("query-p1-1-kis", _kis_rows("V1", 12))
    assert kis.task == "kis"
    assert kis.components.keys() == {"concentration"}
    assert kis.confidence == pytest.approx(1.0)

    # Flat visual dump drags a concentrated ranking down (margin weighs in)
    flat_sig = {"visual": {f"V1,{i}": 3.0 for i in range(12)}}
    with_margin = query_confidence("query-p1-1-kis", _kis_rows("V1", 12), signals=flat_sig)
    assert with_margin.components["margin"] == 0.0
    assert with_margin.confidence == pytest.approx((0.5 * 0.0 + 0.3 * 1.0) / 0.8)

    # QA adds answer consensus; all-fallback answers → that component is 0
    qa_rows = [["V1", str(100 + i), "không rõ"] for i in range(10)]
    qa = query_confidence("query-p1-2-qa", qa_rows)
    assert qa.task == "qa"
    assert qa.components["answer_consensus"] == 0.0
    assert qa.confidence == pytest.approx((0.3 * 1.0 + 0.2 * 0.0) / 0.5)

    # Empty rows → zero confidence regardless of components
    assert query_confidence("query-p1-9-kis", []).confidence == 0.0


# ── weak selection + plan ────────────────────────────────────────────────────


def _conf(stem: str, c: float, task: str = "kis") -> QueryConfidence:
    return QueryConfidence(stem=stem, task=task, confidence=c)


def test_select_weak_threshold_orders_and_caps():
    confs = {
        "q-1-kis": _conf("q-1-kis", 0.9),
        "q-2-kis": _conf("q-2-kis", 0.30),
        "q-3-kis": _conf("q-3-kis", 0.10),
        "q-4-kis": _conf("q-4-kis", 0.20),
    }
    assert select_weak(confs, threshold=0.35, max_fraction=1.0) == \
        ["q-3-kis", "q-4-kis", "q-2-kis"]           # weakest first
    # budget cap: ceil(0.5 * 4) = 2 → only the two weakest go to attempt 2
    assert select_weak(confs, threshold=0.35, max_fraction=0.5) == \
        ["q-3-kis", "q-4-kis"]
    assert select_weak({}, threshold=0.35) == []
    assert select_weak(confs, threshold=0.05) == []  # nobody below → no rerun
    # explicit zero budget means NO rerun — never a floor of 1
    assert select_weak(confs, threshold=0.35, max_fraction=0.0) == []


def test_plan_env_merges_only_weak_tasks():
    only_kis = plan_env({"kis"})
    assert only_kis["CVP_SEARCH__LOW_CONFIDENCE_RETRY"] == "true"
    assert only_kis["CVP_SEARCH__VLM_RERANK_VOTES"] == "5"
    assert not any(k.startswith("CVP_TEMPORAL__") for k in only_kis)
    assert not any(k.startswith("CVP_VQA__") for k in only_kis)

    full = plan_env({"kis", "qa", "trake"})
    assert full["CVP_VQA__SELF_CONSISTENCY"] == "5"
    assert full["CVP_TEMPORAL__EVENT_QUERY_VARIANTS"] == "all"
    assert full["CVP_TEMPORAL__PER_EVENT_TOPK"] == "300"
    # every override key must carry the CVP_ env prefix (typos fail loud here)
    assert all(k.startswith("CVP_") for block in HIGH_EFFORT_ENV.values() for k in block)


def test_build_plan_flags_scattered_and_missing_queries(tmp_path):
    run = _write_run(tmp_path / "att1", {
        # strong: 12 rows on one video
        "query-p1-1-kis": _kis_rows("L01_V001", 12),
        # weak QA: scattered videos + fallback answers
        "query-p1-2-qa": [[f"L0{i % 5 + 1}_V00{i % 3 + 1}", str(100 + i), "không rõ"]
                          for i in range(10)],
        # weak TRAKE: every row a different video
        "query-p1-3-trake": [[f"L0{i}_V001", "10", "20", "30"] for i in range(1, 9)],
    })
    qdir = tmp_path / "queries"
    qdir.mkdir()
    for stem in ("query-p1-1-kis", "query-p1-2-qa", "query-p1-3-trake",
                 "query-p1-4-kis"):                    # p1-4 has NO csv (dropped)
        (qdir / f"{stem}.txt").write_text("cảnh thử nghiệm\n", encoding="utf-8")

    plan = build_plan(run, query_dir=qdir, threshold=0.35, max_fraction=1.0)
    assert plan["queries"]["query-p1-4-kis"]["confidence"] == 0.0
    assert plan["queries"]["query-p1-4-kis"]["components"] == {"missing_csv": 0.0}
    assert set(plan["weak"]) == {"query-p1-2-qa", "query-p1-3-trake", "query-p1-4-kis"}
    assert plan["weak"][0] == "query-p1-4-kis"        # conf 0 → weakest first
    assert "query-p1-1-kis" not in plan["weak"]
    # env merged from the weak tasks only (qa + trake + kis)
    assert plan["env"]["CVP_TEMPORAL__EVENT_QUERY_VARIANTS"] == "all"
    assert plan["env"]["CVP_VQA__SELF_CONSISTENCY"] == "5"


def test_build_plan_uses_signal_dumps_for_margin(tmp_path):
    run = _write_run(tmp_path / "att1", {"query-p1-1-kis": _kis_rows("L01_V001", 12)})
    dumps = tmp_path / "dumps"
    dumps.mkdir()
    (dumps / "query-p1-1-kis.json").write_text(json.dumps({
        "query-p1-1-kis": {"visual": {f"L01_V001,{i}": 3.0 for i in range(12)}}
    }), encoding="utf-8")
    (dumps / "corrupt.json").write_text("{not json", encoding="utf-8")  # warn+skip

    without = build_plan(run)
    with_dumps = build_plan(run, signals_dir=dumps)
    q0, q1 = (p["queries"]["query-p1-1-kis"] for p in (without, with_dumps))
    assert "margin" not in q0["components"]
    assert q1["components"]["margin"] == 0.0           # flat dump → zero margin
    assert q1["confidence"] < q0["confidence"]         # margin dragged it down


def test_load_signal_dumps_merges_files(tmp_path):
    d = tmp_path / "dumps"
    d.mkdir()
    (d / "a.json").write_text(json.dumps({"q1": {"visual": {"V1,10": 1.0}}}), encoding="utf-8")
    (d / "b.json").write_text(json.dumps({"q1": {"ocr": {"V1,10": 2.0}},
                                          "q2": {"visual": {"V2,20": 3.0}}}), encoding="utf-8")
    merged = load_signal_dumps(d)
    assert merged["q1"]["visual"]["V1,10"] == 1.0
    assert merged["q1"]["ocr"]["V1,10"] == 2.0
    assert merged["q2"]["visual"]["V2,20"] == 3.0


def test_stage_weak_queries_copies_only_weak(tmp_path):
    qdir = tmp_path / "queries"
    qdir.mkdir()
    for stem in ("q-1-kis", "q-2-kis"):
        (qdir / f"{stem}.txt").write_text("x\n", encoding="utf-8")
    staged = stage_weak_queries(qdir, ["q-2-kis", "q-ghost-kis"], tmp_path / "stage")
    assert [p.name for p in staged] == ["q-2-kis.txt"]  # ghost skipped with warning
    assert (tmp_path / "stage" / "q-2-kis.txt").is_file()
    assert not (tmp_path / "stage" / "q-1-kis.txt").exists()


# ── weighted N-run RRF merge ─────────────────────────────────────────────────


def test_equal_weight_two_run_merge_reproduces_scripts_63():
    m63 = _load_script("63_ensemble_runs")
    run_a = {
        "query-p1-1-kis": [["V1", "10"], ["V2", "20"], ["V3", "30"]],
        "query-p1-2-qa": [["V1", "10", "đáp A"], ["V5", "50", "x"]],
        "query-p1-3-trake": [["V1", "1", "2"], ["V1", "2", "9"]],
    }
    run_b = {
        "query-p1-1-kis": [["V2", "20"], ["V4", "40"], ["V1", "10"]],
        "query-p1-2-qa": [["V1", "10", "đáp B"], ["V6", "60", "y"]],
        "query-p1-3-trake": [["V1", "1", "3"], ["V1", "1", "2"]],
    }
    ours = rrf_merge_runs([run_a, run_b], k=60)
    for stem in run_a:
        task = next(t for t in ("trake", "avs", "qa", "kis") if t in stem)
        theirs = m63.rrf_merge(run_a[stem], run_b[stem], k=60, task=task)
        assert ours[stem] == theirs, f"diverged from scripts/63 on {stem}"


def test_weighted_merge_prefers_the_heavier_run():
    a = {"q-1-kis": [["V1", "10"]]}
    b = {"q-1-kis": [["V2", "20"]]}
    equal = rrf_merge_runs([a, b])["q-1-kis"]
    assert equal[0] == ["V1", "10"]                    # tie → primary run first
    heavy_b = rrf_merge_runs([a, b], weights=[1.0, 3.0])["q-1-kis"]
    assert heavy_b[0] == ["V2", "20"]                  # weight tips the tie
    assert {tuple(r) for r in heavy_b} == {("V1", "10"), ("V2", "20")}


def test_merge_keeps_answer_of_best_ranked_run():
    runs = [
        {"q-2-qa": [["V1", "10", "ans A"]]},                       # rank 0
        {"q-2-qa": [["V9", "5", "x"], ["V1", "10", "ans B"]]},     # rank 1
        {"q-2-qa": [["V1", "10", "ans C"]]},                       # rank 0 (tie)
    ]
    merged = rrf_merge_runs(runs)["q-2-qa"]
    assert merged[0] == ["V1", "10", "ans A"]          # rank tie → earlier run
    assert ["V9", "5", "x"] in merged


def test_merge_trake_identity_is_the_whole_tuple():
    a = {"q-3-trake": [["V1", "1", "2"]]}
    b = {"q-3-trake": [["V1", "1", "3"]]}
    merged = rrf_merge_runs([a, b])["q-3-trake"]
    assert len(merged) == 2                            # different tuples never dedupe


def test_merge_caps_rows_and_validates_weights():
    big = {"q-1-kis": [["V1", str(i)] for i in range(120)]}
    assert len(rrf_merge_runs([big])["q-1-kis"]) == 100
    assert len(merge_rows([big["q-1-kis"]], [1.0], 60, "kis", limit=5)) == 5
    with pytest.raises(ValueError, match="must match"):
        rrf_merge_runs([big, big], weights=[1.0])
    with pytest.raises(ValueError, match="> 0"):
        rrf_merge_runs([big], weights=[0.0])


def test_merge_union_of_stems_and_write_roundtrip(tmp_path):
    a = {"q-1-kis": [["V1", "10"]]}
    b = {"q-9-kis": [["V2", "20"]]}
    merged = rrf_merge_runs([a, b])
    assert set(merged) == {"q-1-kis", "q-9-kis"}
    written = write_merged(merged, tmp_path / "merged")
    assert sorted(p.stem for p in written) == ["q-1-kis", "q-9-kis"]
    assert load_run(tmp_path / "merged") == merged     # byte-level roundtrip


def test_diff_top1_and_markdown():
    base = {"q-1-kis": [["V1", "10"]], "q-2-kis": [["V2", "20"]]}
    merged = {"q-1-kis": [["V1", "10"]], "q-2-kis": [["V3", "30"]]}
    diffs = diff_top1(base, merged)
    assert diffs == [{"stem": "q-2-kis", "base_top1": ["V2", "20"],
                      "merged_top1": ["V3", "30"]}]
    md = render_diff_markdown(diffs)
    assert "q-2-kis" in md and "V2,20" in md and "V3,30" in md
    assert "Không câu nào" in render_diff_markdown([])


# ── scripts/64 CLI end-to-end (offline: plan + merge, no engine, no API) ─────


def test_script_64_plan_then_merge_offline(tmp_path, monkeypatch, capsys):
    m64 = _load_script("64_adaptive_attempts")

    run1 = _write_run(tmp_path / "att1", {
        "query-p1-1-kis": _kis_rows("L01_V001", 12),
        "query-p1-3-trake": [[f"L0{i}_V001", "10", "20", "30"] for i in range(1, 9)],
    })
    qdir = tmp_path / "queries"
    qdir.mkdir()
    for stem in ("query-p1-1-kis", "query-p1-3-trake"):
        (qdir / f"{stem}.txt").write_text("cảnh thử nghiệm\n", encoding="utf-8")

    plan_path = tmp_path / "plan.json"
    stage_dir = tmp_path / "stage"
    monkeypatch.setattr(sys, "argv", [
        "64", "plan", "--run", str(run1), "--query-dir", str(qdir),
        "--out", str(plan_path), "--stage-dir", str(stage_dir),
        "--max-fraction", "1.0",
    ])
    m64.main()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    assert plan["weak"] == ["query-p1-3-trake"]
    assert (stage_dir / "query-p1-3-trake.txt").is_file()
    assert plan["env"]["CVP_TEMPORAL__EVENT_QUERY_VARIANTS"] == "all"
    assert "YẾU" in capsys.readouterr().out

    # attempt 2 (giả lập chạy xong): the weak query now concentrates on one video
    run2 = _write_run(tmp_path / "att2", {
        "query-p1-3-trake": [["L09_V009", "10", "20", "30"],
                             ["L09_V009", "10", "20", "40"]],
    })
    merged_dir = tmp_path / "merged"
    report_md = merged_dir / "DIFF.md"
    monkeypatch.setattr(sys, "argv", [
        "64", "merge", "--runs", str(run1), str(run2),
        "--weights", "1.0", "2.0", "--out", str(merged_dir),
        "--report", str(report_md),
    ])
    m64.main()
    merged = load_run(merged_dir)
    assert set(merged) == {"query-p1-1-kis", "query-p1-3-trake"}
    # weight 2.0 tips attempt 2's top row over attempt 1's
    assert merged["query-p1-3-trake"][0] == ["L09_V009", "10", "20", "30"]
    assert report_md.is_file()
    assert "query-p1-3-trake" in report_md.read_text(encoding="utf-8")
    out = capsys.readouterr().out
    assert "1 câu đổi top-1" in out
