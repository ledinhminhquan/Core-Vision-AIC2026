"""Extended object-constraint parsing/scoring tests + weight-tuning harness tests.

Covers the B7 upgrades: multiple constraints per noun class, Vietnamese number
words with 3-token lookback, position phrases, expanded vocab sanity, and the
scripts/21_tune_weights.py harness over a synthetic signals dir + tiny GT.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest
import yaml

from cvp.data.catalog import KeyframeCatalog
from cvp.search.object_filter import ObjectBooster, ObjectConstraint

REPO = Path(__file__).resolve().parents[1]
VOCAB_PATH = REPO / "configs" / "object_vocab_vi.yaml"


# ── parser: multiple constraints per class ───────────────────────────────────


def test_multiple_constraints_same_class(corpus):
    booster = ObjectBooster(corpus)
    cons = booster.parse("2 người bên trái và 1 người bên phải")
    persons = [c for c in cons if c.entity == "Person"]
    assert len(persons) == 2
    assert (2, "left") in {(c.count, c.position) for c in persons}
    assert (1, "right") in {(c.count, c.position) for c in persons}


def test_duplicate_mentions_collapse(corpus):
    booster = ObjectBooster(corpus)
    cons = booster.parse("người và người")
    assert cons == [ObjectConstraint(entity="Person", count=None, position=None)]


# ── parser: Vietnamese number words + digits ─────────────────────────────────


def test_number_words(corpus):
    booster = ObjectBooster(corpus)
    assert booster.parse("ba con chó")[0].count == 3
    assert booster.parse("mười con vịt")[0].count == 10
    assert booster.parse("bảy chiếc xe tải")[0].count == 7
    assert booster.parse("một chiếc xe máy")[0].count == 1


def test_digits_and_nearest_number_wins(corpus):
    booster = ObjectBooster(corpus)
    assert booster.parse("3 chiếc xe buýt")[0].count == 3
    # nearest number to the noun wins ("ba" beats "hai")
    assert booster.parse("hai và ba con chó")[0].count == 3


def test_year_is_not_a_count(corpus):
    booster = ObjectBooster(corpus)
    cons = booster.parse("năm 2024 người dân sơ tán")
    person = next(c for c in cons if c.entity == "Person")
    assert person.count is None  # 2024 blocks the misread of "năm" → 5


# ── parser: positions ────────────────────────────────────────────────────────


def test_positions_all_directions(corpus):
    booster = ObjectBooster(corpus)
    assert booster.parse("con chó ở góc trái")[0].position == "left"
    assert booster.parse("chiếc xe tải phía dưới")[0].position == "bottom"
    assert booster.parse("quả bóng ở giữa")[0].position == "center"
    assert booster.parse("máy bay phía trên")[0].position == "top"
    assert booster.parse("con mèo bên phải")[0].position == "right"


def test_position_does_not_cross_va_conjunction(corpus):
    booster = ObjectBooster(corpus)
    cons = booster.parse("người và xe máy bên phải")
    by_entity = {c.entity: c for c in cons}
    assert by_entity["Person"].position is None
    assert by_entity["Motorcycle"].position == "right"


# ── parser: unchanged behaviour for plain queries ────────────────────────────


def test_no_constraint_queries_unchanged(corpus):
    booster = ObjectBooster(corpus)
    assert booster.parse("bầu trời hoàng hôn tuyệt đẹp") == []
    assert booster.parse("xe máy") == [
        ObjectConstraint(entity="Motorcycle", count=None, position=None)
    ]


def test_longest_phrase_still_wins(corpus):
    booster = ObjectBooster(corpus)
    assert {c.entity for c in booster.parse("cô ấy mặc áo dài đỏ")} == {"Dress"}
    assert {c.entity for c in booster.parse("đội mũ bảo hiểm")} == {"Helmet"}
    assert {c.entity for c in booster.parse("đèn giao thông")} == {"Traffic light"}


def test_expanded_vocab_spot_checks(corpus):
    booster = ObjectBooster(corpus)
    expected = {
        "chiếc ghe trên sông": "Boat",
        "con trâu": "Cattle",
        "con dê": "Goat",
        "trực thăng cứu hộ": "Helicopter",
        "múa lân": "Lion",
        "hồ bơi": "Swimming pool",
        "cảnh sát giao thông": "Person",
    }
    for query, entity in expected.items():
        entities = {c.entity for c in booster.parse(query)}
        assert entity in entities, f"{query!r} → {entities}"


# ── scoring with multiple constraints (semantics preserved) ──────────────────


def test_multi_constraint_scoring(corpus):
    # Fixture objects: 2 Persons (cx=0.25 left, cx=0.70 right) + 1 Car per frame.
    catalog = KeyframeCatalog(corpus)
    catalog.build()
    booster = ObjectBooster(corpus)
    ref = catalog.ref(0)
    cons = booster.parse("2 người bên trái và 1 người bên phải")
    # left: 1 person vs count 2 → 0.5 (off-by-one); right: exact → 1.0
    assert booster.score(cons, ref) == pytest.approx(0.75)


def test_position_scoring_bottom_car(corpus):
    # Fixture Car box (0.6, 0.2, 0.9, 0.8) → cy=0.75 → bottom.
    catalog = KeyframeCatalog(corpus)
    catalog.build()
    booster = ObjectBooster(corpus)
    cons = booster.parse("một chiếc ô tô phía dưới")
    assert booster.score(cons, catalog.ref(0)) == pytest.approx(1.0)


# ── vocab file sanity ────────────────────────────────────────────────────────


def test_vocab_size_and_no_duplicate_keys():
    text = VOCAB_PATH.read_text(encoding="utf-8")
    vocab = yaml.safe_load(text)
    assert len(vocab) >= 170, "vocab should stay expanded (~180+ nouns)"
    mapping_lines = [
        line for line in text.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    # YAML silently keeps only the last duplicate key — catch dups in the raw text.
    assert len(mapping_lines) == len(vocab)


def test_vocab_entities_look_like_openimages_classes():
    vocab = yaml.safe_load(VOCAB_PATH.read_text(encoding="utf-8"))
    for noun, entity in vocab.items():
        assert isinstance(entity, str) and entity, noun
        assert re.fullmatch(r"[A-Z][A-Za-z &'()-]*", entity), f"{noun}: {entity}"
    required = {
        "người": "Person", "xe máy": "Motorcycle", "xe buýt": "Bus",
        "xe tải": "Truck", "trâu": "Cattle", "vịt": "Duck",
        "mũ bảo hiểm": "Helmet", "áo dài": "Dress",
        "đèn giao thông": "Traffic light", "hồ bơi": "Swimming pool",
        "chảo": "Frying pan", "chai": "Bottle",
    }
    for noun, entity in required.items():
        assert vocab.get(noun) == entity


# ── weight-tuning harness (scripts/21_tune_weights.py) ───────────────────────


def _load_harness():
    scripts = REPO / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    spec = importlib.util.spec_from_file_location(
        "tune_weights_harness", scripts / "21_tune_weights.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


BASE_WEIGHTS = {"visual": 1.0, "ocr": 0.35, "asr": 0.30,
                "caption": 0.25, "metadata": 0.15, "object": 0.0}


@pytest.fixture()
def planted(tmp_path: Path):
    """Signals where the GT row wins only once the object weight rises."""
    signals = {
        "q1": {
            "visual": {"L01_V001,150": 0.85, "L02_V001,50": 0.90, "L03_V001,10": 0.10},
            "object": {"L01_V001,150": 1.0},
        },
        "q2": {
            "visual": {"L05_V002,120": 0.80, "L06_V001,40": 0.95, "L07_V001,10": 0.05},
            "object": {"L05_V002,120": 1.0, "L05_V002,900": 0.2},
        },
    }
    gt = {
        "q1": {"task": "kis", "video_id": "L01_V001", "frame_start": 100, "frame_end": 200},
        "q2": {"task": "trake", "video_id": "L05_V002", "events": [[100, 140], [880, 920]]},
    }
    sig_dir = tmp_path / "signals"
    sig_dir.mkdir()
    (sig_dir / "dev.json").write_text(json.dumps(signals), encoding="utf-8")
    gt_path = tmp_path / "gt.json"
    gt_path.write_text(json.dumps(gt), encoding="utf-8")
    return sig_dir, gt_path, signals, gt


def test_load_signals_merges_files(tmp_path: Path):
    mod = _load_harness()
    d = tmp_path / "sig"
    d.mkdir()
    (d / "a.json").write_text(json.dumps({"q1": {"visual": {"L01_V001,1": 0.5}}}), "utf-8")
    (d / "b.json").write_text(json.dumps({"q1": {"ocr": {"L01_V001,1": 2.0}}}), "utf-8")
    merged = mod.load_signals(d)
    assert set(merged["q1"]) == {"visual", "ocr"}
    with pytest.raises(FileNotFoundError):
        mod.load_signals(tmp_path / "empty_missing")


def test_local_scorer_formulas():
    mod = _load_harness()
    gt = {"task": "kis", "video_id": "L01_V001", "frame_start": 100, "frame_end": 200}
    rows = [("L01_V001", 150), ("L02_V001", 50)]
    assert mod._local_score("kis", rows, gt) == 1.0
    assert mod._local_score("qa", rows[::-1], gt) == pytest.approx(0.8)  # hit at rank 2
    trake = {"task": "trake", "video_id": "L05_V002", "events": [[100, 140]]}
    assert mod._local_score("trake", [("L05_V002", 120)], trake) == 1.0
    assert mod._local_score("trake", [("L05_V002", 500)], trake) == 0.0
    # "segments" / "moments" are accepted aliases for "events"
    seg = {"task": "trake", "video_id": "L05_V002", "segments": [[100, 140]]}
    assert mod._local_score("trake", [("L05_V002", 120)], seg) == 1.0
    with pytest.raises(ValueError):
        mod._local_score("avsx", rows, gt)


def test_local_scorer_multi_window_ranges():
    # Enhancement E5 (review 2026-07-08): the harness accepts the same
    # multi-window "ranges" GT spelling as cvp.eval.official.
    mod = _load_harness()
    gt = {"task": "kis", "video_id": "L01_V001", "ranges": [[100, 110], [500, 510]]}
    assert mod._local_score("kis", [("L01_V001", 505)], gt) == 1.0
    assert mod._local_score("kis", [("L01_V001", 300)], gt) == 0.0
    assert mod._gt_windows(gt) == [(100, 110), (500, 510)]
    # single-window entries keep their exact previous behaviour
    single = {"task": "kis", "video_id": "L01_V001", "frame_start": 100, "frame_end": 200}
    assert mod._gt_windows(single) == [(100, 200)]
    with pytest.raises(ValueError, match="window"):
        mod._gt_windows({"task": "kis", "video_id": "L01_V001"})


def test_resolve_scorer_always_returns_callable():
    mod = _load_harness()
    fn, name = mod.resolve_scorer()
    assert callable(fn)
    assert "cvp.eval" in name
    # Works whichever backend was resolved (official present or not).
    gt = {"task": "kis", "video_id": "L01_V001", "frame_start": 100, "frame_end": 200}
    assert fn("kis", [("L01_V001", 150)], gt) == pytest.approx(1.0)
    assert fn("kis", [("L09_V009", 150)], gt) == pytest.approx(0.0)


def test_harness_scores_canonical_gt_formats_without_crashing():
    # Regression: QA GT in the canonical cvp.eval.official format
    # ({"range": [s, e], "answers": [...]}) used to KeyError('frame_start') in
    # _local_score. QA is scored as KIS, TRAKE via the single-frame proxy.
    mod = _load_harness()
    qa = {"task": "qa", "video_id": "L05_V005", "range": [800, 900],
          "answers": ["màu xanh"]}
    trake = {"task": "trake", "video_id": "L10_V010",
             "moments": [[95, 105], [145, 155]]}
    center = {"task": "kis", "video_id": "L01_V001", "center": 150, "epsilon": 50}

    assert mod._local_score("qa", [("L05_V005", 850)], qa) == 1.0
    assert mod._local_score("qa", [("L05_V005", 999)], qa) == 0.0
    assert mod._local_score("trake", [("L10_V010", 100)], trake) == 1.0
    assert mod._local_score("kis", [("L01_V001", 150)], center) == 1.0
    with pytest.raises(ValueError, match="window"):
        mod._local_score("kis", [("L01_V001", 150)], {"task": "kis", "video_id": "L01_V001"})

    # ...and through the full resolved-scorer + tune() path.
    scorer, _ = mod.resolve_scorer()
    signals = {
        "q-qa": {"visual": {"L05_V005,850": 0.9, "L06_V001,10": 0.5}},
        "q-trake": {"visual": {"L10_V010,100": 0.9, "L11_V001,10": 0.5}},
    }
    gt = {"q-qa": qa, "q-trake": trake}
    report = mod.tune(signals, gt, base_weights=BASE_WEIGHTS,
                      method="grid", scorer=scorer)
    assert report["baseline"]["score"] == pytest.approx(1.0)
    assert report["baseline"]["per_task"] == {"qa": 1.0, "trake": 1.0}


def test_grid_recovers_planted_ranking(planted):
    sig_dir, gt_path, _, gt = planted
    mod = _load_harness()
    signals = mod.load_signals(sig_dir)
    report = mod.tune(signals, gt, base_weights=BASE_WEIGHTS,
                      method="grid", scorer=mod._local_score)
    assert report["baseline"]["score"] == pytest.approx(0.8)
    assert report["best"]["score"] == pytest.approx(1.0)
    assert report["best"]["weights"]["object"] == pytest.approx(0.3)
    assert report["best"]["per_task"] == {"kis": 1.0, "trake": 1.0}
    assert report["per_task_delta"] == {"kis": pytest.approx(0.2),
                                        "trake": pytest.approx(0.2)}
    assert report["active_signals"] == ["object"]


def test_random_recovers_planted_ranking(planted):
    sig_dir, _, _, gt = planted
    mod = _load_harness()
    signals = mod.load_signals(sig_dir)
    report = mod.tune(signals, gt, base_weights=BASE_WEIGHTS,
                      trials=60, method="random", seed=0, scorer=mod._local_score)
    assert report["best"]["score"] == pytest.approx(1.0)
    assert report["best"]["weights"]["object"] > 0.16
    assert report["delta"] == pytest.approx(0.2)


def test_tune_accepts_custom_scorer(planted):
    # Local fallback scorer — proves the harness has no cvp.eval.official dependency.
    sig_dir, _, _, gt = planted
    mod = _load_harness()
    signals = mod.load_signals(sig_dir)
    calls: list[str] = []

    def scorer(task, rows, entry):
        calls.append(task)
        return 1.0 if rows and rows[0][0] == entry["video_id"] else 0.0

    report = mod.tune(signals, gt, base_weights=BASE_WEIGHTS,
                      method="grid", scorer=scorer)
    assert calls and set(calls) == {"kis", "trake"}
    assert report["scorer"] == "custom"
    assert report["best"]["score"] == pytest.approx(1.0)


def test_cli_end_to_end(planted, tmp_path: Path, monkeypatch):
    sig_dir, gt_path, _, _ = planted
    out = tmp_path / "tuned" / "best.json"
    monkeypatch.setattr(sys, "argv", [
        "21_tune_weights.py", "--signals-dir", str(sig_dir), "--gt", str(gt_path),
        "--method", "grid", "--out", str(out),
    ])
    mod = _load_harness()
    mod.main()
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["best"]["score"] == pytest.approx(1.0)
    assert set(report["best"]["weights"]) == {"visual", "ocr", "asr",
                                              "caption", "metadata", "object"}
