"""Tối ưu hiệu năng round-74 (Nhiệm vụ E): OUTPUT BẤT BIẾN trước/sau.

Mỗi tối ưu được so với BẢN SAO ĐÓNG BĂNG của thuật toán cũ (chép nguyên văn
trước khi sửa) trên fixture ngẫu nhiên seed cố định — bằng nhau TUYỆT ĐỐI
(từng bit float, từng thứ tự key), không phải xấp xỉ:

* ``fusion.minmax``            — vectorize (v−lo)/(hi−lo)
* ``fusion.aggregate_queries`` — nhánh "max" một lượt thay list+np.max
* ``TextIndexField.scores_for`` — hoist (k1+1.0) khỏi vòng trong
* ``KeyframeCatalog.video_ids`` — gather ndarray cache thay pandas .iloc
"""

from __future__ import annotations

import numpy as np
import pytest

from cvp.index.text_store import TextIndexField
from cvp.search import fusion

RNG = np.random.default_rng(74)


def _rand_map(n: int, lo: float = -2.0, hi: float = 5.0) -> dict[int, float]:
    keys = RNG.permutation(n * 3)[:n]
    vals = RNG.uniform(lo, hi, size=n)
    return {int(k): float(v) for k, v in zip(keys, vals)}


# ── minmax ───────────────────────────────────────────────────────────────────


def _minmax_ref(scores: dict[int, float]) -> dict[int, float]:
    """Bản đóng băng của fusion.minmax trước round-74 (dict comprehension)."""
    if not scores:
        return {}
    vals = np.fromiter(scores.values(), dtype=np.float64)
    lo, hi = float(vals.min()), float(vals.max())
    if hi - lo < 1e-9:
        return {k: 1.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


@pytest.mark.parametrize("n", [1, 2, 7, 500, 5000])
def test_minmax_bit_identical_to_reference(n):
    m = _rand_map(n)
    new, ref = fusion.minmax(m), _minmax_ref(m)
    assert list(new.keys()) == list(ref.keys())            # thứ tự key giữ nguyên
    assert all(new[k] == ref[k] for k in ref)              # từng bit float


def test_minmax_degenerate_cases_match_reference():
    assert fusion.minmax({}) == _minmax_ref({}) == {}
    flat = {1: 0.5, 2: 0.5, 3: 0.5}
    assert fusion.minmax(flat) == _minmax_ref(flat) == {1: 1.0, 2: 1.0, 3: 1.0}


# ── aggregate_queries ────────────────────────────────────────────────────────


def _aggregate_ref(per_query, how="max"):
    """Bản đóng băng của fusion.aggregate_queries trước round-74."""
    if not per_query:
        return {}
    if len(per_query) == 1:
        return per_query[0]
    out: dict[int, list[float]] = {}
    for m in per_query:
        for k, v in m.items():
            out.setdefault(k, []).append(v)
    if how == "mean":
        return {k: float(np.mean(v)) for k, v in out.items()}
    return {k: float(np.max(v)) for k, v in out.items()}


@pytest.mark.parametrize("how", ["max", "mean"])
def test_aggregate_queries_identical_to_reference(how):
    maps = [_rand_map(300), _rand_map(500), _rand_map(50)]
    new = fusion.aggregate_queries([dict(m) for m in maps], how=how)
    ref = _aggregate_ref([dict(m) for m in maps], how=how)
    assert list(new.keys()) == list(ref.keys())
    assert all(new[k] == ref[k] for k in ref)


def test_aggregate_queries_shortcuts_match_reference():
    assert fusion.aggregate_queries([]) == {}
    one = _rand_map(10)
    assert fusion.aggregate_queries([one]) is one          # shortcut giữ NGUYÊN object


# ── TextIndexField.scores_for ────────────────────────────────────────────────


def _field(n_docs: int = 400, vocab: int = 60) -> TextIndexField:
    words = [f"tok{i}" for i in range(vocab)]
    docs: dict[int | str, list] = {}
    df: dict[str, int] = {}
    total = 0
    for k in range(n_docs):
        toks = [words[int(i)] for i in RNG.integers(0, vocab, size=12)]
        tf: dict[str, int] = {}
        for t in toks:
            tf[t] = tf.get(t, 0) + 1
        for t in tf:
            df[t] = df.get(t, 0) + 1
        docs[k] = [[len(toks), tf]]
        total += len(toks)
    return TextIndexField(name="t", n_docs=n_docs, avgdl=total / n_docs,
                          df=df, docs=docs)


def _scores_for_ref(field: TextIndexField, query_tokens, keys):
    """Bản đóng băng của scores_for trước round-74 ((k1+1.0) trong vòng trong)."""
    if field.n_docs <= 0 or not query_tokens or field.avgdl <= 0:
        return {}
    weighted = [(tok, field.idf(tok)) for tok in query_tokens]
    weighted = [(tok, w) for tok, w in weighted if w > 0.0]
    if not weighted:
        return {}
    k1, b, avgdl = field.k1, field.b, field.avgdl
    out: dict = {}
    for k in set(keys):
        doc_list = field.docs.get(k)
        if not doc_list:
            continue
        best = 0.0
        for dl, tf in doc_list:
            norm = k1 * (1.0 - b + b * dl / avgdl)
            s = 0.0
            for tok, w in weighted:
                f = tf.get(tok)
                if f:
                    s += w * (f * (k1 + 1.0)) / (norm + f)
            if s > best:
                best = s
        if best > 1e-9:
            out[k] = best
    return out


def test_scores_for_bit_identical_to_reference():
    field = _field()
    query = [f"tok{i}" for i in (1, 3, 3, 7, 59, 12)]      # có token lặp
    keys = [int(k) for k in RNG.choice(400, size=120, replace=False)] + [999999]
    new = field.scores_for(query, keys)
    ref = _scores_for_ref(field, query, keys)
    assert new == ref and all(new[k] == ref[k] for k in ref)
    assert field.scores_for(["ngoai_corpus"], keys) == {}


# ── KeyframeCatalog.video_ids ────────────────────────────────────────────────


def test_video_ids_matches_pandas_iloc_reference(corpus):
    from cvp.data.catalog import KeyframeCatalog

    catalog = KeyframeCatalog(corpus)
    catalog.build()
    df = catalog.load()
    gids = [0, 5, 3, len(df) - 1, 0, -1]                   # lặp + âm (iloc wrap)
    ref = [str(v) for v in df["video_id"].iloc[[int(g) for g in gids]]]
    assert catalog.video_ids(gids) == ref
    assert catalog.video_ids(gids) == ref                  # lần 2 đi qua cache
    assert catalog.video_ids([]) == []
    with pytest.raises(IndexError):
        catalog.video_ids([len(df)])                       # quá biên vẫn nổ như iloc


# ── scripts/68_profile_engine.py: smoke CLI ─────────────────────────────────


def test_profile_script_quick_writes_report(tmp_path, monkeypatch, capsys):
    import importlib.util
    import json
    import sys
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "68_profile_engine", repo / "scripts" / "68_profile_engine.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["68_profile_engine"] = mod
    spec.loader.exec_module(mod)

    out_md, out_json = tmp_path / "profile.md", tmp_path / "profile.json"
    monkeypatch.setattr(sys, "argv", [
        "68", "--quick", "--repeats", "1",
        "--out", str(out_md), "--json-out", str(out_json)])
    mod.main()
    md = out_md.read_text(encoding="utf-8")
    assert "| op |" in md and "dante_dp" in md and "Tổng median" in md
    rows = json.loads(out_json.read_text(encoding="utf-8"))
    assert {r["op"] for r in rows} >= {"tokenize_vi", "fusion_minmax",
                                       "aggregate_queries_max", "dante_dp"}
    assert all(r["median_ms"] >= 0 for r in rows)
    assert "QUICK" in capsys.readouterr().out
