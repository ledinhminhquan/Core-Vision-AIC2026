"""Adversarial-review fixes: unique atomic-write tmp names (concurrent
writers), ObjectStore LRU caching, query timeout plumbing, and dead-knob
removal. Pure CPU, no torch, no network.
"""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import TimeoutError as FuturesTimeoutError
from pathlib import Path

import pytest

from cvp.config import Settings
from cvp.data.metadata import ObjectStore
from cvp.models.query_processor import _call_with_timeout
from cvp.utils.io import _tmp_path, atomic_write_json, write_jsonl


# ── utils/io: unique tmp names + concurrent writers ──────────────────────────


def test_tmp_path_unique_per_call_and_same_directory(tmp_path: Path):
    p = tmp_path / "sub" / "x.json"
    a, b = _tmp_path(p), _tmp_path(p)
    assert a != b                              # two writers can never share a tmp
    assert a.parent == b.parent == p.parent    # same dir → same-filesystem rename
    assert a.name.startswith("x.json.") and a.name.endswith(".tmp")


def test_atomic_write_leaves_no_tmp_litter(tmp_path: Path):
    p = tmp_path / "x.json"
    atomic_write_json(p, {"k": 1})
    assert json.loads(p.read_text(encoding="utf-8")) == {"k": 1}
    assert not list(tmp_path.glob("*.tmp"))


def test_concurrent_writers_to_same_path_never_collide(tmp_path: Path):
    """With a FIXED sibling tmp name this races (one writer renames the
    other's bytes / os.replace hits a missing file on Windows)."""
    p = tmp_path / "shared.json"
    errors: list[Exception] = []

    def hammer(i: int) -> None:
        try:
            for _ in range(25):
                atomic_write_json(p, {"writer": i})
        except Exception as e:  # noqa: BLE001 — the test asserts none happen
            errors.append(e)

    threads = [threading.Thread(target=hammer, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert json.loads(p.read_text(encoding="utf-8"))["writer"] in range(4)
    assert not list(tmp_path.glob("*.tmp"))


def test_write_jsonl_cleans_tmp_on_failure(tmp_path: Path):
    p = tmp_path / "rows.jsonl"

    def rows():
        yield {"a": 1}
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        write_jsonl(p, rows())
    assert not p.exists()
    assert not list(tmp_path.glob("*.tmp"))    # failed write leaves nothing behind


# ── data/metadata: ObjectStore LRU cache ─────────────────────────────────────


def test_object_store_get_is_cached(tmp_path: Path):
    settings = Settings.model_validate({"paths": {"data_root": str(tmp_path / "data")}})
    obj_dir = tmp_path / "data" / "objects" / "L01_V001"
    obj_dir.mkdir(parents=True)
    (obj_dir / "001.json").write_text(json.dumps({
        "detection_class_entities": ["Person"],
        "detection_scores": [0.9],
        "detection_boxes": [[0.1, 0.1, 0.5, 0.4]],
    }), encoding="utf-8")

    store = ObjectStore(settings)
    first = store.get("L01_V001", 1)
    assert [d.entity for d in first] == ["Person"]
    # Objects are immutable artifacts: repeat calls come from the LRU cache,
    # not from re-opening the JSON (delete the file to prove it).
    (obj_dir / "001.json").unlink()
    assert store.get("L01_V001", 1) is first


# ── query_processor: timeout plumbing ────────────────────────────────────────


def test_call_with_timeout_returns_and_raises():
    assert _call_with_timeout(lambda: 42, 5.0) == 42
    with pytest.raises(FuturesTimeoutError):
        _call_with_timeout(lambda: time.sleep(2.0), 0.05)


def test_query_cache_key_covers_model_and_english_enhance():
    """Fix L4 (review 2026-07-08): switching gemini_model or enhance_english
    mid-competition must MISS the cache, not replay stale enhancements."""
    from cvp.models.query_processor import QueryProcessor

    def _proc(**overrides) -> QueryProcessor:
        s = Settings.model_validate({"query": {"provider": "gemini", **overrides}})
        return QueryProcessor(s)

    base = _proc()._cache_key("người dẫn chương trình")
    assert _proc(gemini_model="gemini-3-flash")._cache_key("người dẫn chương trình") != base
    assert _proc(enhance_english=False)._cache_key("người dẫn chương trình") != base
    assert _proc(expansions=4)._cache_key("người dẫn chương trình") != base
    assert _proc()._cache_key("người dẫn chương trình") == base  # deterministic


# ── qwen_embed: instruction on the QUERY side only ───────────────────────────


def test_qwen_messages_instruction_only_for_queries():
    """Fix L2 (review 2026-07-08): documents (images) are embedded WITHOUT the
    instruction system turn, matching the official Qwen3VLEmbedder convention;
    text queries keep it. Uses __new__ — no torch/weights needed."""
    from cvp.models.qwen_embed import QwenEmbedModel

    m = object.__new__(QwenEmbedModel)
    m.instruction = "Represent this keyframe / query for retrieval."

    query_msgs = m._messages(text="một cảnh cháy lớn")
    assert query_msgs[0]["role"] == "system"
    assert query_msgs[0]["content"][0]["text"] == m.instruction

    doc_msgs = m._messages(image=object(), with_instruction=False)
    assert all(turn["role"] != "system" for turn in doc_msgs)
    assert doc_msgs[0]["role"] == "user"


# ── config: dead knobs removed, stray keys tolerated ─────────────────────────


def test_dead_knobs_removed():
    s = Settings()
    assert not hasattr(s.embedding, "normalize")
    assert not hasattr(s.finetuned, "text_only")
    assert not hasattr(s.ocr, "batch_size")
    assert not hasattr(s.caption, "batch_size")
    assert s.asr.batch_size == 8   # kept — wired into the transformers ASR pipeline


def test_stray_keys_in_old_settings_yaml_are_ignored():
    s = Settings.model_validate({
        "embedding": {"model": "siglip2", "normalize": True},
        "ocr": {"engine": "easyocr", "batch_size": 16},
    })
    assert s.embedding.model == "siglip2" and s.ocr.engine == "easyocr"


def test_vietocr_no_longer_documented_as_an_engine():
    import inspect

    import cvp.auxindex.ocr as ocr_mod
    import cvp.config as config_mod

    assert "vietocr" not in (ocr_mod.__doc__ or "").lower()
    assert "vietocr" not in inspect.getsource(config_mod).lower()
    yaml_path = Path(config_mod.__file__).resolve().parents[2] / "configs" / "settings.yaml"
    assert "vietocr" not in yaml_path.read_text(encoding="utf-8").lower()
