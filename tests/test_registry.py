"""Registry + model-lane logic: lazy constructor resolution (no weights, no
torch), index-space aliasing, qwen MRL truncation math, openclip hub→DFN5B
fallback order. Pure CPU, no network."""

from __future__ import annotations

import sys

import numpy as np
import pytest

from cvp.config import Settings
from cvp.models import openclip_model as ocm
from cvp.models import registry
from cvp.models.qwen_embed import _truncate_and_renorm

ALL_NAMES = ("siglip2", "finetuned", "openclip", "provided_clip32", "qwen_embed", "mclip")

HEAVY_MODULES = (
    "cvp.models.siglip2",
    "cvp.models.mclip_model",
    "cvp.models.qwen_embed",  # already imported by this test file — tracked anyway
    "torch",
    "transformers",
    "open_clip",
    "multilingual_clip",
)


# ── constructor resolution ───────────────────────────────────────────────────


def test_resolve_constructor_covers_all_names_lazily():
    before = {m for m in HEAVY_MODULES if m in sys.modules}
    for name in ALL_NAMES:
        assert callable(registry.resolve_constructor(name))
    after = {m for m in HEAVY_MODULES if m in sys.modules}
    assert after == before, f"resolution must not import heavy modules: {after - before}"


def test_known_models_matches_registry():
    assert set(registry.known_models()) == set(ALL_NAMES)


def test_ensemble_is_not_a_single_model():
    with pytest.raises(ValueError, match="ensemble"):
        registry.resolve_constructor("ensemble")
    with pytest.raises(ValueError, match="ensemble"):
        registry.build_model(Settings.model_validate({"embedding": {"model": "ensemble"}}))


def test_unknown_name_raises():
    with pytest.raises(ValueError, match="Unknown embedding model"):
        registry.resolve_constructor("clip9000")
    with pytest.raises(ValueError, match="Unknown embedding model"):
        registry.build_model(Settings(), "clip9000")


# ── index-space aliasing ─────────────────────────────────────────────────────


def test_index_key_for_aliases_finetuned_only():
    assert registry.index_key_for("finetuned") == "siglip2"
    for name in ("siglip2", "openclip", "provided_clip32", "qwen_embed", "mclip"):
        assert registry.index_key_for(name) == name


def test_model_key_for_is_identity():
    for name in ALL_NAMES:
        assert registry.model_key_for(name) == name


# ── qwen MRL truncation ──────────────────────────────────────────────────────


def test_truncate_and_renorm_prefix_slice_and_unit_norm():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(4, 32)) * 3.0  # float64, unnormalized on purpose
    out = _truncate_and_renorm(x, 8)
    assert out.shape == (4, 8)
    assert out.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)
    manual = x[:, :8] / np.linalg.norm(x[:, :8], axis=1, keepdims=True)
    np.testing.assert_allclose(out, manual.astype(np.float32), atol=1e-5)


def test_truncate_and_renorm_full_width_when_dim_large_or_zero():
    x = np.eye(3, dtype=np.float32) * 5.0
    for dim in (0, 3, 99):
        out = _truncate_and_renorm(x, dim)
        assert out.shape == (3, 3)
        np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-6)


def test_truncate_and_renorm_zero_vectors_stay_finite():
    out = _truncate_and_renorm(np.zeros((2, 16), dtype=np.float32), 4)
    assert out.shape == (2, 4)
    assert np.all(np.isfinite(out))
    np.testing.assert_allclose(out, 0.0)


# ── openclip hub fallback ────────────────────────────────────────────────────

HUB_IDS = ["hf-hub:timm/PE-Core-bigG-14-448", "hf-hub:timm/PE-Core-L-14-336"]


def test_load_attempts_order_hub_ids_then_arch():
    attempts = ocm._load_attempts(HUB_IDS, "ViT-H-14-378-quickgelu", "dfn5b")
    assert attempts == [
        ("hub", "hf-hub:timm/PE-Core-bigG-14-448"),
        ("hub", "hf-hub:timm/PE-Core-L-14-336"),
        ("arch", ("ViT-H-14-378-quickgelu", "dfn5b")),
    ]


def test_load_with_fallback_walks_in_order_until_success():
    attempts = ocm._load_attempts(HUB_IDS, "ViT-H-14-378-quickgelu", "dfn5b")
    calls: list = []

    def loader(attempt):
        calls.append(attempt)
        if len(calls) < 3:
            raise OSError("hub download failed")
        return "LOADED"

    chosen, result = ocm._load_with_fallback(attempts, loader)
    assert calls == attempts  # tried strongest-first, stopped at success
    assert result == "LOADED"
    assert chosen == ("arch", ("ViT-H-14-378-quickgelu", "dfn5b"))


def test_load_with_fallback_first_success_short_circuits():
    calls: list = []

    def loader(attempt):
        calls.append(attempt)
        return "PE"

    attempts = ocm._load_attempts(HUB_IDS, "a", "p")
    chosen, result = ocm._load_with_fallback(attempts, loader)
    assert result == "PE"
    assert chosen == ("hub", HUB_IDS[0])
    assert len(calls) == 1


def test_load_with_fallback_all_fail_raises_runtime_error():
    def loader(attempt):
        raise OSError("no network")

    with pytest.raises(RuntimeError, match="no network"):
        ocm._load_with_fallback(ocm._load_attempts(["h1"], "a", "p"), loader)


def test_tag_for_attempt():
    assert ocm._tag_for_attempt(("hub", "hf-hub:timm/PE-Core-bigG-14-448")) == "PE-Core-bigG-14-448"
    tag = ocm._tag_for_attempt(("arch", ("ViT-H-14-378-quickgelu", "dfn5b")))
    assert tag == "dfn5b-vit-h-14-378-quickgelu"


class _FakeTower:
    """Stands in for an open_clip model: absorbs .to()/.eval() chaining."""

    def to(self, *args, **kwargs):
        return self

    def eval(self):
        return self


def test_openclip_init_falls_back_and_records_model_tag(monkeypatch):
    calls: list = []

    def fake_create(self, attempt):
        calls.append(attempt)
        if attempt == ("hub", "hf-hub:timm/PE-Core-bigG-14-448"):
            raise OSError("gated / offline")
        return _FakeTower(), object(), object()

    monkeypatch.setattr(ocm.OpenClipModel, "_create_from_attempt", fake_create)
    monkeypatch.setattr(ocm, "resolve_device", lambda pref: "cpu")
    model = ocm.OpenClipModel(Settings())
    assert [a[0] for a in calls[:2]] == ["hub", "hub"]
    assert model.model_tag == "PE-Core-L-14-336"  # second hub id won
    assert model.key == "openclip"
    assert model.multilingual is False


def test_provided_clip32_pins_checkpoint_and_skips_hub(monkeypatch):
    calls: list = []

    def fake_create(self, attempt):
        calls.append(attempt)
        return _FakeTower(), object(), object()

    monkeypatch.setattr(ocm.OpenClipModel, "_create_from_attempt", fake_create)
    monkeypatch.setattr(ocm, "resolve_device", lambda pref: "cpu")
    model = ocm.provided_clip32_model(Settings())
    assert calls == [("arch", ("ViT-B-32", "openai"))]  # hub ids never tried
    assert model.key == "provided_clip32"
    assert model.model_tag == "openai-vit-b-32"
