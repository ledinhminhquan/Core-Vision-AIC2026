"""Round-11 regressions — lessons from the FIRST LIVE nb01 run on Colab.

  R11-1  transformers 5.x: get_*_features returns a ModelOutput, not a Tensor
         — feature_tensor() accepts both API generations
  R11-2  requirements-colab no longer caps transformers <5 (the cap forced a
         hub downgrade that broke Colab's preinstalled accelerate)
  R11-3  nb01 heals empty map csvs from the source zips (Drive FUSE lost
         freshly-written files across sessions on a real run) and HARD-STOPS
         before embedding when the catalog has zero mapped frames
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]
torch = pytest.importorskip("torch")


# ── R11-1 · feature_tensor across transformers generations ───────────────────
def test_feature_tensor_passthrough_tensor():
    from cvp.models.hf_compat import feature_tensor

    t = torch.ones(2, 4)
    assert feature_tensor(t) is t


def test_feature_tensor_from_model_output_variants():
    from cvp.models.hf_compat import feature_tensor

    pooled = torch.ones(2, 4)
    # SigLIP-style v5: BaseModelOutputWithPooling(pooler_output=...)
    out = SimpleNamespace(pooler_output=pooled, last_hidden_state=torch.ones(2, 7, 4))
    assert feature_tensor(out) is pooled
    # CLIP-style v5: projection outputs win over pooler_output
    proj = torch.ones(2, 3)
    out2 = SimpleNamespace(text_embeds=proj, pooler_output=pooled)
    assert feature_tensor(out2) is proj
    # tuple-like fallback
    assert feature_tensor((pooled,)) is pooled


def test_feature_tensor_rejects_garbage():
    from cvp.models.hf_compat import feature_tensor

    with pytest.raises(TypeError, match="feature tensor"):
        feature_tensor(SimpleNamespace(nothing=1))


def test_models_use_feature_tensor():
    for rel in ("src/cvp/models/siglip2.py", "src/cvp/models/metaclip2.py"):
        src = (REPO / rel).read_text(encoding="utf-8")
        assert "feature_tensor(out)" in src, rel
        assert "out.float().cpu()" not in src, rel


# ── R11-2 · no transformers upper cap ────────────────────────────────────────
def test_requirements_colab_transformers_uncapped():
    req = (REPO / "requirements-colab.txt").read_text(encoding="utf-8")
    line = next(ln for ln in req.splitlines()
                if ln.strip().startswith("transformers"))
    assert "<5" not in line and "<6" not in line


# ── R11-3 · nb01 map-csv self-heal + zero-map hard stop ──────────────────────
def test_builder_heals_and_hard_stops():
    src = (REPO / "notebooks" / "_build_notebooks.py").read_text(encoding="utf-8")
    assert "tự phục hồi từ zip" in src                      # heal path exists
    assert "map csv integrity: OK" in src
    assert "TOÀN BỘ catalog KHÔNG có map-keyframes" in src  # hard stop
    assert "FORCE rebuild catalog" in src                   # poisoned-manifest rebuild
