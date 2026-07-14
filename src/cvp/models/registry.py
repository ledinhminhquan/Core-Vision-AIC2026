"""Model registry: name → EmbeddingModel instance.

Names (config ``embedding.model`` / ``embedding.ensemble_members``):

    siglip2          google/siglip2-so400m — multilingual default
    finetuned        siglip2 image tower + LiT-tuned Vietnamese text tower
    openclip         big English tower (PE-Core hub ids → DFN5B fallback)
    qwen_embed       Qwen3-VL-Embedding — native-vi MLLM lane (heavy, optional)
    provided_clip32  OpenAI ViT-B/32 — matches organiser clip-features-32
    mclip            M-CLIP multilingual (optional diversity member)
    jina             jina-clip-v2 — 89-language Matryoshka lane (optional)
    metaclip2        MetaCLIP 2 worldwide — multilingual SOTA 2026 (optional)
    ensemble         handled by SearchEngine (fusion over members)

Constructor resolution is lazy: ``resolve_constructor`` never imports model
modules or frameworks — weights load only when the returned callable runs.
"""

from __future__ import annotations

from typing import Callable

from cvp.config import Settings
from cvp.models.base import EmbeddingModel


def _make_siglip2(settings: Settings) -> EmbeddingModel:
    from cvp.models.siglip2 import SigLIP2Model

    return SigLIP2Model(settings, finetuned=False)


def _make_finetuned(settings: Settings) -> EmbeddingModel:
    from cvp.models.siglip2 import SigLIP2Model

    return SigLIP2Model(settings, finetuned=True)


def _make_openclip(settings: Settings) -> EmbeddingModel:
    from cvp.models.openclip_model import OpenClipModel

    return OpenClipModel(settings)


def _make_provided_clip32(settings: Settings) -> EmbeddingModel:
    from cvp.models.openclip_model import provided_clip32_model

    return provided_clip32_model(settings)


def _make_qwen_embed(settings: Settings) -> EmbeddingModel:
    from cvp.models.qwen_embed import QwenEmbedModel

    return QwenEmbedModel(settings)


def _make_mclip(settings: Settings) -> EmbeddingModel:
    from cvp.models.mclip_model import MClipModel

    return MClipModel(settings)


def _make_jina(settings: Settings) -> EmbeddingModel:
    from cvp.models.jina_clip import JinaClipModel

    return JinaClipModel(settings)


def _make_metaclip2(settings: Settings) -> EmbeddingModel:
    from cvp.models.metaclip2 import MetaClip2Model

    return MetaClip2Model(settings)


_CONSTRUCTORS: dict[str, Callable[[Settings], EmbeddingModel]] = {
    "siglip2": _make_siglip2,
    "finetuned": _make_finetuned,
    "openclip": _make_openclip,
    "provided_clip32": _make_provided_clip32,
    "qwen_embed": _make_qwen_embed,
    "mclip": _make_mclip,
    "jina": _make_jina,
    "metaclip2": _make_metaclip2,
}


def known_models() -> tuple[str, ...]:
    """Every registered single-model backend name (excludes 'ensemble')."""
    return tuple(_CONSTRUCTORS)


def resolve_constructor(name: str) -> Callable[[Settings], EmbeddingModel]:
    """Backend name → lazy constructor. Never loads weights or heavy deps."""
    if name == "ensemble":
        raise ValueError("'ensemble' is not a single model — SearchEngine builds each member")
    try:
        return _CONSTRUCTORS[name]
    except KeyError:
        raise ValueError(f"Unknown embedding model: {name!r}") from None


def model_key_for(name: str) -> str:
    """Filesystem key for a backend name (embeddings/index folder name)."""
    return name  # names are already filesystem-safe


def index_key_for(name: str) -> str:
    """Which embedding/index space a backend searches in.

    ``finetuned`` keeps the FROZEN siglip2 image tower (LiT), so its text
    vectors live in the siglip2 image space — it searches the siglip2 index
    directly. No re-embedding, no copying. Every other backend (including
    ``qwen_embed``) owns its own space.
    """
    return "siglip2" if name == "finetuned" else name


def build_model(settings: Settings, name: str | None = None) -> EmbeddingModel:
    name = name or settings.embedding.model
    return resolve_constructor(name)(settings)
