"""transformers 4.x ↔ 5.x compatibility shims.

``get_text_features`` / ``get_image_features`` return a plain Tensor on
transformers 4.x but a ModelOutput object (e.g. ``BaseModelOutputWithPooling``)
on 5.x — the 2026 Colab image ships 5.x, and ``out.float()`` exploded there
(round-11, live nb01 run). One helper, used by every HF two-tower model.
"""

from __future__ import annotations


def feature_tensor(out):
    """The pooled/projected feature Tensor from either API generation.

    Preference order mirrors what ``get_*_features`` meant on 4.x:
    projection outputs first (CLIP-style ``text_embeds``/``image_embeds``),
    then the pooled tower output (SigLIP-style has no projection head).
    """
    import torch

    if torch.is_tensor(out):
        return out
    for attr in ("text_embeds", "image_embeds", "pooler_output"):
        v = getattr(out, attr, None)
        if v is not None and torch.is_tensor(v):
            return v
    try:  # ModelOutput is tuple-like; first element is the primary tensor
        v = out[0]
        if torch.is_tensor(v):
            return v
    except (TypeError, KeyError, IndexError):
        pass
    raise TypeError(
        f"cannot extract a feature tensor from {type(out).__name__} — "
        "unexpected transformers output shape"
    )
