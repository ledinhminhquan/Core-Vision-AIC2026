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


def ensure_remote_code_compat() -> None:
    """transformers-5 drift (live run 6): the v5 finalize step reads
    ``self.all_tied_weights_keys``, which is set during ``post_init()`` —
    remote-code InternVL/Vintern models never call it, so every load died with
    AttributeError. A read-only class-level default satisfies v5's
    ``missing_keys - self.all_tied_weights_keys.keys()`` for models that skip
    post_init, while properly initialised models still shadow it per-instance.
    MappingProxyType keeps the shared default immune to in-place mutation (any
    future transformers write to it fails LOUDLY instead of silently poisoning
    later loads). No-op when the attribute already exists at class level.

    Call before EVERY ``from_pretrained(..., trust_remote_code=True)`` site —
    captioner (nb01), local VQA fallback, and the local VLM reranker all load
    the same Vintern remote code in different processes.
    """
    import types

    from transformers.modeling_utils import PreTrainedModel

    if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
        PreTrainedModel.all_tied_weights_keys = types.MappingProxyType({})


def resilient_from_pretrained(load, model_id: str):
    """Run ``load(model_id)``; on failure re-download to LOCAL disk and retry.

    The shared HF cache lives on Drive FUSE in symlink-less "degraded" mode
    and is read by several VMs — it can serve TORN files (live run 12 killed
    the ASR pipeline load that way). Mirror of the round-22 ASR fix for any
    from_pretrained-style loader: ``load`` is a callable taking a model id OR
    a local snapshot path.
    """
    import logging

    try:
        return load(model_id)
    except Exception as e:  # noqa: BLE001 — torn shared cache
        import os
        import tempfile

        logging.getLogger(__name__).warning(
            "HF load failed from the shared cache (%s) — re-downloading %s to "
            "LOCAL disk and retrying", e, model_id)
        from huggingface_hub import snapshot_download

        local = snapshot_download(
            model_id,
            cache_dir=os.path.join(tempfile.gettempdir(), "hf_resilient_cache"))
        return load(local)


def load_tokenizer(model_id: str):
    """Tokenizer for remote-code models across transformers generations.

    Vintern's docs say ``use_fast=False``, but transformers 5 removed several
    slow (pure-Python) tokenizer classes — if the slow path is gone, fall back
    to the default (fast) tokenizer rather than dying one line after the
    model finally loaded (verify-R15).
    """
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=False)
    except Exception:  # noqa: BLE001 — v5 slow-tokenizer removal
        return AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
