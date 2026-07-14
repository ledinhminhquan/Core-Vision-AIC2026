"""Training losses for Vietnamese text-tower tuning."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def siglip_sigmoid_loss(
    text_emb: torch.Tensor,   # (B, d) L2-normalized
    image_emb: torch.Tensor,  # (B, d) L2-normalized
    logit_scale: torch.Tensor,
    logit_bias: torch.Tensor,
    neg_txt_emb: torch.Tensor | None = None,
) -> torch.Tensor:
    """SigLIP pairwise sigmoid loss (Zhai et al. 2023).

    Every (text_i, image_j) pair is an independent binary problem: label +1 on
    the diagonal, −1 elsewhere. ``logit_scale``/``logit_bias`` are the model's
    own learnable scalars (init t' = log 10, b = −10 in the paper) — reuse the
    pretrained values, do NOT re-init, or training collapses early.

    Args:
        neg_txt_emb: optional explicit hard-negative caption embeddings,
            L2-normalized. Shape ``(B, K, d)`` pairs sample ``i``'s K negatives
            (captions from OTHER moments of the same video) against image ``i``;
            shape ``(M, d)`` scores every negative against every image. All
            extra logits get label −1 and the loss stays the mean over ALL
            binary terms (in-batch pairs + hard negatives), keeping SigLIP's
            per-pair independent-classification form.
    """
    scale = logit_scale.exp()
    logits = text_emb @ image_emb.t() * scale + logit_bias
    labels = 2.0 * torch.eye(logits.size(0), device=logits.device, dtype=logits.dtype) - 1.0
    pair_losses = -F.logsigmoid(labels * logits)
    if neg_txt_emb is None:
        return pair_losses.mean()

    if neg_txt_emb.dim() == 3:      # (B, K, d): negative k of sample i vs image i
        neg_logits = torch.einsum("bkd,bd->bk", neg_txt_emb, image_emb) * scale + logit_bias
    else:                           # (M, d): every negative vs every image
        neg_logits = neg_txt_emb @ image_emb.t() * scale + logit_bias
    neg_losses = -F.logsigmoid(-neg_logits)    # label −1 everywhere
    total = pair_losses.sum() + neg_losses.sum()
    return total / (pair_losses.numel() + neg_losses.numel())


def info_nce_loss(
    img_emb: torch.Tensor,    # (B, d) L2-normalized
    txt_emb: torch.Tensor,    # (B, d) L2-normalized
    logit_scale: torch.Tensor,
) -> torch.Tensor:
    """Bidirectional InfoNCE / CLIP softmax loss (Radford et al. 2021).

    Temperature comes from the checkpoint's own ``logit_scale`` (log-space, as
    stored by SigLIP/CLIP) so an A/B swap against the sigmoid loss starts from
    the pretrained operating point instead of a cold re-init.
    """
    logits = txt_emb @ img_emb.t() * logit_scale.exp()   # (B, B): text→image
    labels = torch.arange(logits.size(0), device=logits.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


def distill_cosine_loss(student_emb: torch.Tensor, teacher_emb: torch.Tensor) -> torch.Tensor:
    """1 − cos(student, teacher): anchors the tuned tower to the base model so
    English + general alignment is not catastrophically forgotten."""
    return (1.0 - F.cosine_similarity(student_emb, teacher_emb, dim=-1)).mean()
