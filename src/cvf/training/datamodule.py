"""Dataset over (caption, precomputed image embedding) pairs."""

from __future__ import annotations

import logging
import os
import random
import re

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from cvf.config import Settings

log = logging.getLogger(__name__)

# Organiser corpus video ids (e.g. L21_V001). Rows that DON'T match come from
# merged public single-image channels (video_id like "ktvic:0") whose siblings
# are captions of the SAME image — useless (false-negative) hard negatives.
_CORPUS_VID_RE = re.compile(r"^[A-Z]\d{2}_V\d{3}$")


class TextImageEmbedDataset(Dataset):
    """Rows of train_data/: caption text + frozen image embedding.

    ``word_dropout`` randomly removes words from captions at train time — the
    cheapest augmentation that reliably fights caption-style overfitting
    (auto-captions share phrasing quirks the model would otherwise latch onto).

    Optional training-recipe extensions (all off by default, keyword-only):

    * ``anchor_mix_ratio`` — interleave that exact fraction of
      (english_caption, image_emb) anchor pairs into the served items
      (train split only). Anchors come from a non-empty ``caption_en``
      column of ``meta.parquet`` and/or a second ``anchor_parquet`` with
      ``caption_en``/``caption`` + ``embed`` columns. English anchors keep
      the tuned tower's English/base alignment alive and compose with the
      teacher-cosine distillation anchor in the trainer.
    * ``hard_negative_per_sample`` — each item additionally returns K captions
      sampled from OTHER moments of the SAME video (visually-similar hard
      negatives; falls back to random other rows for 1-moment videos and for
      public single-image channel rows, whose same-id siblings caption the
      SAME image). The sigmoid loss consumes them as extra negative logits.
    """

    def __init__(
        self,
        settings: Settings,
        split: str = "train",
        word_dropout: float = 0.05,
        *,
        anchor_mix_ratio: float = 0.0,
        anchor_parquet: str | os.PathLike | None = None,
        hard_negative_per_sample: int = 0,
        seed: int = 42,
    ):
        data_dir = settings.paths.art("train_data")
        meta = pd.read_parquet(data_dir / "meta.parquet")
        embeds = np.load(data_dir / "embeds.npy")
        mask = (meta["split"] == split).to_numpy()
        sub = meta.loc[mask]
        self.captions = sub["caption"].tolist()
        self.video_ids = sub["video_id"].tolist()
        self.embeds = np.ascontiguousarray(embeds[mask]).astype(np.float32)
        self.word_dropout = word_dropout if split == "train" else 0.0
        self.hard_negative_per_sample = max(0, int(hard_negative_per_sample))

        # Same-video hard negatives only make sense for corpus rows: public
        # single-image channels group captions of ONE image under one
        # video_id, so their "siblings" describe the exact positive image.
        sources = (sub["source"].fillna("").astype(str).tolist()
                   if "source" in sub.columns else [""] * len(self.captions))
        self._corpus_row = [
            src in ("", "corpus") and bool(_CORPUS_VID_RE.match(str(vid)))
            for vid, src in zip(self.video_ids, sources)
        ]
        self._video_rows: dict[str, list[int]] = {}
        if self.hard_negative_per_sample > 0:
            for i, vid in enumerate(self.video_ids):
                if self._corpus_row[i]:
                    self._video_rows.setdefault(str(vid), []).append(i)

        # English-anchor pool: caption_en column and/or a second anchor parquet.
        anchor_caps: list[str] = []
        anchor_embs: list[np.ndarray] = []
        if "caption_en" in sub.columns:
            en = sub["caption_en"].fillna("").astype(str).tolist()
            for row_pos, txt in enumerate(en):
                txt = txt.strip()
                if txt:
                    anchor_caps.append(txt)
                    anchor_embs.append(self.embeds[row_pos])
        if anchor_parquet is not None:
            extra = pd.read_parquet(anchor_parquet)
            if "split" in extra.columns:  # val-leakage guard: train anchors only
                extra = extra.loc[extra["split"] == "train"]
            else:
                log.info("Anchor parquet %s has no 'split' column — using all rows", anchor_parquet)
            cap_col = "caption_en" if "caption_en" in extra.columns else "caption"
            for cap, emb in zip(extra[cap_col].tolist(), extra["embed"].tolist()):
                cap = str(cap or "").strip()
                if cap:
                    anchor_caps.append(cap)
                    anchor_embs.append(np.asarray(emb, dtype=np.float32))
        self._anchor_captions = anchor_caps
        self._anchor_embeds = (
            np.ascontiguousarray(np.stack(anchor_embs)).astype(np.float32) if anchor_embs else None
        )

        # Exact interleave: round(ratio·N) item slots serve an anchor pair
        # instead of their Vietnamese row (deterministic given ``seed``).
        self._anchor_slot: dict[int, int] = {}
        ratio = min(max(float(anchor_mix_ratio or 0.0), 0.0), 1.0)
        if split == "train" and ratio > 0 and not anchor_caps:
            log.warning(
                "anchor_mix_ratio=%.2f requested but no English captions available "
                "(no non-empty caption_en values and no anchor_parquet rows) — "
                "anchor mixing is a no-op", ratio,
            )
        if split == "train" and ratio > 0 and anchor_caps and len(self.captions) > 0:
            rng = np.random.default_rng(seed)
            n_anchor = min(int(round(ratio * len(self.captions))), len(self.captions))
            slots = rng.choice(len(self.captions), size=n_anchor, replace=False)
            pool = rng.integers(0, len(anchor_caps), size=n_anchor)
            self._anchor_slot = {int(s): int(p) for s, p in zip(slots, pool)}

    def __len__(self) -> int:
        return len(self.captions)

    def _augment(self, caption: str) -> str:
        if self.word_dropout > 0:
            words = caption.split()
            if len(words) > 4:
                kept = [w for w in words if random.random() > self.word_dropout]
                caption = " ".join(kept) if len(kept) >= 3 else caption
        return caption

    def _sample_negatives(self, idx: int) -> list[str]:
        """K captions from other moments of the same video (fallback: any other row).

        Public-channel rows never get same-'video' negatives (their siblings
        caption the SAME image); they always take the random-row fallback.
        """
        k = self.hard_negative_per_sample
        if len(self.captions) <= 1:
            return [self.captions[idx]] * k
        same = [r for r in self._video_rows.get(str(self.video_ids[idx]), []) if r != idx]
        negs: list[str] = []
        for _ in range(k):
            if same:
                r = same[random.randrange(len(same))]
            else:
                r = random.randrange(len(self.captions) - 1)
                if r >= idx:
                    r += 1
            negs.append(self.captions[r])
        return negs

    def __getitem__(self, idx: int):
        pool_idx = self._anchor_slot.get(idx)
        if pool_idx is not None and self._anchor_embeds is not None:
            caption = self._anchor_captions[pool_idx]           # English anchors stay verbatim
            emb = torch.from_numpy(self._anchor_embeds[pool_idx])
        else:
            caption = self._augment(self.captions[idx])
            emb = torch.from_numpy(self.embeds[idx])
        if self.hard_negative_per_sample <= 0:
            return caption, emb
        return caption, emb, self._sample_negatives(idx)


def collate(batch: list[tuple]) -> tuple:
    """(texts, embs) — plus a flat ``B·K`` negative-caption list when items carry one."""
    texts = [b[0] for b in batch]
    embs = torch.stack([b[1] for b in batch])
    if len(batch[0]) < 3:
        return texts, embs
    neg_texts = [t for b in batch for t in b[2]]
    return texts, embs, neg_texts
