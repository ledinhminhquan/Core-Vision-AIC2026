"""Training-recipe upgrade tests: losses, datamodule extras, WiSE-FT math,
exact mid-epoch resume planning, and the public-channel parquet merge.

Pure CPU, tiny synthetic tensors, no HF downloads.
"""

from __future__ import annotations

import json
import logging
import math
import random
from itertools import islice
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# The README-documented dev env is torch-less — skip the whole module cleanly
# (the guard must run BEFORE any cvf.training import, they need torch too).
torch = pytest.importorskip("torch")
F = pytest.importorskip("torch.nn.functional")

from cvf.config import Settings  # noqa: E402
from cvf.training.datamodule import TextImageEmbedDataset, collate  # noqa: E402
from cvf.training.lit_trainer import (  # noqa: E402
    LiTTrainer,
    TrainConfig,
    plan_epoch_steps,
    wiseft_interpolate,
)
from cvf.training.losses import info_nce_loss, siglip_sigmoid_loss  # noqa: E402

DIM = 8


def _make_train_data(tmp_path: Path, n_videos: int = 3, frames: int = 4,
                     with_en: bool = True) -> Settings:
    """Synthetic train_data/: identifiable captions + identifiable embeddings."""
    settings = Settings.model_validate({
        "paths": {"data_root": str(tmp_path / "data"), "artifacts_root": str(tmp_path / "artifacts")},
    })
    out = settings.paths.art("train_data")
    out.mkdir(parents=True, exist_ok=True)
    rows, embs = [], []
    i = 0
    for v in range(n_videos):
        vid = f"L21_V{v + 1:03d}"
        for n in range(1, frames + 1):
            row = {"video_id": vid, "n": n,
                   "caption": f"chú thích tiếng việt số {i} của {vid}", "split": "train"}
            if with_en:
                row["caption_en"] = f"EN caption {i} of {vid}"
            rows.append(row)
            embs.append(np.full(DIM, float(i + 1), dtype=np.float32))  # row i ↔ value i+1
            i += 1
    pd.DataFrame(rows).to_parquet(out / "meta.parquet", index=False)
    np.save(out / "embeds.npy", np.stack(embs))
    return settings


# ── losses ───────────────────────────────────────────────────────────────────


def test_info_nce_matches_manual_bidirectional_ce():
    torch.manual_seed(0)
    img = F.normalize(torch.randn(4, DIM), dim=-1)
    txt = F.normalize(torch.randn(4, DIM), dim=-1)
    scale = torch.tensor(math.log(10.0))

    logits = txt @ img.t() * scale.exp()
    labels = torch.arange(4)
    manual = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))
    assert torch.allclose(info_nce_loss(img, txt, scale), manual, atol=1e-6)


def test_siglip_loss_without_negatives_unchanged():
    """No-negatives path must equal the original pairwise-sigmoid mean."""
    torch.manual_seed(1)
    img = F.normalize(torch.randn(4, DIM), dim=-1)
    txt = F.normalize(torch.randn(4, DIM), dim=-1)
    scale, bias = torch.tensor(math.log(10.0)), torch.tensor(-10.0)

    logits = txt @ img.t() * scale.exp() + bias
    labels = 2.0 * torch.eye(4) - 1.0
    manual = -F.logsigmoid(labels * logits).mean()
    assert torch.allclose(siglip_sigmoid_loss(txt, img, scale, bias), manual, atol=1e-6)


def test_hard_negative_logits_enlarge_loss_when_similar():
    torch.manual_seed(2)
    img = F.normalize(torch.randn(4, DIM), dim=-1)
    txt = F.normalize(img + 0.1 * torch.randn(4, DIM), dim=-1)
    scale, bias = torch.tensor(math.log(10.0)), torch.tensor(-10.0)

    base = siglip_sigmoid_loss(txt, img, scale, bias)
    neg_similar = txt.unsqueeze(1).repeat(1, 2, 1)                  # (B, 2, d) ≈ positives
    neg_random = F.normalize(torch.randn(4, 2, DIM), dim=-1)        # easy negatives
    loss_similar = siglip_sigmoid_loss(txt, img, scale, bias, neg_txt_emb=neg_similar)
    loss_random = siglip_sigmoid_loss(txt, img, scale, bias, neg_txt_emb=neg_random)

    assert loss_similar > loss_random     # harder negatives → larger loss
    assert loss_similar > base            # and larger than the negative-free loss


def test_hard_negatives_accept_flat_2d_pool():
    torch.manual_seed(3)
    img = F.normalize(torch.randn(3, DIM), dim=-1)
    txt = F.normalize(torch.randn(3, DIM), dim=-1)
    scale, bias = torch.tensor(math.log(10.0)), torch.tensor(-10.0)
    pool = F.normalize(torch.randn(5, DIM), dim=-1)                 # (M, d) vs every image
    loss = siglip_sigmoid_loss(txt, img, scale, bias, neg_txt_emb=pool)
    assert torch.isfinite(loss) and loss.item() > 0.0


# ── datamodule: English-anchor mixing ────────────────────────────────────────


def test_anchor_mix_ratio_respected(tmp_path: Path):
    settings = _make_train_data(tmp_path, with_en=True)
    ds = TextImageEmbedDataset(settings, "train", 0.0, anchor_mix_ratio=0.5, seed=7)
    served = [ds[i] for i in range(len(ds))]
    en_items = [(c, e) for c, e in served if c.startswith("EN caption")]
    assert len(en_items) == round(0.5 * len(ds)) == 6

    # anchor pairs keep (english_caption, image_emb) row alignment
    for cap, emb in en_items:
        row = int(cap.split()[2])                       # "EN caption {i} of {vid}"
        assert torch.allclose(emb, torch.full((DIM,), float(row + 1)))

    # ratio 0 → pure Vietnamese
    ds0 = TextImageEmbedDataset(settings, "train", 0.0, anchor_mix_ratio=0.0)
    assert not any(ds0[i][0].startswith("EN caption") for i in range(len(ds0)))


def test_anchor_mix_noop_without_english_sources(tmp_path: Path):
    settings = _make_train_data(tmp_path, with_en=False)
    ds = TextImageEmbedDataset(settings, "train", 0.0, anchor_mix_ratio=0.5)
    assert all(ds[i][0].startswith("chú thích") for i in range(len(ds)))


def test_anchor_parquet_provides_pairs(tmp_path: Path):
    settings = _make_train_data(tmp_path, with_en=False)
    anchor_pq = tmp_path / "anchors.parquet"
    pd.DataFrame({
        "caption_en": ["EN anchor one", "EN anchor two"],
        "embed": [np.full(DIM, 99.0, dtype=np.float32).tolist()] * 2,
    }).to_parquet(anchor_pq, index=False)

    ds = TextImageEmbedDataset(settings, "train", 0.0,
                               anchor_mix_ratio=0.25, anchor_parquet=anchor_pq, seed=3)
    served = [ds[i] for i in range(len(ds))]
    en_items = [(c, e) for c, e in served if c.startswith("EN anchor")]
    assert len(en_items) == round(0.25 * len(ds)) == 3
    for _, emb in en_items:
        assert torch.allclose(emb, torch.full((DIM,), 99.0))


def test_anchor_mix_warns_when_no_english_sources(tmp_path: Path, caplog):
    settings = _make_train_data(tmp_path, with_en=False)
    with caplog.at_level(logging.WARNING, logger="cvf.training.datamodule"):
        TextImageEmbedDataset(settings, "train", 0.0, anchor_mix_ratio=0.5)
    assert any("no-op" in r.message for r in caplog.records)

    caplog.clear()  # ratio 0 (and ratio>0 with sources) must stay silent
    settings_en = _make_train_data(tmp_path / "b", with_en=True)
    with caplog.at_level(logging.WARNING, logger="cvf.training.datamodule"):
        TextImageEmbedDataset(settings, "train", 0.0, anchor_mix_ratio=0.0)
        TextImageEmbedDataset(settings_en, "train", 0.0, anchor_mix_ratio=0.5)
    assert not caplog.records


def test_anchor_parquet_val_rows_excluded(tmp_path: Path):
    """Val-leakage guard: only split=='train' anchor-parquet rows are served."""
    settings = _make_train_data(tmp_path, with_en=False)
    anchor_pq = tmp_path / "anchors.parquet"
    pd.DataFrame({
        "caption_en": ["EN train anchor"] * 4 + ["EN VAL LEAK"] * 4,
        "embed": [np.full(DIM, 99.0, dtype=np.float32).tolist()] * 8,
        "split": ["train"] * 4 + ["val"] * 4,
    }).to_parquet(anchor_pq, index=False)

    ds = TextImageEmbedDataset(settings, "train", 0.0,
                               anchor_mix_ratio=0.5, anchor_parquet=anchor_pq, seed=3)
    served = [ds[i][0] for i in range(len(ds))]
    assert any(c == "EN train anchor" for c in served)     # train rows do flow
    assert all("VAL LEAK" not in c for c in served)        # val rows never do


def test_anchor_mix_train_split_only(tmp_path: Path):
    settings = _make_train_data(tmp_path, with_en=True)
    # rewrite one video's rows as val
    out = settings.paths.art("train_data")
    meta = pd.read_parquet(out / "meta.parquet")
    meta.loc[meta["video_id"] == "L21_V003", "split"] = "val"
    meta.to_parquet(out / "meta.parquet", index=False)

    val_ds = TextImageEmbedDataset(settings, "val", 0.0, anchor_mix_ratio=0.9)
    assert all(val_ds[i][0].startswith("chú thích") for i in range(len(val_ds)))


# ── datamodule: in-video hard negatives ──────────────────────────────────────


def test_hard_negatives_sampled_from_same_video(tmp_path: Path):
    settings = _make_train_data(tmp_path, with_en=False)
    ds = TextImageEmbedDataset(settings, "train", 0.0, hard_negative_per_sample=3)
    for idx in range(len(ds)):
        cap, _emb, negs = ds[idx]
        vid = ds.video_ids[idx]
        assert len(negs) == 3
        for neg in negs:
            assert neg != cap                       # never its own caption
            assert neg.endswith(vid)                # other moments of the SAME video


def test_hard_negatives_random_for_public_single_image_rows(tmp_path: Path):
    """Public-channel rows (siblings caption the SAME image) must fall back to
    random-row negatives — their same-id siblings are false negatives."""
    settings = _make_train_data(tmp_path, with_en=False)
    out = settings.paths.art("train_data")
    meta = pd.read_parquet(out / "meta.parquet")
    embeds = np.load(out / "embeds.npy")
    pub = pd.DataFrame([
        # non-corpus video_id pattern → public
        {"video_id": "ktvic:7", "n": 0, "source": "ktvic", "split": "train",
         "caption": f"public caption {i} of ktvic:7"} for i in range(4)
    ] + [
        # corpus-LOOKING video_id but explicit public source → still public
        {"video_id": "L99_V001", "n": 0, "source": "uit_viic", "split": "train",
         "caption": f"public caption {i} of L99_V001"} for i in range(3)
    ])
    pd.concat([meta, pub], ignore_index=True).to_parquet(out / "meta.parquet", index=False)
    np.save(out / "embeds.npy",
            np.concatenate([embeds, np.full((7, DIM), 77.0, dtype=np.float32)]))

    random.seed(0)
    ds = TextImageEmbedDataset(settings, "train", 0.0, hard_negative_per_sample=4)
    # public groups are never candidates for same-'video' negatives
    assert "ktvic:7" not in ds._video_rows
    assert "L99_V001" not in ds._video_rows
    # corpus rows keep their in-video hard negatives
    _cap, _emb, negs = ds[0]
    assert all(neg.endswith(str(ds.video_ids[0])) for neg in negs)
    # public rows draw from ALL other rows (old behavior: siblings only)
    pub_idx = ds.video_ids.index("ktvic:7")
    drawn = [neg for _ in range(10) for neg in ds[pub_idx][2]]
    assert any(not neg.startswith("public caption") for neg in drawn)
    assert all(neg != ds.captions[pub_idx] for neg in drawn)  # never itself


def test_collate_backward_compatible_and_flat_negatives(tmp_path: Path):
    settings = _make_train_data(tmp_path, with_en=False)
    plain = TextImageEmbedDataset(settings, "train", 0.0)
    out = collate([plain[0], plain[1]])
    assert len(out) == 2                            # legacy 2-tuple preserved
    texts, embs = out
    assert len(texts) == 2 and embs.shape == (2, DIM)

    hard = TextImageEmbedDataset(settings, "train", 0.0, hard_negative_per_sample=2)
    texts, embs, neg_texts = collate([hard[0], hard[1]])
    assert len(texts) == 2 and embs.shape == (2, DIM)
    assert len(neg_texts) == 4                      # flat B·K list


# ── WiSE-FT interpolation ────────────────────────────────────────────────────


def test_wiseft_interpolation_endpoints_and_midpoint():
    base = {"w": torch.zeros(3), "b": torch.tensor([1.0, 2.0])}
    tuned = {"w": torch.ones(3), "b": torch.tensor([3.0, 4.0])}

    a0 = wiseft_interpolate(base, tuned, 0.0)
    assert torch.equal(a0["w"], base["w"]) and torch.equal(a0["b"], base["b"])

    a1 = wiseft_interpolate(base, tuned, 1.0)
    assert torch.equal(a1["w"], tuned["w"]) and torch.equal(a1["b"], tuned["b"])

    mid = wiseft_interpolate(base, tuned, 0.5)
    assert torch.allclose(mid["w"], torch.full((3,), 0.5))
    assert torch.allclose(mid["b"], torch.tensor([2.0, 3.0]))


def test_wiseft_interpolation_copies_buffers_and_unknown_keys():
    base = {"w": torch.zeros(2)}
    tuned = {"w": torch.ones(2), "position_ids": torch.arange(4), "extra": torch.full((2,), 7.0)}
    out = wiseft_interpolate(base, tuned, 0.25)
    assert torch.equal(out["position_ids"], torch.arange(4))    # int buffer: tuned as-is
    assert torch.equal(out["extra"], tuned["extra"])            # missing in base: tuned as-is
    assert torch.allclose(out["w"], torch.full((2,), 0.25))


# ── exact mid-epoch resume planning ──────────────────────────────────────────


def test_plan_epoch_steps_math():
    S, T, A = 10, 30, 4  # steps/epoch, total steps (3 epochs), grad_accum
    # fresh start and epoch boundaries: nothing to skip, a full epoch to run
    assert plan_epoch_steps(0, S, T, A) == (0, 10)
    assert plan_epoch_steps(10, S, T, A) == (0, 10)
    # killed 3 steps into epoch 1 (step 13): skip the 3*4 already-trained
    # micro-batches, run only the 7 remaining optimizer steps
    assert plan_epoch_steps(13, S, T, A) == (12, 7)
    # remaining-budget cap binds when total_steps is not an epoch multiple
    assert plan_epoch_steps(23, S, 25, A) == (12, 2)
    # at/after the budget: nothing left to run → the loop must stop
    assert plan_epoch_steps(30, S, T, A) == (0, 0)
    assert plan_epoch_steps(31, S, T, A)[1] == 0


def test_plan_epoch_steps_resume_never_double_trains_or_overshoots():
    """Simulate the trainer's epoch loop for every possible resume step: the
    run must always finish at exactly total_steps and train exactly the
    remaining (total_steps - resume_at) * grad_accum micro-batches."""
    S, T, A, epochs = 10, 30, 4, 3
    for resume_at in range(T + 2):
        step, micro_trained = resume_at, 0
        for _epoch in range(resume_at // S, epochs):
            skip, run = plan_epoch_steps(step, S, T, A)
            if run <= 0:
                break
            assert skip + run * A <= S * A  # never reads past one epoch of data
            micro_trained += run * A
            step += run
        assert step == (T if resume_at <= T else resume_at)  # no overshoot
        assert micro_trained == max(0, T - resume_at) * A    # no double-training


def _mini_cfg(tmp_path: Path) -> TrainConfig:
    return TrainConfig(run_dir=str(tmp_path / "run"), micro_batch=2, num_workers=0, seed=42)


def test_resume_replays_same_epoch_permutation(tmp_path: Path):
    """Crash-resume sample identity: epoch E's shuffle is deterministic given
    (seed, E), so islice fast-forwarding skips EXACTLY the trained samples."""
    settings = _make_train_data(tmp_path, n_videos=4, frames=4, with_en=False)
    ds = TextImageEmbedDataset(settings, "train", 0.0)

    trainer = LiTTrainer(settings, _mini_cfg(tmp_path))
    loader = trainer._build_loader(ds, "cpu")
    trainer._reseed_epoch(0)
    full = [c for texts, _ in loader for c in texts]        # the crashed run's epoch 0

    skip = 3                                                # died after 3 micro-batches
    trainer2 = LiTTrainer(settings, _mini_cfg(tmp_path))    # fresh process on resume
    loader2 = trainer2._build_loader(ds, "cpu")
    trainer2._reseed_epoch(0)
    resumed = [c for texts, _ in islice(loader2, skip, None) for c in texts]
    assert resumed == full[skip * 2:]                       # micro_batch=2

    trainer2._reseed_epoch(1)                               # epochs still differ
    epoch1 = [c for texts, _ in loader2 for c in texts]
    assert epoch1 != full and sorted(epoch1) == sorted(full)


def test_resume_warns_on_train_size_mismatch(tmp_path: Path, caplog):
    settings = _make_train_data(tmp_path, with_en=False)
    trainer = LiTTrainer(settings, _mini_cfg(tmp_path))
    trainer._train_size = 100

    with caplog.at_level(logging.WARNING, logger="cvf.training.lit_trainer"):
        trainer._warn_train_size_mismatch({"train_size": 80}, "step-200")
    assert any("RESUME MISMATCH" in r.message for r in caplog.records)

    caplog.clear()  # matching size / old checkpoint without the key: silent
    with caplog.at_level(logging.WARNING, logger="cvf.training.lit_trainer"):
        trainer._warn_train_size_mismatch({"train_size": 100}, "step-200")
        trainer._warn_train_size_mismatch({"step": 200}, "step-200")
    assert not caplog.records


def test_wiseft_alpha_sweep_includes_raw_tuned_tower():
    # α=1.0 IS the tuned tower, so wiseft_best can never score below it.
    assert TrainConfig().wiseft_alphas[-1] == 1.0


# ── public datasets: caption filter + parquet merge ──────────────────────────


def test_filter_captions_policy():
    from cvf.training.public_datasets import _filter_captions

    caps = [
        "  một   người  đàn ông  ",       # whitespace-normalised, kept
        "ngắn quá",                        # 2 words < min → dropped
        "MỘT NGƯỜI ĐÀN ÔNG",               # case-insensitive dup → dropped
        " ".join(["dài"] * 50),            # 50 words > max → dropped
        "",
    ]
    assert _filter_captions(caps) == ["một người đàn ông"]
    assert _filter_captions(None) == []


def test_default_public_parquet_path(tmp_path: Path):
    from cvf.training.public_datasets import default_public_parquet_path

    settings = Settings.model_validate({"paths": {"artifacts_root": str(tmp_path / "art")}})
    p = default_public_parquet_path("ktvic", settings)
    assert p == tmp_path / "art" / "train_data" / "public" / "extra_ktvic.parquet"


def _write_captions(settings: Settings, videos: dict[str, int]) -> None:
    cap_dir = settings.paths.art("captions")
    cap_dir.mkdir(parents=True, exist_ok=True)
    for vid, n_frames in videos.items():
        n_to_caption = {
            str(n): f"cảnh quay số {n} trong video {vid} bản tin"
            for n in range(1, n_frames + 1)
        }
        (cap_dir / f"{vid}.json").write_text(
            json.dumps({"n_to_caption": n_to_caption}, ensure_ascii=False), encoding="utf-8")


CORPUS_DIM = 16  # matches tests/conftest.py DIM
CORPUS_VIDEOS = {"L21_V001": 6, "L21_V002": 5, "K01_V001": 4}  # matches conftest VIDEOS


def test_build_training_set_merges_extra_parquets(corpus_with_index: Settings):
    from cvf.training.build_dataset import build_training_set

    _write_captions(corpus_with_index, CORPUS_VIDEOS)
    public_dir = corpus_with_index.paths.art("train_data", "public")
    public_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "embed": [np.full(CORPUS_DIM, 0.5, dtype=np.float32).tolist()] * 3,
        "caption": ["một em bé chơi bóng đá", "hai cầu thủ tranh bóng", "khán giả reo hò trên khán đài"],
        "caption_en": ["a child plays football", "", "spectators cheering"],
        "video_id": ["ktvic:0", "ktvic:0", "ktvic:1"],
        "source": ["ktvic"] * 3,
        "domain": ["daily_life"] * 3,
        "split": ["train", "train", "val"],
    }).to_parquet(public_dir / "extra_test.parquet", index=False)

    n_train, n_val = build_training_set(corpus_with_index, model_key="fake")
    out = corpus_with_index.paths.art("train_data")
    meta = pd.read_parquet(out / "meta.parquet")
    embeds = np.load(out / "embeds.npy")

    assert len(meta) == len(embeds) == n_train + n_val
    assert "caption_en" in meta.columns
    merged = meta[meta["video_id"].str.startswith("ktvic:")]
    assert len(merged) == 3
    assert (merged["split"] == "train").sum() == 2
    assert set(merged["caption_en"]) == {"a child plays football", "", "spectators cheering"}
    # merged embeddings carried over verbatim (float16 rows of 0.5)
    merged_rows = embeds[meta["video_id"].str.startswith("ktvic:").to_numpy()]
    assert np.allclose(merged_rows.astype(np.float32), 0.5)
    # corpus rows have empty caption_en
    aic = meta[~meta["video_id"].str.startswith("ktvic:")]
    assert (aic["caption_en"] == "").all()


def test_build_training_set_rejects_dim_mismatch(corpus_with_index: Settings):
    from cvf.training.build_dataset import build_training_set

    _write_captions(corpus_with_index, CORPUS_VIDEOS)
    public_dir = corpus_with_index.paths.art("train_data", "public")
    public_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "embed": [np.zeros(4, dtype=np.float32).tolist()],   # wrong dim (corpus is 16)
        "caption": ["một chú thích hợp lệ đủ dài"],
        "video_id": ["bad:0"],
        "split": ["train"],
    }).to_parquet(public_dir / "extra_bad.parquet", index=False)

    with pytest.raises(ValueError, match="dim"):
        build_training_set(corpus_with_index, model_key="fake")
