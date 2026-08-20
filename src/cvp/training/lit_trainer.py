"""LoRA-LiT trainer: tune SigLIP-2's TEXT tower on Vietnamese captions.

Design (evidence-driven, see docs/TRAINING.md):
* **Image tower frozen** — image embeddings are precomputed once; the FAISS
  index built from them stays valid after training (LiT, Zhai et al. 2022).
* **LoRA on the text tower** (+ fully-trained projection head) instead of full
  fine-tuning — preserves the multilingual base against catastrophic
  forgetting; a distillation anchor to the base tower (weight decays to 0)
  adds a second guard.
* **SigLIP sigmoid loss** with the model's own pretrained logit_scale/bias
  kept learnable — re-initializing them collapses training. An InfoNCE A/B
  option (``TrainConfig.loss = "infonce"``) reuses the same temperature.
* **English-anchor mixing** (``anchor_mix_ratio``) + optional in-video hard
  negatives (``hard_negative_per_sample``) — see datamodule.
* Crash-safe: atomic checkpoints on Drive every eval; auto-resume finds the
  latest VALID checkpoint (Colab disconnects mid-write are expected) and
  fast-forwards the dataloader past the micro-batches the interrupted epoch
  already consumed (``plan_epoch_steps``) — no sample is double-trained and
  ``step`` never overshoots ``total_steps``, so the cosine LR schedule and
  the distill-weight decay stay on schedule.

The exported artifact is a merged text-tower ``text_tower.safetensors`` that
``cvp.models.siglip2`` grafts at load time (``embedding.model: finetuned``).
After training, a **WiSE-FT sweep** (Wortsman et al. 2022) interpolates the
best tuned tower with the base tower at ``wiseft_alphas``, evaluates each on
the val split, and exports the winner to ``export_dir/wiseft_best``.
"""

from __future__ import annotations

import logging
import math
import os
import random
import shutil
import time
from dataclasses import asdict, dataclass, field
from itertools import islice
from pathlib import Path

import numpy as np

from cvp.eval.metrics import retrieval_metrics
from cvp.training.datamodule import TextImageEmbedDataset, collate
from cvp.training.losses import distill_cosine_loss, info_nce_loss, siglip_sigmoid_loss
from cvp.utils.io import atomic_write_json, read_json

log = logging.getLogger(__name__)


def plan_epoch_steps(global_step: int, steps_per_epoch: int, total_steps: int,
                     grad_accum: int) -> tuple[int, int]:
    """Plan one epoch of the (possibly mid-epoch-resumed) training loop.

    Returns ``(skip_micro, run_steps)``:
      * ``skip_micro`` — micro-batches the interrupted epoch already consumed
        (``(global_step % steps_per_epoch) * grad_accum``); the caller fast-
        forwards the dataloader past them so no sample is trained twice.
      * ``run_steps`` — optimizer steps to take this epoch: what is left of
        the epoch, capped by the run's remaining budget ``total_steps -
        global_step`` so a resume can never push ``global_step`` past
        ``total_steps`` (the cosine LR would climb back up and the
        distill-weight decay would extrapolate past its endpoint).

    At an epoch boundary ``skip_micro == 0`` and a full epoch is planned, so
    the fresh-start path is just the degenerate case of the resumed one.
    """
    done_in_epoch = global_step % steps_per_epoch
    remaining_in_epoch = steps_per_epoch - done_in_epoch
    remaining_total = max(0, total_steps - global_step)
    return done_in_epoch * grad_accum, min(remaining_in_epoch, remaining_total)


def wiseft_interpolate(base_state: dict, tuned_state: dict, alpha: float) -> dict:
    """WiSE-FT weight-space ensemble: ``(1−α)·base + α·tuned`` per tensor.

    ``alpha=0`` returns the base weights exactly, ``alpha=1`` the tuned ones.
    Non-floating tensors (e.g. position-id buffers) and keys missing from
    ``base_state`` are copied from ``tuned_state`` unchanged.
    """
    out: dict = {}
    for key, tuned in tuned_state.items():
        base = base_state.get(key)
        if base is None or not tuned.is_floating_point():
            out[key] = tuned.clone()
        else:
            out[key] = base.to(dtype=tuned.dtype, device=tuned.device) * (1.0 - alpha) + tuned * alpha
    return out


@dataclass
class TrainConfig:
    base_id: str = "google/siglip2-so400m-patch16-384"
    train_data_dir: str = "./artifacts/train_data"
    run_dir: str = "./artifacts/runs/vi_siglip2"          # checkpoints (put on Drive)
    export_dir: str = "./artifacts/checkpoints/vi_siglip2_best"

    epochs: int = 8
    micro_batch: int = 256          # auto-sized by GPU in the notebook
    grad_accum: int = 4
    lr: float = 1.0e-4              # LoRA adapters + head
    lr_scalars: float = 1.0e-5      # logit_scale / logit_bias
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    max_grad_norm: float = 1.0

    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_targets: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
    )

    distill_beta0: float = 0.30     # anchor weight at step 0 → decays linearly to 0
    word_dropout: float = 0.05
    max_text_len: int = 64

    # Loss recipe (design §Training upgrades).
    loss: str = "siglip"                    # siglip | infonce (A/B flag)
    anchor_mix_ratio: float = 0.25          # fraction of (EN caption, image) anchor pairs
    anchor_parquet: str | None = None       # optional second anchor parquet (caption_en+embed)
    hard_negative_per_sample: int = 0       # K in-video hard negatives per sample (0 = off)
    # WiSE-FT: α sweep evaluated on val after training; winner → export_dir/wiseft_best.
    # α=1.0 is the raw tuned tower, so wiseft_best can never score below it.
    wiseft_alphas: list[float] = field(default_factory=lambda: [0.4, 0.5, 0.6, 1.0])

    eval_every_steps: int = 200
    early_stop_patience: int = 5    # evals without val R@5 improvement
    keep_checkpoints: int = 2
    seed: int = 42
    num_workers: int = 2


class LiTTrainer:
    def __init__(self, settings, cfg: TrainConfig):
        self.settings = settings
        self.cfg = cfg
        self.run_dir = Path(cfg.run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._base_text_state: dict | None = None   # pristine text tower for WiSE-FT
        self._train_size: int | None = None         # dataset length, checkpointed
        self._loader_generator = None               # per-epoch reseeded shuffle RNG

    # ── setup ────────────────────────────────────────────────────────────

    def _seed(self) -> None:
        import torch

        random.seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        torch.manual_seed(self.cfg.seed)

    def _build_model(self):
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoProcessor, SiglipModel

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32
        log.info("Loading %s (%s, %s)", self.cfg.base_id, device, dtype)
        model = SiglipModel.from_pretrained(self.cfg.base_id, torch_dtype=torch.float32)
        processor = AutoProcessor.from_pretrained(self.cfg.base_id)

        for p in model.parameters():
            p.requires_grad_(False)

        # Snapshot the pristine text tower (CPU fp32) BEFORE LoRA wrapping —
        # WiSE-FT interpolates against exactly these weights at export time.
        self._base_text_state = {
            k: v.detach().cpu().clone() for k, v in model.text_model.state_dict().items()
        }

        lora_cfg = LoraConfig(
            r=self.cfg.lora_r,
            lora_alpha=self.cfg.lora_alpha,
            lora_dropout=self.cfg.lora_dropout,
            target_modules=self.cfg.lora_targets,
            modules_to_save=["head"],   # SigLIP text projection head — fully trained
            bias="none",
        )
        text_model = get_peft_model(model.text_model, lora_cfg)
        model.text_model = text_model

        # The sigmoid loss scalars stay learnable (pretrained values kept).
        model.logit_scale.requires_grad_(True)
        model.logit_bias.requires_grad_(True)

        model = model.to(device)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log.info("Trainable parameters: %.2fM", trainable / 1e6)
        return model, processor, device, dtype

    def _build_loader(self, train_ds: TextImageEmbedDataset, device: str):
        """Train loader with an EXPLICIT shuffle generator (see _reseed_epoch)."""
        import torch
        from torch.utils.data import DataLoader

        self._loader_generator = torch.Generator()
        return DataLoader(
            train_ds, batch_size=self.cfg.micro_batch, shuffle=True, collate_fn=collate,
            num_workers=self.cfg.num_workers, pin_memory=(device == "cuda"), drop_last=True,
            persistent_workers=self.cfg.num_workers > 0, generator=self._loader_generator,
        )

    def _reseed_epoch(self, epoch: int) -> None:
        """Make epoch ``epoch``'s shuffle permutation deterministic given (seed, epoch).

        A mid-epoch resume fast-forwards the loader with islice — that only
        skips the RIGHT samples if the resumed epoch replays the exact
        permutation the crashed run drew, which the default (global-RNG-seeded)
        DataLoader generator does not guarantee.
        """
        self._loader_generator.manual_seed(self.cfg.seed * 100003 + epoch)

    # ── checkpointing ────────────────────────────────────────────────────

    def _ckpt_dirs(self) -> list[Path]:
        return sorted(
            (d for d in self.run_dir.glob("step-*") if (d / "state.json").is_file()),
            key=lambda d: int(d.name.split("-")[1]),
        )

    def _save_checkpoint(self, model, optimizer, scheduler, step: int, best_r5: float,
                         evals_since_best: int) -> None:
        import torch

        tmp = self.run_dir / f".tmp-step-{step}"
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        model.text_model.save_pretrained(tmp / "adapter")
        torch.save(
            {
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "logit_scale": model.logit_scale.detach().cpu(),
                "logit_bias": model.logit_bias.detach().cpu(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "numpy_rng": np.random.get_state(),
                "py_rng": random.getstate(),
            },
            tmp / "extras.pt",
        )
        atomic_write_json(tmp / "state.json", {
            "step": step, "best_r5": best_r5, "evals_since_best": evals_since_best,
            "train_size": self._train_size, "time": time.time(),
        })
        final = self.run_dir / f"step-{step}"
        if final.exists():
            shutil.rmtree(final)
        os.replace(tmp, final)
        # prune old checkpoints (never the best-marker dir)
        for old in self._ckpt_dirs()[: -self.cfg.keep_checkpoints]:
            shutil.rmtree(old, ignore_errors=True)
        log.info("Checkpoint saved at step %d", step)

    def _warn_train_size_mismatch(self, state: dict, ckpt_name: str) -> None:
        """LOUD warning when the dataset changed under a resumed run.

        Old checkpoints predate the ``train_size`` key — tolerated silently.
        """
        saved_size = state.get("train_size")
        if saved_size is None or self._train_size is None:
            return
        if int(saved_size) != int(self._train_size):
            log.warning(
                "!!! RESUME MISMATCH !!! checkpoint %s was saved with %d train rows but the "
                "current dataset has %d — train_data was rebuilt, so steps_per_epoch, the LR "
                "schedule, the distill-weight decay and the shuffle order have all shifted. "
                "Strongly consider a FRESH run_dir instead of resuming.",
                ckpt_name, int(saved_size), int(self._train_size),
            )

    def _try_resume(self, model, optimizer, scheduler) -> tuple[int, float, int]:
        """Resume from the NEWEST checkpoint that actually loads.

        verify-R23: Drive FUSE can serve TORN checkpoint files — one bad
        ``adapter_model.safetensors``/``extras.pt`` must fall back to the
        previous step dir, not wedge every future Run-all on the same crash.
        On total failure the virgin adapter weights are restored so a fresh
        start really is fresh.
        """
        from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict

        dirs = self._ckpt_dirs()
        if not dirs:
            return 0, -1.0, 0
        _virgin = {k: v.detach().clone()
                   for k, v in get_peft_model_state_dict(model.text_model).items()}
        for ckpt in reversed(dirs):
            try:
                return self._resume_from(model, optimizer, scheduler, ckpt)
            except Exception as e:  # noqa: BLE001 — torn checkpoint → older one
                log.warning(
                    "Checkpoint %s không đọc được (%s) — thử checkpoint cũ hơn",
                    ckpt.name, e)
        log.warning("Không checkpoint nào đọc được — chạy lại từ đầu (adapter "
                    "được khôi phục về trạng thái ban đầu)")
        set_peft_model_state_dict(model.text_model, _virgin)
        return 0, -1.0, 0

    def _resume_from(self, model, optimizer, scheduler, ckpt) -> tuple[int, float, int]:
        import torch

        state = read_json(ckpt / "state.json")
        self._warn_train_size_mismatch(state, ckpt.name)

        # Load adapter (+ modules_to_save) weights INTO the already-wrapped
        # PeftModel — load_adapter() would collide with the existing "default".
        from peft.utils import set_peft_model_state_dict
        from safetensors.torch import load_file

        adapter_file = ckpt / "adapter" / "adapter_model.safetensors"
        adapter_state = load_file(str(adapter_file))
        set_peft_model_state_dict(model.text_model, adapter_state)
        extras = torch.load(ckpt / "extras.pt", map_location="cpu", weights_only=False)
        optimizer.load_state_dict(extras["optimizer"])
        scheduler.load_state_dict(extras["scheduler"])
        with torch.no_grad():
            model.logit_scale.copy_(extras["logit_scale"].to(model.logit_scale.device))
            model.logit_bias.copy_(extras["logit_bias"].to(model.logit_bias.device))
        torch.set_rng_state(extras["torch_rng"])
        if extras.get("cuda_rng") is not None and torch.cuda.is_available():
            try:
                torch.cuda.set_rng_state_all(extras["cuda_rng"])
            except RuntimeError:
                pass
        np.random.set_state(extras["numpy_rng"])
        random.setstate(extras["py_rng"])
        log.info("Resumed from %s (step %d, best R@5 %.4f)", ckpt.name, state["step"], state["best_r5"])
        return int(state["step"]), float(state["best_r5"]), int(state["evals_since_best"])

    # ── eval ─────────────────────────────────────────────────────────────

    def _encode_with_module(self, text_module, processor, device, dtype, texts: list[str],
                            batch: int = 256) -> np.ndarray:
        """Encode ``texts`` with an arbitrary SigLIP text tower → (N, d) L2-normed."""
        import torch

        outs = []
        with torch.no_grad():
            for i in range(0, len(texts), batch):
                toks = processor.tokenizer(
                    texts[i : i + batch], padding="max_length", truncation=True,
                    max_length=self.cfg.max_text_len, return_tensors="pt",
                ).to(device)
                with torch.autocast(device_type="cuda", dtype=dtype, enabled=(device == "cuda")):
                    emb = text_module(**toks).pooler_output
                emb = torch.nn.functional.normalize(emb.float(), dim=-1)
                outs.append(emb.cpu().numpy())
        return np.concatenate(outs, axis=0)

    def _encode_texts(self, model, processor, device, dtype, texts: list[str],
                      batch: int = 256, disable_adapter: bool = False) -> np.ndarray:
        ctx = model.text_model.disable_adapter() if disable_adapter else _nullcontext()
        with ctx:
            return self._encode_with_module(model.text_model, processor, device, dtype,
                                            texts, batch=batch)

    def evaluate(self, model, processor, device, dtype, val_ds: TextImageEmbedDataset) -> dict[str, float]:
        if len(val_ds) == 0:
            return {"R@1": 0.0, "R@5": 0.0, "R@10": 0.0, "MRR": 0.0, "MedR": 0.0}
        model.eval()
        text_vecs = self._encode_texts(model, processor, device, dtype, val_ds.captions)
        img = val_ds.embeds / np.maximum(np.linalg.norm(val_ds.embeds, axis=1, keepdims=True), 1e-9)
        metrics = retrieval_metrics(text_vecs, img)
        model.train()
        return metrics

    # ── train loop ───────────────────────────────────────────────────────

    def train(self) -> dict[str, float]:
        import torch

        self._seed()
        model, processor, device, dtype = self._build_model()

        train_ds = TextImageEmbedDataset(
            self.settings, "train", self.cfg.word_dropout,
            anchor_mix_ratio=self.cfg.anchor_mix_ratio,
            anchor_parquet=self.cfg.anchor_parquet,
            hard_negative_per_sample=self.cfg.hard_negative_per_sample,
            seed=self.cfg.seed,
        )
        val_ds = TextImageEmbedDataset(self.settings, "val", 0.0)
        log.info("Data: %d train / %d val pairs", len(train_ds), len(val_ds))
        self._train_size = len(train_ds)
        loader = self._build_loader(train_ds, device)

        # A tiny dataset can have fewer batches per epoch than grad_accum —
        # without this clamp `step` would never advance (infinite loop).
        self.cfg.grad_accum = max(1, min(self.cfg.grad_accum, len(loader)))

        adapter_params = [p for n, p in model.named_parameters() if p.requires_grad and "logit_" not in n]
        scalar_params = [model.logit_scale, model.logit_bias]
        optimizer = torch.optim.AdamW(
            [
                {"params": adapter_params, "lr": self.cfg.lr, "weight_decay": self.cfg.weight_decay},
                {"params": scalar_params, "lr": self.cfg.lr_scalars, "weight_decay": 0.0},
            ]
        )
        steps_per_epoch = max(1, len(loader) // self.cfg.grad_accum)
        total_steps = steps_per_epoch * self.cfg.epochs
        warmup = max(1, int(total_steps * self.cfg.warmup_ratio))
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda s: s / warmup if s < warmup
            else 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, total_steps - warmup))),
        )

        start_step, best_r5, evals_since_best = self._try_resume(model, optimizer, scheduler)
        # export() may have run after the last checkpoint (crash window) — the
        # exported model's metric is the true bar a new best must clear.
        exported = read_json(Path(self.cfg.export_dir) / "export_meta.json", default={}) or {}
        exported_r5 = float((exported.get("metrics") or {}).get("R@5", -1.0))
        best_r5 = max(best_r5, exported_r5)
        if start_step == 0:
            base = self.evaluate(model, processor, device, dtype, val_ds)
            log.info("Zero-shot baseline: %s", {k: round(v, 4) for k, v in base.items()})
            atomic_write_json(self.run_dir / "baseline.json", base)

        model.train()
        step = start_step
        stop = False
        t0 = time.time()
        # Exact mid-epoch resume: each epoch fast-forwards the loader past the
        # micro-batches an interrupted session already consumed and takes an
        # exact multiple of grad_accum micro-batches (no partial group bleeds
        # un-stepped gradients into the next epoch). See plan_epoch_steps.
        start_epoch = min(step // steps_per_epoch, self.cfg.epochs)
        for epoch in range(start_epoch, self.cfg.epochs):
            if stop or step >= total_steps:
                break
            skip_micro, run_steps = plan_epoch_steps(
                step, steps_per_epoch, total_steps, self.cfg.grad_accum)
            if run_steps <= 0:      # optimizer-step budget already spent
                break
            if skip_micro:
                log.info("Mid-epoch resume: fast-forwarding %d already-trained micro-batches, "
                         "%d optimizer steps left in epoch %d.", skip_micro, run_steps, epoch)
            # Deterministic (seed, epoch) permutation — a resumed epoch replays
            # the crashed run's exact shuffle before the islice fast-forward.
            self._reseed_epoch(epoch)
            optimizer.zero_grad(set_to_none=True)
            micro = 0
            for batch in islice(loader, skip_micro, skip_micro + run_steps * self.cfg.grad_accum):
                texts, img_emb = batch[0], batch[1]
                neg_texts = batch[2] if len(batch) > 2 else None
                toks = processor.tokenizer(
                    texts, padding="max_length", truncation=True,
                    max_length=self.cfg.max_text_len, return_tensors="pt",
                ).to(device)
                img_emb = img_emb.to(device)
                img_emb = torch.nn.functional.normalize(img_emb, dim=-1)

                with torch.autocast(device_type="cuda", dtype=dtype, enabled=(device == "cuda")):
                    text_emb = model.text_model(**toks).pooler_output
                    text_emb = torch.nn.functional.normalize(text_emb.float(), dim=-1)
                    if self.cfg.loss == "infonce":
                        loss_main = info_nce_loss(img_emb.float(), text_emb, model.logit_scale)
                    else:
                        neg_emb = None
                        if neg_texts:
                            neg_toks = processor.tokenizer(
                                neg_texts, padding="max_length", truncation=True,
                                max_length=self.cfg.max_text_len, return_tensors="pt",
                            ).to(device)
                            neg_emb = model.text_model(**neg_toks).pooler_output
                            neg_emb = torch.nn.functional.normalize(neg_emb.float(), dim=-1)
                            neg_emb = neg_emb.view(img_emb.size(0), -1, text_emb.size(-1))
                        loss_main = siglip_sigmoid_loss(
                            text_emb, img_emb.float(), model.logit_scale, model.logit_bias,
                            neg_txt_emb=neg_emb,
                        )
                    beta = self.cfg.distill_beta0 * max(0.0, 1.0 - step / max(1, total_steps))
                    loss = loss_main
                    if beta > 0:
                        with torch.no_grad(), model.text_model.disable_adapter():
                            teacher = model.text_model(**toks).pooler_output
                            teacher = torch.nn.functional.normalize(teacher.float(), dim=-1)
                        loss = loss_main + beta * distill_cosine_loss(text_emb, teacher)

                (loss / self.cfg.grad_accum).backward()
                micro += 1
                if micro % self.cfg.grad_accum:
                    continue

                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], self.cfg.max_grad_norm
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1

                if step % 50 == 0:
                    log.info(
                        "step %d/%d | loss %.4f | lr %.2e | %.1f min",
                        step, total_steps, float(loss), scheduler.get_last_lr()[0],
                        (time.time() - t0) / 60,
                    )

                if step % self.cfg.eval_every_steps == 0 or step >= total_steps:
                    metrics = self.evaluate(model, processor, device, dtype, val_ds)
                    log.info("eval @%d: %s", step, {k: round(v, 4) for k, v in metrics.items()})
                    if metrics["R@5"] > best_r5:
                        best_r5 = metrics["R@5"]
                        evals_since_best = 0
                        self.export(model, processor, metrics)
                    else:
                        evals_since_best += 1
                    self._save_checkpoint(model, optimizer, scheduler, step, best_r5, evals_since_best)
                    if evals_since_best >= self.cfg.early_stop_patience:
                        log.info("Early stop: %d evals without R@5 improvement", evals_since_best)
                        stop = True
                if stop:
                    break

        final = self.evaluate(model, processor, device, dtype, val_ds)
        log.info("Training done. Final: %s | best R@5 %.4f", {k: round(v, 4) for k, v in final.items()}, best_r5)
        try:
            wise = self.export_wiseft(model, processor, device, dtype, val_ds)
            if wise:
                log.info("WiSE-FT winner: alpha=%.2f R@5=%.4f", wise["alpha"], wise["metrics"]["R@5"])
        except Exception as e:  # WiSE-FT is a bonus export — never fail the run over it
            log.warning("WiSE-FT export skipped: %s", e)
        return final

    # ── export ───────────────────────────────────────────────────────────

    def export(self, model, processor, metrics: dict[str, float]) -> Path:
        """Merge LoRA into the text tower and save the graftable checkpoint."""
        import copy

        from safetensors.torch import save_file

        export_dir = Path(self.cfg.export_dir)
        export_dir.mkdir(parents=True, exist_ok=True)
        merged = copy.deepcopy(model.text_model).merge_and_unload()
        state = {k: v.detach().to("cpu", dtype=_float32()) for k, v in merged.state_dict().items()}
        tmp = export_dir / "text_tower.safetensors.tmp"
        save_file(state, str(tmp))
        os.replace(tmp, export_dir / "text_tower.safetensors")
        atomic_write_json(export_dir / "export_meta.json", {
            "base_id": self.cfg.base_id,
            "metrics": metrics,
            "config": asdict(self.cfg),
            "time": time.time(),
        })
        log.info("Exported best text tower → %s", export_dir)
        return export_dir

    def export_wiseft(self, model, processor, device, dtype,
                      val_ds: TextImageEmbedDataset) -> dict | None:
        """WiSE-FT α sweep over the best-exported tower; winner → wiseft_best/.

        Interpolates ``base·(1−α) + tuned·α`` for each ``cfg.wiseft_alphas``,
        scores every candidate on the val split (R@5, same eval as training),
        and saves the winning tower + its α to ``export_dir/wiseft_best``.
        Returns ``{"alpha", "metrics"}`` or None when preconditions are missing
        (no best export yet, empty val split, or no base snapshot).
        """
        import copy

        from safetensors.torch import load_file, save_file

        export_dir = Path(self.cfg.export_dir)
        tuned_path = export_dir / "text_tower.safetensors"
        if (not tuned_path.is_file() or len(val_ds) == 0
                or not self.cfg.wiseft_alphas or self._base_text_state is None):
            return None

        tuned_state = load_file(str(tuned_path))
        tower = copy.deepcopy(model.text_model).merge_and_unload()
        tower.eval()
        img = val_ds.embeds / np.maximum(np.linalg.norm(val_ds.embeds, axis=1, keepdims=True), 1e-9)

        best_alpha, best_metrics, best_state = None, None, None
        tried: dict[str, float] = {}
        for alpha in self.cfg.wiseft_alphas:
            state = wiseft_interpolate(self._base_text_state, tuned_state, float(alpha))
            tower.load_state_dict(state, strict=False)
            text_vecs = self._encode_with_module(tower, processor, device, dtype, val_ds.captions)
            metrics = retrieval_metrics(text_vecs, img)
            tried[f"{float(alpha):.2f}"] = metrics["R@5"]
            log.info("WiSE-FT alpha=%.2f: %s", alpha, {k: round(v, 4) for k, v in metrics.items()})
            if best_metrics is None or metrics["R@5"] > best_metrics["R@5"]:
                best_alpha, best_metrics, best_state = float(alpha), metrics, state

        out_dir = export_dir / "wiseft_best"
        out_dir.mkdir(parents=True, exist_ok=True)
        save_state = {k: v.detach().to("cpu", dtype=_float32()).contiguous()
                      for k, v in best_state.items()}
        tmp = out_dir / "text_tower.safetensors.tmp"
        save_file(save_state, str(tmp))
        os.replace(tmp, out_dir / "text_tower.safetensors")
        atomic_write_json(out_dir / "export_meta.json", {
            "base_id": self.cfg.base_id,
            "wiseft_alpha": best_alpha,
            "metrics": best_metrics,
            "alphas_tried": tried,
            "config": asdict(self.cfg),
            "time": time.time(),
        })
        log.info("Exported WiSE-FT tower (alpha=%.2f) → %s", best_alpha, out_dir)
        return {"alpha": best_alpha, "metrics": best_metrics}


def _float32():
    import torch

    return torch.float32


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False
