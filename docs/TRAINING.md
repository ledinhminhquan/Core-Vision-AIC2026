# 🧠 TRAINING — Vietnamese encoder fine-tune (LoRA-LiT), design & how-to

## Why train at all?

SigLIP-2 reads Vietnamese but was trained ~90% on English — on Vietnamese
benchmarks multilingual towers lose 15–25 R@1 points vs their English scores
(ViCLIP-OT paper measurements). We close that gap **on the competition corpus
itself**: fine-tune the text tower on (Vietnamese caption ↔ keyframe) pairs
auto-generated from the exact videos we must search.

## Why THIS recipe (each choice is load-bearing)

| Choice | Reason |
|---|---|
| **Freeze the image tower** (LiT) | 100% of image embeddings + the FAISS index stay valid — no re-embedding after training. Also the classic anti-overfit move (Zhai et al., CVPR'22). |
| **LoRA (r=32) on the text tower**, head fully trained | ~1–2% trainable params → hard to catastrophically forget the multilingual base; merged away at export (zero inference cost). |
| **SigLIP sigmoid loss, reusing pretrained `logit_scale`/`logit_bias`** | The loss the tower was born with; re-initializing the scalars collapses training (known pitfall). Sigmoid loss is batch-size-friendly → single-GPU is legitimate. |
| **Distillation anchor** `1−cos(student, teacher)`, β: 0.3→0 | Teacher = base tower via `disable_adapter()` (no extra VRAM). Keeps English/general alignment early, releases the leash later. |
| **Video-level split** | Frames of one video are near-duplicates; a random split would leak and fake the metrics. |
| **Word-dropout 5% on captions** | Auto-captions share phrasing quirks; dropout stops the tower from keying on them. |
| **Early stop on val R@5** (patience 5 evals) | The metric that mirrors qualifier top-5 scoring. |
| **English-anchor mixing** (`anchor_mix_ratio=0.25`) | 25% of each batch are (English caption, image) pairs → the tower cannot drift away from the multilingual base while it learns Vietnamese (FLYP/LDIFS-style anchoring). No-op when no EN sources exist. |
| **In-video hard negatives** (`hard_negative_per_sample`, off by default) | News shots inside one broadcast are natural hard negatives for temporal disambiguation; extra negative logits in the sigmoid loss. Turn on only with a val set to watch. |
| **InfoNCE A/B** (`loss: infonce`) | open_clip practitioners report softmax-CE frequently beats sigmoid on small fine-tunes; one flag flips it — let val R@5 decide. |
| **WiSE-FT export** (`wiseft_alphas=[0.4,0.5,0.6,1.0]`) | Post-training weight interpolation base↔tuned, winner picked on val R@5 → `.../wiseft_best`. Costs nothing at train time, large robustness gains (Wortsman et al.). |

## Data

`notebooks/01` captions keyframes with **Vintern-1B-v3.5** (Vietnamese VLM) →
`scripts/11_build_train_dataset.py` assembles pairs:
length gate 15–400 chars, near-dup suppression (cosine>0.97), ~3% val by video hash.
Typical yield: 100–300k pairs from a full AIC corpus — squarely in the LiT
fine-tune regime.

**Public Vietnamese caption channels** (wired in): `scripts/12_build_public_datasets.py`
(or the nb02 cell) builds KTVIC (~21.6k captions) / UIT-ViIC (~19.3k) parquets under
`artifacts/train_data/public/` — images embedded with the frozen SigLIP-2 image tower
so rows stay aligned; `build_training_set` auto-merges every `public/*.parquet`
(dim-mismatch is rejected loudly).

## Hyperparameters (defaults in `TrainConfig` / notebook 02)

- epochs 8 · effective batch ≈2048 (micro auto: H100 512 / A100 256 / L4 128 / T4 64)
- AdamW: LoRA+head lr 1e-4, scalars lr 1e-5, wd 0.01, cosine + 5% warmup, clip 1.0
- bf16 autocast (H100/A100), TF32 matmul, eval+checkpoint every 200 steps
- H100 wall-clock: ~30–60 min for 200k pairs × 8 epochs (with early stop, usually less)

## Crash-safety / autopilot

Checkpoints = `runs/vi_siglip2/step-N/{adapter/, extras.pt, state.json}` written
atomically (tmp dir → rename) to Drive; optimizer/scheduler/RNG included.
On any disconnect: rerun the notebook — `LiTTrainer` resumes from the newest
checkpoint whose `state.json` exists, **and fast-forwards the dataloader
mid-epoch** (`plan_epoch_steps`) so no batch is trained twice. The run pointer
`runs/vi_siglip2/active_train_run.json` (bound to a sha256 of the config)
shows status running/crashed/finished at a glance. Best model is exported
CONTINUOUSLY to `checkpoints/vi_siglip2_best` (merged, safetensors) — even a
dead session leaves a usable model; the WiSE-FT winner lands in
`checkpoints/vi_siglip2_best/wiseft_best`.

## Using the result

```bash
CVP_EMBEDDING__MODEL=finetuned                      # single-model mode
# WiSE-FT winner instead of the raw best:
CVP_FINETUNED__CHECKPOINT=./artifacts/checkpoints/vi_siglip2_best/wiseft_best
# or best: ensemble the tuned tower with the English lane —
# in configs/settings.yaml: ensemble_members: [finetuned, openclip]
CVP_EMBEDDING__MODEL=ensemble
```
`finetuned` **shares the siglip2 embeddings and FAISS index automatically**
(the LiT image tower is frozen, so the spaces are identical — see
`index_key_for` in `cvp/models/registry.py`). No re-embedding, no copying:
train, set the env var, done.

## Evaluating

`scripts/eval_model.py` (or notebook 02 cell 8) prints baseline vs fine-tuned
R@1/5/10, MRR, MedR on the val split. Expect +5–15 R@1 on Vietnamese captions;
verify no regression by also eyeballing English queries in the app.
