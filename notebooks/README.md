# Colab notebooks

| # | Notebook | Purpose |
|---|----------|---------|
| 01 | `01_build_artifacts_colab.ipynb` | catalog → keyframe extraction → dense embeddings + FAISS → OCR / ASR / captions → persisted BM25 text index |
| 02 | `02_train_vi_encoder_H100.ipynb` | LoRA-LiT fine-tune of the SigLIP-2 text tower (H100 autopilot: empirical OOM probe, run pointer + resume dashboard, public VI caption channels, WiSE-FT sweep, deliverables mirror) |
| 03 | `03_test_system.ipynb` | end-to-end smoke test: search (KIS/TRAKE/AVS), submission CSVs, Codabench validate + package, official GT scoring, automatic-track dry-run, Streamlit UI |

The `.ipynb` files are **build products** — never hand-edit them. The single
source of truth is `_build_notebooks.py`:

```
python notebooks/_build_notebooks.py        # regenerate after editing CELL_* constants
python -m pytest tests/test_notebooks.py -q # structure + marker tests (CPU only)
```

Every cell is idempotent / resumable: after a Colab disconnect just
*Runtime → Run all*. Caches (HF models, pip wheels), checkpoints and the run
pointer all live under `MyDrive/<project>/artifacts/`, so nothing is lost
between sessions.

🆕 2026-07-08: notebook 01 gains `INSTALL_TRANSNETV2` (default True) — installs
`transnetv2-pytorch` with `--no-deps` (Colab torch is never touched) so K-batch
keyframe extraction uses the TransNetV2 shot detector; any install failure
falls back to PySceneDetect automatically.

🆕 2026-07-12 (round-3 fixes): the TransNetV2 install now includes
`ffmpeg-python` (hard requirement of `predict_video` — without it every video
silently fell back to PySceneDetect); `WISEFT_ALPHAS` gains the α=1.0 guard
(wiseft_best can never score below the raw tuned tower); the `pip check`
micro-fix regex now matches both real pip formats; every generated cell
carries a deterministic nbformat-4.5 `id`.
