# 🎯 Core Vision Ultimate Final

> **Vietnamese text-to-keyframe interactive video retrieval** for the
> **HCMC AI Challenge (AIC) 2026** — full coverage of KIS · KIS-V · QA · TRAKE ·
> AVS · KIS-C **+ the new 2026 automatic track**, with an H100 training
> pipeline for a Vietnamese-tuned encoder.

Built end-to-end and evidence-driven: the architecture follows what the
2025 top qualifiers actually ran (MERVIN 79/88 on PE-Core, Vortex, Unified-IMMR,
DANTE "Outstanding TRAKE", 4×-champion UIT's ablations), then goes further —
multi-variant SuperGlobal, candidate-restricted persisted BM25, ensemble DANTE-DP
for TRAKE, MMR-diversified AVS, listwise VLM re-ranking, per-row QA answers,
official-formula offline scoring, Codabench packaging and a DRES client.

**📖 Start here → [docs/PROJECT_CONTEXT.md](docs/PROJECT_CONTEXT.md)** (toàn bộ
ngữ cảnh, thuật toán, quy trình — tiếng Việt).

```
OFFLINE (build once, resumable)                 ONLINE (per query, <1s)
videos ─► shot detect ─► keyframes              VI query ─► Gemini translate/
keyframes ─► SigLIP-2 so400m (multilingual) ─┐              enhance/expand
keyframes ─► PE-Core-bigG (EN, #1 2026)     ─┼─► FAISS ─► top-K per lane
keyframes ─► [Qwen3-VL-Embedding-2B (vi)]   ─┤     ─► SuperGlobal (all variants)
keyframes ─► OCR · captions (Vintern-1B)    ─┼─► BM25 (persisted, O(K)) fusion
videos    ─► PhoWhisper ASR                 ─┤     ─► object & neighbor boosts
media-info · objects (Open Images)          ─┘     ─► [VLM rerank] ─► UI / CSV / DRES
```

## Quickstart (laptop, no GPU needed)

```bash
pip install -e ".[search,app,dev]"
pytest                                   # 292 pass + 1 skip without torch; full 316 with [ml]/torch-cpu

# with the AIC dataset in ./data (see docs/DRIVE_SETUP.md):
pip install -e ".[ml]"                        # query text encoder needs torch+open_clip
export CVF_EMBEDDING__MODEL=provided_clip32   # reuse organiser features — no image GPU work
python scripts/00_build_catalog.py
python scripts/02_embed_and_index.py
streamlit run app/streamlit_app.py
```

For real accuracy, build the SigLIP-2 + PE-Core indexes on Colab —
**notebooks/01_build_artifacts_colab.ipynb** — then train the Vietnamese tower
on an H100 — **notebooks/02_train_vi_encoder_H100.ipynb** (crash-safe
autopilot: re-run after any disconnect and it resumes exactly, mid-epoch).
Verify + package submissions with **notebooks/03_test_system.ipynb**.

## Competition workflow

```bash
python scripts/20_run_queries.py --query-dir <pack> --zip   # CSVs + validated Codabench zip
python scripts/40_eval_official.py --submission-dir ... --gt gt.json   # official formulas offline
python scripts/23_dump_signals.py + scripts/21_tune_weights.py         # tune fusion weights on dev GT
python scripts/25_auto_agent.py --query-dir <pack>          # 2026 automatic track, end-to-end
```

Query packs are parsed in ALL observed organiser layouts (2026-07-08 + 2026-07-12
updates, verified on the real AIC-2025 finals packs): TRAKE files with a context
line + `E1:`…`Ek:` events, single-line QA with the question embedded ("… Hỏi …?"),
MULTI-paragraph KIS/AVS (all paragraphs joined for retrieval) and multi-line QA
(question = last interrogative line) — see `cvf.pipeline.run_queries.parse_query_lines`.

## Repository map

| Path | What |
|---|---|
| `configs/settings.yaml` | every knob; override via `CVF_SECTION__KEY` env vars |
| `src/cvf/` | the library — data, models, index, search, submission, training, eval |
| `scripts/` | numbered pipeline steps + tuning harness + auto-agent + official scorer |
| `app/streamlit_app.py` | competition UI (5 task tabs, baskets, 5-min clock, CSV export) |
| `notebooks/` | Colab: 01 build · 02 train (H100 autopilot) · 03 test/package |
| `docs/` | PROJECT_CONTEXT · DRIVE_SETUP · DATA_FORMAT · TRAINING · PLAYBOOK · PAPER_NOTES |

## The three commandments

1. **Submit `frame_idx`, never the keyframe ordinal** — the bridge is
   `map-keyframes/*.csv`; the catalog enforces it everywhere.
2. **A stale index refuses to serve** — corpus signature checked at load
   (and the checkpoint that built each index is recorded and verified);
   re-run `scripts/30_ingest.py` after any dataset drop.
3. **Everything long-running is resumable** — embedding, OCR/ASR/captions,
   training (exact mid-epoch resume). Disconnects cost minutes, not hours.

MIT license. Built for the AIC 2026 season.
