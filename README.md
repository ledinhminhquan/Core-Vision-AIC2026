# 🎯 Core Vision Perfect V1

> **Vietnamese text-to-keyframe interactive + automatic video retrieval** for
> the **HCMC AI Challenge (AIC) 2026** — full coverage of KIS · KIS-V · QA ·
> TRAKE · AVS (flagged) · KIS-C **+ the new 2026 assistant-vs-assistant
> automatic track**, with an H100 training pipeline for a Vietnamese-tuned
> encoder. Successor of Core-Vision_Ultimate_Final (5 adversarial review
> rounds), inheriting its proven core and closing every remaining finding.

Built end-to-end and evidence-driven: the architecture follows what the
2025 top qualifiers actually ran (MERVIN 79/88 on PE-Core, Vortex 79.6/88,
Unified-IMMR 76.4/88, DANTE "Outstanding TRAKE", 4×-champion UIT's ablations),
then goes further — multi-variant SuperGlobal, candidate-restricted persisted
BM25, ensemble DANTE-DP for TRAKE, MMR-diversified AVS, **pairwise
cross-encoder rerank (BLIP-2 ITM / Qwen3-VL-Reranker)**, listwise VLM
re-ranking, **multi-frame VQA strips**, per-row QA answers, **temporal-context
boost**, **confidence-gated query-reformulation retry**, official-formula
offline scoring, Codabench packaging, a DRES client and a **machine-callable
HTTP service** for the automatic track.

**📖 Start here → [docs/PROJECT_CONTEXT.md](docs/PROJECT_CONTEXT.md)** (toàn bộ
ngữ cảnh, thuật toán, quy trình — tiếng Việt) ·
**🙋 hướng dẫn tay-cầm-tay → [HUONG_DAN.md](HUONG_DAN.md)**

```
OFFLINE (build once, resumable)                 ONLINE (per query, <1s)
videos ─► shot detect ─► keyframes              VI query ─► Gemini translate/
keyframes ─► SigLIP-2 so400m (multilingual) ─┐              enhance/expand
keyframes ─► PE-Core-bigG (EN, #1 2026)     ─┼─► FAISS ─► top-K per lane
keyframes ─► [Qwen3-VL-Embed / jina / …]    ─┤     ─► SuperGlobal (all variants)
keyframes ─► OCR · captions (Vintern-1B)    ─┼─► BM25 (persisted, O(K)) fusion
videos    ─► PhoWhisper ASR                 ─┤     ─► object · neighbor · temporal boosts
media-info · objects (Open Images→parquet)  ─┘     ─► [cross-encoder] ─► [VLM rerank]
                                                   ─► UI / CSV / DRES / HTTP service
```

## Quickstart (laptop, no GPU needed)

```bash
pip install -e ".[search,app,dev]"
pytest             # 513+ tests pure CPU; full 537 with torch installed ([ml] extra)

# with the AIC dataset in ./data (see docs/DRIVE_SETUP.md):
pip install -e ".[ml]"                        # query text encoder needs torch+open_clip
export CVP_EMBEDDING__MODEL=provided_clip32   # reuse organiser features — no image GPU work
python scripts/00_build_catalog.py
python scripts/02_embed_and_index.py
streamlit run app/streamlit_app.py
```

For real accuracy, build the SigLIP-2 + PE-Core indexes on Colab —
**notebooks/01_build_artifacts_colab.ipynb** — then (optionally) train the
Vietnamese tower on an H100 — **notebooks/02_train_vi_encoder_H100.ipynb**
(crash-safe autopilot: re-run after any disconnect and it resumes exactly,
mid-epoch). Verify + package submissions with **notebooks/03_test_system.ipynb**.

## Competition workflow

```bash
python scripts/20_run_queries.py --query-dir <pack> --zip --gt gt.json  # run → validate → zip → score
python scripts/40_eval_official.py --submission-dir ... --gt gt.json    # official formulas offline
python scripts/23_dump_signals.py + scripts/21_tune_weights.py          # tune fusion weights on dev GT
python scripts/26_run_ablations.py --query-dir <dev> --gt gt.json       # A1–A10 ablation battery
python scripts/50_bench_latency.py                                      # p50/p95 latency gate
python scripts/51_warm_cache.py --query-dir <pack>                      # pre-warm Gemini cache before a round
cvp eval --submission-dir <dir> --gt gt.json                            # official scoring from any terminal
python scripts/41_diff_submissions.py --a <runA> --b <runB>             # what did a config change move?
python scripts/25_auto_agent.py --query-dir <pack>                      # 2026 automatic track, end-to-end
cvp serve                                                               # HTTP/JSON retrieval service
```

Query packs are parsed in ALL observed organiser layouts (verified on the real
AIC-2025 finals packs — 89/89 files clean): TRAKE files with a context line +
`E1:`…`Ek:` events (duplicate-label typos included), single-line QA with the
question embedded ("… Hỏi …?" / marker-less "… là gì?" / imperative "Hảy cho
biết …"), MULTI-paragraph KIS/AVS and multi-line QA — see
`cvp.pipeline.run_queries.parse_query_lines`.

## Repository map

| Path | What |
|---|---|
| `configs/settings.yaml` | every knob; override via `CVP_SECTION__KEY` env vars |
| `src/cvp/` | the library — data, models, index, search, submission, service, training, eval |
| `scripts/` | numbered pipeline steps + tuning + ablations + latency + auto-agent + map-keyframes rebuild |
| `app/streamlit_app.py` | competition UI (5 task tabs, baskets, 5-min clock, group-by-video, CSV export) |
| `notebooks/` | Colab: 01 build · 02 train (H100 autopilot) · 03 test/package |
| `docs/` | PROJECT_CONTEXT · ARCHITECTURE · EVALUATION · DATASET_INGESTION · COLAB_GUIDE · PROJECT_PLAN · DRIVE_SETUP · DATA_FORMAT · TRAINING · PLAYBOOK · PAPER_NOTES |
| `report/` | LaTeX kit for the mandatory prelim solution report |
| `HUONG_DAN.md` | hướng dẫn A-Z tiếng Việt (Drive → Colab → thi đấu) |

## The three commandments

1. **Submit `frame_idx`, never the keyframe ordinal** — the bridge is
   `map-keyframes/*.csv`; the catalog enforces it everywhere (and
   `scripts/05_rebuild_map_keyframes.py` reconstructs missing csvs from video).
2. **A stale index refuses to serve** — corpus signature checked at load
   (and the checkpoint that built each index is recorded and verified);
   re-run `scripts/30_ingest.py` after any dataset drop.
3. **Everything long-running is resumable** — embedding, OCR/ASR/captions,
   training (exact mid-epoch resume). Disconnects cost minutes, not hours.

MIT license. Built for the AIC 2026 season.
