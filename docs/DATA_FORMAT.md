# 📦 DATA_FORMAT — Dataset contract & submission formats

## 1. Organiser dataset (AIC 2025 layout — 2026 expected similar)

Video ids: `[A-Z]\d{2}_V\d{3}` (e.g. `L21_V001`, `K08_V003`). Everything merges
under one `data_root`:

| Path | Content | Notes |
|---|---|---|
| `keyframes/{vid}/{nnn}.jpg` | keyframe images | `nnn` = 1-indexed ordinal, zero-padded |
| `map-keyframes/{vid}.csv` | `n,pts_time,fps,frame_idx` | **the submission bridge** — see below |
| `clip-features-32/{vid}.npy` | `(N,512)` fp16 CLIP ViT-B/32 features | row `n-1` ↔ keyframe `n` |
| `media-info/{vid}.json` | YouTube metadata | title/description/keywords/watch_url |
| `objects/{vid}/{nnn}.json` | Open Images V4 detections | entities/scores/boxes (≤100/frame) |
| `videos/{vid}.mp4` | raw video | K-batches ship ONLY this |

### map-keyframes semantics (critical)

```csv
n,pts_time,fps,frame_idx
1,0.0,25.0,0
2,3.84,25.0,96
```

- `n` — keyframe ordinal (file name, feature row `n-1`).
- `frame_idx` — frame number in the ORIGINAL video. **This is what you submit.**
- `pts_time` = `frame_idx / fps` seconds. fps varies per video — never hardcode.
- Self-extracted videos (K-batch): `cvp.data.extraction` generates this CSV
  together with the keyframes, so the bridge always exists.

## 2. Submission CSVs (Codabench qualifiers; DRES finals analogous)

UTF-8, comma-separated, **no header**, **≤100 rows**, ranked best-first,
de-duplicated. Ground truth is a contiguous frame segment — any frame inside
counts as correct.

```csv
# KIS / AVS                     # QA (answer ≤100 chars)             # TRAKE (one row = full sequence)
L21_V001,12450                  L28_V001,8765,xã Cam Hải Đông        K08_V001,3120,3450,3990,4502
L25_V003,8800                   L25_V003,6200,xã Cam Hải Đông        K10_V002,5600,6200,7100,8050
```

Writers in `cvp/submission/writer.py` enforce all of this (validation included:
video-id regex, non-negative frames, TRAKE strictly increasing, QA answer quoting).

## 3. Qualifier scoring (what to optimize)

Per query: R-Score per row (task formulas below), then
`Final = (1/5) · Σ_{k∈{1,5,20,50,100}} max R-Score among the first k rows` —
implemented EXACTLY (organiser formulas, pure stdlib) in `cvp/eval/official.py`;
the quick approximation lives in `cvp/eval/metrics.py::qualifier_score`.

- KIS: 1 iff correct video AND `frame_idx ∈ [s,e]`.
- QA: additionally the answer must match (casefolded, diacritics preserved).
- TRAKE: 0 on video mismatch, else fraction of frames inside per-moment windows
  (organiser worked example: 3 of 4 → 0.75).
- A correct hit at rank ≤1 is worth 5× a hit at rank ≤100 — **ranking quality matters**,
  not just recall@100. Always fill all 100 rows (a rank-80 hit still pays 0.2).
- For AVS-style tasks, diversity across rows is what pays (see `search/avs.py`).

Ground-truth JSON for offline scoring (`scripts/40_eval_official.py`):
`{"<query-stem>": {"task": "kis", "video_id": "L01_V001", "range": [500, 510]}}`
(also accepts `moments: [[s,e],...]`, `center`+`epsilon`, `answers: [...]`,
and the legacy `frame_start`/`frame_end`/`events` shapes).

🆕 2026-07-08: a KIS/QA entry may declare SEVERAL acceptable windows in one
video — `"ranges": [[s1,e1], [s2,e2], ...]` (a row scores when its frame lands
inside ANY window; useful for AVS-style dev sets and answers that repeat in a
video). Task aliases now also include `kis-v`/`video-kis` (scored as KIS).

### Organiser query-file layouts (both parsed since 2026-07-08)

Verified on the real AIC-2025 finals packs — `cvp.pipeline.run_queries`
normalises BOTH automatically (`parse_trake_events` / `split_qa_line`):

```
# TRAKE, organiser layout: context line + En-prefixed events   # plain layout
Đoạn video múa lân …, tìm các sự kiện sau:                      sự kiện một
E1: Lân quay vòng trên cột …                                    sự kiện hai
E2: Khoảnh khắc 4 chân chạm đất …                               sự kiện ba

# QA, organiser layout (single line, question embedded)
<mô tả cảnh> … Hỏi xã này có tên là gì?
```

The context line is dropped (or merged into every event with
`temporal.event_context: prepend`); `E1:`-style prefixes are stripped; the QA
question after the LAST "Hỏi"/"Câu hỏi" goes to VQA while dense search gets
the description only. Examples: `queries/example/query-0-4-trake-organiser.txt`
and `query-0-5-qa-organiser.txt`.

## 4. Built artifacts (this repo's outputs)

```
artifacts/
├── catalog/manifest.parquet      global_id|video_id|n|frame_idx|pts_time|fps|path|has_map
├── catalog/signature.json        corpus hash — staleness guard
├── embeddings/{model}/{vid}.npy  (N,dim) fp32, positional row per catalog order
├── indexes/{model}/kf.faiss      FAISS row == global_id  (+ meta.json: signature,
│                                  dim, count, model_tag = checkpoint that built it)
├── text_index/{field}.json.gz    persisted BM25 stats (tf/df/doclen) + meta.json
├── ocr/{vid}.json                {"n_to_text": {"1": "..."}}
├── asr/{vid}.json                {"segments": [{"start","end","text"}]}
├── captions/{vid}.json           {"n_to_caption": {"1": "..."}}
├── train_data/{embeds.npy, meta.parquet}  (+ public/*.parquet extra channels)
├── checkpoints/vi_siglip2_best/  text_tower.safetensors + export_meta.json
│   └── wiseft_best/              WiSE-FT winner (same graftable format)
├── signal_dumps/<pack>/*.json    raw per-signal maps for scripts/21_tune_weights.py
└── submissions/*.csv (+ *.zip Codabench package + MANIFEST.json sha256)
```

## 5. 2026 clarifications (query forms & finals rules — verified 07/2026)

- **Qualifiers (Codabench, 8/2026): queries are TEXT-only.** No clips at the
  qualifier stage — the batch runner (`scripts/20_run_queries.py`) covers the
  whole round. The uploaded zip **MUST contain a folder named `submission`** —
  already our default (`submission.package_name` in `configs/settings.yaml`;
  `cvp/submission/packager.py` builds exactly this layout).
- **Finals add watched-only clips (KIS-V):** the clip is SHOWN on the organiser
  screen and may only be WATCHED — no recording, no photographing, no
  screen-capturing in any form. You MAY re-describe it in words, draw it, or
  generate an image from your description to feed your own system (workflow in
  `COMPETITION_PLAYBOOK.md` §2). **Audio may be muted** — never build a
  workflow that depends on hearing the clip.
- **AVS is UNCERTAIN for 2026** (organiser-side instructor recalls no AVS this
  year, but rules change yearly). It stays fully supported — AVS tab,
  `/search/avs`, MMR diversification — but keep it flag-gated and spend zero
  tuning effort on it until the đề bài confirms.
- **KIS-C is a desired system STYLE, not (yet) a confirmed task.** The concrete
  precedent is the 2025 finals progressive format: textual KIS ran **5 minutes
  with 5 hints released at 1-minute intervals** (VKIS: 4 minutes, 20-second
  clip). The KIS-C tab (hint merging + realtime 5-minute clock) is built for
  exactly that regime.
