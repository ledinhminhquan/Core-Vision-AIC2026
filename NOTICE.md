# NOTICE — what this release contains, and what it does not

This repository is the source code of **Core-Vision Perfect V1**, the Vietnamese
video moment retrieval system that team *AIO_Heuristics* built for the
**Ho Chi Minh City AI Challenge (AIC) 2026**. It is released so that the results
reported in our papers can be inspected.

The `LICENSE` file (MIT) covers **code written by the team**. It does not, and
cannot, cover the third-party material listed below, nor any data.

---

## 1. Not included in this repository

Nothing here is a substitute for the competition data, and none of it is
redistributed:

| Not included | Why |
|---|---|
| The AIC 2026 video corpus, keyframes, and `map-keyframes` CSVs | Organiser-licensed competition data. |
| The organisers' query packs and any ground truth | Organiser-authored material, issued to participants only. |
| The AIC 2025 finals query pack | Organiser-authored. An earlier internal revision of this tree carried a byte-exact copy as a regression fixture; it has been removed from the released history. Use `queries/example/` instead. |
| Built artefacts — FAISS indexes, BM25 tables, OCR / ASR / caption sidecars, object tables | Derived from the organisers' data. Rebuild them with `scripts/30_ingest.py`. |
| The LoRA-adapted Vietnamese text tower checkpoint | Not released. |
| The `report/` LaTeX class and style files (`ai_conquer2026.cls`, `tvietlistings.sty`, `vipythonhighlight.sty`) | Third-party templates the team did not author. `report/main.tex` is kept for reference but will not compile without them. |

Paths referring to any of the above (for example `queries/dev-2025-finals`,
`CVP_PATHS__DATA_ROOT`) are configuration, not content: they describe where the
data would sit on a machine that has it.

## 2. Third-party components

The system depends on publicly released models and libraries, each under its own
licence. The ones that carry a licence worth naming before you redistribute
anything built from this code:

| Component | Licence |
|---|---|
| SigLIP 2 (`google/siglip2-so400m-patch16-384`) | Apache-2.0 |
| MetaCLIP 2 | as published by its authors |
| Qwen3-VL reranker | as published by its authors |
| PhoWhisper | BSD-3-Clause |
| Vintern-1B | as published by its authors |
| PaddleOCR | Apache-2.0 |
| FAISS | MIT |
| Open Images class names | CC BY 4.0 |
| `jina-clip-v2` (optional lane, **off by default**) | CC BY-NC — non-commercial |
| Preprocessing derived from InternVL | MIT |

The `jina-clip-v2` lane is non-commercial; it is disabled in the shipped
configuration and was not used in competition.

## 3. Provenance of this release

The published history is the development history: **119 commits, 14 July to
5 September 2026**, plus a few post-competition release and documentation
commits (no functional change). It was filtered before publication
to remove third-party material the team had no right to redistribute (the items
marked above), and commit e-mail addresses were rewritten to the maintainer's
GitHub no-reply address. No functional source file was altered by the filter.

The repository was maintained by a single committer on behalf of the team.

## 4. Test suite

The authoritative numbers, measured on this tree:

- **104** test files plus a shared `tests/conftest.py`, **16,241** lines
- **970** `test_` functions
- **81** Python files under `src/`, **16,083** lines
- CI (`.github/workflows/ci.yml`) runs `ruff` and `pytest` on CPU-only
  Python 3.11 and 3.12; every Gemini and Hugging Face call is stubbed or
  skipped in the suite.

Some historical planning documents under `docs/` quote earlier, superseded
suite sizes from mid-development. Where they disagree with this file, this file
and CI are correct.
