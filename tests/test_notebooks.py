"""Pure-CPU tests for the generated Colab notebooks.

The .ipynb files are BUILD PRODUCTS of notebooks/_build_notebooks.py; these
tests re-run the builder (deterministic, writes to the real notebooks dir),
then assert structure + the load-bearing markers:

  * valid nbformat-4 JSON with the expected number of code cells,
  * regenerating is idempotent (same bytes on a second run),
  * PYTORCH_CUDA_ALLOC_CONF is set in the PARAMS cell BEFORE any torch import,
  * no cell imports cvp before the repo+deps cell,
  * per-notebook feature markers (BM25 text index + provided_clip32 auto-add
    in nb01; run pointer + WiSE-FT + OOM probe in nb02; Codabench packaging +
    official scoring + auto-track dry-run in nb03).
"""

from __future__ import annotations

import importlib.util
import json
import re
import types
from pathlib import Path

import pytest

NOTEBOOKS_DIR = Path(__file__).resolve().parents[1] / "notebooks"
NB1 = "01_build_artifacts_colab.ipynb"
NB2 = "02_train_vi_encoder_H100.ipynb"
NB3 = "03_test_system.ipynb"
ALL_NBS = (NB1, NB2, NB3)

EXPECTED_CODE_CELLS = {NB1: 11, NB2: 13, NB3: 12}

# import-detection: `import torch`, `import gc, torch`, `from torch... import`
TORCH_IMPORT_RE = re.compile(r"^\s*(?:import\s+[^#\n]*\btorch\b|from\s+torch\b)", re.MULTILINE)
CVP_IMPORT_RE = re.compile(r"^\s*(?:import\s+[^#\n]*\bcvp\b|from\s+cvp\b)", re.MULTILINE)


def _load_builder():
    spec = importlib.util.spec_from_file_location(
        "cvp_notebooks_builder_for_tests", NOTEBOOKS_DIR / "_build_notebooks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def builder():
    """Run the builder once — regenerates the committed .ipynb files."""
    mod = _load_builder()
    mod.main()
    return mod


def _nb(name: str) -> dict:
    return json.loads((NOTEBOOKS_DIR / name).read_text(encoding="utf-8"))


def _code_sources(name: str) -> list[str]:
    return ["".join(c["source"]) for c in _nb(name)["cells"] if c["cell_type"] == "code"]


def _cell(name: str, marker: str) -> str:
    hits = [src for src in _code_sources(name) if marker in src]
    assert len(hits) == 1, f"{name}: expected exactly 1 cell matching {marker!r}, got {len(hits)}"
    return hits[0]


def _fragment(src: str, start_marker: str, end_marker: str | None = None) -> str:
    """Cell source from just after ``start_marker``'s line up to ``end_marker``."""
    frag = src.split(start_marker, 1)[1]
    if end_marker is not None:
        frag = frag.split(end_marker, 1)[0]
    return frag[frag.index("\n"):]  # drop the remainder of the marker line


# ── builder output integrity ────────────────────────────────────────────────


def test_regenerating_is_idempotent(builder):
    """A second main() run must produce byte-identical .ipynb files."""
    before = {name: (NOTEBOOKS_DIR / name).read_bytes() for name in ALL_NBS}
    builder.main()
    for name in ALL_NBS:
        assert (NOTEBOOKS_DIR / name).read_bytes() == before[name], (
            f"{name}: builder output is not deterministic")


@pytest.mark.parametrize("name", ALL_NBS)
def test_parses_as_nbformat4(builder, name):
    nb = _nb(name)
    assert nb["nbformat"] == 4
    assert nb["cells"], "notebook has no cells"
    assert nb["cells"][0]["cell_type"] == "markdown"  # title cell first
    for c in nb["cells"]:
        assert c["cell_type"] in ("markdown", "code")
        assert isinstance(c["source"], list)
        if c["cell_type"] == "code":
            assert c["execution_count"] is None
            assert c["outputs"] == []


@pytest.mark.parametrize("name", ALL_NBS)
def test_expected_code_cell_count(builder, name):
    assert len(_code_sources(name)) == EXPECTED_CODE_CELLS[name]


@pytest.mark.parametrize("name", ALL_NBS)
def test_every_code_cell_is_valid_python(builder, name):
    """compile() every cell — a builder typo must fail here, not on the H100."""
    for i, src in enumerate(_code_sources(name)):
        compile(src, f"<{name} cell {i}>", "exec")


# ── env / import ordering invariants ────────────────────────────────────────


@pytest.mark.parametrize("name", ALL_NBS)
def test_alloc_conf_set_in_params_cell_before_any_torch_import(builder, name):
    """PYTORCH_CUDA_ALLOC_CONF only works if set BEFORE torch is imported."""
    sources = _code_sources(name)
    assert "PYTORCH_CUDA_ALLOC_CONF" in sources[0], (
        f"{name}: the PARAMS cell (first code cell) must set PYTORCH_CUDA_ALLOC_CONF")
    assert not TORCH_IMPORT_RE.search(sources[0]), (
        f"{name}: the PARAMS cell must not import torch")
    # and no cell before the params cell exists at all (params is cell 0),
    # so every torch import necessarily comes after the env var is set.
    torch_cells = [i for i, src in enumerate(sources) if TORCH_IMPORT_RE.search(src)]
    assert all(i > 0 for i in torch_cells)


@pytest.mark.parametrize("name", ALL_NBS)
def test_no_cvp_import_before_repo_deps_cell(builder, name):
    sources = _code_sources(name)
    deps_idx = next(i for i, src in enumerate(sources) if "requirements-colab.txt" in src)
    for i, src in enumerate(sources[:deps_idx]):
        assert not CVP_IMPORT_RE.search(src), (
            f"{name}: cell {i} imports cvp before the repo+deps cell installed it")


# ── per-notebook feature markers ────────────────────────────────────────────

SHARED_MARKERS = (
    "PIP_CACHE_DIR",            # pip cache on Drive (mount cell)
    "_write_test_",             # Drive preflight write test
    "importlib.metadata",       # version checks without importing (deps cell)
    "pip\", \"check",           # pip check micro-fix loop
)

NB_MARKERS = {
    NB1: (
        "build_text_index",             # persisted BM25 index cell
        "FORCE_TEXT_INDEX",             # wired force toggle
        "provided_clip32\" not in EMBED_MODELS",  # auto-add lane (weakness #6)
        "_missing_feats",               # ...only auto-added on FULL .npy coverage
        "AUX_CHANGED",                  # aux work → text-index force-rebuild
        "n_synced",                     # K-batch sync-back to Drive
        "model_tag=model_tag",          # scripts/02 index-stamp contract
        "_log_stage",                   # lightweight stage tee to nb01.log
        "qwen_embed",                   # optional heavy lane documented
        "INSTALL_TRANSNETV2",           # TransNetV2 --no-deps install toggle
        "transnetv2-pytorch",           # ...and the actual pip target
    ),
    NB2: (
        "active_train_run.json",        # v12 run pointer
        "write_pointer(\"crashed\"",    # crash marking around trainer.train()
        "wiseft",                       # WiSE-FT eval row + winner alpha
        "_probe_micro_batch",           # empirical OOM probe
        "build_all_public_parquets",    # KTVIC / UIT-ViIC channels
        "anchor_mix_ratio=ANCHOR_MIX_RATIO",  # new TrainConfig knobs wired
        "wiseft_alphas=WISEFT_ALPHAS",
        "CVP_FINETUNED__CHECKPOINT",    # wiseft_best eval via env override
        "deliverables",                 # deliverables mirror (copytree)
        "dirs_exist_ok=True",
        "ENSEMBLE_MEMBERS",             # exact env lines to use the model
    ),
    NB3: (
        "package_codabench",            # validate + package cell
        "files=nb03_files",             # ONLY this run's CSVs — the submissions
        "validate_file",                # dir is shared with real UI exports (R4)
        "score_run",                    # official GT scoring cell
        "RUN_AUTO_AGENT",               # automatic-track dry-run toggle
        "run_auto",
        "submit=False",
    ),
}


@pytest.mark.parametrize("name", ALL_NBS)
def test_shared_cell_markers(builder, name):
    joined = "\n".join(_code_sources(name))
    for marker in SHARED_MARKERS:
        assert marker in joined, f"{name}: missing shared marker {marker!r}"


@pytest.mark.parametrize("name", ALL_NBS)
def test_notebook_feature_markers(builder, name):
    joined = "\n".join(_code_sources(name))
    for marker in NB_MARKERS[name]:
        assert marker in joined, f"{name}: missing feature marker {marker!r}"


def test_torch_lines_never_installed_by_deps_cell(builder):
    """The deps cell must strip torch* requirement lines (Colab rule #1)."""
    for name in ALL_NBS:
        sources = _code_sources(name)
        deps = next(src for src in sources if "requirements-colab.txt" in src)
        assert 'startswith("torch")' in deps, (
            f"{name}: deps cell lost the torch*-requirement guard")


# ── adversarial-review fixes ────────────────────────────────────────────────


class _PathsStub:
    """settings.paths stand-in for executing nb01 cell fragments."""

    clip_features_dir = "clip-features-32"

    def __init__(self, root):
        self.root = Path(root)
        self.data_root = str(root)

    def data(self, name):
        return self.root / name


def _autoadd_fragment() -> str:
    return _fragment(_cell(NB1, "clip_features_dir"), "# Organiser CLIP features")


def test_nb1_provided_clip32_not_added_without_full_coverage(builder, tmp_path, capsys):
    """K-batch video without organiser .npy → the lane must NOT be auto-added
    (ingest would skip it and store.build would hard-fail on the gap)."""
    import pandas as pd

    feat = tmp_path / "clip-features-32"
    feat.mkdir()
    (feat / "L21_V001.npy").write_bytes(b"x")
    ns = {"settings": types.SimpleNamespace(paths=_PathsStub(tmp_path)),
          "EMBED_MODELS": ["siglip2"],
          "df": pd.DataFrame({"video_id": ["L21_V001", "K01_V001"]})}
    exec(compile(_autoadd_fragment(), "<nb1-autoadd>", "exec"), ns)
    assert "provided_clip32" not in ns["EMBED_MODELS"]
    out = capsys.readouterr().out
    assert "skipped" in out and "K01_V001" in out


def test_nb1_provided_clip32_added_on_full_coverage(builder, tmp_path):
    import pandas as pd

    feat = tmp_path / "clip-features-32"
    feat.mkdir()
    for vid in ("L21_V001", "K01_V001"):
        (feat / f"{vid}.npy").write_bytes(b"x")
    ns = {"settings": types.SimpleNamespace(paths=_PathsStub(tmp_path)),
          "EMBED_MODELS": ["siglip2"],
          "df": pd.DataFrame({"video_id": ["L21_V001", "K01_V001"]})}
    exec(compile(_autoadd_fragment(), "<nb1-autoadd>", "exec"), ns)
    assert ns["EMBED_MODELS"] == ["siglip2", "provided_clip32"]


def _syncback_fragment() -> str:
    return _fragment(_cell(NB1, "sync-back"),
                     "# K-batch sync-back", "# Organiser CLIP features")


def test_nb1_syncback_copies_extracted_kbatch_to_drive(builder, tmp_path):
    """Self-extracted K-batch keyframes + map CSVs on the LOCAL disk must be
    mirrored to Drive; existing Drive folders are never touched."""
    drive, local = tmp_path / "drive", tmp_path / "local"
    (drive / "keyframes" / "L21_V001").mkdir(parents=True)
    (drive / "keyframes" / "L21_V001" / "org.jpg").write_bytes(b"organiser")
    (drive / "map-keyframes").mkdir(parents=True)
    (drive / "map-keyframes" / "L21_V001.csv").write_text("org", encoding="utf-8")
    for vid in ("L21_V001", "K01_V001"):
        d = local / "keyframes" / vid
        d.mkdir(parents=True)
        (d / "001.jpg").write_bytes(b"\xff\xd8local")
        (local / "map-keyframes").mkdir(parents=True, exist_ok=True)
        (local / "map-keyframes" / f"{vid}.csv").write_text(
            "n,pts_time,fps,frame_idx\n1,0.00,25.0,0\n", encoding="utf-8")

    ns = {"settings": types.SimpleNamespace(paths=_PathsStub(local)), "DATA_DIR": drive}
    exec(compile(_syncback_fragment(), "<nb1-syncback>", "exec"), ns)

    assert ns["n_synced"] == 2   # K01_V001 folder + K01_V001.csv
    assert (drive / "keyframes" / "K01_V001" / "001.jpg").read_bytes() == b"\xff\xd8local"
    assert "frame_idx" in (drive / "map-keyframes" / "K01_V001.csv").read_text(encoding="utf-8")
    # organiser data untouched (target existed → skipped, not overwritten)
    assert (drive / "keyframes" / "L21_V001" / "org.jpg").read_bytes() == b"organiser"
    assert (drive / "map-keyframes" / "L21_V001.csv").read_text(encoding="utf-8") == "org"
    assert not list(drive.rglob("*.__tmp"))   # no tmp leftovers


def test_nb1_syncback_noop_in_drive_mode(builder, tmp_path):
    """COPY_KEYFRAMES_LOCAL=False → data_root IS DATA_DIR → nothing to sync."""
    drive = tmp_path / "drive"
    (drive / "keyframes" / "K01_V001").mkdir(parents=True)
    ns = {"settings": types.SimpleNamespace(paths=_PathsStub(drive)), "DATA_DIR": drive}
    exec(compile(_syncback_fragment(), "<nb1-syncback-noop>", "exec"), ns)
    assert ns["n_synced"] == 0


def test_nb1_text_index_forced_when_aux_processed_videos(builder):
    """scripts/03 rule: aux processed ≥1 video → force the BM25 rebuild."""
    aux = _cell(NB1, "ocr_all_keyframes")
    assert "AUX_CHANGED = False" in aux                # always defined
    assert "AUX_CHANGED = AUX_CHANGED or _did_work(n)" in aux
    assert "n is None or n > 0" in aux                 # None-returning API = changed
    ti = _cell(NB1, "build_text_index(settings, catalog")
    assert "force=FORCE_TEXT_INDEX or AUX_CHANGED" in ti


def test_nb1_model_tag_printed_before_embedding(builder):
    """Checkpoint-mixing diagnosability: the loaded model_tag must be printed
    BEFORE embed_all_keyframes runs, with the FORCE_EMBED remediation hint."""
    embed = _cell(NB1, "embed_all_keyframes")
    assert 'print(f"[{name}] model_tag = {model_tag}")' in embed
    assert (embed.index("model_tag = getattr")
            < embed.index('print(f"[{name}] model_tag')
            < embed.index("embed_all_keyframes(model"))
    assert "FORCE_EMBED=True" in embed                 # remediation hint
    assert "model tag mismatch" in embed               # names the embedder error
