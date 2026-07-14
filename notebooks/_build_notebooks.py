"""Generates the three Colab notebooks (single source of truth).

Run:  python notebooks/_build_notebooks.py
Regenerate after editing any CELL_* constant below — never hand-edit the
.ipynb JSON. tests/test_notebooks.py re-runs this builder and asserts the
output is deterministic + carries the load-bearing markers.

The heavy-duty Colab patterns (dependency discipline, Drive preflight,
run pointer + resume dashboard, empirical OOM probe, deliverables mirror)
are ported from the Toxicity v12 H100 autopilot notebook lineage.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": source.strip("\n").splitlines(keepends=True),
    }


def write_nb(name: str, cells: list[dict]) -> None:
    # nbformat 4.5 requires a unique cell `id` (round-3 fix C-R3-3: strict
    # validators reject id-less cells). Deterministic = sha1(index:source), so
    # regeneration stays byte-identical.
    for i, cell in enumerate(cells):
        cell["id"] = hashlib.sha1(
            f"{i}:{''.join(cell['source'])}".encode("utf-8")
        ).hexdigest()[:12]
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "A100"},
        },
        "cells": cells,
    }
    out = HERE / name
    out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {out.name} ({len(cells)} cells)")


# ══════════════════════════════════════════════════════════════════════════
# Shared cells
# ══════════════════════════════════════════════════════════════════════════

CELL_PARAMS = r'''
# ╔══════════════════════════════════════════════════════════════════╗
# ║  1 · PARAMS — the ONLY cell you may need to edit                 ║
# ╚══════════════════════════════════════════════════════════════════╝
DRIVE_PROJECT_DIR = "AIC2025"        # MyDrive/<this>/{data, artifacts}
REPO_URL  = "https://github.com/ledinhminhquan/Core-Vision_Perfect_V1.git"
REPO_REF  = "main"

# Which dense encoders to build indexes for (order = ensemble order).
#   "siglip2"          multilingual default (needed for training too)
#   "openclip"         English lane (DFN5B ViT-H/14-378) — strongest with translation
#   "qwen_embed"       optional HEAVY lane (Qwen embedding tower — strong, slow)
#   "provided_clip32"  organiser features — instant, no GPU (L-batches only);
#                      auto-added in the catalog cell when clip-features-32 exists
EMBED_MODELS = ["siglip2", "openclip"]

# Copy keyframes from Drive → local disk before embedding (much faster I/O).
COPY_KEYFRAMES_LOCAL = True

# Aux indexes to build (each is resumable; captions are the slowest).
RUN_OCR, RUN_ASR, RUN_CAPTIONS = True, True, True
CAPTION_STRIDE = 2                   # caption every 2nd keyframe (2× faster)

# K-batch shot detection: install TransNetV2 (the winning-team detector) for
# keyframe self-extraction. Installed --no-deps (Colab torch is never touched);
# without it extraction falls back to PySceneDetect automatically. (nb01 only)
INSTALL_TRANSNETV2 = True

# Force-rebuild toggles — mặc định False = resume/skip khi artifact đã có.
FORCE_CATALOG    = False             # rebuild the catalog parquet
FORCE_EMBED      = False             # re-embed every keyframe
FORCE_INDEX      = False             # rebuild the FAISS indexes
FORCE_AUX        = False             # redo OCR/ASR/captions from scratch
FORCE_TEXT_INDEX = False             # rebuild the persisted BM25 text index

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
print("params ok")
'''

CELL_MOUNT = r'''
# ╔══════════════════════════════════════════════════════════════════╗
# ║  2 · Mount Drive + folder layout + preflight write test          ║
# ╚══════════════════════════════════════════════════════════════════╝
import os, time
from pathlib import Path

from google.colab import drive
drive.mount("/content/drive")
assert Path("/content/drive/MyDrive").exists(), "Drive mount failed — rerun this cell"

PROJECT   = Path("/content/drive/MyDrive") / DRIVE_PROJECT_DIR
DATA_DIR  = PROJECT / "data"           # organiser dataset (merged packages)
ARTIFACTS = PROJECT / "artifacts"      # everything we build → survives disconnects
for p in (DATA_DIR, ARTIFACTS):
    p.mkdir(parents=True, exist_ok=True)

# PREFLIGHT (v12): Drive PHẢI ghi/đọc được — quota đầy hay mất quyền thì
# dừng NGAY tại đây thay vì hỏng giữa chừng sau 2 giờ chạy.
_probe = ARTIFACTS / f"_write_test_{int(time.time())}.tmp"
try:
    _probe.write_text("ok", encoding="utf-8")
    assert _probe.read_text(encoding="utf-8") == "ok"
    _probe.unlink()
    print("✅ Drive write test: OK")
except Exception as e:
    raise RuntimeError(
        f"❌ Không ghi được vào Drive ({ARTIFACTS}): {e!r}\n"
        "Kiểm tra dung lượng (quota) Google Drive và quyền truy cập thư mục, "
        "rồi chạy lại ô này."
    ) from e

# HF + pip caches on Drive → models/wheels download once, not per session.
os.environ["HF_HOME"] = str(ARTIFACTS / "hf_cache")
os.environ["PIP_CACHE_DIR"] = str(ARTIFACTS / "pip_cache")
for _d in (os.environ["HF_HOME"], os.environ["PIP_CACHE_DIR"]):
    Path(_d).mkdir(parents=True, exist_ok=True)

import shutil
free_gb = shutil.disk_usage(str(PROJECT)).free / 1e9
print(f"Project: {PROJECT}")
print(f"Drive free space: {free_gb:.0f} GB")
if free_gb < 20:
    print("⚠ Less than 20 GB free on Drive — embeddings/checkpoints may not fit!")
'''

CELL_REPO_DEPS = r'''
# ╔══════════════════════════════════════════════════════════════════╗
# ║  3 · Get repo + install dependencies (v12 discipline)            ║
# ╚══════════════════════════════════════════════════════════════════╝
# Quy tắc (học từ notebook Toxicity v12):
#   * check version qua importlib.metadata — KHÔNG import package trước khi
#     nâng cấp (import sớm sẽ ghim version cũ vào sys.modules);
#   * KHÔNG BAO GIỜ đụng torch/torchvision/torchaudio của Colab;
#   * chỉ cài đúng những gói thiếu/sai version (--prefer-binary);
#   * sau khi cài: `pip check` + micro-fix (tối đa 2 vòng, không crash),
#     rồi purge sys.modules TRƯỚC khi import cvp.
FORCE_REINSTALL_DEPS = False

import re, subprocess, sys
from pathlib import Path

REPO_DIR = Path("/content/Core-Vision_Perfect_V1")

def _run(cmd, **kw):
    print("$", " ".join(map(str, cmd)))
    return subprocess.run([str(c) for c in cmd], check=False, **kw).returncode

def _pip(args):
    return _run([sys.executable, "-m", "pip", *args])

# Private repo? Add a fine-grained PAT as Colab secret "GITHUB_TOKEN".
clone_url = REPO_URL
try:
    from google.colab import userdata
    _tok = userdata.get("GITHUB_TOKEN")
    if _tok and clone_url.startswith("https://github.com/"):
        clone_url = clone_url.replace("https://", f"https://{_tok}@")
except Exception:
    pass

if REPO_DIR.exists():
    _run(["git", "-C", REPO_DIR, "fetch", "--all", "-q"])
    _run(["git", "-C", REPO_DIR, "checkout", REPO_REF, "-q"])
    _run(["git", "-C", REPO_DIR, "pull", "-q"])
else:
    rc = _run(["git", "clone", "--branch", REPO_REF, clone_url, REPO_DIR])
    if rc != 0:  # private repo / no network → fall back to a Drive copy
        drive_copy = Path("/content/drive/MyDrive") / DRIVE_PROJECT_DIR / "Core-Vision_Perfect_V1"
        assert drive_copy.exists(), (
            "Clone failed and no Drive copy found. Either make the GitHub repo "
            f"reachable or upload the repo folder to {drive_copy}"
        )
        import shutil as _sh
        _sh.copytree(drive_copy, REPO_DIR)
        print("Using repo copy from Drive")

try:
    from packaging.requirements import Requirement
except ImportError:
    _pip(["install", "-q", "packaging"])
    from packaging.requirements import Requirement
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _meta_version

# Parse requirements-colab.txt; strip any torch* line (Colab rule #1: the
# preinstalled torch/torchvision/torchaudio build must never be touched).
reqs = []
for _line in (REPO_DIR / "requirements-colab.txt").read_text(encoding="utf-8").splitlines():
    _line = _line.split("#", 1)[0].strip()
    if not _line:
        continue
    try:
        _r = Requirement(_line)
    except Exception:
        print("⚠ bỏ qua requirement không parse được:", _line)
        continue
    if _r.name.lower().replace("-", "_").startswith("torch"):
        print("skip (never touch Colab torch):", _line)
        continue
    reqs.append(_r)

def _satisfied(r):
    """Installed + in range — via importlib.metadata, WITHOUT importing it."""
    try:
        v = _meta_version(r.name)
    except PackageNotFoundError:
        return False
    return (not r.specifier) or r.specifier.contains(v, prereleases=True)

missing = [r for r in reqs if FORCE_REINSTALL_DEPS or not _satisfied(r)]
did_install = bool(missing)
if missing:
    print(f"installing {len(missing)} package(s):", ", ".join(r.name for r in missing))
    _pip(["install", "-q", "--prefer-binary", *[str(r) for r in missing]])
else:
    print("dependencies satisfied — no pip install needed")

_pip(["install", "-q", "-e", str(REPO_DIR), "--no-deps"])

# faiss: gpu wheel with cpu fallback (metadata check — no import)
def _installed(*names):
    for n in names:
        try:
            _meta_version(n)
            return n
        except PackageNotFoundError:
            pass
    return None

if _installed("faiss-gpu-cu12", "faiss-gpu", "faiss-cpu", "faiss") is None:
    if _pip(["install", "-q", "faiss-gpu-cu12"]) != 0:
        _pip(["install", "-q", "faiss-cpu"])
    did_install = True

# `pip check` + micro-fixes for known conflicts (max 2 rounds, then warn)
def _pip_check():
    r = subprocess.run([sys.executable, "-m", "pip", "check"],
                       capture_output=True, text=True)
    return r.returncode, ((r.stdout or "") + "\n" + (r.stderr or "")).strip()

if did_install:
    rc, out = _pip_check()
    for _round in (1, 2):
        if rc == 0:
            break
        # pip's two REAL formats (round-3 fix L-R3-8 — the old regex missed the
        # version-conflict wording so that repair branch never ran):
        #   "pkgA 1.0 requires pkgB, which is not installed."
        #   "pkgA 1.0 has requirement pkgB<2,>=1, but you have pkgB 3.0."
        _specs = sorted({
            m.strip()
            for m in re.findall(
                r"(?:requires|has requirement) (.+?), (?:but you have|which is not installed)", out)
            if not m.strip().lower().startswith("torch")
        })
        if not _specs:
            break
        print(f"pip check micro-fix (round {_round}):", ", ".join(_specs))
        _pip(["install", "-q", "--prefer-binary", *_specs])
        rc, out = _pip_check()
    print("pip check: OK" if rc == 0 else f"⚠ pip check còn cảnh báo (không chặn):\n{out}")

# Purge stale sys.modules of upgraded packages BEFORE importing cvp (v12).
if did_install:
    _ALIAS = {"pillow": "pil", "pyyaml": "yaml", "opencv_python_headless": "cv2",
              "open_clip_torch": "open_clip", "scikit_learn": "sklearn"}
    _roots = {r.name.lower().replace("-", "_") for r in reqs} | {"cvp", "faiss"}
    _roots |= {_ALIAS[n] for n in _roots & set(_ALIAS)}
    _purged = [m for m in list(sys.modules)
               if m.split(".", 1)[0].lower().replace("-", "_") in _roots]
    for _m in _purged:
        sys.modules.pop(_m, None)
    if _purged:
        print(f"purged {len(_purged)} stale sys.modules entries")

if str(REPO_DIR / "src") not in sys.path:
    sys.path.insert(0, str(REPO_DIR / "src"))
import cvp
print("cvp", cvp.__version__, "ready")
'''

CELL_ENV_GPU = r'''
# ╔══════════════════════════════════════════════════════════════════╗
# ║  4 · Point cvp at the data + GPU setup (TF32 / SDPA / bf16)      ║
# ╚══════════════════════════════════════════════════════════════════╝
import os, torch

os.environ["CVP_PATHS__DATA_ROOT"]      = str(DATA_DIR)
os.environ["CVP_PATHS__ARTIFACTS_ROOT"] = str(ARTIFACTS)
os.environ["CVP_SETTINGS"] = str(REPO_DIR / "configs" / "settings.yaml")

print("torch", torch.__version__, "| CUDA build", torch.version.cuda)
print("GPU available:", torch.cuda.is_available())
GPU_NAME, VRAM_GB, USE_BF16 = "cpu", 0.0, False
if torch.cuda.is_available():
    GPU_NAME = torch.cuda.get_device_name(0)
    VRAM_GB = torch.cuda.get_device_properties(0).total_memory / 1e9
    USE_BF16 = torch.cuda.is_bf16_supported()
    # TF32 fast paths (new API with old fallback)
    try:
        torch.backends.cuda.matmul.fp32_precision = "tf32"
        torch.backends.cudnn.conv.fp32_precision = "tf32"
    except Exception:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    for fn in ("enable_flash_sdp", "enable_mem_efficient_sdp"):
        if hasattr(torch.backends.cuda, fn):
            getattr(torch.backends.cuda, fn)(True)
print(f"GPU: {GPU_NAME} | VRAM {VRAM_GB:.0f} GB | bf16={USE_BF16}")

# Colab secrets → env (optional: Gemini query enhancement/VQA, HF pushes)
try:
    from google.colab import userdata
    for _sec in ("GEMINI_API_KEY", "HF_TOKEN"):
        try:
            _v = userdata.get(_sec)
            if _v:
                os.environ[_sec] = _v
                print(f"secret {_sec}: loaded")
        except Exception:
            pass
except ImportError:
    pass

from cvp.config import load_settings
from cvp.utils.logging import setup_logging
settings = load_settings()
setup_logging("INFO")
print("data_root      =", settings.paths.data_root)
print("artifacts_root =", settings.paths.artifacts_root)
'''


# ══════════════════════════════════════════════════════════════════════════
# Notebook 1 — build artifacts
# ══════════════════════════════════════════════════════════════════════════

NB1_TITLE = r'''
# 🏗 Core Vision Ultimate Final — 01 · Build Artifacts (Colab)

Builds **everything the search system needs** from the organiser dataset on
your Drive: catalog → keyframe self-extraction (K-batches) → dense embeddings
→ FAISS indexes → OCR / ASR / captions → persisted BM25 text index.

**Every step is resumable** — if Colab disconnects, just *Runtime → Run all*
again; finished videos are skipped. Stage timings land in
`artifacts/logs/nb01.log`.

Prerequisites (see `docs/DRIVE_SETUP.md`):
`MyDrive/AIC2025/data/{keyframes, map-keyframes, media-info, clip-features-32, objects, videos}`

GPU: any (T4 works; A100/H100 much faster for SigLIP-2 + captions).
'''

NB1_UNZIP = r'''
# ── 5 · (optional) Auto-extract organiser zips still sitting in data/ ──
# If you uploaded raw zips (Keyframes_L21.zip, ...) instead of extracted
# folders, this unpacks them into the right places, then removes nothing.
import zipfile
from pathlib import Path

ZIP_DEST = {
    "keyframes":       DATA_DIR / "keyframes",
    "videos":          DATA_DIR / "videos",
    "clip-features":   DATA_DIR / "clip-features-32",
    "map-keyframes":   DATA_DIR / "map-keyframes",
    "media-info":      DATA_DIR / "media-info",
    "objects":         DATA_DIR / "objects",
}

def guess_dest(zname: str):
    z = zname.lower()
    if z.startswith("keyframes"): return ZIP_DEST["keyframes"]
    if z.startswith("videos"):    return ZIP_DEST["videos"]
    if "clip-features" in z:      return ZIP_DEST["clip-features"]
    if "map-keyframes" in z:      return ZIP_DEST["map-keyframes"]
    if "media-info" in z:         return ZIP_DEST["media-info"]
    if "objects" in z:            return ZIP_DEST["objects"]
    return None

for zp in sorted(DATA_DIR.glob("*.zip")):
    dest = guess_dest(zp.name)
    if dest is None:
        print("skip (unknown):", zp.name); continue
    marker = dest / f".unzipped-{zp.stem}"
    if marker.exists():
        continue
    print("unzipping", zp.name, "→", dest)
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zp) as z:
        names = z.namelist()
        # strip a wrapping top-level folder if the zip has one
        top = names[0].split("/")[0] if names and "/" in names[0] else None
        wrapped = top and all(n.startswith(top + "/") for n in names if n.strip("/"))
        z.extractall(dest.parent / "__tmp_unzip" if wrapped else dest)
        if wrapped:
            import shutil as _sh
            src = dest.parent / "__tmp_unzip" / top
            dest.mkdir(parents=True, exist_ok=True)
            for item in src.iterdir():
                target = dest / item.name
                if not target.exists():
                    _sh.move(str(item), str(target))
            _sh.rmtree(dest.parent / "__tmp_unzip", ignore_errors=True)
    marker.touch()
print("zip check done")
'''

NB1_LOCAL_COPY = r'''
# ── 6 · Copy keyframes Drive → local disk (I/O speed) ──
# Embedding reads hundreds of thousands of small JPGs; Drive FUSE is ~50×
# slower than local disk. Artifacts still go to Drive.
import shutil
from pathlib import Path

if COPY_KEYFRAMES_LOCAL and (DATA_DIR / "keyframes").exists():
    LOCAL_DATA = Path("/content/data")
    (LOCAL_DATA).mkdir(exist_ok=True)
    for sub in ("keyframes", "map-keyframes", "media-info", "objects", "clip-features-32"):
        src, dst = DATA_DIR / sub, LOCAL_DATA / sub
        if not src.exists() or dst.exists():
            continue
        print(f"copying {sub} → local ...")
        # copy into a tmp dir then rename: an interrupted copy must not leave
        # a partial folder that a re-run would silently accept
        tmp_dst = LOCAL_DATA / (sub + ".__tmp")
        if tmp_dst.exists():
            shutil.rmtree(tmp_dst)
        shutil.copytree(src, tmp_dst)
        tmp_dst.rename(dst)
    # videos stay on Drive (huge); link them in
    if (DATA_DIR / "videos").exists() and not (LOCAL_DATA / "videos").exists():
        (LOCAL_DATA / "videos").symlink_to(DATA_DIR / "videos")
    import os
    os.environ["CVP_PATHS__DATA_ROOT"] = str(LOCAL_DATA)
    from cvp.config import load_settings
    settings = load_settings()
    print("data_root now:", settings.paths.data_root)
else:
    print("using Drive data_root directly")
'''

NB1_CATALOG = r'''
# ── 7 · Catalog + self-extract K-batch keyframes (+ sync back to Drive) ──
import time
from contextlib import contextmanager

@contextmanager
def _log_stage(name: str):
    """Tee nhẹ kiểu v12: ghi start/end/duration của mỗi stage lên Drive."""
    log_path = ARTIFACTS / "logs" / "nb01.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} START {name}\n")
    try:
        yield
    finally:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} END   {name} "
                    f"({time.time() - t0:.0f}s)\n")

from cvp.data.catalog import KeyframeCatalog
from cvp.data.extraction import extract_missing

# TransNetV2 for K-batch shot detection (best detector per the winning teams).
# --no-deps: torch/numpy/opencv already exist — Colab torch must not be touched.
# ffmpeg-python is REQUIRED by predict_video() (round-3 fix M-R3-1) and itself
# hard-imports `past.builtins` from the `future` distribution at import time
# (round-4 fix: `future` is a REAL runtime dep, not a py2 leftover). All three
# packages DO declare dependencies, but every one of them is either already on
# Colab or installed by this very call — so --no-deps stays torch-safe (round-5
# wording fix C-R5-2; the ffmpeg BINARY ships with Colab).
# Missing/failed install is fine: extraction falls back to PySceneDetect.
if INSTALL_TRANSNETV2:
    def _transnet_ready() -> bool:
        try:
            import transnetv2_pytorch  # noqa: F401
            import ffmpeg  # noqa: F401
            return True
        except ImportError:
            return False

    if _transnet_ready():
        print("TransNetV2 + ffmpeg-python: already installed")
    else:
        _pip(["install", "-q", "--no-deps", "transnetv2-pytorch", "ffmpeg-python", "future"])
        # Verify-then-report: a green pip rc alone proved nothing in round 3.
        print("TransNetV2:", "READY" if _transnet_ready()
              else "unavailable — SceneDetect fallback will be used")

with _log_stage("catalog"):
    n_extracted = extract_missing(settings)     # K-videos without keyframes
    print("extracted videos:", n_extracted)
    catalog = KeyframeCatalog(settings)
    df = catalog.build(force=FORCE_CATALOG or n_extracted > 0)
print(f"catalog: {len(df):,} keyframes / {df.video_id.nunique()} videos "
      f"({int(df.has_map.sum()):,} frames with map-keyframes)")

# K-batch sync-back: khi COPY_KEYFRAMES_LOCAL=True, extract_missing ghi
# keyframes + map-keyframes mới vào data_root LOCAL (/content/data) — local
# disk BỐC HƠI khi hết session, NB3/laptop sẽ không bao giờ thấy chúng.
# Mirror mọi folder/CSV mà Drive CHƯA có về DATA_DIR (tmp + rename như ô 6).
# No-op khi chạy thẳng trên Drive hoặc không có gì mới.
from pathlib import Path
import shutil as _sh

n_synced = 0
_local_root = Path(str(settings.paths.data_root)).resolve()
if _local_root != DATA_DIR.resolve():
    for src_root, dst_root, want_dir in (
        (_local_root / "keyframes", DATA_DIR / "keyframes", True),
        (_local_root / "map-keyframes", DATA_DIR / "map-keyframes", False),
    ):
        if not src_root.is_dir():
            continue
        for item in sorted(src_root.iterdir()):
            if item.name.endswith(".__tmp"):
                continue
            if not (item.is_dir() if want_dir else item.suffix == ".csv"):
                continue
            target = dst_root / item.name
            if target.exists():
                continue
            dst_root.mkdir(parents=True, exist_ok=True)
            tmp = dst_root / (item.name + ".__tmp")
            if tmp.is_dir():
                _sh.rmtree(tmp)
            elif tmp.exists():
                tmp.unlink()
            (_sh.copytree if want_dir else _sh.copy2)(item, tmp)
            tmp.rename(target)
            n_synced += 1
print(f"sync-back to Drive: {n_synced} item(s)" if n_synced
      else "sync-back: nothing new for Drive")

# Organiser CLIP features → free extra retrieval lane, but ONLY with FULL
# coverage: ingest_provided_features bỏ qua video không có .npy, rồi
# store.build sẽ hard-fail trên lane thiếu vector (video K-batch tự extract
# KHÔNG BAO GIỜ có organiser features).
feat_dir = settings.paths.data(settings.paths.clip_features_dir)
if feat_dir.is_dir() and "provided_clip32" not in EMBED_MODELS:
    _vids = set(map(str, df.video_id.unique()))
    _have = {p.stem for p in feat_dir.glob("*.npy")}
    _missing_feats = sorted(_vids - _have)
    if not _missing_feats:
        EMBED_MODELS = EMBED_MODELS + ["provided_clip32"]
        print("auto-added 'provided_clip32' to EMBED_MODELS (full .npy coverage)")
    else:
        print(f"provided_clip32 lane skipped: {len(_missing_feats)} video(s) have no "
              f"organiser features (K-batch present?) — e.g. {_missing_feats[:3]}")
'''

NB1_EMBED = r'''
# ── 8 · Dense embeddings + FAISS index per model (resumable) ──
# ⚠ model_tag được in TRƯỚC khi embed từng lane. Nếu embedder dừng với lỗi
# "model tag mismatch": session này nạp CHECKPOINT KHÁC session trước (vd.
# openclip: hub PE-Core không tải được → fallback DFN5B) — trộn 2 checkpoint
# trong một folder embeddings sẽ hỏng index. Chạy lại với FORCE_EMBED=True
# để re-embed sạch, hoặc khôi phục mạng để nạp đúng checkpoint cũ.
import gc, torch
from cvp.index.embedder import embed_all_keyframes, ingest_provided_features
from cvp.index.store import IndexStore
from cvp.models.registry import build_model

for name in EMBED_MODELS:
    print(f"\n════ {name} ════")
    model_tag = None
    with _log_stage(f"embed:{name}"):
        if name == "provided_clip32":
            ingest_provided_features(settings, catalog)
        else:
            model = build_model(settings, name)
            # the EXACT loaded checkpoint, printed up front so a cross-session
            # mismatch is diagnosable; also stamps the index (scripts/02 contract)
            model_tag = getattr(model, "model_tag", None) or model.key
            print(f"[{name}] model_tag = {model_tag}")
            embed_all_keyframes(model, settings, catalog, overwrite=FORCE_EMBED)
            del model; gc.collect(); torch.cuda.empty_cache()
        store = IndexStore(settings, name)
        store.build(catalog, force=FORCE_INDEX or FORCE_EMBED, model_tag=model_tag)
    print(f"[{name}] index: {store.count():,} vectors, dim {store.dim()}")
'''

NB1_AUX = r'''
# ── 9 · Aux indexes: OCR / ASR / captions (each resumable) ──
# AUX_CHANGED feeds cell 10: scripts/03's rule is "force-rebuild the BM25
# text index whenever this run processed ≥1 aux video".
AUX_CHANGED = False

def _did_work(n) -> bool:
    return n is None or n > 0   # None (older API) → assume something changed

if RUN_OCR:
    from cvp.auxindex.ocr import ocr_all_keyframes
    with _log_stage("ocr"):
        n = ocr_all_keyframes(settings, catalog, overwrite=FORCE_AUX)
    AUX_CHANGED = AUX_CHANGED or _did_work(n)
    print(f"OCR: processed {n} videos")
if RUN_ASR:
    from cvp.auxindex.asr import asr_all_videos
    with _log_stage("asr"):
        n = asr_all_videos(settings, catalog, overwrite=FORCE_AUX)
    AUX_CHANGED = AUX_CHANGED or _did_work(n)
    print(f"ASR: processed {n} videos")
if RUN_CAPTIONS:
    from cvp.auxindex.captioner import caption_all_keyframes
    with _log_stage("captions"):
        n = caption_all_keyframes(settings, catalog, stride=CAPTION_STRIDE, overwrite=FORCE_AUX)
    AUX_CHANGED = AUX_CHANGED or _did_work(n)
    print(f"Captions: processed {n} videos")
print("aux indexes done | AUX_CHANGED =", AUX_CHANGED)
'''

NB1_TEXT_INDEX = r'''
# ── 10 · Persisted BM25 text index (OCR/ASR/captions/metadata) ──
# Query-time BM25 becomes candidate-restricted lookups instead of rescanning
# every artifact on engine start. Direct function call into scripts/03 (the
# module name starts with a digit → import_module, not a plain import).
import sys
from importlib import import_module

if str(REPO_DIR / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_DIR / "scripts"))
build_text_index = import_module("03_build_aux_indexes").build_text_index

# scripts/03 rule: rebuild whenever the aux cell processed ≥1 video this run
# (AUX_CHANGED), not only when the catalog signature changed.
with _log_stage("text_index"):
    built = build_text_index(settings, catalog, force=FORCE_TEXT_INDEX or AUX_CHANGED)
if built:
    print("text index built: [" + ", ".join(built) + "]")
else:
    print("text index up-to-date (signature match) — đặt FORCE_TEXT_INDEX=True để build lại")
'''

NB1_DOCTOR = r'''
# ── 11 · Health report ──
import json
from cvp.pipeline.ingest import doctor
print(json.dumps(doctor(settings), indent=2, ensure_ascii=False))
print("\n✅ Artifacts build complete. Next: notebooks/02_train_vi_encoder_H100.ipynb")
'''


# ══════════════════════════════════════════════════════════════════════════
# Notebook 2 — H100 training
# ══════════════════════════════════════════════════════════════════════════

NB2_TITLE = r'''
# 🚀 Core Vision Ultimate Final — 02 · Train Vietnamese Encoder (H100 autopilot)

Fine-tunes the **SigLIP-2 text tower on Vietnamese** captions of *this exact
corpus* via **LoRA-LiT**: image tower frozen → your FAISS index stays valid;
LoRA + distillation anchor → no catastrophic forgetting. After training a
**WiSE-FT α sweep** exports the best base↔tuned interpolation.

**Autopilot / crash-safe:** checkpoints go to Drive every eval; a run pointer
(`runs/vi_siglip2/active_train_run.json`) tracks status + config hash. On any
disconnect just *Runtime → Run all* — the resume dashboard shows where the
last session stopped and training resumes from the last valid checkpoint.
The best model is continuously exported to `artifacts/checkpoints/vi_siglip2_best`
and mirrored to `artifacts/deliverables/latest`.

Prerequisite: notebook 01 finished `siglip2` embeddings + captions.
GPU: **H100 ≈ 30–60 min**; A100 ~2×, L4/T4 slower but works (auto-sized + OOM probe).
'''

NB2_PARAMS_EXTRA = r'''
# ── 5 · Training params (auto-sized by GPU; safe to leave as-is) ──
EPOCHS            = 8
TARGET_EFF_BATCH  = 2048       # micro_batch × grad_accum kept ≈ constant
LR                = 1e-4
LORA_R            = 32
DISTILL_BETA0     = 0.30
EVAL_EVERY_STEPS  = 200
EARLY_STOP_EVALS  = 5
WORD_DROPOUT      = 0.05

# Loss recipe (docs/TRAINING.md) — new TrainConfig knobs:
LOSS                = "siglip"       # "siglip" | "infonce" (A/B the two losses)
ANCHOR_MIX_RATIO    = 0.25           # fraction of (EN caption, image) anchor pairs
HARD_NEG_PER_SAMPLE = 0              # K in-video hard negatives per sample (0 = off)
# WiSE-FT sweep → export_dir/wiseft_best. α=1.0 = raw tuned tower, so the
# winner can never score below it (round-3 fix L-R3-7 — matches TrainConfig).
WISEFT_ALPHAS       = [0.4, 0.5, 0.6, 1.0]

USE_PUBLIC_DATA   = True       # KTVIC + UIT-ViIC human VI captions (needs network)
PROBE_BATCH       = True       # empirical OOM probe (False → VRAM table only)
PUSH_TO_HF_HUB    = False      # set True + add HF_TOKEN in Colab secrets to upload
HF_REPO_ID        = "ledinhminhquan/vi-siglip2-so400m-corevision"

# micro-batch by VRAM (text tower only — generous headroom); this is only the
# STARTING GUESS — the next cell probes the real model and halves on OOM.
def pick_micro_batch(vram_gb: float) -> int:
    if vram_gb >= 70: return 512   # H100/A100-80G
    if vram_gb >= 38: return 256   # A100-40G
    if vram_gb >= 22: return 128   # L4
    if vram_gb >= 14: return 64    # T4
    return 16
MICRO_BATCH = pick_micro_batch(VRAM_GB)
GRAD_ACCUM  = max(1, TARGET_EFF_BATCH // MICRO_BATCH)
print(f"VRAM-table guess: micro_batch={MICRO_BATCH} × grad_accum={GRAD_ACCUM} "
      f"(effective {MICRO_BATCH * GRAD_ACCUM})")
'''

NB2_PROBE = r'''
# ── 6 · Empirical OOM batch probe (v12 pattern) ──
# pick_micro_batch() is a guess; the probe runs a REAL forward+backward of the
# text tower that will actually be trained, halving on OOM (floor 8).
# GRAD_ACCUM is recomputed so TARGET_EFF_BATCH never changes — probing tunes
# throughput, never training semantics. Costs one extra model load (~20 s).
import gc, torch

def _probe_micro_batch(guess: int, floor: int = 8) -> int:
    from transformers import AutoProcessor, SiglipModel

    print("loading probe model (text tower)...")
    proc = AutoProcessor.from_pretrained(settings.finetuned.base_id)
    model = SiglipModel.from_pretrained(settings.finetuned.base_id, torch_dtype=torch.float32)
    tower = model.text_model.to("cuda")
    for p in tower.parameters():
        p.requires_grad_(True)
    toks = proc.tokenizer(
        ["một người dẫn chương trình đang nói trong trường quay truyền hình"],
        padding="max_length", truncation=True, max_length=64, return_tensors="pt")
    toks = {k: v.to("cuda") for k, v in toks.items()}
    amp = torch.bfloat16 if USE_BF16 else torch.float16
    b = max(floor, int(guess))
    try:
        while True:
            try:
                kw = {k: v.repeat(b, 1) for k, v in toks.items()}
                with torch.autocast("cuda", dtype=amp):
                    emb = tower(**kw).pooler_output
                    loss = emb.float().pow(2).mean()
                loss.backward()
                del kw, emb, loss
                print(f"  micro_batch={b}: ✅ fits")
                return b
            except RuntimeError as e:   # torch.cuda.OutOfMemoryError IS a RuntimeError
                if "out of memory" not in str(e).lower():
                    raise
                print(f"  micro_batch={b}: 💥 OOM — halving")
                if b <= floor:
                    return floor
                b = max(floor, b // 2)
            finally:
                tower.zero_grad(set_to_none=True)
                gc.collect(); torch.cuda.empty_cache()
    finally:                            # give the VRAM back to the trainer
        del model, tower, proc
        gc.collect(); torch.cuda.empty_cache()

if PROBE_BATCH and torch.cuda.is_available():
    MICRO_BATCH = _probe_micro_batch(MICRO_BATCH)
else:
    print(f"probe skipped (PROBE_BATCH={PROBE_BATCH}, cuda={torch.cuda.is_available()}) "
          "— keeping the VRAM-table value")
GRAD_ACCUM = max(1, TARGET_EFF_BATCH // MICRO_BATCH)
print(f"final: micro_batch={MICRO_BATCH} × grad_accum={GRAD_ACCUM} "
      f"(effective {MICRO_BATCH * GRAD_ACCUM}, target {TARGET_EFF_BATCH})")
'''

NB2_PUBLIC_DATA = r'''
# ── 7 · Public Vietnamese caption parquets (KTVIC + UIT-ViIC) ──
# Human-written VI captions (KTVIC daily-life ~21.6k, UIT-ViIC sports ~18k)
# as extra training channels → the fine-tune learns Vietnamese, not just
# Vintern's caption style. Resumable; already-built parquets are skipped
# instantly. Network optional — failures only print a warning.
if USE_PUBLIC_DATA:
    try:
        from cvp.training.public_datasets import build_all_public_parquets
        paths = build_all_public_parquets(settings=settings)
        for p in paths:
            print("extra parquet:", p)
    except Exception as e:
        print(f"⚠ public datasets skipped (mạng/HF không khả dụng — không sao): {e}")
else:
    print("USE_PUBLIC_DATA=False — training on corpus captions only")
'''

NB2_DATASET = r'''
# ── 8 · Build the caption↔embedding training set (skips if fresh) ──
from pathlib import Path
from cvp.training.build_dataset import build_training_set

train_dir = settings.paths.art("train_data")
public_dir = settings.paths.art("train_data", "public")
_pub = sorted(public_dir.glob("*.parquet")) if public_dir.is_dir() else []
meta_p, emb_p = train_dir / "meta.parquet", train_dir / "embeds.npy"
# stale when a public parquet (cell 7) is newer than the merged train_data
_stale = (meta_p.exists() and _pub
          and max(p.stat().st_mtime for p in _pub) > meta_p.stat().st_mtime)
if meta_p.exists() and emb_p.exists() and not _stale:
    import pandas as pd
    meta = pd.read_parquet(meta_p)
    print(f"train_data exists: {len(meta):,} pairs "
          f"({(meta.split=='train').sum():,} train / {(meta.split=='val').sum():,} val) — skipping build")
else:
    if _stale:
        print("public parquets newer than train_data → rebuilding to merge them")
    n_train, n_val = build_training_set(settings, model_key="siglip2")
    print(f"built: {n_train:,} train / {n_val:,} val pairs")
'''

NB2_RUN_POINTER = r'''
# ── 9 · Run identity (sha256 of TrainConfig) + resume dashboard (v12) ──
import dataclasses, hashlib, json, os
from datetime import datetime, timezone
from pathlib import Path

from cvp.training.lit_trainer import TrainConfig

cfg = TrainConfig(
    base_id=settings.finetuned.base_id,
    train_data_dir=str(settings.paths.art("train_data")),
    run_dir=str(settings.paths.art("runs", "vi_siglip2")),
    export_dir=str(settings.paths.art("checkpoints", "vi_siglip2_best")),
    epochs=EPOCHS,
    micro_batch=MICRO_BATCH,
    grad_accum=GRAD_ACCUM,
    lr=LR,
    lora_r=LORA_R,
    lora_alpha=LORA_R * 2,
    distill_beta0=DISTILL_BETA0,
    eval_every_steps=EVAL_EVERY_STEPS,
    early_stop_patience=EARLY_STOP_EVALS,
    word_dropout=WORD_DROPOUT,
    loss=LOSS,
    anchor_mix_ratio=ANCHOR_MIX_RATIO,
    hard_negative_per_sample=HARD_NEG_PER_SAMPLE,
    wiseft_alphas=WISEFT_ALPHAS,
)

# micro_batch/grad_accum/num_workers are RESUME-NEUTRAL: the OOM probe may
# retune them between sessions without changing training semantics (the
# effective batch stays constant), so they are excluded from the identity.
_RESUME_NEUTRAL = {"micro_batch", "grad_accum", "num_workers"}

def cfg_hash(c: TrainConfig) -> str:
    d = {k: v for k, v in sorted(dataclasses.asdict(c).items()) if k not in _RESUME_NEUTRAL}
    return hashlib.sha256(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:12]

CFG_HASH = cfg_hash(cfg)
POINTER = Path(cfg.run_dir) / "active_train_run.json"

def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def read_pointer():
    try:
        return json.loads(POINTER.read_text(encoding="utf-8"))
    except Exception:
        return None

def write_pointer(status: str, **extra) -> None:
    """Atomic tmp+rename — a mid-write disconnect never leaves broken JSON."""
    prev = read_pointer() or {}
    payload = {
        "status": status,                        # running | crashed | finished
        "cfg_hash": CFG_HASH,
        "run_dir": cfg.run_dir,
        "export_dir": cfg.export_dir,
        "started_utc": prev.get("started_utc") if prev.get("cfg_hash") == CFG_HASH else None,
        "updated_utc": _utc(),
    }
    payload["started_utc"] = payload["started_utc"] or payload["updated_utc"]
    payload.update(extra)
    POINTER.parent.mkdir(parents=True, exist_ok=True)
    tmp = POINTER.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, POINTER)

# ── Read-only resume dashboard ──
ptr = read_pointer()
print("═══ RESUME DASHBOARD ═══")
if ptr:
    print(f"previous run : status={ptr.get('status')!r}  cfg_hash={ptr.get('cfg_hash')}")
    print(f"               started={ptr.get('started_utc')}  updated={ptr.get('updated_utc')}")
else:
    print("previous run : (none — fresh start)")

ckpts = sorted((d for d in Path(cfg.run_dir).glob("step-*") if (d / "state.json").is_file()),
               key=lambda d: int(d.name.split("-")[1]))
if ckpts:
    state = json.loads((ckpts[-1] / "state.json").read_text(encoding="utf-8"))
    print(f"last ckpt    : {ckpts[-1]}")
    print(f"               global_step={state.get('step')}  best R@5={state.get('best_r5', -1):.4f}  "
          f"evals_since_best={state.get('evals_since_best')}")
else:
    print("last ckpt    : (none)")
_meta = Path(cfg.export_dir) / "export_meta.json"
if _meta.is_file():
    _m = (json.loads(_meta.read_text(encoding="utf-8")).get("metrics") or {})
    print(f"best export  : R@5={_m.get('R@5', float('nan')):.4f} → {cfg.export_dir}")

if ptr and ptr.get("cfg_hash") != CFG_HASH:
    print(f"⚠ CONFIG MISMATCH: pointer {ptr.get('cfg_hash')} ≠ current {CFG_HASH}")
    print("  Config đã thay đổi — checkpoint trong run_dir thuộc config CŨ.")
    print(f"  Muốn train sạch từ đầu: xoá {cfg.run_dir} rồi chạy lại; nếu giữ nguyên,")
    print("  trainer sẽ resume checkpoint cũ với config MỚI (LR schedule có thể lệch).")
elif ptr:
    print("✓ config hash khớp — auto-resume an toàn.")

write_pointer("running")
print(f"\nrun identity {CFG_HASH} | pointer → running")
'''

NB2_TRAIN = r'''
# ── 10 · TRAIN (autopilot: after any disconnect just Runtime → Run all) ──
from cvp.training.lit_trainer import LiTTrainer

trainer = LiTTrainer(settings, cfg)
try:
    final_metrics = trainer.train()
except BaseException as e:   # KeyboardInterrupt/SystemExit also mark crashed
    write_pointer("crashed", error=repr(e)[:500])
    raise
write_pointer("finished",
              final_metrics={k: round(float(v), 4) for k, v in final_metrics.items()})
print("final:", final_metrics)
'''

NB2_DELIVERABLES = r'''
# ── 11 · Deliverables mirror → artifacts/deliverables/latest (v12) ──
# copytree, NOT symlink — Drive/FUSE does not support symlinks.
import shutil
from pathlib import Path

export_dir = Path(str(settings.paths.art("checkpoints", "vi_siglip2_best")))
if (export_dir / "text_tower.safetensors").exists():
    dest = ARTIFACTS / "deliverables" / "latest"
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copytree(export_dir, dest, dirs_exist_ok=True)
    print("mirrored", sorted(p.name for p in dest.iterdir()), "→", dest)
else:
    print(f"no export at {export_dir} yet — run the TRAIN cell first")
'''

NB2_EVAL = r'''
# ── 12 · Compare zero-shot vs fine-tuned vs WiSE-FT on the val split ──
import gc, os
import numpy as np, torch
from pathlib import Path
from cvp.config import load_settings
from cvp.eval.metrics import retrieval_metrics
from cvp.models.registry import build_model
from cvp.training.datamodule import TextImageEmbedDataset

ds = TextImageEmbedDataset(settings, "val", word_dropout=0.0)
img = ds.embeds / np.maximum(np.linalg.norm(ds.embeds, axis=1, keepdims=True), 1e-9)

_export = Path(str(settings.paths.art("checkpoints", "vi_siglip2_best")))
# name → checkpoint dir (None = plain zero-shot backend). The wiseft row
# points finetuned.checkpoint at export_dir/wiseft_best via env.
CANDIDATES = {"siglip2": None, "finetuned": _export, "wiseft": _export / "wiseft_best"}

rows = {}
for name, ckpt in CANDIDATES.items():
    if ckpt is not None and not (ckpt / "text_tower.safetensors").exists():
        print(f"[{name}] skipped: no checkpoint at {ckpt}")
        continue
    try:
        if ckpt is not None:
            os.environ["CVP_FINETUNED__CHECKPOINT"] = str(ckpt)
            model = build_model(load_settings(), "finetuned")
        else:
            model = build_model(settings, name)
        rows[name] = retrieval_metrics(model.encode_text(ds.captions), img)
        del model
    except Exception as e:
        print(f"[{name}] skipped: {e}")
    finally:
        os.environ.pop("CVP_FINETUNED__CHECKPOINT", None)
        gc.collect(); torch.cuda.empty_cache()

if "siglip2" in rows and len(rows) > 1:
    others = [k for k in rows if k != "siglip2"]
    print(f"\n{'metric':8} {'baseline':>10} " + " ".join(f"{k:>10}" for k in others))
    for m in rows["siglip2"]:
        line = f"{m:8} {rows['siglip2'][m]:>10.4f}"
        for k in others:
            line += f" {rows[k][m]:>10.4f}"
        print(line)
'''

NB2_PUSH = r'''
# ── 13 · WiSE-FT winner + (optional) HF push + how to USE the model ──
import json
from pathlib import Path

_export = Path(str(settings.paths.art("checkpoints", "vi_siglip2_best")))
wise_meta = _export / "wiseft_best" / "export_meta.json"
if wise_meta.is_file():
    _wm = json.loads(wise_meta.read_text(encoding="utf-8"))
    print(f"🏆 WiSE-FT winner: alpha={_wm.get('wiseft_alpha')} "
          f"R@5={(_wm.get('metrics') or {}).get('R@5', float('nan')):.4f} "
          f"| alphas tried: {_wm.get('alphas_tried')}")
else:
    print("(no wiseft_best export yet — sweep runs at the end of training)")

if PUSH_TO_HF_HUB:
    from google.colab import userdata
    from huggingface_hub import HfApi
    token = userdata.get("HF_TOKEN")
    api = HfApi(token=token)
    api.create_repo(HF_REPO_ID, private=True, exist_ok=True)
    api.upload_folder(folder_path=str(_export), repo_id=HF_REPO_ID)
    print("pushed →", HF_REPO_ID)
else:
    print("PUSH_TO_HF_HUB=False — skipped (checkpoint already on Drive)")

print(f"""
✅ Training done. To switch the engine to the tuned Vietnamese tower, set
BEFORE launching the app / scripts:
   CVP_EMBEDDING__MODEL=finetuned
   CVP_FINETUNED__CHECKPOINT={_export}
   # (or point it at {_export / 'wiseft_best'} for the WiSE-FT winner)
To make it an ensemble member instead ([finetuned, openclip] — needs the
openclip lane from notebook 01):
   CVP_EMBEDDING__MODEL=ensemble
   CVP_EMBEDDING__ENSEMBLE_MEMBERS='["finetuned", "openclip"]'
""")
'''


# ══════════════════════════════════════════════════════════════════════════
# Notebook 3 — test the system
# ══════════════════════════════════════════════════════════════════════════

NB3_TITLE = r'''
# ✅ Core Vision Ultimate Final — 03 · Test the System & Trained Model

Sanity-checks the full retrieval stack on Colab: loads the engine, runs
Vietnamese queries end-to-end (KIS / TRAKE / AVS), measures latency, writes a
sample submission CSV, validates + packages it Codabench-style, optionally
scores it against a local ground truth (official formulas) and dry-runs the
automatic track. The Streamlit UI can be served via a tunnel at the end.

Prerequisite: notebook 01 (and optionally 02 for the fine-tuned tower).
'''

NB3_ENGINE = r'''
# ── 5 · Load the search engine ──
# Pick the model for this session: "siglip2" | "finetuned" | "ensemble"
import os, time
os.environ["CVP_EMBEDDING__MODEL"] = "siglip2"     # ← change to test others
os.environ["CVP_QUERY__PROVIDER"]  = "none"        # offline test (no Gemini needed)

from cvp.search.engine import SearchEngine
from cvp.config import load_settings
settings = load_settings()
t0 = time.time()
engine = SearchEngine(settings)
print(f"engine ready in {time.time()-t0:.1f}s — {len(engine.catalog):,} keyframes")
'''

NB3_QUERIES = r'''
# ── 6 · Run sample Vietnamese KIS queries + show results inline ──
import time
import matplotlib.pyplot as plt
from PIL import Image

QUERIES = [
    "người dẫn chương trình mặc áo dài đứng trong trường quay",
    "đám cháy lớn khói đen bốc lên từ tòa nhà",
    "các vận động viên đua xe đạp trên đường phố",
    "món ăn được trình bày trên đĩa trắng",
    "cảnh ngập lụt trên đường phố, người dân lội nước",
]

for q in QUERIES:
    t0 = time.time()
    results = engine.search_text(q, display_k=8)
    dt = time.time() - t0
    print(f"\n🔎 {q}   ({dt:.2f}s)")
    if not results:
        print("   (no results)")
        continue
    fig, axes = plt.subplots(1, min(8, len(results)), figsize=(20, 3), squeeze=False)
    for ax, r in zip(axes[0], results):
        try:
            ax.imshow(Image.open(r.ref.path)); ax.axis("off")
            ax.set_title(f"{r.video_id}\nf={r.frame_idx} s={r.score:.2f}", fontsize=7)
        except Exception:
            ax.axis("off")
    plt.show()
'''

NB3_TRAKE_AVS = r'''
# ── 7 · TRAKE + AVS smoke test ──
events = [
    "vận động viên chuẩn bị xuất phát",
    "vận động viên chạy trên đường đua",
    "vận động viên về đích ăn mừng",
]
seqs = engine.search_trake(events, max_results=5)
for c in seqs[:5]:
    print(f"TRAKE {c.video_id} frames={c.frame_idxs} score={c.score:.3f}")

avs = engine.search_avs("cảnh giao thông đông đúc ở thành phố", limit=20)
print(f"\nAVS: {len(avs)} rows across {len({r.video_id for r in avs})} distinct videos")
'''

NB3_SUBMISSION = r'''
# ── 8 · Submission CSV round-trip (exact Codabench format) ──
from cvp.submission.writer import write_kis, write_trake

results = engine.search_text(QUERIES[0])
p = write_kis(settings.paths.art("submissions", "demo-kis.csv"),
              [(r.video_id, r.frame_idx) for r in results])
print(p, "\n" + p.read_text(encoding="utf-8")[:300])

if seqs:
    p2 = write_trake(settings.paths.art("submissions", "demo-trake.csv"),
                     [(c.video_id, c.frame_idxs) for c in seqs])
    print(p2, "\n" + p2.read_text(encoding="utf-8")[:300])
'''

NB3_PACKAGE = r'''
# ── 9 · Validate + package (Codabench) ──
# The organiser contract is re-checked on the finished CSVs (a wrong row
# silently costs a submission slot); errors BLOCK the zip.
import json
from cvp.submission.packager import has_errors, package_codabench, validate_submission_dir

sub_dir = settings.paths.art("submissions")
issues = validate_submission_dir(sub_dir, strict=True)
for i in issues:
    print(" ", i)
if has_errors(issues):
    print("❌ Còn lỗi chặn — sửa CSV rồi chạy lại ô này (zip KHÔNG được tạo).")
else:
    zip_path = settings.paths.art("submissions", "codabench.zip")
    package_codabench(sub_dir, zip_path, package_name=settings.submission.package_name)
    manifest = json.loads((zip_path.parent / "MANIFEST.json").read_text(encoding="utf-8"))
    print(f"\n📦 {zip_path.name}  sha256={manifest['zip_sha256'][:12]}…")
    for f in manifest["files"]:
        print(f"   {f['name']:24} task={f['task']:5} rows={f['rows']}")
'''

NB3_SCORE_GT = r'''
# ── 10 · (optional) Score against ground truth — official formulas ──
# Drop a GT file at MyDrive/<project>/queries/gt.json to see the exact
# Codabench-style scores (R@k / Final) of the CSVs written above.
GT_PATH = PROJECT / "queries" / "gt.json"
if GT_PATH.exists():
    from cvp.eval.official import score_run
    report = score_run(settings.paths.art("submissions"), GT_PATH)
    print(f"{'query':28} {'task':6} {'final':>7} {'best_rank':>9}")
    for stem, qs in sorted(report.per_query.items()):
        print(f"{stem:28} {qs.task:6} {qs.final:>7.4f} {str(qs.best_rank):>9}")
    for stem, why in sorted(report.unscored.items()):
        print(f"{stem:28} UNSCORED: {why}")
    print(f"\nmean_final = {report.mean_final:.4f}  "
          f"({report.num_scored}/{report.num_gt} GT queries scored)")
else:
    print("Không thấy", GT_PATH, "— muốn chấm điểm offline, tạo JSON keyed theo")
    print("tên file CSV (không đuôi). Ví dụ tối thiểu:")
    print("""{
  "demo-kis": {"task": "kis", "video_id": "L21_V001", "range": [500, 510]}
}""")
    print("Hỗ trợ range / center+epsilon / moments / answers — xem cvp/eval/official.py")
'''

NB3_AUTO_AGENT = r'''
# ── 11 · (optional) Automatic track dry-run (no human, no submit) ──
# Runs the full auto pipeline over MyDrive/<project>/queries/example/*.txt:
# infer task per filename → search → CSVs → validate → Codabench zip.
RUN_AUTO_AGENT = False
if RUN_AUTO_AGENT:
    from cvp.pipeline.auto_agent import run_auto
    qdir = PROJECT / "queries" / "example"
    qdir.mkdir(parents=True, exist_ok=True)
    if not any(qdir.glob("*.txt")):     # seed one sample query for the dry-run
        (qdir / "query-p1-1-kis.txt").write_text(
            "người dẫn chương trình mặc áo dài đứng trong trường quay", encoding="utf-8")
    rep = run_auto(qdir, settings.paths.art("submissions", "auto_dry_run"),
                   settings, submit=False)
    print(f"AutoRunReport: written={len(rep.written)} ok={rep.ok} zip={rep.zip_path}")
    for i in rep.issues:
        print("  ", i)
else:
    print("RUN_AUTO_AGENT=False — bật để chạy thử automatic track (submit=False).")
'''

NB3_UI = r'''
# ── 12 · (optional) Launch the Streamlit UI from Colab ──
# Colab can't open localhost — use the built-in proxy:
LAUNCH_UI = False
if LAUNCH_UI:
    import subprocess, time
    proc = subprocess.Popen(
        ["streamlit", "run", str(REPO_DIR / "app" / "streamlit_app.py"),
         "--server.port", "8501", "--server.headless", "true"])
    time.sleep(8)
    from google.colab import output
    output.serve_kernel_port_as_window(8501)
    # proc.terminate() when done
else:
    print("Set LAUNCH_UI=True to serve the app (better: run it on your laptop).")
'''


def main() -> None:
    write_nb("01_build_artifacts_colab.ipynb", [
        md(NB1_TITLE),
        code(CELL_PARAMS),
        code(CELL_MOUNT),
        code(CELL_REPO_DEPS),
        code(CELL_ENV_GPU),
        code(NB1_UNZIP),
        code(NB1_LOCAL_COPY),
        code(NB1_CATALOG),
        code(NB1_EMBED),
        code(NB1_AUX),
        code(NB1_TEXT_INDEX),
        code(NB1_DOCTOR),
    ])
    write_nb("02_train_vi_encoder_H100.ipynb", [
        md(NB2_TITLE),
        code(CELL_PARAMS),
        code(CELL_MOUNT),
        code(CELL_REPO_DEPS),
        code(CELL_ENV_GPU),
        code(NB2_PARAMS_EXTRA),
        code(NB2_PROBE),
        code(NB2_PUBLIC_DATA),
        code(NB2_DATASET),
        code(NB2_RUN_POINTER),
        code(NB2_TRAIN),
        code(NB2_DELIVERABLES),
        code(NB2_EVAL),
        code(NB2_PUSH),
    ])
    write_nb("03_test_system.ipynb", [
        md(NB3_TITLE),
        code(CELL_PARAMS),
        code(CELL_MOUNT),
        code(CELL_REPO_DEPS),
        code(CELL_ENV_GPU),
        code(NB3_ENGINE),
        code(NB3_QUERIES),
        code(NB3_TRAKE_AVS),
        code(NB3_SUBMISSION),
        code(NB3_PACKAGE),
        code(NB3_SCORE_GT),
        code(NB3_AUTO_AGENT),
        code(NB3_UI),
    ])


if __name__ == "__main__":
    main()
