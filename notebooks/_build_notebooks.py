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
    # newline="\n": without it Windows emits CRLF and every pytest run (the
    # suite regenerates notebooks) leaves git status dirty (review finding C7).
    out.write_text(json.dumps(nb, ensure_ascii=False, indent=1),
                   encoding="utf-8", newline="\n")
    print(f"wrote {out.name} ({len(cells)} cells)")


# ══════════════════════════════════════════════════════════════════════════
# Shared cells
# ══════════════════════════════════════════════════════════════════════════

CELL_PARAMS = r'''
# ╔══════════════════════════════════════════════════════════════════╗
# ║  1 · PARAMS — the ONLY cell you may need to edit                 ║
# ╚══════════════════════════════════════════════════════════════════╝
DRIVE_PROJECT_DIR = "AIC2025"        # MyDrive/<this>/{data, artifacts}
FIRST_TIME_SETUP = False             # True CHỈ cho lần ĐẦU TIÊN tạo dự án trên
#   Drive trống. Mặc định False: mount "lười metadata" sẽ KHÔNG BAO GIỜ được
#   tự đẻ thư mục dự án sinh đôi nữa (round-66 — bài học 07 live).
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
CAPTION_STRIDE = 4                   # caption mỗi keyframe thứ 4 (round-16: đủ dày
#   cho kênh recall BM25 mà nhanh gấp đôi stride 2. NÂNG stride luôn an toàn với
#   resume: video đã caption ở stride nhỏ hơn vẫn được tính là XONG ở stride lớn
#   hơn. Mọi phiên chạy song song PHẢI dùng CÙNG một stride.

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
import os, shutil, subprocess, time
from pathlib import Path

from google.colab import drive

_MP = "/content/drive"

def _drive_alive() -> bool:
    try:
        return Path(_MP, "MyDrive").exists()
    except OSError:
        return False

def _ensure_drive():
    """Mount / HỒI SINH Drive FUSE — dùng ở mọi cell dài hơi phía sau.

    Round-19 (live run 8): daemon DriveFS chết để lại mountpoint 'bẩn' →
    drive.mount kêu 'Mountpoint must not already contain files' và cả
    force_remount cũng bó tay. Trình tự cứu đúng: (1) fusermount -uz gỡ
    mount chết; (2) CHỈ khi chắc chắn không còn mount (os.path.ismount ==
    False — lúc này các entry trong mountpoint là RÁC LOCAL trên đĩa VM,
    không phải Drive thật) mới dọn sạch chúng; (3) mount lại.
    """
    for _try in range(4):
        if _drive_alive():
            return
        if _try:
            print(f"⚠ Drive FUSE chưa sống — hồi sinh (lần {_try}/3) ...")
        try:
            if os.path.ismount(_MP):
                subprocess.run(["fusermount", "-uz", _MP], capture_output=True)
                time.sleep(2)
            if os.path.isdir(_MP) and not os.path.ismount(_MP):
                for _c in os.listdir(_MP):     # rác local — KHÔNG phải Drive
                    _p = os.path.join(_MP, _c)
                    shutil.rmtree(_p, ignore_errors=True) if os.path.isdir(_p) \
                        else os.unlink(_p)
            drive.mount(_MP, force_remount=bool(_try))
        except Exception as _e:  # noqa: BLE001 — thử tiếp vòng sau
            print("   mount lỗi:", _e)
            time.sleep(5)
    if not _drive_alive():
        raise RuntimeError(
            "Không mount được Google Drive sau 4 lần thử — Runtime ▸ "
            "Disconnect and delete runtime rồi Run all lại (tiến độ đã lưu "
            "trên Drive còn nguyên).")

_ensure_drive()
assert Path("/content/drive/MyDrive").exists(), "Drive mount failed — rerun this cell"

PROJECT   = Path("/content/drive/MyDrive") / DRIVE_PROJECT_DIR
DATA_DIR  = PROJECT / "data"           # organiser dataset (merged packages)
ARTIFACTS = PROJECT / "artifacts"      # everything we build → survives disconnects
# Round-63 (live 07): mkdir NGAY trên mount còn "lười metadata" từng ĐẺ RA một
# AIC2025 SINH ĐÔI rỗng (Drive cho phép trùng tên) — từ đó mỗi phiên mới bind
# ngẫu nhiên vào bản thật hay bản rỗng và "không thấy data". Dự án đã tồn tại
# thì KHÔNG BAO GIỜ mkdir; chỉ khi chờ 3 phút vẫn không thấy (lần setup đầu
# tiên trong đời) mới được tạo.
_t0p = time.time()
while not PROJECT.exists() and time.time() - _t0p < 180:
    print(f"⏳ chưa thấy MyDrive/{DRIVE_PROJECT_DIR} — đợi metadata "
          f"({int(time.time() - _t0p)}s; TUYỆT ĐỐI không tự tạo vội) ...")
    time.sleep(10)
    try:
        list(Path("/content/drive/MyDrive").iterdir())   # cú hích ép nạp metadata
    except OSError:
        pass
if not PROJECT.exists():
    # round-66: KHÔNG BAO GIỜ tự tạo khi chưa được phép — chính là cỗ máy đẻ
    # thư mục dự án sinh đôi. Lần setup đầu tiên THẬT thì bật cờ ở cell 1.
    if not FIRST_TIME_SETUP:
        raise RuntimeError(
            f"3 phút vẫn không thấy MyDrive/{DRIVE_PROJECT_DIR} — máy ảo này "
            "hỏng metadata Drive. Runtime ▸ Disconnect and delete runtime rồi "
            "Run all lại máy mới (dữ liệu trên Drive vẫn nguyên vẹn). Nếu đây "
            "THẬT SỰ là lần đầu tạo dự án: đặt FIRST_TIME_SETUP = True ở cell 1.")
    print(f"⚠ FIRST_TIME_SETUP=True — tạo mới MyDrive/{DRIVE_PROJECT_DIR}.")
# Round-69 (audit): gate chống-sinh-đôi phải phủ cả THƯ MỤC CON — mount thấy
# AIC2025 nhưng chưa nạp children mà mkdir ngay thì data/artifacts sinh đôi
# y hệt vụ round-63, chỉ là một tầng sâu hơn.
for p in (DATA_DIR, ARTIFACTS):
    if p.exists() or FIRST_TIME_SETUP:
        p.mkdir(parents=True, exist_ok=True)
        continue
    _t0c = time.time()
    while not p.exists() and time.time() - _t0c < 120:
        print(f"⏳ chưa thấy {p.name}/ trong dự án — đợi metadata ({int(time.time() - _t0c)}s) ...")
        time.sleep(10)
        try:
            list(PROJECT.iterdir())                  # cú hích ép nạp children
        except OSError:
            pass
    if not p.exists():
        raise RuntimeError(
            f"2 phút không thấy {p.name}/ trong MyDrive/{DRIVE_PROJECT_DIR} — máy "
            "ảo hỏng metadata Drive. Runtime ▸ Disconnect and delete runtime rồi "
            "chạy máy mới; nếu đây là lần setup đầu tiên: FIRST_TIME_SETUP=True.")

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

# DATA-PRESENCE GATE (round-20, live run 9): trên VM mới, DriveFS có thể liệt
# kê data/ ra RỖNG suốt vài phút đầu (metadata sync lười) — mkdir exist_ok ở
# trên còn CHE mất triệu chứng, để cell 7 chết khó hiểu với "Keyframes folder
# not found". Poll tới 3 phút (mỗi listdir là một cú hích ép DriveFS fetch);
# hết kiên nhẫn thì dừng TO với chẩn đoán rõ ràng.
_t0 = time.time()
_data_ok = False
while time.time() - _t0 < 180:
    try:
        if any(DATA_DIR.iterdir()):
            _data_ok = True
            break
    except OSError:
        pass
    print(f"⏳ data/ đang rỗng — đợi DriveFS sync metadata ({int(time.time() - _t0)}s) ...")
    time.sleep(10)
if not _data_ok:
    raise RuntimeError(
        "data/ trên Drive vẫn RỖNG sau 3 phút chờ. Ba nguyên nhân thường gặp:\n"
        "  1) Phiên Colab đăng nhập NHẦM tài khoản Google (kiểm tra avatar góc "
        f"phải trên) — phải là tài khoản có MyDrive/{DRIVE_PROJECT_DIR}/data;\n"
        "  2) DriveFS sync quá chậm — Runtime ▸ Disconnect and delete runtime "
        "rồi Run all lại trên máy mới;\n"
        "  3) Lần chạy đầu tiên mà chưa upload dữ liệu — ném các zip của BTC "
        f"vào MyDrive/{DRIVE_PROJECT_DIR}/data trước (docs/DRIVE_SETUP.md).\n"
        "KHÔNG có gì bị mất — dữ liệu vẫn nằm nguyên trên Drive của tài khoản đúng.")
print(f"✅ data/ nhìn thấy dữ liệu sau {int(time.time() - _t0)}s")

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

def _run(cmd, show=None, **kw):
    # `show` masks credentials in the echoed command — a PAT-carrying clone
    # URL must NEVER be printed into the saved notebook output.
    print("$", " ".join(map(str, show or cmd)))
    return subprocess.run([str(c) for c in cmd], check=False, **kw).returncode

def _pip(args):
    return _run([sys.executable, "-m", "pip", *args])

# Private repo? Add a fine-grained PAT as Colab secret "GITHUB_TOKEN"
# (Contents: Read-only on this repo) and ENABLE its notebook-access toggle.
clone_url, _tok = REPO_URL, None
try:
    from google.colab import userdata
    _tok = userdata.get("GITHUB_TOKEN")
except Exception as _e:
    print(f"⚠ KHÔNG đọc được secret GITHUB_TOKEN ({type(_e).__name__}) — repo "
          "private sẽ KHÔNG clone được. Kiểm tra: 🔑 panel có secret tên đúng "
          "y hệt GITHUB_TOKEN và công tắc 'Notebook access' đã BẬT chưa?")
if _tok and clone_url.startswith("https://github.com/"):
    clone_url = clone_url.replace("https://", f"https://{_tok}@")
    print(f"GITHUB_TOKEN: loaded ({len(_tok)} chars, {_tok[:11]}…)")
elif not _tok:
    print("⚠ GITHUB_TOKEN trống/vắng mặt — thử clone KHÔNG xác thực "
          "(chắc chắn fail nếu repo private).")

if REPO_DIR.exists():
    _run(["git", "-C", REPO_DIR, "fetch", "--all", "-q"])
    _run(["git", "-C", REPO_DIR, "checkout", REPO_REF, "-q"])
    _run(["git", "-C", REPO_DIR, "pull", "-q"])
else:
    rc = _run(["git", "clone", "--branch", REPO_REF, clone_url, REPO_DIR],
              show=["git", "clone", "--branch", REPO_REF, REPO_URL, REPO_DIR])
    if rc != 0:  # private repo / no network → fall back to a Drive copy
        print("⚠ Clone THẤT BẠI. Nguyên nhân thường gặp, theo thứ tự:\n"
              "  1) Secret GITHUB_TOKEN sai tên / chưa bật Notebook access "
              "(xem cảnh báo phía trên);\n"
              "  2) PAT sai/hết hạn/thiếu quyền — cần fine-grained PAT với "
              "Contents: Read-only cấp cho ĐÚNG repo này;\n"
              "  3) Mạng Colab trục trặc tạm thời — chạy lại cell.")
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
# ONLY the packages actually (re)installed THIS run (round-11): purging every
# requirement dropped numpy/pandas from sys.modules while torch still held
# references to the old modules — the "NumPy module was reloaded" warning.
if did_install:
    _ALIAS = {"pillow": "pil", "pyyaml": "yaml", "opencv_python_headless": "cv2",
              "open_clip_torch": "open_clip", "scikit_learn": "sklearn"}
    _roots = {r.name.lower().replace("-", "_") for r in missing} | {"cvp", "faiss"}
    _roots |= {_ALIAS[n] for n in _roots & set(_ALIAS)}
    _purged = [m for m in list(sys.modules)
               if m.split(".", 1)[0].lower().replace("-", "_") in _roots]
    for _m in _purged:
        sys.modules.pop(_m, None)
    if _purged:
        print(f"purged {len(_purged)} stale sys.modules entries")

    # Sanity (round-11): the HF stack must import cleanly in a FRESH
    # interpreter — a broken hub/accelerate pairing must surface HERE with an
    # actionable message, not 5 cells later as a cryptic circular import.
    _rc = _run([sys.executable, "-c", "import transformers, accelerate"])
    if _rc != 0:
        print("⚠ transformers/accelerate KHÔNG import được — thường do phiên cài "
              "này đã hạ cấp huggingface-hub dưới mức accelerate cần. Cách sửa "
              "sạch nhất: Runtime ▸ Disconnect and delete runtime, rồi Run all "
              "lại từ đầu (mọi tiến độ đã nằm trên Drive, không mất gì).")

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

# Colab secrets → env (optional: Gemini query enhancement/VQA, HF pushes).
# GOOGLE_API_KEY is the name Colab's built-in "Gemini API key ▸ Import from
# Google AI Studio" button creates — the engine accepts either spelling.
try:
    from google.colab import userdata
    for _sec in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "HF_TOKEN"):
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
# 🏗 Core Vision Perfect V1 — 01 · Build Artifacts (Colab)

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

NB01B_TITLE = r'''
# 🚀 Core Vision Perfect V1 — 01b · Caption BOOST (chạy SONG SONG với nb01)

Notebook phụ **chỉ chạy captions**, đi **NGƯỢC** danh sách video (L30 → L21)
trong khi phiên nb01 chính đi xuôi (L21 → L30) — hai phiên tự **gặp nhau ở
giữa** nhờ resume-skip theo từng video, chia đôi thời gian captions.

**An toàn:** mỗi video là MỘT file json ghi atomic trên Drive; phiên này KHÔNG
đụng embeddings / FAISS / BM25 / OCR / ASR. Tệ nhất hai phiên trùng nhau đúng
1 video ở điểm gặp — bên sau ghi đè bản y hệt, không thể phá artifact.

Cách dùng: mở **một phiên Colab GPU thứ hai** (A100/H100 đều được) → Run all.
Đứt phiên → Run all lại, tự resume. Khi CẢ HAI phiên xong captions: chạy nb01
một lượt cuối với `FORCE_TEXT_INDEX=True` để BM25 nạp đủ trường caption.
'''

NB01B_CAPTIONS = r'''
# ── 5 (01b) · Captions ONLY — đi NGƯỢC danh sách video ──
# Phiên nb01 chính caption L21→L30; phiên boost này L30→L21. Resume-skip theo
# từng video làm hai phiên hội tụ ở giữa mà không cần điều phối gì thêm.
# LƯU Ý: CAPTION_STRIDE ở ô PARAMS phải GIỐNG HỆT giá trị bên nb01.
from cvp.auxindex.captioner import caption_all_keyframes

vids = [str(v) for v in df.video_id.unique()][::-1]
print(f"captions-boost: {len(vids)} videos, đi ngược từ {vids[0]} về {vids[-1]}, "
      f"stride={CAPTION_STRIDE}")
with _log_stage("captions-boost"):
    n = caption_all_keyframes(settings, catalog, videos=vids, stride=CAPTION_STRIDE)
print(f"✅ captions-boost: {n} video mới trong phiên này")
print("Khi CẢ HAI phiên đều xong captions: chạy lại nb01 một lượt với "
      "FORCE_TEXT_INDEX=True để BM25 nạp đủ trường caption cho toàn bộ 873 video.")
'''

NB01C_TITLE = r'''
# 🎙 Core Vision Perfect V1 — 01c · ASR BOOST (phiên thứ 3, tùy chọn)

Notebook phụ **chỉ chạy ASR**, đi **NGƯỢC** danh sách video (L30 → L21) trong
khi phiên nb01 chính đi xuôi — hai phiên hội tụ ở giữa nhờ resume-skip theo
từng video, chia đôi thời gian ASR. Cùng khung an toàn đã kiểm chứng của 01b:
mỗi video một file json ghi atomic trên Drive; KHÔNG đụng captions /
embeddings / FAISS / OCR / BM25; bản mp4 hỏng trên Drive tự phục hồi từ
Videos_*.zip gốc.

⚠ CHỈ MỞ PHIÊN NÀY khi ngân sách Colab còn dư dả (panel Tài nguyên ▸ "Có
sẵn: X đơn vị" — nên còn ≥150): phiên A100 thứ 3 đốt ~6.8 đơn vị/giờ, và hết
đơn vị TRƯỚC buổi thi còn tệ hơn ASR chậm.
'''

NB01C_ASR = r'''
# ── 5 (01c) · ASR ONLY — đi NGƯỢC danh sách video ──
# Phiên nb01 chính ASR L21→; phiên này L30→ — hội tụ ở giữa nhờ resume-skip
# theo từng video. Điểm gặp nhau: cùng lắm 1 video được transcribe 2 lần,
# bên sau ghi đè bản tương đương (atomic) — không thể hỏng artifact.
from cvp.auxindex.asr import asr_all_videos

vids = [str(v) for v in df.video_id.unique()][::-1]
print(f"asr-boost: {len(vids)} videos, đi ngược từ {vids[0]} về {vids[-1]}")
with _log_stage("asr-boost"):
    n = asr_all_videos(settings, catalog, videos=vids)
print(f"✅ asr-boost: {n} video mới trong phiên này")
print("ASR xong toàn bộ thì phiên nb01 chính sẽ tự chuyển sang captions; "
      "phiên này có thể tắt để tiết kiệm đơn vị điện toán.")
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
    # Normalise _/space → '-' so 2026 spelling variants (clip_features,
    # Map Keyframes, keyframe singular, video_…) still route correctly.
    # SPECIFIC families are tested FIRST (review R3-C25): a 2026 rename like
    # "keyframe-map-b1.zip" must never fall into the generic keyframes bucket.
    z = zname.lower().replace("_", "-").replace(" ", "-")
    if "map" in z and "keyframe" in z:                        return ZIP_DEST["map-keyframes"]
    if "clip-feature" in z or "features-32" in z:             return ZIP_DEST["clip-features"]
    if "media-info" in z or "metadata" in z:                  return ZIP_DEST["media-info"]
    if "object" in z:                                         return ZIP_DEST["objects"]
    if z.startswith(("keyframes", "keyframe", "key-frames")): return ZIP_DEST["keyframes"]
    if z.startswith(("videos", "video-")):                    return ZIP_DEST["videos"]
    return None

import re as _re
import shutil as _sh
_VID_DIR_RE = _re.compile(r"^[A-Z]\d{2}_V\d{3}$")   # per-video PAYLOAD dirs, never wrappers

_unknown_zips = []
for zp in sorted(DATA_DIR.glob("*.zip")):
    dest = guess_dest(zp.name)
    if dest is None:
        _unknown_zips.append(zp.name); continue
    marker = dest / f".unzipped-{zp.stem}"
    if marker.exists():
        continue
    print("unzipping", zp.name, "→", dest)
    dest.mkdir(parents=True, exist_ok=True)
    tmp_root = dest.parent / "__tmp_unzip"
    if tmp_root.exists():
        _sh.rmtree(tmp_root)
    with zipfile.ZipFile(zp) as z:
        z.extractall(tmp_root)
    # Walk down through GENUINE wrapper folders only: a single child dir that
    # is NOT video-id-shaped. Handles nested shapes like Videos_L28_a/video/*
    # or Keyframes_L26/keyframes/L26_*/ (review R3-C28) while never mistaking
    # a lone per-video payload dir for a wrapper (review R3-C14).
    src = tmp_root
    while True:
        children = list(src.iterdir())
        if len(children) == 1 and children[0].is_dir() and not _VID_DIR_RE.match(children[0].name):
            src = children[0]
            continue
        break
    # MERGE into dest — never overwrite, never drop a second zip's files.
    kept_existing = 0
    for item in src.iterdir():
        target = dest / item.name
        if not target.exists():
            _sh.move(str(item), str(target))
        elif item.is_dir() and target.is_dir():
            for sub in item.iterdir():
                sub_target = target / sub.name
                if not sub_target.exists():
                    _sh.move(str(sub), str(sub_target))
                else:
                    kept_existing += 1
        else:
            kept_existing += 1
    _sh.rmtree(tmp_root, ignore_errors=True)
    if kept_existing:
        print(f"   giữ nguyên {kept_existing} mục đã tồn tại (không ghi đè)")
    marker.touch()
if _unknown_zips:
    print("\n" + "!" * 70)
    print("⚠ CÁC ZIP KHÔNG NHẬN DIỆN ĐƯỢC (KHÔNG được giải nén — kiểm tra tay!):")
    for _n in _unknown_zips:
        print("   •", _n)
    print("  Nếu đây là gói 2026 với tên mới → giải nén thủ công vào đúng thư mục")
    print("  data/{keyframes,map-keyframes,media-info,clip-features-32,objects,videos}")
    print("  (mapping: docs/DATASET_INGESTION.md §2) rồi chạy lại từ ô này.")
    print("!" * 70)
print("zip check done")
'''

NB1_LOCAL_COPY = r'''
# ── 6 · Materialize data → local disk TỪ ZIP GỐC (nhanh + miễn nhiễm FUSE) ──
# Round-14 (live-run 5): copytree 177k JPG lẻ qua Drive FUSE mất 3h+ rồi làm
# SẬP luôn cả mount ([Errno 107] Transport endpoint is not connected — mọi
# file sau đó đọc ra ENOENT). Chiến lược mới: copy CÁC FILE ZIP về local
# (ít file, to, đọc tuần tự — đúng kiểu I/O FUSE làm tốt) rồi giải nén tại
# chỗ — nhanh hơn nhiều lần, resume theo TỪNG zip, tự remount khi FUSE chết.
# Nội dung chỉ-có-trên-Drive (keyframes K-batch tự cắt, csv tái dựng…) được
# merge bù ở pha 2. Embedding/OCR/caption đọc 177k JPG từ local như cũ.
import re as _re
import shutil, time, zipfile
from pathlib import Path

# _ensure_drive/_drive_alive: bản HARDENED định nghĩa ở Ô 2 (round-19) —
# biết gỡ mount chết (fusermount -uz) + dọn mountpoint bẩn trước khi mount lại.
_ensure_drive()          # verify-R14: gate dưới stat qua FUSE — mount phải sống

def _kf_visible() -> bool:
    """DriveFS trên VM mới có thể thấy data/ nhưng CHƯA thấy subdir keyframes/
    (round-28, live nb03 run 5: gate này rơi nhầm sang nhánh Drive-direct rồi
    chết ở catalog). listdir cha = cú hích ép nạp metadata; zip Keyframes*
    cũng được chấp nhận — materialize vốn bung từ zip, không cần dir Drive."""
    try:
        list(DATA_DIR.iterdir())
        if (DATA_DIR / "keyframes").exists():
            return True
        _zn = [p.name.lower().replace("_", "-").replace(" ", "-")
               for p in DATA_DIR.glob("*.zip")]
        return any(n.startswith(("keyframes", "keyframe", "key-frames")) for n in _zn)
    except OSError:
        return False

_kf_ok = False
if COPY_KEYFRAMES_LOCAL:
    for _w in range(30):                       # tới 5 phút (round-62: VM "lười
        if _kf_visible():                      # metadata" từng cần hơn 2 phút)
            _kf_ok = True
            break
        print(f"⏳ DriveFS chưa thấy keyframes/ hay Keyframes*.zip — đợi ({_w * 10}s) ...")
        time.sleep(10)
    if not _kf_ok:
        print("⚠ 5 phút vẫn không thấy keyframes/ lẫn zip nguồn — máy ảo này dính "
              "DriveFS hỏng metadata. KHUYÊN MẠNH: Runtime ▸ Disconnect and "
              "delete runtime rồi Run all lại trên máy mới (dữ liệu Drive vẫn "
              "nguyên). Tạm thời rơi về đọc thẳng Drive (RẤT chậm).")
if COPY_KEYFRAMES_LOCAL and _kf_ok:
    LOCAL_DATA = Path("/content/data")
    LOCAL_DATA.mkdir(exist_ok=True)
    _ZCACHE = Path("/content/__zip_cache")
    _ZCACHE.mkdir(exist_ok=True)
    _VID_DIR_RE = _re.compile(r"^[A-Z]\d{2}_V\d{3}$")

    def _zip_family(zname: str):
        # CÙNG thứ tự ưu tiên với guess_dest ở ô 5 — một zip phải về đúng
        # MỘT family ở cả hai ô. Videos* trả None: video ở lại Drive (symlink).
        z = zname.lower().replace("_", "-").replace(" ", "-")
        if "map" in z and "keyframe" in z:                        return "map-keyframes"
        if "clip-feature" in z or "features-32" in z:             return "clip-features-32"
        if "media-info" in z or "metadata" in z:                  return "media-info"
        if "object" in z:                                         return "objects"
        if z.startswith(("keyframes", "keyframe", "key-frames")): return "keyframes"
        return None

    def _walk_wrapper(root: Path) -> Path:
        # bỏ các folder bọc ngoài thật sự (Keyframes_L26/keyframes/…) nhưng
        # không bao giờ nhầm một payload dir dạng L21_V001 đơn độc là wrapper
        src = root
        while True:
            ch = list(src.iterdir())
            if len(ch) == 1 and ch[0].is_dir() and not _VID_DIR_RE.match(ch[0].name):
                src = ch[0]
                continue
            return src

    def _merge_into(src: Path, dest: Path) -> int:
        """Move src/* vào dest — không ghi đè, đi sâu 1 cấp cho dir trùng."""
        kept = 0
        dest.mkdir(parents=True, exist_ok=True)
        for item in src.iterdir():
            target = dest / item.name
            if not target.exists():
                shutil.move(str(item), str(target))
            elif item.is_dir() and target.is_dir():
                for sub in item.iterdir():
                    st = target / sub.name
                    if not st.exists():
                        shutil.move(str(sub), str(st))
                    else:
                        kept += 1
            else:
                kept += 1
        return kept

    for _sub in ("keyframes", "map-keyframes", "media-info", "objects", "clip-features-32"):
        dst = LOCAL_DATA / _sub
        _stamp = LOCAL_DATA / f".materialized-{_sub}"
        if _stamp.exists():
            # verify-R14: stamp KHÔNG được che zip mới upload giữa session —
            # còn zip matching chưa có marker local thì phải bung bổ sung.
            _ensure_drive()
            _new = [z for z in sorted(DATA_DIR.glob("*.zip"))
                    if _zip_family(z.name) == _sub
                    and not (LOCAL_DATA / f".unzipped-{_sub}-{z.stem}").exists()]
            if not _new:
                print(f"{_sub}: đã materialize trong session này — skip")
                continue
            print(f"{_sub}: {len(_new)} zip mới sau lần materialize trước → bung bổ sung")
        if dst.exists():
            for stale in dst.glob("*.__tmp"):
                shutil.rmtree(stale, ignore_errors=True) if stale.is_dir() else stale.unlink()

        # PHA 1 — bung từ zip nguồn (marker LOCAL theo từng zip → resume mịn;
        # crash giữa merge không sao: lần sau bung lại, merge chỉ bù file thiếu)
        _ensure_drive()
        for zp in sorted(DATA_DIR.glob("*.zip")):
            if _zip_family(zp.name) != _sub:
                continue
            _done = LOCAL_DATA / f".unzipped-{_sub}-{zp.stem}"
            if _done.exists():
                continue
            t0 = time.time()
            lz = _ZCACHE / zp.name
            tmp_root = _ZCACHE / "__tmp_extract"
            for _attempt in (1, 2, 3):
                try:
                    # verify-R14: MỌI syscall chạm FUSE (stat, copyfile) phải
                    # nằm TRONG retry — zip trước mất nhiều phút extract thuần
                    # local, FUSE có thể chết trong cửa sổ đó.
                    _ensure_drive()
                    _free = shutil.disk_usage("/content").free
                    if _free < zp.stat().st_size * 2.2 + 5e9:
                        raise RuntimeError(          # không retry lỗi hết disk
                            f"Disk local sắp đầy ({_free / 1e9:.0f} GB) — không đủ "
                            f"chỗ bung {zp.name}. Runtime ▸ Disconnect and delete "
                            "runtime để lấy máy mới, hoặc đặt "
                            "COPY_KEYFRAMES_LOCAL=False (chậm hơn nhiều).")
                    shutil.copyfile(zp, lz)             # 1 file to, đọc tuần tự
                    if tmp_root.exists():
                        shutil.rmtree(tmp_root)
                    with zipfile.ZipFile(lz) as z:      # CRC check từng member
                        z.extractall(tmp_root)
                    break
                except (OSError, zipfile.BadZipFile) as e:
                    print(f"   ⚠ {zp.name}: {e!r} — thử lại ({_attempt}/3)")
                    if _attempt == 3:
                        raise
                    time.sleep(5)
            kept = _merge_into(_walk_wrapper(tmp_root), dst)
            shutil.rmtree(tmp_root, ignore_errors=True)
            lz.unlink(missing_ok=True)                  # trả disk ngay
            _done.touch()
            print(f"   {zp.name} → local {_sub}/ ({time.time() - t0:.0f}s"
                  + (f", giữ {kept} mục trùng)" if kept else ")"))

        # PHA 2 — merge phần CHỈ có trên Drive (K-batch tự cắt, upload tay…):
        # 1 lần listdir + exists-check local là rẻ; copy lẻ chỉ cho phần thiếu.
        added = 0
        srcD = DATA_DIR / _sub
        # verify-R14: family chỉ-có-folder (không zip nguồn) → pha 1 chưa hề
        # tạo dst; copy2 vào parent chưa tồn tại sẽ FileNotFoundError.
        dst.mkdir(parents=True, exist_ok=True)
        for _attempt in (1, 2, 3):
            try:
                _ensure_drive()                 # srcD.exists cũng chạm FUSE
                if srcD.exists():
                    for item in sorted(srcD.iterdir()):
                        if item.name.startswith(".unzipped-") or item.name.endswith(".__tmp"):
                            continue
                        target = dst / item.name
                        if target.exists():
                            continue
                        tmp_target = dst / (item.name + ".__tmp")
                        if tmp_target.is_dir():
                            shutil.rmtree(tmp_target)
                        elif tmp_target.exists():
                            tmp_target.unlink()
                        (shutil.copytree if item.is_dir() else shutil.copy2)(item, tmp_target)
                        tmp_target.rename(target)
                        added += 1
                break
            except OSError as e:
                print(f"   ⚠ merge Drive-extras {_sub}: {e!r} — thử lại ({_attempt}/3)")
                if _attempt == 3:
                    raise
                time.sleep(5)
        _stamp.touch()
        _n = sum(1 for _ in dst.iterdir())
        print(f"{_sub}: sẵn sàng local ({_n} mục"
              + (f", +{added} bù từ Drive" if added else "") + ")")
    # INTEGRITY + SELF-HEAL (round-11/12, live-run lessons): Google Drive FUSE
    # can serve freshly-written files back EMPTY (buffered writes lost when a
    # session dies mid-sync). Round-11 hit 873 header-less map csvs; round-12
    # hit empty clip-features .npy files that killed the provided_clip32 lane
    # AFTER 8h of GPU work. Validate every LOCAL small-file artifact and heal
    # broken ones straight FROM THE SOURCE ZIP (uploaded long ago = reliably
    # synced), repairing the Drive copy too.
    import zipfile as _zf
    import numpy as _np

    def _bad_csv(f):
        try:
            if f.stat().st_size < 40:
                return True
            with open(f, encoding="utf-8-sig") as fh:
                return sum(1 for _ in fh) < 2          # header only / empty
        except OSError:
            return True

    def _bad_npy(f):
        try:
            if f.stat().st_size < 90:                  # npy header alone is ~64B
                return True
            return _np.load(f, mmap_mode="r").shape[0] == 0
        except Exception:
            return True

    def _bad_empty(f):
        try:
            return f.stat().st_size == 0
        except OSError:
            return True

    # (subdir, glob, zip-name matcher, validator, key depth 1=basename 2=vid/name)
    _HEAL_SPECS = [
        ("map-keyframes", "*.csv",
         lambda z: "map" in z and "keyframe" in z, _bad_csv, 1),
        ("clip-features-32", "*.npy",
         lambda z: "clip-feature" in z or "features-32" in z, _bad_npy, 1),
        ("media-info", "*.json",
         lambda z: "media-info" in z or "metadata" in z, _bad_empty, 1),
        ("objects", "*/*.json",
         lambda z: "object" in z, _bad_empty, 2),
    ]
    for _sub, _pat, _match, _isbad, _depth in _HEAL_SPECS:
        _dirL = LOCAL_DATA / _sub
        if not _dirL.is_dir():
            continue
        _key = (lambda p: p.name) if _depth == 1 else (lambda p: f"{p.parent.name}/{p.name}")
        _bad = [f for f in sorted(_dirL.glob(_pat)) if _isbad(f)]
        if not _bad:
            print(f"{_sub} integrity: OK")
            continue
        print(f"⚠ {len(_bad)} file LOCAL rỗng/hỏng trong {_sub}/ (Drive FUSE mất "
              "dữ liệu?) — tự phục hồi từ zip gốc ...")
        # Zip handles opened ONCE per family (round-13): re-opening a Drive
        # zip per bad file would stall for hours on a family-scale corruption.
        _ensure_drive()                        # verify-R14: ZipFile đọc qua FUSE
        _members, _open_zips = {}, []
        for _z in DATA_DIR.glob("*.zip"):
            _zl = _z.name.lower().replace("_", "-")
            if _match(_zl):
                _zh = _zf.ZipFile(_z)
                _open_zips.append(_zh)
                for _n in _zh.namelist():
                    if not _n.endswith("/"):
                        _parts = Path(_n).parts
                        _members["/".join(_parts[-_depth:])] = (_zh, _n)
        _healed = 0
        for f in _bad:
            _srcz = _members.get(_key(f))
            if not _srcz:
                continue
            _data = _srcz[0].read(_srcz[1])
            if not _data:
                continue
            f.write_bytes(_data)                       # heal LOCAL
            _drv = DATA_DIR / _sub / _key(f)           # heal DRIVE too
            try:
                if not _drv.exists() or _isbad(_drv):
                    _tmpf = _drv.parent / (_drv.name + ".__tmp")
                    _tmpf.write_bytes(_data)
                    _tmpf.replace(_drv)
            except OSError:
                pass
            _healed += 1
        for _zh in _open_zips:
            _zh.close()
        print(f"   phục hồi {_healed}/{len(_bad)}")
        _still = [_key(f) for f in _bad if _isbad(f)]
        if _still:
            raise RuntimeError(
                f"{len(_still)} file trong {_sub}/ vẫn hỏng sau phục hồi "
                f"(vd {_still[:3]}) — kiểm tra zip nguồn còn trong data/ trên "
                "Drive (đừng xóa zip!) rồi chạy lại ô này.")
    # videos stay on Drive (huge); link them in
    _ensure_drive()
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
    # Round-11 (live-run): a manifest built while the map csvs read EMPTY says
    # 0 has_map forever (the corpus signature ignores map csvs) — detect the
    # poisoned state and rebuild once the csvs are healthy again.
    if len(df) and int(df.has_map.sum()) == 0:
        _map_dir = Path(str(settings.paths.data_root)) / "map-keyframes"
        _valid = (sum(1 for f in _map_dir.glob("*.csv") if f.stat().st_size > 40)
                  if _map_dir.is_dir() else 0)
        if _valid:
            print(f"manifest nói 0 map nhưng {_valid} csv hợp lệ đang có "
                  "→ FORCE rebuild catalog")
            df = catalog.build(force=True)
if len(df) and int(df.has_map.sum()) == 0:
    raise RuntimeError(
        "TOÀN BỘ catalog KHÔNG có map-keyframes — frame_idx nộp bài sẽ SAI. "
        "DỪNG tại đây thay vì tốn nhiều giờ embed vô ích. Kiểm tra thông báo "
        "phục hồi map csv ở ô 6, xác nhận data/map-keyframes trên Drive có "
        "csv thật (mở thử 1 file), rồi chạy lại ô này."
    )
print(f"catalog: {len(df):,} keyframes / {df.video_id.nunique()} videos "
      f"({int(df.has_map.sum()):,} frames with map-keyframes)")
_no_map = int((~df.has_map).sum())
if _no_map:
    print(f"⚠ {_no_map:,} keyframes KHÔNG có map-keyframes → frame_idx đang là "
          "ƯỚC LƯỢNG, nộp bài sẽ SAI. Upload gói map-keyframes của BTC, hoặc "
          "tái dựng từ video gốc: python scripts/05_rebuild_map_keyframes.py "
          "(cần videos/*.mp4), rồi chạy lại ô này với FORCE_CATALOG=True.")

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
    _ens = globals().get("_ensure_drive")   # ô 6 (round-14): FUSE chết giữa
    if _ens:                                # lane trước → remount trước lane sau
        _ens()
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
# ── 9 · Aux indexes: OCR / ASR / captions (each resumable, each GUARDED) ──
# AUX_CHANGED feeds cell 10. Round-13: MỖI stage được bọc riêng — một stage
# hỏng (model-load fail, API drift) KHÔNG giết các stage còn lại và không chặn
# cell 10-11; lỗi được nêu lại TO ở ô cuối sau khi mọi thứ khác hoàn tất.
AUX_CHANGED = False
_aux_errors = {}

def _did_work(n) -> bool:
    return n is None or n > 0   # None (older API) → assume something changed

def _aux_stage(stage, enabled, fn):
    global AUX_CHANGED
    if not enabled:
        return
    try:
        _ens = globals().get("_ensure_drive")   # ô 6 (round-14): FUSE chết
        if _ens:                                # ở stage trước → remount đã
            _ens()
        with _log_stage(stage):
            n = fn()
        AUX_CHANGED = AUX_CHANGED or _did_work(n)
        print(f"{stage}: processed {n} videos")
    except Exception as e:  # noqa: BLE001 — stage isolation, re-raised in cell 11
        _aux_errors[stage] = e
        import traceback
        traceback.print_exc()
        print(f"⚠ stage {stage} FAILED: {e!r} — TIẾP TỤC các stage còn lại; "
              "lỗi sẽ được nêu lại ở ô cuối.")

def _run_ocr():
    from cvp.auxindex.ocr import ocr_all_keyframes
    return ocr_all_keyframes(settings, catalog, overwrite=FORCE_AUX)

def _run_asr():
    from cvp.auxindex.asr import asr_all_videos
    return asr_all_videos(settings, catalog, overwrite=FORCE_AUX)

def _run_captions():
    from cvp.auxindex.captioner import caption_all_keyframes
    return caption_all_keyframes(settings, catalog, stride=CAPTION_STRIDE,
                                 overwrite=FORCE_AUX)

_aux_stage("ocr", RUN_OCR, _run_ocr)
_aux_stage("asr", RUN_ASR, _run_asr)
_aux_stage("captions", RUN_CAPTIONS, _run_captions)
print("aux indexes done | AUX_CHANGED =", AUX_CHANGED,
      "| stage lỗi:", list(_aux_errors) or "không")
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

# verify-R23 (HIGH): thiếu objects.parquet thì ObjectBooster đọc ~500-1500
# file json lẻ qua Drive FUSE MỖI QUERY lúc thi — gộp 177k json thành MỘT
# parquet ngay tại đây (đọc từ đĩa local, ~vài phút, chỉ chạy một lần).
from cvp.data.objects_compact import build_objects_index

_pq = settings.paths.art("objects_index") / "objects.parquet"
if _pq.exists():
    print("objects.parquet: đã có")
else:
    with _log_stage("objects_index"):
        print("objects.parquet built:", build_objects_index(settings, catalog))
'''

NB3_OBJECTS = r'''
# ── 5b · Compact objects index (một lần, nếu nb01 chưa build) ──
# verify-R23 (HIGH): thiếu objects.parquet thì ObjectBooster rơi về đọc từng
# file json qua Drive FUSE — cộng thêm HÀNG PHÚT mỗi query. Build từ đĩa
# local (ô 5 đã materialize) rồi ghi MỘT file parquet lên Drive artifacts.
from cvp.config import load_settings
from cvp.data.catalog import KeyframeCatalog
from cvp.data.objects_compact import build_objects_index

import time as _t
from pathlib import Path as _P

settings = load_settings()
_pq = settings.paths.art("objects_index") / "objects.parquet"
# round-28: stat lười trên VM mới từng nói parquet "không tồn tại" dù nó nằm
# sẵn trên Drive → suýt rebuild vô ích (và chết nếu data_root chưa local).
# Nudge-poll trước khi kết luận vắng mặt.
for _w in range(6):
    try:
        list(_P(str(settings.paths.artifacts_root)).iterdir())   # nudge metadata
    except OSError:
        pass
    if _pq.exists():
        break
    _t.sleep(5)
if _pq.exists():
    print("objects.parquet: đã có —", _pq)
else:
    catalog = KeyframeCatalog(settings)
    catalog.build()
    print("objects.parquet built:", build_objects_index(settings, catalog))
'''

NB1_DOCTOR = r'''
# ── 11 · Health report ──
import json
from cvp.pipeline.ingest import doctor
print(json.dumps(doctor(settings), indent=2, ensure_ascii=False))
# Round-13: aux-stage failures were isolated in cell 9 so the rest of the
# build could finish — but they must NOT pass silently. Surface them here,
# AFTER the doctor report, as the run's final verdict.
_errs = globals().get("_aux_errors") or {}
if _errs:
    raise RuntimeError(
        f"Build hoàn tất NHƯNG {len(_errs)} aux stage đã FAIL: {list(_errs)} — "
        "xem traceback ở ô 9. Các stage khác đã lưu; sửa nguyên nhân rồi "
        "Run all lại (resume tự skip phần xong)."
    )
print("\n✅ Artifacts build complete. Next: notebooks/02_train_vi_encoder_H100.ipynb")
'''


# ══════════════════════════════════════════════════════════════════════════
# Notebook 2 — H100 training
# ══════════════════════════════════════════════════════════════════════════

NB2_TITLE = r'''
# 🚀 Core Vision Perfect V1 — 02 · Train Vietnamese Encoder (H100 autopilot)

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
# torchao clash (live nb02 run 1): image Colab cài sẵn torchao 0.10 nhưng
# peft khi tiêm LoRA dò thấy torchao và TỪ CHỐI mọi bản < 0.16. Nâng cấp
# (torch KHÔNG bị đụng — torchao là add-on rời); không có wheel phù hợp thì
# gỡ hẳn: pipeline không dùng torchao, vắng mặt là peft bỏ qua luôn.
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _ta_ver

def _ta_old():
    try:
        return tuple(int(x) for x in _ta_ver("torchao").split(".")[:2]) < (0, 16)
    except PackageNotFoundError:
        return False
    except ValueError:
        return True

if _ta_old():
    _pip(["install", "-q", "-U", "torchao>=0.16"])
    if _ta_old():
        _pip(["uninstall", "-q", "-y", "torchao"])
        print("torchao cũ đã gỡ (peft không cần nó)")
    else:
        print("torchao →", _ta_ver("torchao"))

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
# stale when a public parquet (cell 7) OR a corpus caption (nb01 — grows when
# a new data batch lands) is newer than the merged train_data
cap_dir = settings.paths.art("captions")
_caps = sorted(cap_dir.glob("*.json")) if cap_dir.is_dir() else []
_stale_pub = (meta_p.exists() and _pub
              and max(p.stat().st_mtime for p in _pub) > meta_p.stat().st_mtime)
_stale_caps = (meta_p.exists() and _caps
               and max(p.stat().st_mtime for p in _caps) > meta_p.stat().st_mtime)
_stale = _stale_pub or _stale_caps
if meta_p.exists() and emb_p.exists() and not _stale:
    import pandas as pd
    meta = pd.read_parquet(meta_p)
    print(f"train_data exists: {len(meta):,} pairs "
          f"({(meta.split=='train').sum():,} train / {(meta.split=='val').sum():,} val) — skipping build")
else:
    if _stale_pub:
        print("public parquets newer than train_data → rebuilding to merge them")
    if _stale_caps:
        print("corpus captions newer than train_data (new batch?) → rebuilding to include them")
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

# Round-68: DATASET cũng là danh tính của run — kho captions dày 3.5× (26/08)
# sinh train_data mới; thiếu dấu vân tay này, phiên mới sẽ "resume" nhầm run
# cũ đã early-stop trên dữ liệu cũ và kết thúc sau 2 phút mà KHÔNG train gì.
_td = Path(cfg.train_data_dir)
DATA_FP = "|".join(
    f"{_f.name}:{_f.stat().st_size}"
    for _f in sorted(list(_td.glob("*.parquet")) + list(_td.glob("*.npy")))
    if _f.is_file())

def cfg_hash(c: TrainConfig) -> str:
    d = {k: v for k, v in sorted(dataclasses.asdict(c).items()) if k not in _RESUME_NEUTRAL}
    d["dataset_fingerprint"] = DATA_FP
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
    try:                              # verify-R23: state.json rách không được
        state = json.loads((ckpts[-1] / "state.json").read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):   # giết dashboard resume
        state = {}
        print("⚠ state.json của checkpoint mới nhất hỏng — trainer sẽ tự lùi "
              "về checkpoint cũ hơn")
    print(f"last ckpt    : {ckpts[-1]}")
    print(f"               global_step={state.get('step')}  best R@5={state.get('best_r5', -1):.4f}  "
          f"evals_since_best={state.get('evals_since_best')}")
else:
    print("last ckpt    : (none)")
_meta = Path(cfg.export_dir) / "export_meta.json"
if _meta.is_file():
    _m = (json.loads(_meta.read_text(encoding="utf-8")).get("metrics") or {})
    print(f"best export  : R@5={_m.get('R@5', float('nan')):.4f} → {cfg.export_dir}")

# Round-69 (audit): việc cất run cũ phải BẤT KHẢ THẤT BẠI — tên lưu trữ tự
# đánh số khi trùng (A/B qua lại không bị kẹt), pointer rách vẫn cất (danh
# tính không rõ = không được tin), thông báo chỉ in SAU khi đổi tên thật, và
# cất luôn export_dir: export_meta cũ đặt "xà ngang" R@5 đo trên val CŨ (dễ
# hơn 3.5×) — không cất thì run mới không bao giờ export nổi và early-stop
# sau vài phút mà không train được gì.
def _archive_unique(_src, _base):
    _dst, _n = Path(str(_base)), 1
    while _dst.exists():
        _n += 1
        _dst = Path(f"{_base}-{_n}")
    Path(_src).rename(_dst)
    return _dst

_has_old = POINTER.exists() or any(Path(cfg.run_dir).glob("step-*"))
# Round-71 (audit): guard 2-phiên áp cho MỌI trường hợp — kể cả CÙNG hash (tai
# nạn dễ nhất: mở nhầm nb02 trên máy thứ hai giữa chiến dịch); pointer được
# nhịp-tim làm mới mỗi 10 phút suốt lúc train (xem cuối cell) nên luôn "tươi".
if ptr and ptr.get("status") == "running":
    try:
        _age = (datetime.now(timezone.utc) - datetime.strptime(
            ptr["updated_utc"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        ).total_seconds()
    except Exception:
        _age = 1e9
    if _age < 1800:
        raise RuntimeError(
            "Pointer đang 'running' và mới cập nhật <30 phút — có vẻ một phiên "
            "nb02 KHÁC đang chạy. Nếu chắc chắn không phải (phiên trước vừa bị "
            "giết cứng), đợi ~30 phút rồi Run all lại. Chỉ MỘT phiên nb02 một lúc.")
if _has_old and (ptr is None or ptr.get("cfg_hash") != CFG_HASH):
    _old_tag = (ptr or {}).get("cfg_hash") or "unknown"
    _a1 = _archive_unique(cfg.run_dir,
                          Path(cfg.run_dir).parent / f"vi_siglip2-{_old_tag}")
    print(f"⚠ Danh tính run đổi ({_old_tag} → {CFG_HASH}) — đã cất run cũ → "
          f"{_a1.name}; train SẠCH từ đầu trên dataset hiện tại.")
    if Path(cfg.export_dir).exists():
        _a2 = _archive_unique(cfg.export_dir, f"{cfg.export_dir}-{_old_tag}")
        print(f"   export cũ (kèm xà ngang R@5 val cũ) cất → {_a2.name}")
elif ptr:
    print("✓ config hash khớp — auto-resume an toàn.")

write_pointer("running")

# Round-71: nhịp tim pointer — train nhiều giờ mà updated_utc đứng im thì guard
# 2-phiên bên trên vô dụng sau 30 phút. Thread nền làm mới mỗi 10 phút; cell
# TRAIN hạ cờ _HB_STOP trước khi ghi trạng thái cuối để không ghi đè nó.
import threading as _thm
import time as _tmm
_HB_STOP = False

def _pointer_heartbeat():
    while not globals().get("_HB_STOP"):
        _tmm.sleep(600)
        if globals().get("_HB_STOP"):
            break
        try:
            write_pointer("running")
        except Exception:
            pass

_thm.Thread(target=_pointer_heartbeat, daemon=True).start()
print(f"\nrun identity {CFG_HASH} | pointer → running (+nhịp tim 10 phút)")
'''

NB2_TRAIN = r'''
# ── 10 · TRAIN (autopilot: after any disconnect just Runtime → Run all) ──
from cvp.training.lit_trainer import LiTTrainer

trainer = LiTTrainer(settings, cfg)
try:
    final_metrics = trainer.train()
except BaseException as e:   # KeyboardInterrupt/SystemExit also mark crashed
    globals()["_HB_STOP"] = True             # round-71: tắt nhịp tim pointer
    write_pointer("crashed", error=repr(e)[:500])
    raise
globals()["_HB_STOP"] = True                 # round-71: tắt nhịp tim pointer
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
    # Round-68: giữ đường lui — bản đang dùng cất vào deliverables/previous
    # TRƯỚC khi ghi đè (bench chê tower mới thì hoán đổi lại là xong).
    # Round-69: xoay vòng phải BẤT BIẾN khi chạy lại — latest đã trùng bản
    # export thì đứng yên, kẻo lần Run all thứ hai xóa mất đường lui previous.
    prev = ARTIFACTS / "deliverables" / "previous"

    def _meta_bytes(_d):
        _f = Path(str(_d)) / "export_meta.json"
        return _f.read_bytes() if _f.is_file() else b"?"

    _same = (dest / "text_tower.safetensors").exists() and \
        _meta_bytes(dest) == _meta_bytes(export_dir) != b"?"
    if _same:
        print("latest đã trùng bản export hiện tại — không xoay vòng "
              "(previous giữ nguyên làm đường lui).")
    else:
        if (dest / "text_tower.safetensors").exists():
            if prev.exists():
                shutil.rmtree(prev)
            shutil.copytree(dest, prev)
            print("bản đang dùng đã cất →", prev)
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
# ✅ Core Vision Perfect V1 — 03 · Test the System & Trained Model

Sanity-checks the full retrieval stack on Colab: loads the engine, runs
Vietnamese queries end-to-end (KIS / TRAKE / AVS), measures latency, writes a
sample submission CSV, validates + packages it Codabench-style, optionally
scores it against a local ground truth (official formulas) and dry-runs the
automatic track. The Streamlit UI can be served via a tunnel at the end.

Prerequisite: notebook 01 (and optionally 02 for the fine-tuned tower).
'''

NB3_ARTIFACTS_LOCAL = r'''
# ── 5c · Artifacts đọc-nhiều → đĩa LOCAL (round-26, live nb03 run 2) ──
# Engine mmap embeddings/index từ Drive FUSE → query "lạnh" 114-130s, query
# "ấm" 1.75s. Copy các thư mục CHỈ-ĐỌC về local (~4-6GB, vài phút) rồi trỏ
# artifacts_root vào đó. Drive KHÔNG bị đụng — submissions được đồng bộ ngược
# về Drive ở ô đóng gói.
import os, shutil, time
from pathlib import Path

LOCAL_ART = Path("/content/artifacts")
LOCAL_ART.mkdir(exist_ok=True)
_READ_HOT = ("catalog", "embeddings", "indexes", "text_index", "objects_index",
             "asr", "checkpoints", "thumbs")   # round-42: webp thumbs → lưới UI 10x
_t0 = time.time()
try:
    list(ARTIFACTS.iterdir())    # round-28: nudge metadata trước loạt exists()
except OSError:
    _ensure_drive()
_missing = []
for _d in _READ_HOT:
    src, dst = ARTIFACTS / _d, LOCAL_ART / _d
    if dst.exists():
        print(f"   {_d}/: đã có local — skip")
        continue
    _ensure_drive()
    if not src.exists():
        _missing.append(_d)          # round-76: thiếu là phải LA LÊN, xem dưới
        continue
    _tmp = LOCAL_ART / (_d + ".__tmp")
    if _tmp.exists():
        shutil.rmtree(_tmp)
    shutil.copytree(src, _tmp)
    _tmp.rename(dst)
    print(f"   {_d}/ → local")
if _missing:
    # Round-76 (audit tiền-trận): trước đây thiếu thư mục nào là LẶNG LẼ bỏ
    # qua — text_index vắng mặt nghĩa là OCR/ASR/caption âm thầm = 0 suốt
    # trận. Metadata DriveFS lười là thủ phạm quen mặt; thuốc: đổi máy ảo.
    print(f"\n⚠⚠⚠ THIẾU {len(_missing)} kho artifacts trên Drive: {_missing}")
    print("    Máy ảo lười metadata? ĐỔI MÁY ẢO MỚI rồi Run all lại —")
    print("    KHÔNG ra trận khi thiếu bất kỳ kho nào ngoài 'thumbs'.")
(LOCAL_ART / "submissions").mkdir(parents=True, exist_ok=True)
os.environ["CVP_PATHS__ARTIFACTS_ROOT"] = str(LOCAL_ART)
# round-42: có kho thumbnail (chạy scripts/60_make_thumbs.py MỘT lần) → web
# đội tải ảnh ~8KB thay vì 60-150KB — lưới hiện gần như tức thì qua tunnel.
if (LOCAL_ART / "thumbs").is_dir():
    os.environ["CVP_WEB__THUMBS_DIR"] = str(LOCAL_ART / "thumbs")
    print("thumbs: BẬT (webp 320px)")
print(f"artifacts_root now: {LOCAL_ART} ({time.time() - _t0:.0f}s)")
'''

NB3_ENGINE = r'''
# ── 5 · Load the search engine ──
# Chọn model cho phiên test này:
#   "siglip2"    lane gốc zero-shot (dự phòng)
#   "finetuned"  text tower tiếng Việt từ nb02 (cùng index ảnh siglip2)
#   "ensemble"   round-47: finetuned + METACLIP2 60/40 — bench Lab 23/08:
#                0.5522 vs finetuned đơn 0.5370 (retrieval-thuần, 23/23 câu).
#                (Cặp cũ finetuned+openclip đã thua A/B 20/08 và bị thay.)
ENGINE_MODEL   = "ensemble"         # ← cấu hình ra trận đợt 2 (bench 23/08)
# "none"   = test offline, không gọi Gemini (nhanh, không tốn quota)
# "gemini" = dịch + mở rộng query (CẦN secret GEMINI_API_KEY; tự rơi về
#            Google-Translate miễn phí rồi passthrough nếu API lỗi — A/B 20/08:
#            riêng bản dịch EN đã nâng chất lượng rõ rệt cho mọi lane)
QUERY_PROVIDER = "gemini"           # ← cấu hình ra trận round 1
# Gemini NHÌN top-24 ảnh ứng viên và xếp lại đầu bảng (UIT CVPRW'25: +10%
# hit@1). Cần API trả phí; ~3–8s/query. Tắt (False) nếu cần UI phản hồi nhanh.
VLM_RERANK = True
# Khẩu pháo cuối: cross-encoder Qwen3-VL-Reranker-2B chạy LOCAL trên A100,
# chấm lại từng cặp (câu, ảnh) trong top-100 rồi trộn 50/50 với điểm fusion
# (recipe Unified-IMMR AIC-2025). Bổ trợ cho VLM rerank (pairwise ↔ listwise);
# mọi đường lỗi tự trả về thứ hạng cũ. ĐÃ ĐO ở vòng nháp 20/08: 7.2 → 7.6
# (VLM rerank trước đó: 6.4 → 7.2) — giữ True cho round 1.
CROSS_RERANK = True
# Round-37, học từ bài 19.8/23 (vòng nháp): đáp án chuẩn hay đứng rank 25–79
# trong bảng của ta — NGOÀI tầm nhìn top-24 của VLM rerank. Nới lên 48 để
# Gemini với tới (vẫn MỘT cuộc gọi, chỉ nhiều ảnh hơn, thêm ~2–4s/query).
VLM_RERANK_TOPK = 48
# Round-40 "suy nghĩ lâu hơn": gọi Gemini nhiều lần và biểu quyết. Bằng chứng
# trận 21/08: cùng cấu hình ra 9.4 rồi 9.0 (xúc xắc VLM); QA q3 lật '300 kg'
# ↔ '30 kg' giữa hai lần chạy. 3 phiếu đổi ~2× thời gian pack lấy độ ổn định.
VLM_VOTES = 3      # VLM rerank: trung bình 3 lượt chấm (1 = tắt)
QA_VOTES  = 3      # VQA: 3 lần trả lời, lấy đáp án đa số (1 = tắt)
import os, time
from pathlib import Path
os.environ["CVP_EMBEDDING__MODEL"] = ENGINE_MODEL
os.environ["CVP_QUERY__PROVIDER"]  = QUERY_PROVIDER
os.environ["CVP_SEARCH__VLM_RERANK"] = "true" if VLM_RERANK else "false"
os.environ["CVP_SEARCH__VLM_RERANK_TOPK"] = str(VLM_RERANK_TOPK)
os.environ["CVP_SEARCH__VLM_RERANK_VOTES"] = str(VLM_VOTES)
os.environ["CVP_VQA__SELF_CONSISTENCY"] = str(QA_VOTES)
os.environ["CVP_SEARCH__RERANKER"] = "qwen_reranker" if CROSS_RERANK else "none"
# Round-44: Lab (nb04) dò được bộ trọng số fusion thắng bench → tự nạp.
_tw = PROJECT / "artifacts" / "tuning" / "best_weights.json"
try:                                     # round-76: nudge metadata tuning/
    list(_tw.parent.iterdir())           # (nb04 có từ round-71, nb03 thì chưa)
except OSError:
    pass
if not _tw.exists():
    print("⚠⚠ KHÔNG thấy tuning/best_weights.json — trận sẽ chạy trọng số "
          "MẶC ĐỊNH, KHÁC cấu hình bench 0.6826! Metadata Drive lười? "
          "Chạy lại cell này; vẫn thiếu thì đổi máy ảo mới.")
if _tw.exists():
    import json as _json
    _w = _json.loads(_tw.read_text(encoding="utf-8")).get("best", {}).get("weights")
    if _w:
        # Round-49 SHRINKAGE 50% về baseline: tuner học trên vỏn vẹn 23 câu
        # của pack NHÁP và kéo OCR về ≈0.01 — đề đợt sau phân bố khác, tin
        # 100% cực trị đó là đánh bạc overfit. Trung bình với baseline giữ
        # nguyên CHIỀU HƯỚNG đã học (hạ OCR/ASR, nâng caption/metadata)
        # nhưng chỉ đi nửa biên độ — không tín hiệu nào bị giết hẳn.
        _base = {"visual": 1.0, "ocr": 0.35, "asr": 0.30, "caption": 0.25,
                 "metadata": 0.15, "object": 0.25}
        _w = {k: round(0.5 * float(_w.get(k, v)) + 0.5 * v, 4)
              for k, v in _base.items()}
        for _sig, _val in _w.items():
            os.environ[f"CVP_SEARCH__WEIGHTS__{_sig.upper()}"] = str(_val)
        print("⚖ Trọng số fusion TUNED+shrinkage 50% (từ Lab):", _w)
# Ranking phẳng (top không tách khỏi đám đông) → tự tìm lại bằng các biến thể
# Gemini đã cache rồi trộn RRF — không tốn thêm cuộc gọi API nào.
os.environ["CVP_SEARCH__LOW_CONFIDENCE_RETRY"] = "true"
# ── Round-75: gói knob AB vào TRẬN — thắng bench#3/#4 (0.6478 → 0.6739 →
# 0.6826; hiệu ứng nhất quán 2 lượt: q22-qa +0.8, TRAKE +0.2, đuôi KIS +0.4,
# đổi q24 −0.2). Gói B (3 knob vqa) không đo được lãi trên đề nháp nhưng là
# bảo hiểm định dạng cho đề thật, không phá gì (q22 giữ 0.8, budget đủ).
for _k, _v in {
    "CVP_SEARCH__NEIGHBOR_CONSISTENCY_BOOST": "0.15",
    "CVP_SEARCH__ROW_STRATEGY": "diversify_tail",
    "CVP_TEMPORAL__SUBMIT_STRATEGY": "jitter",
    "CVP_TEMPORAL__POOL_CONTEXT": "prepend",
    "CVP_TEMPORAL__EVENT_QUERY_VARIANTS": "all",
    "CVP_TEMPORAL__CAPTION_SIGNAL_WEIGHT": "0.2",
    "CVP_VQA__ANSWER_CANONICALIZE": "true",
    "CVP_VQA__ANSWER_NEIGHBOR_FRAMES": "1",
    "CVP_VQA__MAX_CALLS_PER_QUERY": "10",
}.items():
    os.environ[_k] = _v
print("🎛 Gói knob AB (round-75) đã vào trận: boost 0.15 + diversify_tail + "
      "4 knob TRAKE + vote canonical/neighbor")
if ENGINE_MODEL in ("finetuned", "ensemble"):
    os.environ["CVP_FINETUNED__CHECKPOINT"] = str(
        Path(os.environ["CVP_PATHS__ARTIFACTS_ROOT"]) / "checkpoints" / "vi_siglip2_best")
if ENGINE_MODEL == "ensemble":
    os.environ["CVP_EMBEDDING__ENSEMBLE_MEMBERS"] = '["finetuned", "metaclip2"]'
    os.environ["CVP_EMBEDDING__ENSEMBLE_WEIGHTS"] = "[0.6, 0.4]"

# Round-76 (audit tiền-trận): thiếu GEMINI key là hỏng ÂM THẦM và MUỘN —
# enhancement rơi về Google-Translate, VLM rerank tắt, QA rơi về Vintern;
# cấu hình 0.6826 cần Gemini. La lên NGAY tại đây thay vì giữa trận.
if QUERY_PROVIDER == "gemini" and not (os.environ.get("GEMINI_API_KEY")
                                       or os.environ.get("GOOGLE_API_KEY")):
    print("⚠⚠⚠ KHÔNG có GEMINI_API_KEY/GOOGLE_API_KEY trong env — thêm secret "
          "ở panel 🔑 (bật Notebook access), chạy lại cell secrets rồi cell "
          "này. Ra trận thiếu key = mất enhancement + VLM rerank + QA Pro!")
from cvp.search.engine import SearchEngine
from cvp.config import load_settings
settings = load_settings()
t0 = time.time()
engine = SearchEngine(settings)
# Round-76: chốt chặn 2-lane cho TRẬN — nb04 có từ round-71, nb03 thì chưa.
# Engine "degrade gracefully" khi một lane hỏng: đêm thi mà chạy ensemble
# thiếu lane là đánh cả đêm với nửa vũ khí, chỉ có một dòng warning chìm
# trong log. Fail TO TIẾNG tại đây; thuốc: đổi máy ảo MỚI rồi Run all lại.
if ENGINE_MODEL == "ensemble":
    assert getattr(engine, "member_names", None) == ["finetuned", "metaclip2"], (
        f"Ensemble thiếu lane: {getattr(engine, 'member_names', None)} — "
        "index/checkpoint chưa nạp đủ (máy ảo lười metadata?). Đổi máy ảo "
        "MỚI và Run all lại — TUYỆT ĐỐI không ra trận thiếu lane.")
    print("✓ ensemble đủ 2 lane:", engine.member_names)
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

nb03_files = [p]   # ONLY the CSVs this notebook run writes get packaged (cell 9)
if seqs:
    p2 = write_trake(settings.paths.art("submissions", "demo-trake.csv"),
                     [(c.video_id, c.frame_idxs) for c in seqs])
    nb03_files.append(p2)
    print(p2, "\n" + p2.read_text(encoding="utf-8")[:300])
'''

NB3_PACKAGE = r'''
# ── 9 · Validate + package (Codabench) ──
# The organiser contract is re-checked on the finished CSVs (a wrong row
# silently costs a submission slot); errors BLOCK the zip. Validate and zip
# ONLY the CSVs cell 8 just wrote (nb03_files) — artifacts/submissions is
# shared with the UI's real exports, and stale/demo files must never mix
# into a contest zip (packager docstring contract).
import json
from cvp.submission.packager import has_errors, package_codabench, validate_file

sub_dir = settings.paths.art("submissions")
issues = [i for f in nb03_files for i in validate_file(f, strict=True)]
for i in issues:
    print(" ", i)
if has_errors(issues):
    print("❌ Còn lỗi chặn — sửa CSV rồi chạy lại ô này (zip KHÔNG được tạo).")
else:
    zip_path = settings.paths.art("submissions", "submission.zip")
    package_codabench(sub_dir, zip_path, package_name=settings.submission.package_name,
                      files=nb03_files)
    manifest = json.loads((zip_path.parent / "MANIFEST.json").read_text(encoding="utf-8"))
    print(f"\n📦 {zip_path.name}  sha256={manifest['zip_sha256'][:12]}…")
    for f in manifest["files"]:
        print(f"   {f['name']:24} task={f['task']:5} rows={f['rows']}")
    # round-26: artifacts_root đang là LOCAL — đồng bộ submissions về Drive để
    # zip/csv sống sót sau khi phiên tắt (chỉ THÊM/GHI ĐÈ file cùng tên của
    # chính lượt chạy này, không xóa gì trên Drive).
    import shutil as _sh
    _drv_sub = ARTIFACTS / "submissions"
    if Path(str(sub_dir)).resolve() != _drv_sub.resolve():
        _ensure_drive()
        _sh.copytree(sub_dir, _drv_sub, dirs_exist_ok=True)
        print("submissions đồng bộ về Drive:", _drv_sub)
'''

NB3_RUN_PACK = r'''
# ── 9b · 🏁 THI ĐẤU THẬT: chạy CẢ PACK đề của BTC → CSV → validate → zip ──
# Chuẩn bị: bung zip đề của BTC, upload các file query-*.txt vào
# MyDrive/<project>/queries/<QUERY_PACK>/  (file .txt nằm TRỰC TIẾP trong
# thư mục — đừng để lồng thêm một thư mục con sau khi bung zip).
QUERY_PACK = "p1"      # tên thư mục con trong queries/
RUN_PACK   = False     # bật True khi đề đã nằm đúng chỗ
REZIP_ONLY = False     # True: KHÔNG search lại — chỉ validate + zip lại các
                       # query-*.csv hiện có trong pack (dùng SAU khi soát tay
                       # bằng UI và ghi đè vài file CSV bằng bản người chọn)
RESUME_PACK = True     # round-77: VM chết giữa pack → chạy lại cell này sẽ GIỮ
                       # các query-*.csv đã xong và chỉ chạy phần thiếu (đêm
                       # 28/08 mất trắng 27 câu vì thiếu cái này)
PACK_DEADLINE_MIN = 0  # round-77: >0 = quá số phút này thì các câu còn lại
                       # chạy chế độ NƯỚC RÚT (QA votes 1, tắt VLM rerank);
                       # đặt ~60-70% thời gian còn lại tới giờ đóng cổng nộp
_qdir = PROJECT / "queries" / QUERY_PACK
if RUN_PACK and not (_qdir.is_dir() and any(_qdir.glob("*.txt"))):
    # Round-34: đêm thi Run all chạy TRƯỚC giờ BTC phát đề — cell này crash
    # là đứt Run all và cell UI phía dưới không bao giờ mở. Báo rồi đi tiếp.
    print(f"⚠ Chưa thấy file query-*.txt trong {_qdir} — upload đề vào đó rồi "
          "chạy lại RIÊNG cell này. (Run all vẫn đi tiếp, engine + UI không bị chặn.)")
elif RUN_PACK:
    import shutil as _sh

    _out = settings.paths.art("submissions", QUERY_PACK)
    if REZIP_ONLY:
        from cvp.submission.packager import has_errors, package_codabench, validate_file

        class rep:  # noqa: N801 — cùng hình dạng với AutoRunReport phía dưới
            written = sorted(_out.glob("query-*.csv"))
            failed: dict = {}
            issues = [i for p in written for i in validate_file(p, strict=True)]
            zip_path = None
        if rep.written and not has_errors(rep.issues):
            _zp = _out / f"{settings.submission.package_name}.zip"
            if not has_errors(package_codabench(
                    _out, _zp, package_name=settings.submission.package_name,
                    files=rep.written)):
                rep.zip_path = _zp
    else:
        from cvp.pipeline.auto_agent import run_auto

        # engine_factory tái dùng engine ô 5 (đang nóng, đúng cấu hình ra
        # trận) — để mặc định sẽ build engine THỨ HAI và nhân đôi RAM/VRAM.
        # gán VÔ ĐIỀU KIỆN — settings sống dai trong kernel, đặt 60 rồi hạ
        # về 0 mà chỉ gán-khi-truthy thì governor cũ vẫn âm thầm còn vũ trang
        settings.submission.pack_deadline_min = float(PACK_DEADLINE_MIN)
        rep = run_auto(_qdir, _out, settings, submit=False, resume=RESUME_PACK,
                       engine_factory=lambda _s: engine)
    print(f"\nCSV viết được: {len(rep.written)}")
    if rep.failed:
        print(f"⚠ {len(rep.failed)} query KHÔNG ra CSV (sẽ 0 điểm): {sorted(rep.failed)}")
    for _i in rep.issues:
        print("  ", _i)
    if rep.zip_path:
        _drv = PROJECT / "artifacts" / "submissions" / QUERY_PACK
        _drv.mkdir(parents=True, exist_ok=True)
        _sh.copytree(_out, _drv, dirs_exist_ok=True)
        print(f"\n📦 {rep.zip_path.name} đã đồng bộ về Drive: {_drv}")
        print("→ Tải submission.zip từ Drive về máy, nộp ở tab 'Nộp bài' của BTC.")
    else:
        print("⚠ KHÔNG có zip — sửa lỗi validate ở trên rồi chạy lại cell này "
              "(đừng nộp tay CSV lẻ).")
else:
    print("RUN_PACK=False — upload đề vào queries/<PACK>/ rồi bật True và chạy lại.")
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
    print("Hỗ trợ range / center+epsilon / moments / answers / targets (AVS coverage) "
          "— xem cvp/eval/official.py")
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
    # Reuse the cell-5 engine (review C6): the default engine_factory would
    # build a SECOND full engine and double index+catalog memory this session.
    rep = run_auto(qdir, settings.paths.art("submissions", "auto_dry_run"),
                   settings, submit=False, engine_factory=lambda _s: engine)
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
SHARE_URL = True    # round-32: link tạm cho ĐỒNG ĐỘI cùng vào UI (cửa sổ
                    # proxy của Colab CHỈ chủ phiên xem được). Link chỉ mở
                    # giao diện tìm kiếm — KHÔNG lộ source/Drive/notebook.
                    # Chỉ gửi link trong nhóm kín; tắt phiên là link chết.
if LAUNCH_UI:
    import os, re, subprocess, time
    # round-30 ĐẢO NGƯỢC verify-R23: ô engine giờ đặt sẵn CẤU HÌNH RA TRẬN
    # (finetuned + gemini + artifacts local) — UI PHẢI thừa hưởng env này;
    # pop như trước sẽ âm thầm hạ UI về siglip2 zero-shot đọc Drive.
    _env = dict(os.environ)
    proc = subprocess.Popen(
        ["streamlit", "run", str(REPO_DIR / "app" / "streamlit_app.py"),
         "--server.port", "8501", "--server.headless", "true"], env=_env)
    time.sleep(8)
    if SHARE_URL:
        # cloudflared quick tunnel: không cần tài khoản, URL ngẫu nhiên dài
        # khó đoán, tự chết khi phiên tắt.
        _cf = "/content/cloudflared"
        if not os.path.exists(_cf):
            subprocess.run(
                ["wget", "-q", "-O", _cf,
                 "https://github.com/cloudflare/cloudflared/releases/"
                 "latest/download/cloudflared-linux-amd64"], check=True)
            os.chmod(_cf, 0o755)
        _tun = subprocess.Popen([_cf, "tunnel", "--url", "http://localhost:8501"],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True)
        _url, _t0 = None, time.time()
        while _url is None and time.time() - _t0 < 90:
            _m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com",
                           _tun.stdout.readline() or "")
            if _m:
                _url = _m.group(0)
        if _url:
            print("🔗 LINK CHO ĐỒNG ĐỘI (gửi trong nhóm kín):", _url)
        else:
            print("⚠ Không lấy được URL tunnel sau 90s — chạy lại cell, "
                  "hoặc dùng cửa sổ proxy bên dưới (chỉ mình bạn xem được).")
    from google.colab import output
    output.serve_kernel_port_as_window(8501)   # cửa sổ riêng của CHỦ PHIÊN
    # proc.terminate() when done
else:
    print("Set LAUNCH_UI=True to serve the app (better: run it on your laptop).")
'''


NB3_FASTUI = r'''
# ── 12b · ⚡ Web soát tay NHANH (tùy chọn 2 — SPA + API, không lag rerun) ──
# Chạy NGAY TRONG kernel này, tái dùng engine ô 5 (không tốn thêm VRAM), phát
# qua tunnel thứ hai. Cả đội đăng nhập MỘT tài khoản chung — ai không có
# mật khẩu thì link cũng vô dụng. Streamlit (ô 12) vẫn là phương án dự phòng.
LAUNCH_FAST_UI = True
TEAM_USER = "aic2026-222"
TEAM_PASS = ""            # để trống = tự sinh mật khẩu ngẫu nhiên và in ra
if LAUNCH_FAST_UI:
    import os, re, secrets, subprocess, sys, threading, time
    try:
        import fastapi, uvicorn  # noqa: F401
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                        "fastapi", "uvicorn"], check=True)
        import uvicorn  # noqa: F401
    import uvicorn

    from cvp.web.server import create_team_app

    if not TEAM_PASS:
        TEAM_PASS = secrets.token_urlsafe(6)
    _webapp = create_team_app(engine, settings, TEAM_USER, TEAM_PASS)
    threading.Thread(
        target=lambda: uvicorn.run(_webapp, host="0.0.0.0", port=8600,
                                   log_level="warning"),
        daemon=True, name="cvp-fast-ui").start()
    time.sleep(3)
    _cf = "/content/cloudflared"
    if not os.path.exists(_cf):
        subprocess.run(["wget", "-q", "-O", _cf,
                        "https://github.com/cloudflare/cloudflared/releases/"
                        "latest/download/cloudflared-linux-amd64"], check=True)
        os.chmod(_cf, 0o755)
    _tun2 = subprocess.Popen([_cf, "tunnel", "--url", "http://localhost:8600"],
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True)
    _u2, _t2 = None, time.time()
    while _u2 is None and time.time() - _t2 < 90:
        _m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com",
                       _tun2.stdout.readline() or "")
        if _m:
            _u2 = _m.group(0)
    if _u2:
        print("⚡ LINK WEB NHANH CHO ĐỒNG ĐỘI:", _u2)
        print(f"   Đăng nhập MỘT tài khoản chung: {TEAM_USER} / {TEAM_PASS}")
        print("   (gửi cả link lẫn mật khẩu trong nhóm kín)")
    else:
        print("⚠ Không lấy được URL tunnel web nhanh — chạy lại cell, "
              "hoặc cả đội dùng link Streamlit ô 12.")
else:
    print("LAUNCH_FAST_UI=False — đội chỉ dùng Streamlit (ô 12).")
'''

NB3_KEEPALIVE = r'''
# ── 13 · 🫀 Giữ phiên + trông UI (chạy MÃI — bấm nút Dừng ⏹ của ô để thoát) ──
# Colab thu hồi máy ảo "không hoạt động". Kernel đang bận chạy ô này = hoạt
# động, nên phiên sống tới khi bạn tự dừng (trần cứng 24h của Colab vẫn áp
# dụng). Kiêm watchdog: streamlit / tunnel chết là tự dựng lại và in link MỚI
# (round-35 — phiên nháp 21/08 tự ngắt giữa buổi soát tay, đứt UI cả đội).
KEEP_ALIVE = True
if KEEP_ALIVE:
    import os as _os, re as _re, subprocess as _sp, time as _tm

    def _alive(p):
        return p is not None and p.poll() is None

    _n = 0
    print("🫀 Watchdog chạy — phiên được giữ sống. Bấm nút Dừng (⏹) của ô này "
          "khi muốn kết thúc.")
    try:
        while True:
            if globals().get("LAUNCH_UI"):
                if not _alive(globals().get("proc")):
                    print(f"⚠ {_tm.strftime('%H:%M')} streamlit chết — dựng lại ...")
                    proc = _sp.Popen(
                        ["streamlit", "run", str(REPO_DIR / "app" / "streamlit_app.py"),
                         "--server.port", "8501", "--server.headless", "true"],
                        env=dict(_os.environ))
                    _tm.sleep(8)
                if (globals().get("SHARE_URL") and globals().get("_cf")
                        and not _alive(globals().get("_tun"))):
                    print(f"⚠ {_tm.strftime('%H:%M')} tunnel Streamlit chết — mở lại ...")
                    _tun = _sp.Popen([_cf, "tunnel", "--url", "http://localhost:8501"],
                                     stdout=_sp.PIPE, stderr=_sp.STDOUT, text=True)
                    _u, _t1 = None, _tm.time()
                    while _u is None and _tm.time() - _t1 < 90:
                        _m2 = _re.search(r"https://[a-z0-9-]+\.trycloudflare\.com",
                                         _tun.stdout.readline() or "")
                        if _m2:
                            _u = _m2.group(0)
                    print("🔗 LINK MỚI CHO ĐỒNG ĐỘI:", _u or "⚠ không lấy được — chạy lại ô UI")
            if (globals().get("LAUNCH_FAST_UI") and globals().get("_cf")
                    and globals().get("_tun2") is not None
                    and not _alive(globals().get("_tun2"))):
                # round-36: web nhanh (ô 12b) cũng được watchdog trông hộ
                print(f"⚠ {_tm.strftime('%H:%M')} tunnel web nhanh chết — mở lại ...")
                _tun2 = _sp.Popen([_cf, "tunnel", "--url", "http://localhost:8600"],
                                  stdout=_sp.PIPE, stderr=_sp.STDOUT, text=True)
                _u2, _t2 = None, _tm.time()
                while _u2 is None and _tm.time() - _t2 < 90:
                    _m3 = _re.search(r"https://[a-z0-9-]+\.trycloudflare\.com",
                                     _tun2.stdout.readline() or "")
                    if _m3:
                        _u2 = _m3.group(0)
                print("⚡ LINK WEB NHANH MỚI:", _u2 or "⚠ không lấy được — chạy lại ô 12b")
            _tm.sleep(30)
            _n += 1
            if _n % 10 == 0:   # ~5 phút một nhịp, giữ output gọn
                print(f"🫀 {_tm.strftime('%H:%M')} phiên sống"
                      + (" · UI ok" if _alive(globals().get("proc")) else ""))
    except KeyboardInterrupt:
        print("⏹ Đã dừng watchdog — từ giờ Colab tính phiên là nhàn rỗi.")
else:
    print("KEEP_ALIVE=False — phiên sẽ tự ngắt khi nhàn rỗi.")
'''


NB4_TITLE = r'''
# 🔬 Core Vision Perfect V1 — 04 · Lab: nâng chất artifacts CÓ ĐO LƯỜNG

Phòng thí nghiệm giữa các vòng đấu (round-44). Mọi nâng cấp được CHẤM ĐIỂM
trên bench GT (dựng từ bài tham chiếu 19.8/23 của vòng nháp) TRƯỚC khi nhận
vào cấu hình ra trận. Các công tắc, bật từng cái tùy phiên:

| Knob | Việc | Thời gian |
|---|---|---|
| `RUN_GT` | Dựng gt.json từ zip tham chiếu | vài giây |
| `RUN_BENCH_FULL` | Chấm dàn vũ khí ĐẦY ĐỦ hiện tại trên đề nháp | ~45–60 phút |
| `RUN_TUNE` | Dump tín hiệu + dò trọng số fusion → best_weights.json | ~30 phút |
| `RUN_METACLIP` | Embed lane MetaCLIP-2 + A/B 3 đội hình retrieval | ~2 giờ |

Chuẩn bị MỘT lần: upload zip bài tham chiếu 19.8 lên Drive thành
`MyDrive/AIC2025/queries/thunghiem-ref.zip` (đề nháp đã nằm sẵn ở
`queries/p1/` từ vòng thử nghiệm).
'''

LAB_GT = r'''
# ── L1 · Bench GT từ bài tham chiếu 19.8 ──
RUN_GT = True
import subprocess, sys
def _run(*args):
    # Round-47: stream con-process output VÀO CELL — subprocess.run kế thừa
    # fd thật của kernel nên log 10 giờ ASR từng "im lặng" trong cell (nó chảy
    # vào runtime log, không phải notebook).
    print("$", " ".join(map(str, args)), flush=True)
    p = subprocess.Popen([sys.executable, "-u", *map(str, args)],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True)
    for _ln in p.stdout:
        print(_ln, end="", flush=True)
    if p.wait() != 0:
        raise RuntimeError(f"lệnh lỗi (exit {p.returncode})")
REF_ZIP   = PROJECT / "queries" / "thunghiem-ref.zip"
TRIAL_DIR = PROJECT / "queries" / "p1"          # 24 đề vòng nháp (đã upload từ trước)
GT_PATH   = PROJECT / "queries" / "gt-thunghiem.json"
if RUN_GT:
    if not REF_ZIP.exists():
        print(f"⚠ Chưa thấy {REF_ZIP} — upload 'submission (3).zip' (bài 19.8) "
              "lên Drive với tên đó rồi chạy lại cell này. Các cell sau vẫn chạy "
              "được nếu GT đã dựng từ trước.")
    else:
        _run(REPO_DIR / "scripts" / "62_build_gt_from_reference.py",
             "--reference", REF_ZIP, "--queries", TRIAL_DIR, "--out", GT_PATH)
print("GT:", "sẵn sàng ✓" if GT_PATH.exists() else "CHƯA có")
'''

LAB_BENCH_FULL = r'''
# ── L2 · Chấm dàn vũ khí ĐẦY ĐỦ trên đề nháp (baseline mới) ──
# Đúng cấu hình ra trận nb03: finetuned + gemini + VLM48×3 + Qwen-8B + Pro-QA.
RUN_BENCH_FULL = True
# ── Round-74: gói knob A/B cho chiến dịch bench (dựa trên chẩn đoán 66/67).
# "off" = baseline tiền-r75 · "A" = ranking/hàng, KHÔNG đổi API (boost
# cao-nguyên + đa dạng hóa đuôi 100 dòng + 4 knob TRAKE round-72) · "AB" =
# A + gói QA (canonicalize vote + hỏi thêm strip lân cận, ×2 call Gemini
# mỗi nhóm QA). Round-75: gói AB thắng bench#3/#4 và đã vào nb03 ra trận →
# mặc định bench cũng là "AB" cho đúng luật bench-là-bản-sao-trận.
BENCH_PACK = "AB"
import os, time
from pathlib import Path
_PACK_A = {
    "CVP_SEARCH__NEIGHBOR_CONSISTENCY_BOOST": "0.15",
    "CVP_SEARCH__ROW_STRATEGY": "diversify_tail",
    "CVP_TEMPORAL__SUBMIT_STRATEGY": "jitter",
    "CVP_TEMPORAL__POOL_CONTEXT": "prepend",
    "CVP_TEMPORAL__EVENT_QUERY_VARIANTS": "all",
    "CVP_TEMPORAL__CAPTION_SIGNAL_WEIGHT": "0.2",
}
_PACK_B = {
    "CVP_VQA__ANSWER_CANONICALIZE": "true",
    "CVP_VQA__ANSWER_NEIGHBOR_FRAMES": "1",
    "CVP_VQA__MAX_CALLS_PER_QUERY": "10",
}
assert BENCH_PACK in ("off", "A", "AB"), f"BENCH_PACK lạ: {BENCH_PACK!r}"
_knobs = {} if BENCH_PACK == "off" else (
    _PACK_A if BENCH_PACK == "A" else {**_PACK_A, **_PACK_B})
for _k in {**_PACK_A, **_PACK_B}:        # dọn sạch trước — cell chạy lại không
    os.environ.pop(_k, None)             # được thừa kế knob của lượt trước
for _k, _v in _knobs.items():
    os.environ[_k] = _v
print(f"🎛 BENCH_PACK = {BENCH_PACK}"
      + (f" — {len(_knobs)} knob BẬT: {sorted(_knobs)}" if _knobs
         else " — knob TẮT hết, cấu hình y hệt bench#2 (baseline)"))
if RUN_BENCH_FULL and GT_PATH.exists():
    # Round-70: bench phải là BẢN SAO Y đội hình ra trận nb03 (round-49) —
    # ensemble finetuned+metaclip2 60/40 + trọng số tuned-shrinkage. Bản cũ
    # đo lane finetuned ĐƠN với trọng số mặc định = không phải đội hình thật.
    os.environ["CVP_EMBEDDING__MODEL"] = "ensemble"
    os.environ["CVP_EMBEDDING__ENSEMBLE_MEMBERS"] = '["finetuned", "metaclip2"]'
    os.environ["CVP_EMBEDDING__ENSEMBLE_WEIGHTS"] = "[0.6, 0.4]"
    os.environ["CVP_QUERY__PROVIDER"]  = "gemini"
    os.environ["CVP_FINETUNED__CHECKPOINT"] = str(
        Path(os.environ["CVP_PATHS__ARTIFACTS_ROOT"]) / "checkpoints" / "vi_siglip2_best")
    _tw = PROJECT / "artifacts" / "tuning" / "best_weights.json"
    try:                                     # round-71: nudge metadata tuning/
        list(_tw.parent.iterdir())
    except OSError:
        pass
    if not _tw.exists():
        print("⚠ KHÔNG thấy tuning/best_weights.json — bench sẽ chạy trọng số "
              "MẶC ĐỊNH (khác trận). Nếu file có trên Drive: đổi máy ảo chạy lại.")
    if _tw.exists():
        import json as _wj
        _w = _wj.loads(_tw.read_text(encoding="utf-8")).get("best", {}).get("weights")
        if _w:
            _base = {"visual": 1.0, "ocr": 0.35, "asr": 0.30, "caption": 0.25,
                     "metadata": 0.15, "object": 0.25}
            _w = {k: round(0.5 * float(_w.get(k, v)) + 0.5 * v, 4)
                  for k, v in _base.items()}
            for _sig, _val in _w.items():
                os.environ[f"CVP_SEARCH__WEIGHTS__{_sig.upper()}"] = str(_val)
            print("⚖ Bench dùng trọng số TUNED+shrinkage 50% (y hệt trận):", _w)
    os.environ["CVP_SEARCH__VLM_RERANK"] = "true"
    os.environ["CVP_SEARCH__VLM_RERANK_TOPK"] = "48"
    os.environ["CVP_SEARCH__VLM_RERANK_VOTES"] = "3"
    os.environ["CVP_VQA__SELF_CONSISTENCY"] = "3"
    os.environ["CVP_SEARCH__RERANKER"] = "qwen_reranker"
    os.environ["CVP_SEARCH__LOW_CONFIDENCE_RETRY"] = "true"
    from cvp.config import load_settings
    from cvp.eval.official import score_run
    from cvp.pipeline.auto_agent import run_auto
    from cvp.search.engine import SearchEngine
    settings = load_settings()
    # Round-71 (audit): engine "degrade gracefully" khi một lane hỏng — bench
    # nhiều giờ mà chạy thiếu lane là đo SAI đội hình. Bắt đủ 2 lane TRƯỚC,
    # rồi trao đúng engine đã kiểm cho run_auto (không nạp model 2 lần).
    engine = SearchEngine(settings)
    assert engine.member_names == ["finetuned", "metaclip2"], (
        f"Ensemble thiếu lane: {engine.member_names} — index/checkpoint chưa nạp "
        "đủ (máy ảo lười metadata?). Đổi máy mới chạy lại, đừng bench thiếu lane.")
    print("✓ ensemble đủ 2 lane:", engine.member_names)
    _t0 = time.time()
    rep = run_auto(TRIAL_DIR, settings.paths.art("submissions", "lab_full"),
                   settings, submit=False, engine_factory=lambda _s: engine)
    r = score_run(settings.paths.art("submissions", "lab_full"), GT_PATH)
    import json as _json
    import shutil as _shd
    _drv_lab = PROJECT / "artifacts" / "lab"
    _drv_lab.mkdir(parents=True, exist_ok=True)
    # Round-71: giữ hồ sơ TRƯỚC/SAU — bench cũ xoay sang -prev thay vì bị đè
    # (chiến dịch cần so bench-1 với bench-2 sau khi retrain tower).
    if (_drv_lab / "bench_full.json").exists():
        _shd.copy2(_drv_lab / "bench_full.json", _drv_lab / "bench_full-prev.json")
        if (_drv_lab / "lab_full").exists():
            _shd.rmtree(_drv_lab / "lab_full-prev", ignore_errors=True)
            _shd.copytree(_drv_lab / "lab_full", _drv_lab / "lab_full-prev")
        print("bench cũ đã xoay → bench_full-prev.json + lab_full-prev/")
    # Round-74: lưu kèm gói knob để mỗi bản bench tự khai nó đo cấu hình nào.
    (_drv_lab / "bench_full.json").write_text(
        _json.dumps({**r.to_dict(), "bench_pack": BENCH_PACK,
                     "bench_knobs": _knobs},
                    ensure_ascii=False, indent=2), encoding="utf-8")
    _shd.copytree(settings.paths.art("submissions", "lab_full"),
                  _drv_lab / "lab_full", dirs_exist_ok=True)
    print(f"\n⭐ BENCH FULL [pack={BENCH_PACK}]: mean_final={r.mean_final:.4f} "
          f"({r.num_scored}/{r.num_gt} câu, {(time.time()-_t0)/60:.0f} phút) "
          f"— đã lưu Drive: {_drv_lab}")
    print("   Mốc cũ (bản 8.4 đợt nháp): 0.6413 · bản trộn 2 lần: 0.6587")
    for stem, qs in sorted(r.per_query.items()):
        print(f"   {stem:26s} final={qs.final:.3f}")
elif RUN_BENCH_FULL:
    print("⚠ Chưa có GT — chạy cell L1 trước.")
'''

LAB_TUNE = r'''
# ── L3 · Dò trọng số fusion trên bench (dump 1 lần, thử hàng nghìn bộ) ──
RUN_TUNE = True
if RUN_TUNE and GT_PATH.exists():
    import os
    # Round-70: dò trọng số trên đúng lane retrieval ra trận (ensemble 60/40)
    # — tín hiệu visual của lane đơn khác lane trộn, trọng số học ra sẽ lệch.
    os.environ["CVP_EMBEDDING__MODEL"] = "ensemble"
    os.environ["CVP_EMBEDDING__ENSEMBLE_MEMBERS"] = '["finetuned", "metaclip2"]'
    os.environ["CVP_EMBEDDING__ENSEMBLE_WEIGHTS"] = "[0.6, 0.4]"
    os.environ["CVP_FINETUNED__CHECKPOINT"] = str(
        Path(os.environ["CVP_PATHS__ARTIFACTS_ROOT"]) / "checkpoints" / "vi_siglip2_best")
    # Round-71 (audit): dump tín hiệu KHÔNG cần reranker — tắt Qwen-8B + VLM
    # 48×3 thừa kế env từ cell bench, kẻo dump chậm gấp chục lần và đốt oan
    # quota Gemini (dump chụp tín hiệu TRƯỚC bước rerank, kết quả rerank bị bỏ).
    os.environ["CVP_SEARCH__RERANKER"] = "none"
    os.environ["CVP_SEARCH__VLM_RERANK"] = "false"
    os.environ["CVP_SEARCH__LOW_CONFIDENCE_RETRY"] = "false"
    # Round-74 (audit): BENCH_PACK của L2 sống dai trong kernel — tune phải
    # dọn sạch, kẻo dump/tune đo cấu hình knob thay vì baseline.
    for _k in ("CVP_SEARCH__NEIGHBOR_CONSISTENCY_BOOST", "CVP_SEARCH__ROW_STRATEGY",
               "CVP_TEMPORAL__SUBMIT_STRATEGY", "CVP_TEMPORAL__POOL_CONTEXT",
               "CVP_TEMPORAL__EVENT_QUERY_VARIANTS",
               "CVP_TEMPORAL__CAPTION_SIGNAL_WEIGHT",
               "CVP_VQA__ANSWER_CANONICALIZE", "CVP_VQA__ANSWER_NEIGHBOR_FRAMES",
               "CVP_VQA__MAX_CALLS_PER_QUERY"):
        os.environ.pop(_k, None)
    _dump = Path(os.environ["CVP_PATHS__ARTIFACTS_ROOT"]) / "signal_dumps" / "thunghiem"
    _tune_out = PROJECT / "artifacts" / "tuning" / "best_weights.json"
    _tune_out.parent.mkdir(parents=True, exist_ok=True)
    try:                                     # round-71: nudge metadata tuning/
        list(_tune_out.parent.iterdir())
    except OSError:
        pass
    # Round-70/71: két sắt -prev chỉ ghi MỘT lần — chạy lại cell không được đè
    # bộ trọng số trận GỐC bằng chính bản vừa tune xong.
    _prev = _tune_out.with_name("best_weights-prev.json")
    if _tune_out.exists() and not _prev.exists():
        import shutil as _sh
        _sh.copy2(_tune_out, _prev)
        print("bộ trọng số đang dùng đã cất → best_weights-prev.json")
    elif _prev.exists():
        print("best_weights-prev.json đã có — giữ nguyên két sắt.")
    _run(REPO_DIR / "scripts" / "23_dump_signals.py",
         "--query-dir", TRIAL_DIR, "--out-dir", _dump)
    _run(REPO_DIR / "scripts" / "21_tune_weights.py", "--signals-dir", _dump,
         "--gt", GT_PATH, "--method", "random", "--trials", "400",
         "--out", _tune_out)
    print("\n📌 Trọng số thắng đã lưu:", _tune_out)
    print("   nb03 (round-44) TỰ ĐỘNG nạp file này ở ô engine — không phải sửa gì.")
elif RUN_TUNE:
    print("⚠ Chưa có GT — chạy cell L1 trước.")
'''

LAB_METACLIP = r'''
# ── L4 · Lane MetaCLIP-2: embed + A/B 3 đội hình retrieval (đo mới nhận) ──
RUN_METACLIP = True
if RUN_METACLIP and GT_PATH.exists():
    import gc, os, shutil, torch
    from pathlib import Path
    _run(REPO_DIR / "scripts" / "02_embed_and_index.py", "--model", "metaclip2")
    # đồng bộ thành quả embed về Drive để các phiên sau khỏi làm lại
    _la = Path(os.environ["CVP_PATHS__ARTIFACTS_ROOT"])
    for _sub in ("embeddings", "indexes"):
        for _f in (_la / _sub).glob("*metaclip2*"):
            _dst = PROJECT / "artifacts" / _sub / _f.name
            _dst.parent.mkdir(parents=True, exist_ok=True)
            # round-46: embeddings/metaclip2 là THƯ MỤC (mỗi video một .npy) —
            # copy2 lên thư mục từng làm sập cả cell (IsADirectoryError).
            if _f.is_dir():
                shutil.copytree(_f, _dst, dirs_exist_ok=True)
                print("   → Drive:", _dst.name + "/")
            elif not _dst.exists() or _dst.stat().st_size != _f.stat().st_size:
                shutil.copy2(_f, _dst)
                print("   → Drive:", _dst.name)
    # A/B retrieval-only (tắt reranker nặng để so LANE cho sạch và nhanh)
    from cvp.config import load_settings
    from cvp.eval.official import score_run
    from cvp.pipeline.run_queries import run_query_folder
    os.environ["CVP_SEARCH__RERANKER"] = "none"
    os.environ["CVP_SEARCH__VLM_RERANK"] = "false"
    os.environ["CVP_QUERY__PROVIDER"] = "gemini"
    # Round-74 (audit): BENCH_PACK của L2 sống dai trong kernel — lane A/B
    # phải so LANE thuần, không được thừa kế knob của lượt bench trước.
    for _k in ("CVP_SEARCH__NEIGHBOR_CONSISTENCY_BOOST", "CVP_SEARCH__ROW_STRATEGY",
               "CVP_TEMPORAL__SUBMIT_STRATEGY", "CVP_TEMPORAL__POOL_CONTEXT",
               "CVP_TEMPORAL__EVENT_QUERY_VARIANTS",
               "CVP_TEMPORAL__CAPTION_SIGNAL_WEIGHT",
               "CVP_VQA__ANSWER_CANONICALIZE", "CVP_VQA__ANSWER_NEIGHBOR_FRAMES",
               "CVP_VQA__MAX_CALLS_PER_QUERY"):
        os.environ.pop(_k, None)
    CONFIGS = {
        "finetuned": {"CVP_EMBEDDING__MODEL": "finetuned"},
        "metaclip2": {"CVP_EMBEDDING__MODEL": "metaclip2"},
        "ensemble(f+m)": {"CVP_EMBEDDING__MODEL": "ensemble",
                          "CVP_EMBEDDING__ENSEMBLE_MEMBERS": '["finetuned", "metaclip2"]',
                          "CVP_EMBEDDING__ENSEMBLE_WEIGHTS": "[0.6, 0.4]"},
    }
    _ab = {}
    print(f"\n{'đội hình':16s} mean_final (retrieval-only)")
    for _name, _env in CONFIGS.items():
        for k, v in _env.items():
            os.environ[k] = v
        _out = Path(os.environ["CVP_PATHS__ARTIFACTS_ROOT"]) / "submissions" / f"lab_{_name[:4]}"
        run_query_folder(load_settings(), TRIAL_DIR, _out, with_vqa=False)
        _r = score_run(_out, GT_PATH)
        print(f"{_name:16s} {_r.mean_final:.4f}")
        _ab[_name] = _r.mean_final
        gc.collect(); torch.cuda.empty_cache()
    import json as _json
    _drv_lab = PROJECT / "artifacts" / "lab"
    _drv_lab.mkdir(parents=True, exist_ok=True)
    (_drv_lab / "lane_ab.json").write_text(
        _json.dumps(_ab, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nĐội hình thắng → báo Claude để khóa vào nb03. (bảng đã lưu Drive)")
elif RUN_METACLIP:
    print("⚠ Chưa có GT — chạy cell L1 trước.")
'''

LAB_KEEPALIVE = r'''
# ── L6 · 🫀 Giữ phiên sống sau khi Lab xong (bấm ⏹ của ô này để dừng) ──
# Round-46: phiên Lab từng bị Colab thu hồi vì "không hoạt động". Kernel bận
# chạy ô này = hoạt động. Kết quả các stage đã được LƯU THẲNG LÊN DRIVE ngay
# khi có (bench_full.json / best_weights.json / lane_ab.json / lab_full/),
# nên dù phiên chết cũng không mất bài — ô này chỉ giữ máy ảo cho bạn quay
# lại chạy thêm stage. GIỮ TAB TRÌNH DUYỆT MỞ trong lúc Lab chạy.
import time as _tm
print("🫀 Lab watchkeeper — phiên được giữ sống.")
try:
    _n = 0
    while True:
        _tm.sleep(30)
        _n += 1
        if _n % 10 == 0:
            print(f"🫀 {_tm.strftime('%H:%M')} phiên sống")
except KeyboardInterrupt:
    print("⏹ Dừng — phiên sẽ tính là nhàn rỗi từ giờ.")
'''

LAB_KEEPALIVE = r'''
# ── L6 · 🫀 Giữ phiên sống sau khi Lab xong (bấm ⏹ của ô này để dừng) ──
# Round-46: phiên Lab từng bị Colab thu hồi vì "không hoạt động". Kernel bận
# chạy ô này = hoạt động. Kết quả các stage đã được LƯU THẲNG LÊN DRIVE ngay
# khi có (bench_full.json / best_weights.json / lane_ab.json / lab_full/),
# nên dù phiên chết cũng không mất bài — ô này chỉ giữ máy ảo cho bạn quay
# lại chạy thêm stage. GIỮ TAB TRÌNH DUYỆT MỞ trong lúc Lab chạy.
import time as _tm
print("🫀 Lab watchkeeper — phiên được giữ sống.")
try:
    _n = 0
    while True:
        _tm.sleep(30)
        _n += 1
        if _n % 10 == 0:
            print(f"🫀 {_tm.strftime('%H:%M')} phiên sống")
except KeyboardInterrupt:
    print("⏹ Dừng — phiên sẽ tính là nhàn rỗi từ giờ.")
'''

LAB_ASR_LARGE = r'''
# ── L5 · CA ĐÊM: ASR PhoWhisper-large toàn bộ 873 video (~10–20 giờ) ──
# Thoại là nguồn sống của track QA. Bật True rồi để chạy qua đêm; per-video
# resume nên đứt phiên chạy lại là nối tiếp. Bản medium được giữ nguyên trên
# Drive (asr-medium-backup) cho tới khi bạn hài lòng với bản large.
RUN_ASR_LARGE = False
if RUN_ASR_LARGE:
    import os, shutil, threading
    import time as _t2
    from pathlib import Path
    _drv_asr = PROJECT / "artifacts" / "asr"
    _bak = PROJECT / "artifacts" / "asr-medium-backup"
    _partial = PROJECT / "artifacts" / "asr-large-partial"
    if _drv_asr.exists() and not _bak.exists():
        _drv_asr.rename(_bak)               # giữ bản medium làm đường lui
        print("Đã cất bản medium →", _bak)
    _la = Path(os.environ["CVP_PATHS__ARTIFACTS_ROOT"])
    if (_la / "asr").exists():
        shutil.rmtree(_la / "asr")          # xóa bản medium staging cho sạch
    # Round-48: ca ASR dài hơn trần 24h của Colab — kết quả từng phần phải
    # SỐNG TRÊN DRIVE. Seed lại từ partial (resume xuyên phiên) + thread nền
    # đồng bộ mỗi 10 phút; phiên chết chỉ mất tối đa 10 phút công.
    if _partial.exists():
        shutil.copytree(_partial, _la / "asr", dirs_exist_ok=True)
        print(f"Resume: {len(list((_la / 'asr').glob('*.json')))} video large đã xong từ phiên trước")
    else:
        _partial.mkdir(parents=True, exist_ok=True)
    _stop_sync = False

    def _syncer():
        while not _stop_sync:
            _t2.sleep(600)
            try:
                _n = 0
                for _f in (_la / "asr").glob("*.json"):
                    _d = _partial / _f.name
                    if not _d.exists() or _d.stat().st_size != _f.stat().st_size:
                        shutil.copy2(_f, _d)
                        _n += 1
                if _n:
                    print(f"💾 {_t2.strftime('%H:%M')} sync {_n} video ASR → Drive")
            except Exception:  # noqa: BLE001 — syncer chết lặng lẽ là chấp nhận được
                pass

    threading.Thread(target=_syncer, daemon=True).start()
    os.environ["CVP_ASR__MODEL"] = "vinai/PhoWhisper-large"
    _run(REPO_DIR / "scripts" / "03_build_aux_indexes.py", "--asr")
    _run(REPO_DIR / "scripts" / "03_build_aux_indexes.py",
         "--text-index", "--force-text-index")
    _stop_sync = True
    for _d in ("asr", "text_index"):
        _dst = PROJECT / "artifacts" / _d
        if _dst.exists():
            shutil.rmtree(_dst)
        shutil.copytree(_la / _d, _dst)
        print("   → Drive:", _dst)
    print("XONG — chạy lại cell L2 để đo bản large trên bench.")
else:
    print("RUN_ASR_LARGE=False — bật True cho ca đêm ASR.")
'''


NB5_TITLE = r'''
# 🎙 Core Vision Perfect V1 — 05 · ASR LARGE (build chuyên biệt, shard song song)

Nâng đôi tai của hệ: nghe lại TOÀN BỘ 873 video bằng **PhoWhisper-large**
(ASR tiếng Việt mạnh nhất công khai). Tổng ~24–40 giờ GPU — chạy **N phiên
song song** (N bản sao notebook, mỗi bản một `SHARD_INDEX`) để chia N lần
thời gian. Kết quả sống trên Drive từng 10 phút; bản medium được giữ làm
đường lui (`asr-medium-backup`). Phiên thấy kho đủ tự rebuild BM25 + hoán đổi.
'''

NB5_SWEEP = r'''
# ── ⚒ ASR PhoWhisper-LARGE toàn bộ 873 video (shard song song) ──
# Round-51: shard ĐÃ ĐẶT SẴN theo tên file (bản a/b/c) — không chỉnh gì cả,
# chỉ Run all. Mỗi phiên gánh 1/N số video;
# kết quả từng video đổ chung về MỘT kho Drive (asr-large-partial) mỗi 10 phút —
# phiên chết chỉ mất tối đa 10 phút công, chạy lại là tự nối tiếp.
# Phiên nào hoàn tất mà thấy KHO ĐỦ toàn bộ video sẽ tự FINALIZE (rebuild
# BM25 text-index + hoán đổi vào artifacts thật, có backup đường lui).
SHARD_INDEX = 0
SHARD_TOTAL = 1
import os, shutil, subprocess, sys, threading
import time as _tm
from pathlib import Path

def _run(*args):
    print("$", " ".join(map(str, args)), flush=True)
    _pp = subprocess.Popen([sys.executable, "-u", *map(str, args)],
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           text=True)
    for _ln in _pp.stdout:
        print(_ln, end="", flush=True)
    if _pp.wait() != 0:
        raise RuntimeError(f"lệnh lỗi (exit {_pp.returncode})")

# Round-57 (audit): danh sách video phải lấy từ bản LOCAL đã materialize + tự
# vá (cell trước) — listing Drive FUSE có thể trả THIẾU/RỖNG (round-28), và
# _vids ngắn sẽ khiến prune xóa nhầm bài thật + finalize giao kho thiếu bài.
_mk_local = Path("/content/data/map-keyframes")
_mk = _mk_local if _mk_local.is_dir() else (PROJECT / "data" / "map-keyframes")
_vids = sorted(_q.stem for _q in _mk.glob("*.csv"))
assert len(_vids) > 800, (
    f"map-keyframes chỉ liệt kê {len(_vids)} video — máy ảo này dính DriveFS "
    "hỏng metadata (tuyệt đối không chạy với danh sách thiếu). Cách xử: "
    "Runtime ▸ Disconnect and delete runtime rồi Run all lại trên máy mới — "
    "dữ liệu trên Drive vẫn nguyên vẹn.")
_my = _vids[SHARD_INDEX::SHARD_TOTAL]
print(f"Shard {SHARD_INDEX + 1}/{SHARD_TOTAL}: {len(_my)}/{len(_vids)} video")

_partial = PROJECT / "artifacts" / "asr-large-partial"
# Round-52 (live 05 run 24/08): N phiên cùng mkdir một lúc → Google Drive tạo
# NHIỀU thư mục TRÙNG TÊN (Drive cho phép trùng tên!), mỗi phiên đổ bài vào
# một bản → không phiên nào đếm đủ, FINALIZE không bao giờ nổ. Hai lớp chống:
# (1) chỉ ca 1 được TẠO kho — các ca sau ĐỢI thấy kho rồi mới vào;
if not (SHARD_INDEX == 0 or _partial.exists()):
    for _w in range(40):                     # tới 20 phút
        if _partial.exists():
            break
        print(f"⏳ đợi ca 1 tạo kho chung ({_w * 30}s) — cứ để yên ...", flush=True)
        _tm.sleep(30)
    else:
        print("⚠ 20 phút không thấy kho chung — đành tự tạo (hãy chắc ca 1 đang chạy).")
_partial.mkdir(parents=True, exist_ok=True)
# (2) lưới an toàn: gộp mọi kho sinh đôi "<tên> (1)"… về bản chính rồi xóa —
# nhờ vậy chạy lại ô này trên MỘT phiên là tự lành + finalize được.
for _dup in sorted(_partial.parent.glob(_partial.name + " (*")):
    _suf = _dup.name[len(_partial.name):]
    # round-61 (audit): chỉ nhận đúng dạng " (N)" Drive tự sinh khi trùng tên —
    # folder người dùng tự đặt kiểu " (backup)" không bị gộp-rồi-xóa nhầm.
    if not (_dup.is_dir() and _suf.startswith(" (")
            and _suf.endswith(")") and _suf[2:-1].isdigit()):
        continue
    _n_dup = 0
    for _f in _dup.glob("*.json"):
        _d = _partial / _f.name
        if not _d.exists() or _d.stat().st_size < _f.stat().st_size:
            shutil.copy2(_f, _d)
            _n_dup += 1
    shutil.rmtree(_dup, ignore_errors=True)
    print(f"⚠ Gộp kho trùng tên '{_dup.name}': +{_n_dup} video", flush=True)
# Round-53 (live 05 run 24/08): CVP_PATHS__ARTIFACTS_ROOT của cell 4 vẫn trỏ
# VÀO DRIVE → "staging local" hóa ra là artifacts/asr THẬT trên Drive: bản
# medium bị rmtree bay mất, 3 phiên đua nhau tạo staging sinh đôi trùng tên,
# và job còn ghi đè text_index trận đấu bằng bản nửa vời. Staging phải nằm
# trên ĐĨA LOCAL của VM — Drive chỉ nhận kết quả qua syncer + FINALIZE.
_la = Path("/content/artifacts")
_la.mkdir(parents=True, exist_ok=True)
os.environ["CVP_PATHS__ARTIFACTS_ROOT"] = str(_la)
# Round-55 (live 05a run 24/08): script aux ĐỌC catalog/manifest.parquet từ
# artifacts root — staging local rỗng phải kéo bản Drive về trước, không thì
# chết ngay "Catalog missing" (di chứng của round-53).
_drv_cat = PROJECT / "artifacts" / "catalog"
_ensure_drive()                          # round-57: mount phải sống trước FUSE I/O
assert (_drv_cat / "manifest.parquet").exists(), (
    "Thiếu artifacts/catalog/manifest.parquet trên Drive — chạy nb01 trước.")
for _try in (1, 2, 3):                   # round-57: copy đầu phiên cũng phải lì đòn
    try:
        shutil.copytree(_drv_cat, _la / "catalog", dirs_exist_ok=True)
        if (_la / "catalog" / "manifest.parquet").exists():
            print("catalog: staged về local")
            break
    except OSError as _e:
        print(f"   ⚠ copy catalog lỗi: {_e!r}")
    _ensure_drive()
    _tm.sleep(15)
else:
    raise RuntimeError("Không kéo được catalog về local sau 3 lần — chạy lại ô này.")
_job_local = _la / "asr"
# Round-57 (audit): KHÔNG rmtree staging và KHÔNG đè file local bằng bản Drive
# — bài local (ghi nguyên tử) là bản đáng tin nhất; sync chốt lỗi rồi chạy lại
# ô này sẽ không mất bài nữa. Chỉ bù những file THIẾU từ kho chung.
_job_local.mkdir(parents=True, exist_ok=True)
for _try in (1, 2, 3):
    try:
        for _f in _partial.glob("*.json"):           # resume xuyên phiên
            _d = _job_local / _f.name
            if not _d.exists():
                shutil.copy2(_f, _d)
        break
    except OSError as _e:
        print(f"   ⚠ seed staging lỗi: {_e!r} — thử lại {_try}/3")
        _ensure_drive()
        _tm.sleep(15)
else:
    raise RuntimeError("Không seed được staging từ kho chung — chạy lại ô này.")
_vidset = set(_vids)

def _prune_staging():
    # Round-54: kho chung có thể lẫn rác sau các thao tác dọn tay trên Drive
    # web (bản trùng tên "xxx (1).json", file up nhầm chỗ…) — gạt khỏi staging
    # để store/text-index không nuốt phải video ma.
    # Round-57 (audit): file RÁCH (phiên chết giữa lúc sync) cũng phải bị gạt
    # — resume chỉ nhìn tên file, bản rách sẽ bị khóa vĩnh viễn vào kho trận
    # đấu nếu để lọt; xóa để job tính lại rồi sync đè bản lành lên kho chung.
    import json as _json
    for _p in list(_job_local.iterdir()):
        _ok = _p.is_file() and _p.suffix == ".json" and _p.stem in _vidset
        if _ok:
            try:
                with open(_p, encoding="utf-8") as _fh:
                    _json.load(_fh)
            except Exception:
                _ok = False
        if not _ok:
            shutil.rmtree(_p, ignore_errors=True) if _p.is_dir() else _p.unlink()
            print("   bỏ qua file lạ/rách trong kho:", _p.name, flush=True)

_prune_staging()
print(f"Resume: {len(list(_job_local.glob('*.json')))} video đã xong từ trước")

_stop_sync = False

def _syncer():
    # Round-57 (audit): daemon DriveFS có thể CHẾT giữa phiên dài (Errno 107)
    # — syncer câm lặng sẽ âm thầm ngừng đổ bài về Drive hàng chục giờ. Giờ nó
    # tự hồi sinh mount trước mỗi lượt và LA LỚN khi sync hỏng.
    while not _stop_sync:
        _tm.sleep(600)
        try:
            _ensure_drive()
            _n = 0
            for _f in _job_local.glob("*.json"):
                _d = _partial / _f.name
                if not _d.exists() or _d.stat().st_size != _f.stat().st_size:
                    shutil.copy2(_f, _d)
                    _n += 1
            _p_new = 0
            for _f in _partial.glob("*.json"):   # round-58: KÉO chiều về — học
                _d = _job_local / _f.name        # ngay bài các ca khác vừa xong
                if not _d.exists():              # để job tự skip, không trùng việc
                    shutil.copy2(_f, _d)
                    _p_new += 1
            if _n or _p_new:
                print(f"SYNC {_tm.strftime('%H:%M')}: đẩy {_n} / kéo {_p_new} video",
                      flush=True)
        except Exception as _e:  # noqa: BLE001 — không được giết job vì sync
            print(f"⚠ SYNC {_tm.strftime('%H:%M')} LỖI: {_e!r} — thử lại sau 10 phút",
                  flush=True)

threading.Thread(target=_syncer, daemon=True).start()
os.environ["CVP_ASR__MODEL"] = "vinai/PhoWhisper-large"
_run(REPO_DIR / "scripts" / "03_build_aux_indexes.py", "--asr",
     "--videos", *_my)
_stop_sync = True
for _try in (1, 2, 3):                   # đợt sync chốt của shard — phải lì đòn
    try:
        _ensure_drive()                  # round-57: mount có thể đã chết giữa job
        for _f in _job_local.glob("*.json"):
            _d = _partial / _f.name
            if not _d.exists() or _d.stat().st_size != _f.stat().st_size:
                shutil.copy2(_f, _d)
        break
    except OSError as _e:
        print(f"   ⚠ sync chốt lỗi: {_e!r} — thử lại {_try}/3")
        _tm.sleep(20)
else:
    raise RuntimeError("Sync chốt về Drive thất bại — KHÔNG xóa runtime! "
                       "Chạy lại ô này để đẩy nốt kết quả local lên Drive.")
_done = {_p.stem for _p in _partial.glob("*.json")}
_missing = [_v for _v in _vids if _v not in _done]
print(f"Shard xong. Kho chung: {len(_vids) - len(_missing)}/{len(_vids)} video")
if _missing:
    print(f"   còn thiếu (vd): {_missing[:5]}")

# ── FINALIZE: shard nào thấy kho đủ sẽ chốt hạ (có khóa chống chạy đôi) ──
# Round-54: đếm theo TÊN video (stem) — file trùng tên/lạ không thổi phồng số.
if not _missing:
    _lock = _partial / "_finalize.lock"
    _done_mark = _partial / "_finalize.done"
    # Round-56: khóa phải phân biệt "đã xong" / "đang chạy" / "chết giữa chừng"
    # — finalize crash không được để khóa mồ côi chặn vĩnh viễn.
    # Round-57 (audit): exists→write KHÔNG nguyên tử qua FUSE (cache trễ giữa
    # các máy) → khóa mang TOKEN riêng + xác nhận lại sau 90s + nhịp tim làm
    # mới khóa giữa các bước dài (finalize thật có thể >2h) + ngưỡng cũ 6h.
    import uuid as _uuid
    _token = _uuid.uuid4().hex

    def _touch_lock():
        _lock.write_text(_token, encoding="utf-8")

    _lock_fresh = False
    if _lock.exists():
        try:
            _lock_fresh = (_tm.time() - _lock.stat().st_mtime) < 6 * 3600
        except OSError:
            pass
    if _done_mark.exists():
        print("Finalize đã hoàn tất từ trước — không cần làm lại.")
    elif _lock.exists() and _lock_fresh:
        print("Finalize đang được phiên khác chạy (khóa <6h) — bỏ qua. Nếu chắc "
              "chắn không phiên nào khác đang chạy: xóa _finalize.lock trong "
              "kho chung trên Drive rồi chạy lại ô này.")
    else:
        if _lock.exists():
            print("Khóa finalize cũ (>6h) mà chưa có dấu hoàn tất — tiếp quản chốt lại.")
        _touch_lock()
        print("FINALIZE: giành khóa — chờ 90s xác nhận không phiên nào giành cùng lúc ...")
        try:
            _tm.sleep(90)
            _mine = False
            try:
                _mine = _lock.read_text(encoding="utf-8").strip() == _token
            except OSError:
                pass
            if not _mine:
                raise RuntimeError(
                    "Phiên khác giành khóa finalize cùng lúc — phiên này NHƯỜNG "
                    "(KHÔNG phải lỗi; theo dõi phiên kia là được).")
            print("FINALIZE: rebuild BM25 + hoán đổi artifacts ...")
            # Round-56: staging phải ĐẦY ĐỦ THẬT — copytree qua FUSE từng giao
            # thiếu (863/873 live). Kéo lại vài lượt rồi tự ASR bù phần thiếu.
            _ensure_drive()
            _still = []
            for _try in (1, 2, 3):
                for _f in _partial.glob("*.json"):   # chỉ bù file THIẾU — không
                    _d = _job_local / _f.name        # đè bản local lành bằng
                    if not _d.exists():              # bản Drive có thể rách
                        shutil.copy2(_f, _d)
                _prune_staging()
                _still = [_v for _v in _vids
                          if not (_job_local / (_v + ".json")).is_file()]
                if not _still:
                    break
                print(f"   staging thiếu {len(_still)} video (vd {_still[:3]}) — kéo lại {_try}/3 ...")
                _tm.sleep(30)
            if _still:
                print(f"   tự chạy bù {len(_still)} video thiếu ...")
                _run(REPO_DIR / "scripts" / "03_build_aux_indexes.py", "--asr",
                     "--videos", *_still)
                for _v in _still:                    # trả bản bù về kho chung
                    _f = _job_local / (_v + ".json")
                    if _f.is_file():
                        shutil.copy2(_f, _partial / _f.name)
                _still2 = [_v for _v in _still
                           if not (_job_local / (_v + ".json")).is_file()]
                if _still2:
                    # round-61 (audit): TUYỆT ĐỐI không chốt kho thiếu bài —
                    # chạy bù mà vẫn thiếu thì dừng (khóa tự nhả), chạy lại sau.
                    raise RuntimeError(
                        f"Chạy bù xong vẫn thiếu {len(_still2)} video "
                        f"(vd {_still2[:3]}) — không chốt kho thiếu; chạy lại ô này.")
            # Round-56: kho aux nào kéo thiếu → index trận đấu âm thầm yếu đi.
            for _aux in ("ocr", "asr", "captions"):  # BM25 cần đủ các kho aux
                _src = PROJECT / "artifacts" / _aux
                if _aux == "asr" or not _src.is_dir():
                    continue
                for _try in (1, 2, 3):
                    shutil.copytree(_src, _la / _aux, dirs_exist_ok=True)
                    # round-57: listing Drive có thể TRẢ THIẾU y hệt copy —
                    # neo số cần vào danh sách video, không tin listing suông.
                    _need = max(len(list(_src.glob("*.json"))), len(_vids))
                    _got = len(list((_la / _aux).glob("*.json")))
                    if _got >= _need:
                        print(f"   staging {_aux}: {_got}/{_need} file")
                        break
                    print(f"   staging {_aux} thiếu ({_got}/{_need}) — thử lại {_try}/3 ...")
                    _tm.sleep(60)
                else:
                    raise RuntimeError(
                        f"Kéo kho {_aux} về máy mãi vẫn thiếu — Drive trục trặc; "
                        "chạy lại ô này sau ít phút (khóa sẽ tự nhả).")
            _touch_lock()                # nhịp tim khóa trước bước rebuild dài
            _run(REPO_DIR / "scripts" / "03_build_aux_indexes.py",
                 "--text-index", "--force-text-index")
            _ensure_drive()              # round-57: mount phải sống trước hoán đổi
            _touch_lock()
            _drv_job = PROJECT / "artifacts" / "asr"
            _bak = PROJECT / "artifacts" / "asr-medium-backup"
            if _drv_job.exists() and not _bak.exists():
                _drv_job.rename(_bak)
                print("Đã cất bản cũ →", _bak)
            for _d in ("asr", "text_index"):
                _dst = PROJECT / "artifacts" / _d
                if _dst.exists():
                    shutil.rmtree(_dst)
                shutil.copytree(_la / _d, _dst)
                # Round-56: upload cũng đếm lại — copy bù nếu Drive nuốt thiếu
                _n_src = sum(1 for _q in (_la / _d).rglob("*") if _q.is_file())
                _n_dst = sum(1 for _q in _dst.rglob("*") if _q.is_file())
                if _n_dst < _n_src:
                    shutil.copytree(_la / _d, _dst, dirs_exist_ok=True)
                    _n_dst = sum(1 for _q in _dst.rglob("*") if _q.is_file())
                print(f"   → Drive: {_dst} ({_n_dst}/{_n_src} file)")
            _done_mark.write_text(_tm.strftime("%Y-%m-%d %H:%M"), encoding="utf-8")
        except BaseException:
            # round-57: chỉ nhả khóa nếu vẫn là khóa CỦA MÌNH — thua cuộc đua
            # thì tuyệt đối không được gỡ khóa của phiên thắng.
            try:
                if _lock.read_text(encoding="utf-8").strip() == _token:
                    _lock.unlink(missing_ok=True)   # nhả khóa để chạy lại được ngay
            except OSError:
                pass
            raise
        print("XONG TOÀN BỘ — artifacts đã nâng cấp. Đo lại bằng nb04 (L2).")
else:
    print("Kho chưa đủ — chờ các shard khác (hoặc chạy lại phiên để nối tiếp).")
'''

NB6_TITLE = r'''
# 📝 Core Vision Perfect V1 — 06 · Captions DÀY (build chuyên biệt, shard song song)

Caption hiện chỉ phủ ~1/4 số keyframe (stride 4 từ tuần build). Notebook này
caption **MỌI keyframe** (stride 1) bằng Vintern — kho chữ cho BM25 dày gấp
~4 lần (bộ tinh chỉnh trọng số đã "thích" tín hiệu caption). Tổng ~50–100
giờ GPU — nên chạy 3–4 shard song song. Cùng cơ chế an toàn như notebook 05.
'''

NB6_SWEEP = r'''
# ── ⚒ Captions Vintern DÀY — stride 1, mọi keyframe (shard song song) ──
# Round-51: shard ĐÃ ĐẶT SẴN theo tên file (bản a/b/c) — không chỉnh gì cả,
# chỉ Run all. Mỗi phiên gánh 1/N số video;
# kết quả từng video đổ chung về MỘT kho Drive (captions-dense-partial) mỗi 10 phút —
# phiên chết chỉ mất tối đa 10 phút công, chạy lại là tự nối tiếp.
# Phiên nào hoàn tất mà thấy KHO ĐỦ toàn bộ video sẽ tự FINALIZE (rebuild
# BM25 text-index + hoán đổi vào artifacts thật, có backup đường lui).
SHARD_INDEX = 0
SHARD_TOTAL = 1
import os, shutil, subprocess, sys, threading
import time as _tm
from pathlib import Path

def _run(*args):
    print("$", " ".join(map(str, args)), flush=True)
    _pp = subprocess.Popen([sys.executable, "-u", *map(str, args)],
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           text=True)
    for _ln in _pp.stdout:
        print(_ln, end="", flush=True)
    if _pp.wait() != 0:
        raise RuntimeError(f"lệnh lỗi (exit {_pp.returncode})")

# Round-57 (audit): danh sách video phải lấy từ bản LOCAL đã materialize + tự
# vá (cell trước) — listing Drive FUSE có thể trả THIẾU/RỖNG (round-28), và
# _vids ngắn sẽ khiến prune xóa nhầm bài thật + finalize giao kho thiếu bài.
_mk_local = Path("/content/data/map-keyframes")
_mk = _mk_local if _mk_local.is_dir() else (PROJECT / "data" / "map-keyframes")
_vids = sorted(_q.stem for _q in _mk.glob("*.csv"))
assert len(_vids) > 800, (
    f"map-keyframes chỉ liệt kê {len(_vids)} video — máy ảo này dính DriveFS "
    "hỏng metadata (tuyệt đối không chạy với danh sách thiếu). Cách xử: "
    "Runtime ▸ Disconnect and delete runtime rồi Run all lại trên máy mới — "
    "dữ liệu trên Drive vẫn nguyên vẹn.")
_my = _vids[SHARD_INDEX::SHARD_TOTAL]
print(f"Shard {SHARD_INDEX + 1}/{SHARD_TOTAL}: {len(_my)}/{len(_vids)} video")

_partial = PROJECT / "artifacts" / "captions-dense-partial"
# Round-52 (live 05 run 24/08): N phiên cùng mkdir một lúc → Google Drive tạo
# NHIỀU thư mục TRÙNG TÊN (Drive cho phép trùng tên!), mỗi phiên đổ bài vào
# một bản → không phiên nào đếm đủ, FINALIZE không bao giờ nổ. Hai lớp chống:
# (1) chỉ ca 1 được TẠO kho — các ca sau ĐỢI thấy kho rồi mới vào;
if not (SHARD_INDEX == 0 or _partial.exists()):
    for _w in range(40):                     # tới 20 phút
        if _partial.exists():
            break
        print(f"⏳ đợi ca 1 tạo kho chung ({_w * 30}s) — cứ để yên ...", flush=True)
        _tm.sleep(30)
    else:
        print("⚠ 20 phút không thấy kho chung — đành tự tạo (hãy chắc ca 1 đang chạy).")
_partial.mkdir(parents=True, exist_ok=True)
# (2) lưới an toàn: gộp mọi kho sinh đôi "<tên> (1)"… về bản chính rồi xóa —
# nhờ vậy chạy lại ô này trên MỘT phiên là tự lành + finalize được.
for _dup in sorted(_partial.parent.glob(_partial.name + " (*")):
    _suf = _dup.name[len(_partial.name):]
    # round-61 (audit): chỉ nhận đúng dạng " (N)" Drive tự sinh khi trùng tên —
    # folder người dùng tự đặt kiểu " (backup)" không bị gộp-rồi-xóa nhầm.
    if not (_dup.is_dir() and _suf.startswith(" (")
            and _suf.endswith(")") and _suf[2:-1].isdigit()):
        continue
    _n_dup = 0
    for _f in _dup.glob("*.json"):
        _d = _partial / _f.name
        if not _d.exists() or _d.stat().st_size < _f.stat().st_size:
            shutil.copy2(_f, _d)
            _n_dup += 1
    shutil.rmtree(_dup, ignore_errors=True)
    print(f"⚠ Gộp kho trùng tên '{_dup.name}': +{_n_dup} video", flush=True)
# Round-53 (live 05 run 24/08): CVP_PATHS__ARTIFACTS_ROOT của cell 4 vẫn trỏ
# VÀO DRIVE → "staging local" hóa ra là artifacts/captions THẬT trên Drive:
# store thật bị rmtree, N phiên đua nhau tạo staging sinh đôi trùng tên, và
# job còn ghi đè text_index trận đấu bằng bản nửa vời. Staging phải nằm
# trên ĐĨA LOCAL của VM — Drive chỉ nhận kết quả qua syncer + FINALIZE.
_la = Path("/content/artifacts")
_la.mkdir(parents=True, exist_ok=True)
os.environ["CVP_PATHS__ARTIFACTS_ROOT"] = str(_la)
# Round-55 (live 05a run 24/08): script aux ĐỌC catalog/manifest.parquet từ
# artifacts root — staging local rỗng phải kéo bản Drive về trước, không thì
# chết ngay "Catalog missing" (di chứng của round-53).
_drv_cat = PROJECT / "artifacts" / "catalog"
_ensure_drive()                          # round-57: mount phải sống trước FUSE I/O
assert (_drv_cat / "manifest.parquet").exists(), (
    "Thiếu artifacts/catalog/manifest.parquet trên Drive — chạy nb01 trước.")
for _try in (1, 2, 3):                   # round-57: copy đầu phiên cũng phải lì đòn
    try:
        shutil.copytree(_drv_cat, _la / "catalog", dirs_exist_ok=True)
        if (_la / "catalog" / "manifest.parquet").exists():
            print("catalog: staged về local")
            break
    except OSError as _e:
        print(f"   ⚠ copy catalog lỗi: {_e!r}")
    _ensure_drive()
    _tm.sleep(15)
else:
    raise RuntimeError("Không kéo được catalog về local sau 3 lần — chạy lại ô này.")
_job_local = _la / "captions"
# Round-57 (audit): KHÔNG rmtree staging và KHÔNG đè file local bằng bản Drive
# — bài local (ghi nguyên tử) là bản đáng tin nhất; sync chốt lỗi rồi chạy lại
# ô này sẽ không mất bài nữa. Chỉ bù những file THIẾU từ kho chung.
_job_local.mkdir(parents=True, exist_ok=True)
for _try in (1, 2, 3):
    try:
        for _f in _partial.glob("*.json"):           # resume xuyên phiên
            _d = _job_local / _f.name
            if not _d.exists():
                shutil.copy2(_f, _d)
        break
    except OSError as _e:
        print(f"   ⚠ seed staging lỗi: {_e!r} — thử lại {_try}/3")
        _ensure_drive()
        _tm.sleep(15)
else:
    raise RuntimeError("Không seed được staging từ kho chung — chạy lại ô này.")
_vidset = set(_vids)

def _prune_staging():
    # Round-54: kho chung có thể lẫn rác sau các thao tác dọn tay trên Drive
    # web (bản trùng tên "xxx (1).json", file up nhầm chỗ…) — gạt khỏi staging
    # để store/text-index không nuốt phải video ma.
    # Round-57 (audit): file RÁCH (phiên chết giữa lúc sync) cũng phải bị gạt
    # — resume chỉ nhìn tên file, bản rách sẽ bị khóa vĩnh viễn vào kho trận
    # đấu nếu để lọt; xóa để job tính lại rồi sync đè bản lành lên kho chung.
    import json as _json
    for _p in list(_job_local.iterdir()):
        _ok = _p.is_file() and _p.suffix == ".json" and _p.stem in _vidset
        if _ok:
            try:
                with open(_p, encoding="utf-8") as _fh:
                    _json.load(_fh)
            except Exception:
                _ok = False
        if not _ok:
            shutil.rmtree(_p, ignore_errors=True) if _p.is_dir() else _p.unlink()
            print("   bỏ qua file lạ/rách trong kho:", _p.name, flush=True)

_prune_staging()
print(f"Resume: {len(list(_job_local.glob('*.json')))} video đã xong từ trước")

_stop_sync = False

def _syncer():
    # Round-57 (audit): daemon DriveFS có thể CHẾT giữa phiên dài (Errno 107)
    # — syncer câm lặng sẽ âm thầm ngừng đổ bài về Drive hàng chục giờ. Giờ nó
    # tự hồi sinh mount trước mỗi lượt và LA LỚN khi sync hỏng.
    while not _stop_sync:
        _tm.sleep(600)
        try:
            _ensure_drive()
            _n = 0
            for _f in _job_local.glob("*.json"):
                _d = _partial / _f.name
                if not _d.exists() or _d.stat().st_size != _f.stat().st_size:
                    shutil.copy2(_f, _d)
                    _n += 1
            _p_new = 0
            for _f in _partial.glob("*.json"):   # round-58: KÉO chiều về — học
                _d = _job_local / _f.name        # ngay bài các ca khác vừa xong
                if not _d.exists():              # để job tự skip, không trùng việc
                    shutil.copy2(_f, _d)
                    _p_new += 1
            if _n or _p_new:
                print(f"SYNC {_tm.strftime('%H:%M')}: đẩy {_n} / kéo {_p_new} video",
                      flush=True)
        except Exception as _e:  # noqa: BLE001 — không được giết job vì sync
            print(f"⚠ SYNC {_tm.strftime('%H:%M')} LỖI: {_e!r} — thử lại sau 10 phút",
                  flush=True)

threading.Thread(target=_syncer, daemon=True).start()
# stride 1: caption MỌI keyframe (kho cũ stride 4 chỉ phủ ~1/4)
_run(REPO_DIR / "scripts" / "03_build_aux_indexes.py", "--captions", "--caption-stride", "1",
     "--videos", *_my)
_stop_sync = True
for _try in (1, 2, 3):                   # đợt sync chốt của shard — phải lì đòn
    try:
        _ensure_drive()                  # round-57: mount có thể đã chết giữa job
        for _f in _job_local.glob("*.json"):
            _d = _partial / _f.name
            if not _d.exists() or _d.stat().st_size != _f.stat().st_size:
                shutil.copy2(_f, _d)
        break
    except OSError as _e:
        print(f"   ⚠ sync chốt lỗi: {_e!r} — thử lại {_try}/3")
        _tm.sleep(20)
else:
    raise RuntimeError("Sync chốt về Drive thất bại — KHÔNG xóa runtime! "
                       "Chạy lại ô này để đẩy nốt kết quả local lên Drive.")
_done = {_p.stem for _p in _partial.glob("*.json")}
_missing = [_v for _v in _vids if _v not in _done]
print(f"Shard xong. Kho chung: {len(_vids) - len(_missing)}/{len(_vids)} video")
if _missing:
    print(f"   còn thiếu (vd): {_missing[:5]}")

# ── FINALIZE: shard nào thấy kho đủ sẽ chốt hạ (có khóa chống chạy đôi) ──
# Round-54: đếm theo TÊN video (stem) — file trùng tên/lạ không thổi phồng số.
if not _missing:
    _lock = _partial / "_finalize.lock"
    _done_mark = _partial / "_finalize.done"
    # Round-56: khóa phải phân biệt "đã xong" / "đang chạy" / "chết giữa chừng"
    # — finalize crash không được để khóa mồ côi chặn vĩnh viễn.
    # Round-57 (audit): exists→write KHÔNG nguyên tử qua FUSE (cache trễ giữa
    # các máy) → khóa mang TOKEN riêng + xác nhận lại sau 90s + nhịp tim làm
    # mới khóa giữa các bước dài (finalize thật có thể >2h) + ngưỡng cũ 6h.
    import uuid as _uuid
    _token = _uuid.uuid4().hex

    def _touch_lock():
        _lock.write_text(_token, encoding="utf-8")

    _lock_fresh = False
    if _lock.exists():
        try:
            _lock_fresh = (_tm.time() - _lock.stat().st_mtime) < 6 * 3600
        except OSError:
            pass
    if _done_mark.exists():
        print("Finalize đã hoàn tất từ trước — không cần làm lại.")
    elif _lock.exists() and _lock_fresh:
        print("Finalize đang được phiên khác chạy (khóa <6h) — bỏ qua. Nếu chắc "
              "chắn không phiên nào khác đang chạy: xóa _finalize.lock trong "
              "kho chung trên Drive rồi chạy lại ô này.")
    else:
        if _lock.exists():
            print("Khóa finalize cũ (>6h) mà chưa có dấu hoàn tất — tiếp quản chốt lại.")
        _touch_lock()
        print("FINALIZE: giành khóa — chờ 90s xác nhận không phiên nào giành cùng lúc ...")
        try:
            _tm.sleep(90)
            _mine = False
            try:
                _mine = _lock.read_text(encoding="utf-8").strip() == _token
            except OSError:
                pass
            if not _mine:
                raise RuntimeError(
                    "Phiên khác giành khóa finalize cùng lúc — phiên này NHƯỜNG "
                    "(KHÔNG phải lỗi; theo dõi phiên kia là được).")
            print("FINALIZE: rebuild BM25 + hoán đổi artifacts ...")
            # Round-56: staging phải ĐẦY ĐỦ THẬT — copytree qua FUSE từng giao
            # thiếu (863/873 live). Kéo lại vài lượt rồi tự caption bù phần thiếu.
            _ensure_drive()
            _still = []
            for _try in (1, 2, 3):
                for _f in _partial.glob("*.json"):   # chỉ bù file THIẾU — không
                    _d = _job_local / _f.name        # đè bản local lành bằng
                    if not _d.exists():              # bản Drive có thể rách
                        shutil.copy2(_f, _d)
                _prune_staging()
                _still = [_v for _v in _vids
                          if not (_job_local / (_v + ".json")).is_file()]
                if not _still:
                    break
                print(f"   staging thiếu {len(_still)} video (vd {_still[:3]}) — kéo lại {_try}/3 ...")
                _tm.sleep(30)
            if _still:
                print(f"   tự chạy bù {len(_still)} video thiếu ...")
                _run(REPO_DIR / "scripts" / "03_build_aux_indexes.py",
                     "--captions", "--caption-stride", "1",
                     "--videos", *_still)
                for _v in _still:                    # trả bản bù về kho chung
                    _f = _job_local / (_v + ".json")
                    if _f.is_file():
                        shutil.copy2(_f, _partial / _f.name)
                _still2 = [_v for _v in _still
                           if not (_job_local / (_v + ".json")).is_file()]
                if _still2:
                    # round-61 (audit): TUYỆT ĐỐI không chốt kho thiếu bài —
                    # chạy bù mà vẫn thiếu thì dừng (khóa tự nhả), chạy lại sau.
                    raise RuntimeError(
                        f"Chạy bù xong vẫn thiếu {len(_still2)} video "
                        f"(vd {_still2[:3]}) — không chốt kho thiếu; chạy lại ô này.")
            # Round-56: kho aux nào kéo thiếu → index trận đấu âm thầm yếu đi.
            for _aux in ("ocr", "asr", "captions"):  # BM25 cần đủ các kho aux
                _src = PROJECT / "artifacts" / _aux
                if _aux == "captions" or not _src.is_dir():
                    continue
                for _try in (1, 2, 3):
                    shutil.copytree(_src, _la / _aux, dirs_exist_ok=True)
                    # round-57: listing Drive có thể TRẢ THIẾU y hệt copy —
                    # neo số cần vào danh sách video, không tin listing suông.
                    _need = max(len(list(_src.glob("*.json"))), len(_vids))
                    _got = len(list((_la / _aux).glob("*.json")))
                    if _got >= _need:
                        print(f"   staging {_aux}: {_got}/{_need} file")
                        break
                    print(f"   staging {_aux} thiếu ({_got}/{_need}) — thử lại {_try}/3 ...")
                    _tm.sleep(60)
                else:
                    raise RuntimeError(
                        f"Kéo kho {_aux} về máy mãi vẫn thiếu — Drive trục trặc; "
                        "chạy lại ô này sau ít phút (khóa sẽ tự nhả).")
            _touch_lock()                # nhịp tim khóa trước bước rebuild dài
            _run(REPO_DIR / "scripts" / "03_build_aux_indexes.py",
                 "--text-index", "--force-text-index")
            _ensure_drive()              # round-57: mount phải sống trước hoán đổi
            _touch_lock()
            _drv_job = PROJECT / "artifacts" / "captions"
            _bak = PROJECT / "artifacts" / "captions-stride4-backup"
            if _drv_job.exists() and not _bak.exists():
                _drv_job.rename(_bak)
                print("Đã cất bản cũ →", _bak)
            for _d in ("captions", "text_index"):
                _dst = PROJECT / "artifacts" / _d
                if _dst.exists():
                    shutil.rmtree(_dst)
                shutil.copytree(_la / _d, _dst)
                # Round-56: upload cũng đếm lại — copy bù nếu Drive nuốt thiếu
                _n_src = sum(1 for _q in (_la / _d).rglob("*") if _q.is_file())
                _n_dst = sum(1 for _q in _dst.rglob("*") if _q.is_file())
                if _n_dst < _n_src:
                    shutil.copytree(_la / _d, _dst, dirs_exist_ok=True)
                    _n_dst = sum(1 for _q in _dst.rglob("*") if _q.is_file())
                print(f"   → Drive: {_dst} ({_n_dst}/{_n_src} file)")
            _done_mark.write_text(_tm.strftime("%Y-%m-%d %H:%M"), encoding="utf-8")
        except BaseException:
            # round-57: chỉ nhả khóa nếu vẫn là khóa CỦA MÌNH — thua cuộc đua
            # thì tuyệt đối không được gỡ khóa của phiên thắng.
            try:
                if _lock.read_text(encoding="utf-8").strip() == _token:
                    _lock.unlink(missing_ok=True)   # nhả khóa để chạy lại được ngay
            except OSError:
                pass
            raise
        print("XONG TOÀN BỘ — artifacts đã nâng cấp. Đo lại bằng nb04 (L2).")
else:
    print("Kho chưa đủ — chờ các shard khác (hoặc chạy lại phiên để nối tiếp).")
'''

NB6D_TITLE = r'''
# 🧹 Core Vision Perfect V1 — 06d · Captions NGƯỜI QUÉT DỌN (phiên thứ 4 tùy chọn)

Chạy SONG SONG với 06a/b/c: đi NGƯỢC từ cuối danh sách, mỗi lượt nhận một mẻ
nhỏ những video CHƯA AI làm; giữa các mẻ tự làm mới cái nhìn từ kho chung nên
gần như không bao giờ trùng việc với 3 ca xuôi. Bốn mũi khoan gặp nhau ở giữa
là cả gia đình về đích. Cùng kho chung, cùng khóa finalize, cùng resume — an
toàn y hệt các ca kia; mở bất cứ lúc nào SAU khi kho chung đã tồn tại.
'''

NB7_TITLE = r'''
# 🔎 Core Vision Perfect V1 — 07 · OCR PaddleOCR (build chuyên biệt, shard song song)

Nâng cấp đôi mắt đọc chữ: đọc lại TOÀN BỘ keyframe bằng **PaddleOCR (PP-OCRv5,
tiếng Việt)** — thay kho EasyOCR cũ vốn nhiễu (bộ tinh chỉnh trọng số từng phải
dìm kênh OCR xuống gần 0). Chữ nung trên khung hình (banner, tiêu đề, chyron)
là tín hiệu KIS đắt giá nhất khi đọc CHUẨN dấu tiếng Việt. Nhanh hơn captions
nhiều lần (~2–5 giờ/ca GPU); cùng bộ giáp an toàn round-52..58 như 05/06.
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
    write_nb("01b_caption_boost_colab.ipynb", [
        md(NB01B_TITLE),
        code(CELL_PARAMS),
        code(CELL_MOUNT),
        code(CELL_REPO_DEPS),
        code(CELL_ENV_GPU),
        code(NB1_UNZIP),
        code(NB1_LOCAL_COPY),
        code(NB1_CATALOG),
        code(NB01B_CAPTIONS),
    ])
    write_nb("01c_asr_boost_colab.ipynb", [
        md(NB01C_TITLE),
        code(CELL_PARAMS),
        code(CELL_MOUNT),
        code(CELL_REPO_DEPS),
        code(CELL_ENV_GPU),
        code(NB1_UNZIP),
        code(NB1_LOCAL_COPY),
        code(NB1_CATALOG),
        code(NB01C_ASR),
    ])
    write_nb("04_lab_artifacts.ipynb", [
        md(NB4_TITLE),
        code(CELL_PARAMS),
        code(CELL_MOUNT),
        code(CELL_REPO_DEPS),
        code(CELL_ENV_GPU),
        code(NB1_LOCAL_COPY),
        code(NB3_OBJECTS),
        code(NB3_ARTIFACTS_LOCAL),
        code(LAB_GT),
        code(LAB_BENCH_FULL),
        code(LAB_TUNE),
        code(LAB_METACLIP),
        code(LAB_KEEPALIVE),
    ])
    # Round-51: shard notebooks đặt sẵn chỉ số — upload & Run all, zero chỉnh.
    for _fam, _title, _sweep in (("05", NB5_TITLE, NB5_SWEEP),
                                 ("06", NB6_TITLE, NB6_SWEEP)):
        for _i, _letter in enumerate("abc"):
            _src = _sweep.replace("SHARD_INDEX = 0", f"SHARD_INDEX = {_i}")
            _src = _src.replace("SHARD_TOTAL = 1", "SHARD_TOTAL = 3")
            _ttl = _title.replace("(build chuyên biệt, shard song song)",
                                  f"— CA {_i + 1}/3 (bản {_letter})")
            _job = "asr_large" if _fam == "05" else "caption_dense"
            write_nb(f"{_fam}{_letter}_{_job}_shard{_i + 1}.ipynb", [
                md(_ttl),
                code(CELL_PARAMS),
                code(CELL_MOUNT),
                code(CELL_REPO_DEPS),
                code(CELL_ENV_GPU),
                code(NB1_LOCAL_COPY),
                code(_src),
                code(LAB_KEEPALIVE),
            ])
    # Round-58: 06d "người quét dọn" — phiên thứ 4 TÙY CHỌN cho gia đình 06.
    # Đi NGƯỢC danh sách theo mẻ 12 video CHƯA AI làm, làm mới cái nhìn từ kho
    # chung giữa các mẻ (cộng hưởng với chiều KÉO mới trong syncer) → không
    # trùng việc với 3 ca xuôi; bốn mũi khoan gặp nhau ở giữa là xong sớm.
    _d_run_old = '''threading.Thread(target=_syncer, daemon=True).start()
# stride 1: caption MỌI keyframe (kho cũ stride 4 chỉ phủ ~1/4)
_run(REPO_DIR / "scripts" / "03_build_aux_indexes.py", "--captions", "--caption-stride", "1",
     "--videos", *_my)
_stop_sync = True'''
    _d_run_new = '''threading.Thread(target=_syncer, daemon=True).start()
# ── Round-58: vòng quét NGƯỢC — mỗi mẻ 12 video CHƯA AI làm ──
while True:
    try:
        _ensure_drive()
        for _f in _partial.glob("*.json"):       # làm mới cái nhìn từ kho chung
            _d = _job_local / _f.name
            if not _d.exists():
                shutil.copy2(_f, _d)
        _prune_staging()
    except OSError as _e:
        print(f"⚠ làm mới kho lỗi: {_e!r} — thử lại sau 30s", flush=True)
        _tm.sleep(30)
        continue
    _todo = [_v for _v in reversed(_vids)
             if not (_job_local / (_v + ".json")).is_file()]
    if not _todo:
        print("🧹 Kho đã phủ kín — người quét dọn hết việc.", flush=True)
        break
    _batch = _todo[:12]
    print(f"🧹 còn {len(_todo)} video thiếu — nhận mẻ {len(_batch)}: "
          f"{_batch[0]} … {_batch[-1]}", flush=True)
    _run(REPO_DIR / "scripts" / "03_build_aux_indexes.py",
         "--captions", "--caption-stride", "1", "--videos", *_batch)
    try:
        _ensure_drive()
        for _v in _batch:                        # đẩy NGAY mẻ vừa xong
            _f = _job_local / (_v + ".json")
            if _f.is_file():
                _d = _partial / _f.name
                if not _d.exists() or _d.stat().st_size != _f.stat().st_size:
                    shutil.copy2(_f, _d)
    except OSError:
        pass                                     # syncer 10 phút sẽ đẩy bù
_stop_sync = True'''
    _d_src = NB6_SWEEP
    for _old, _new in (
        ("# ── ⚒ Captions Vintern DÀY — stride 1, mọi keyframe (shard song song) ──",
         "# ── 🧹 Captions Vintern DÀY — NGƯỜI QUÉT DỌN (phiên thứ 4, đi ngược) ──"),
        ("SHARD_INDEX = 0\nSHARD_TOTAL = 1",
         "SHARD_INDEX = 3                # quét dọn: KHÔNG BAO GIỜ tạo kho chung\nSHARD_TOTAL = 4"),
        ('_my = _vids[SHARD_INDEX::SHARD_TOTAL]\n'
         'print(f"Shard {SHARD_INDEX + 1}/{SHARD_TOTAL}: {len(_my)}/{len(_vids)} video")',
         'print(f"Người quét dọn: phủ nốt phần thiếu của {len(_vids)} video (đi ngược từ cuối)")'),
        (_d_run_old, _d_run_new),
    ):
        assert _old in _d_src, f"NB6D surgery mất mốc: {_old[:50]!r}"
        _d_src = _d_src.replace(_old, _new)
    write_nb("06d_caption_dense_sweeper.ipynb", [
        md(NB6D_TITLE),
        code(CELL_PARAMS),
        code(CELL_MOUNT),
        code(CELL_REPO_DEPS),
        code(CELL_ENV_GPU),
        code(NB1_LOCAL_COPY),
        code(_d_src),
        code(LAB_KEEPALIVE),
    ])
    # Round-59: gia đình 07 — OCR PaddleOCR đọc lại MỌI keyframe. Sinh từ
    # NB6_SWEEP bằng phẫu thuật có assert (kế thừa nguyên bộ giáp round-52..58);
    # kho chung MỚI: ocr-v2-partial; bản EasyOCR cũ cất thành ocr-easyocr-backup.
    _7_install = '''# ── Cài PaddleOCR 3.x GPU từ index CHÍNH THỨC + smoke test ──────────────
# Round-60 (audit): PyPI đóng băng paddlepaddle-gpu ở 2.6.2/CUDA-10.2 — wheel
# GPU 3.x chỉ nằm trên index riêng của Paddle, chọn theo kiến trúc GPU. Gate
# GPU phải chạy PHÉP TÍNH THẬT (cờ compile-time không lộ wheel thiếu kernel
# cho GPU đời mới). Smoke fail trên GPU → tự thử đường CPU (pin 3.2.* né bug
# oneDNN/PIR của 3.3) trước khi bỏ cuộc.
import torch as _th
_sm = _th.cuda.get_device_capability(0) if _th.cuda.is_available() else (0, 0)
_idx = ("https://www.paddlepaddle.org.cn/packages/stable/cu129/" if _sm[0] >= 12
        else "https://www.paddlepaddle.org.cn/packages/stable/cu126/")

def _pip(*_a):
    # round-64: subprocess thừa kế fd của kernel → output rơi vào runtime log
    # (bài round-47) — phải bắt và in TAIL ra cell, nhất là khi lỗi.
    _r = subprocess.run([sys.executable, "-m", "pip", *_a],
                        capture_output=True, text=True)
    if _r.returncode != 0:
        print("   pip lỗi:", ((_r.stderr or _r.stdout) or "").strip()[-500:],
              flush=True)
    return _r.returncode

_kf_smoke = None
for _kd in (Path("/content/data/keyframes"), PROJECT / "data" / "keyframes"):
    if _kd.is_dir():
        _kf_smoke = next(iter(_kd.glob("*/*.jpg")), None)
        if _kf_smoke:
            break
assert _kf_smoke is not None, (
    "Không thấy keyframe nào (local lẫn Drive) — chạy lại cell materialize.")

def _smoke_rc():
    # round-64: BẮT output — không capture thì lỗi rơi vào runtime log vô hình
    # (round-47) và ta mù tịt lý do smoke fail.
    _r = subprocess.run(
        [sys.executable, "-c",
         "import os; os.environ['CVP_OCR__ENGINE'] = 'paddle'; "
         "from cvp.config import load_settings; "
         "from cvp.auxindex.ocr import _build_engine; "
         "eng = _build_engine(load_settings()); "
         f"print('SMOKE OCR:', repr(eng.read({str(_kf_smoke)!r})[:150]))"],
        capture_output=True, text=True)
    if _r.returncode != 0:
        print("   smoke lỗi chi tiết:",
              ((_r.stderr or _r.stdout) or "(im lặng)").strip()[-700:], flush=True)
    else:
        print("  ", (_r.stdout or "").strip()[-200:], flush=True)
    return _r.returncode

print(f"Cài PaddleOCR GPU (wheel {_idx.rsplit('/', 2)[-2]}) ...", flush=True)
# Round-65 (live 07a, traceback bắt được nhờ round-64): wheel GPU của Paddle
# kéo theo bộ nvidia-* CŨ HƠN đè lên đúng bộ torch Colab đang dùng → torch
# chết ở MỌI process mới ("undefined symbol: ncclCommShrink") → smoke fail cả
# GPU lẫn CPU (chuỗi import paddleocr→paddlex→modelscope→torch). Luật sắt
# "không đụng torch Colab" áp cả GIÁN TIẾP: cài paddle GPU với --no-deps
# (dùng chung bộ nvidia-* mới hơn của torch — tương thích xuôi chiều) và tự
# bù các dep python thuần vô hại.
_rc0 = _pip("install", "-q", "--no-cache-dir", "--no-deps",
            "paddlepaddle-gpu==3.2.*", "--extra-index-url", _idx)
_pip("install", "-q", "--no-cache-dir", "decorator", "astor", "opt_einsum",
     "protobuf", "httpx", "typing_extensions")
for _try in (1, 2):                      # round-66: cây dep ~170 gói — retry 1 lần
    _rc1 = _pip("install", "-q", "--no-cache-dir", "paddleocr>=3.0,<4")
    if _rc1 == 0:
        break
    print(f"⚠ cài paddleocr lỗi (lần {_try}) — thử lại sau 20s ...", flush=True)
    _tm.sleep(20)
if _rc1 != 0:
    raise RuntimeError("Cài paddleocr thất bại sau 2 lần — mạng trục trặc; "
                       "chạy lại ô này.")
# Chốt chặn round-65: torch phải còn SỐNG trong process mới — chính là vụ tai
# nạn vừa rồi. Hỏng là dừng ngay tại đây, không đốt thêm phút nào.
# round-66: máy KHÔNG có GPU (hết quota, chọn nhầm CPU) thì chỉ kiểm import —
# đừng chẩn oan "Paddle làm hỏng torch" rồi bắt người dùng đổi máy vô ích.
_th_code = ("import torch; assert torch.cuda.is_available(); "
            "print(torch.zeros(2).cuda().sum().item())") if _sm != (0, 0) else (
    "import torch; print(torch.zeros(2).sum().item())")
_th = subprocess.run([sys.executable, "-c", _th_code],
                     capture_output=True, text=True)
if _th.returncode != 0:
    raise RuntimeError(
        "torch của Colab đã bị bộ cài Paddle làm hỏng — máy ảo này kẹt vĩnh "
        "viễn: " + ((_th.stderr or "").strip()[-300:]) +
        " → Runtime ▸ Disconnect and delete runtime rồi Run all lại máy mới.")
print("✅ torch nguyên vẹn sau khi cài Paddle.", flush=True)
_probe = subprocess.run(
    [sys.executable, "-c",
     "import paddle; paddle.set_device('gpu'); "
     "x = paddle.ones([64, 64]); print(float((x @ x).sum()))"],
    capture_output=True, text=True)
_gpu_ok = _rc0 == 0 and _probe.returncode == 0
if _rc0 == 0 and _probe.returncode != 0:
    # round-61: đừng nuốt chứng cứ — dòng lỗi này phân biệt "wheel thiếu
    # kernel cho GPU này" với "driver/cuDNN trục trặc".
    print("   probe GPU lỗi:", (_probe.stderr or "").strip()[-400:], flush=True)
# Round-61: smoke fail KHÔNG đồng nghĩa GPU hỏng (hay gặp: mạng tải model
# PP-OCR chớp nhoáng) — thử lại 1 lần trước khi hạ cấp CPU, kẻo phiên A100
# âm thầm chạy CPU chậm 10-30 lần.
_ok = False
if _gpu_ok:
    for _try in (1, 2):
        if _smoke_rc() == 0:
            _ok = True
            break
        print(f"⚠ smoke GPU lần {_try} lỗi (mạng tải model?) — thử lại sau 30s ...",
              flush=True)
        _tm.sleep(30)
if not _ok:
    print("⚠ Paddle GPU không chạy được trên máy này — chuyển bản CPU "
          "(chậm hơn nhưng vẫn về đích).", flush=True)
    _pip("uninstall", "-q", "-y", "paddlepaddle-gpu")
    _pip("install", "-q", "--no-cache-dir", "paddlepaddle==3.2.*")
    if _smoke_rc() != 0:
        raise RuntimeError("PaddleOCR smoke test THẤT BẠI cả GPU lẫn CPU — dừng "
                           "TRƯỚC khi tốn giờ GPU. Chụp log ô này gửi Claude.")
print("✅ PaddleOCR sẵn sàng.", flush=True)
threading.Thread(target=_syncer, daemon=True).start()
# Round-59: PaddleOCR đọc tiếng Việt chuẩn hơn hẳn EasyOCR (kho cũ) — kỳ vọng
# lật kênh OCR từ tín hiệu nhiễu (tuner từng dìm 0.35 → 0.01) thành vũ khí.
os.environ["CVP_OCR__ENGINE"] = "paddle"
_run(REPO_DIR / "scripts" / "03_build_aux_indexes.py", "--ocr",
     "--videos", *_my)
_stop_sync = True'''
    _7_src = NB6_SWEEP
    for _old, _new in (
        ("# ── ⚒ Captions Vintern DÀY — stride 1, mọi keyframe (shard song song) ──",
         "# ── 🔎 OCR PaddleOCR tiếng Việt — đọc lại MỌI keyframe (shard song song) ──"),
        ("captions-dense-partial", "ocr-v2-partial"),
        ('_job_local = _la / "captions"', '_job_local = _la / "ocr"'),
        (_d_run_old, _7_install),
        ('''                _run(REPO_DIR / "scripts" / "03_build_aux_indexes.py",
                     "--captions", "--caption-stride", "1",
                     "--videos", *_still)''',
         '''                _run(REPO_DIR / "scripts" / "03_build_aux_indexes.py",
                     "--ocr",
                     "--videos", *_still)'''),
        ('if _aux == "captions" or not _src.is_dir():',
         'if _aux == "ocr" or not _src.is_dir():'),
        ('_drv_job = PROJECT / "artifacts" / "captions"',
         '_drv_job = PROJECT / "artifacts" / "ocr"'),
        ('_bak = PROJECT / "artifacts" / "captions-stride4-backup"',
         '_bak = PROJECT / "artifacts" / "ocr-easyocr-backup"'),
        ('for _d in ("captions", "text_index"):',
         'for _d in ("ocr", "text_index"):'),
    ):
        assert _old in _7_src, f"NB7 surgery mất mốc: {_old[:50]!r}"
        _7_src = _7_src.replace(_old, _new)
    for _i, _letter in enumerate("abc"):
        _src = _7_src.replace("SHARD_INDEX = 0", f"SHARD_INDEX = {_i}")
        _src = _src.replace("SHARD_TOTAL = 1", "SHARD_TOTAL = 3")
        _ttl = NB7_TITLE.replace("(build chuyên biệt, shard song song)",
                                 f"— CA {_i + 1}/3 (bản {_letter})")
        write_nb(f"07{_letter}_ocr_paddle_shard{_i + 1}.ipynb", [
            md(_ttl),
            code(CELL_PARAMS),
            code(CELL_MOUNT),
            code(CELL_REPO_DEPS),
            code(CELL_ENV_GPU),
            code(NB1_LOCAL_COPY),
            code(_src),
            code(LAB_KEEPALIVE),
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
        code(NB1_LOCAL_COPY),
        code(NB3_OBJECTS),
        code(NB3_ARTIFACTS_LOCAL),
        code(NB3_ENGINE),
        code(NB3_QUERIES),
        code(NB3_TRAKE_AVS),
        code(NB3_SUBMISSION),
        code(NB3_PACKAGE),
        code(NB3_RUN_PACK),
        code(NB3_SCORE_GT),
        code(NB3_AUTO_AGENT),
        code(NB3_UI),
        code(NB3_FASTUI),
        code(NB3_KEEPALIVE),
    ])


if __name__ == "__main__":
    main()
