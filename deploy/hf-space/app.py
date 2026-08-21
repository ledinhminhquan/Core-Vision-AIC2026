"""Round-42: entry của Web luyện tập 24/7 trên HF Space (CPU, không GPU).

Khởi động: tải 2 dataset PRIVATE (artifacts + thumbs) bằng HF_TOKEN, trỏ env
về đó, dựng engine CPU (finetuned text tower + FAISS + BM25 + objects;
Qwen/VLM rerank TẮT — cần GPU/ảnh gốc), rồi phục vụ đúng cvp.web.server
(SPA + đăng nhập đội + export). Cold start ~3-6 phút; sau đó search ~2-5s.
"""

import logging
import os

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("practice-space")

from huggingface_hub import snapshot_download  # noqa: E402

TOKEN = os.environ.get("HF_TOKEN")
ART_REPO = os.environ["ARTIFACTS_REPO"]
THUMBS_REPO = os.environ.get("THUMBS_REPO", "")

log.info("Tải artifacts từ %s ...", ART_REPO)
art = snapshot_download(ART_REPO, repo_type="dataset", token=TOKEN)
if THUMBS_REPO:
    log.info("Tải thumbnails từ %s ...", THUMBS_REPO)
    os.environ["CVP_WEB__THUMBS_DIR"] = snapshot_download(
        THUMBS_REPO, repo_type="dataset", token=TOKEN)

env = os.environ.setdefault
env("CVP_PATHS__ARTIFACTS_ROOT", art)
env("CVP_PATHS__DATA_ROOT", art)                  # không có data thô — vô hại
env("CVP_EMBEDDING__MODEL", "finetuned")
env("CVP_FINETUNED__CHECKPOINT", os.path.join(art, "checkpoints", "vi_siglip2_best"))
env("CVP_EMBEDDING__DEVICE", "cpu")
env("CVP_EMBEDDING__DTYPE", "fp32")
env("CVP_SEARCH__RERANKER", "none")               # cross-encoder cần GPU
env("CVP_SEARCH__VLM_RERANK", "false")            # cần ảnh gốc — Space chỉ có thumbs
env("CVP_SEARCH__LOW_CONFIDENCE_RETRY", "true")
env("CVP_QUERY__PROVIDER",
    "gemini" if os.environ.get("GEMINI_API_KEY") else "google")
env("CVP_VQA__PROVIDER", "none")                  # VQA cần ảnh gốc + GPU

from cvp.config import load_settings  # noqa: E402
from cvp.search.engine import SearchEngine  # noqa: E402
from cvp.web.server import create_team_app  # noqa: E402

settings = load_settings()
log.info("Dựng engine CPU (lần đầu tải model text ~2 phút) ...")
engine = SearchEngine(settings)
log.info("Engine sẵn sàng: %d keyframes", len(engine.catalog))

app = create_team_app(
    engine, settings,
    os.environ.get("TEAM_USER", "aic2026-222"),
    os.environ["TEAM_PASS"],
)
