"""Round-42: push the read-only artifact subset the 24/7 practice Space needs
to a PRIVATE HF dataset repo.

Included: catalog/, indexes/, embeddings/ (SuperGlobal rerank), text_index/,
objects_index/, checkpoints/vi_siglip2_best/. Excluded: keyframes (thumbs repo
covers images), asr/, cache/, submissions/, logs.

Run inside the LIVE nb03 Colab session (env inherited → artifacts_root local):

    !python /content/Core-Vision_Perfect_V1/scripts/61_push_practice_artifacts.py \
        --repo <hf-user>/aic26-artifacts --token hf_xxx
"""

import argparse
import os
from pathlib import Path

from _bootstrap import init

SUBDIRS = ("catalog", "indexes", "embeddings", "text_index", "objects_index",
           "checkpoints")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", required=True, help="<user>/<name> — created PRIVATE")
    ap.add_argument("--token", default=None)
    args = ap.parse_args()

    settings = init(None)
    root = Path(settings.paths.artifacts_root)
    from huggingface_hub import HfApi

    api = HfApi(token=args.token or os.environ.get("HF_TOKEN"))
    api.create_repo(args.repo, repo_type="dataset", private=True, exist_ok=True)
    for d in SUBDIRS:
        src = root / d
        if not src.is_dir():
            print(f"  ⚠ thiếu {src} — bỏ qua")
            continue
        # The finetuned lane rides the siglip2 index/embeddings — other lanes'
        # gigabytes stay home.
        allow = ["*siglip2*"] if d in ("indexes", "embeddings") else None
        print(f"  {d}/ → {args.repo}" + (f" (chỉ {allow})" if allow else ""))
        api.upload_folder(repo_id=args.repo, repo_type="dataset",
                          folder_path=str(src), path_in_repo=d,
                          allow_patterns=allow)
    print("XONG — Space sẽ tải bộ này lúc khởi động bằng HF_TOKEN.")


if __name__ == "__main__":
    main()
