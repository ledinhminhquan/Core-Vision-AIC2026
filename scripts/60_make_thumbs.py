"""Round-42: build 320px webp thumbnails for EVERY catalog keyframe.

The grids (Colab tunnel AND the 24/7 practice Space) were shipping original
keyframes (~60-150KB each); a 320px webp is ~6-10KB — ~10x faster grids.
Layout: <out>/<video_id>/<n>.webp — one folder per video (~200 files), far
under the HF 10k-files-per-folder hard limit.

⚠ Competition keyframes are organiser-licensed data: the upload target MUST
be a PRIVATE dataset repo. The practice Space downloads it with HF_TOKEN and
serves the files itself behind the team login — nothing is ever public.

Run inside the LIVE nb03 Colab session (env + local keyframes inherited):

    !python /content/Core-Vision_Perfect_V1/scripts/60_make_thumbs.py \
        --out /content/drive/MyDrive/AIC2025/artifacts/thumbs

    # later, once, to publish for the 24/7 Space:
    !python .../60_make_thumbs.py --out .../artifacts/thumbs \
        --upload <hf-user>/aic26-thumbs --token hf_xxx
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from _bootstrap import init


def _one(job: tuple[str, str, int, int]) -> bool:
    src, dst, width, quality = job
    try:
        from PIL import Image

        p = Path(dst)
        if p.exists():
            return True
        img = Image.open(src).convert("RGB")
        img.thumbnail((width, width))
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp.webp")
        img.save(tmp, "WEBP", quality=quality, method=4)
        tmp.rename(p)                       # atomic-ish: resume-safe on Drive
        return True
    except Exception:  # noqa: BLE001 — one broken JPEG must not stop 177k
        return False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, help="thumbnail root (Drive artifacts/thumbs)")
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--quality", type=int, default=70)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--upload", default=None,
                    help="HF dataset repo id (<user>/<name>) — created PRIVATE")
    ap.add_argument("--token", default=None, help="HF write token (or env HF_TOKEN)")
    args = ap.parse_args()

    settings = init(None)
    from cvp.data.catalog import Catalog

    cat = Catalog(settings)
    df = cat.load()
    out = Path(args.out)
    jobs = [(str(r.path), str(out / str(r.video_id) / f"{int(r.n)}.webp"),
             args.width, args.quality) for r in df.itertuples()]
    print(f"{len(jobs)} keyframes → {out}")
    done = fail = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, ok in enumerate(ex.map(_one, jobs, chunksize=256), 1):
            done += ok
            fail += (not ok)
            if i % 20000 == 0:
                print(f"  {i}/{len(jobs)} (lỗi: {fail})")
    print(f"XONG: {done} thumbnail, {fail} lỗi")

    if args.upload:
        import os

        from huggingface_hub import HfApi

        token = args.token or os.environ.get("HF_TOKEN")
        api = HfApi(token=token)
        # PRIVATE — competition data must never be publicly hotlinkable.
        api.create_repo(args.upload, repo_type="dataset", private=True, exist_ok=True)
        api.upload_large_folder(repo_id=args.upload, repo_type="dataset",
                                folder_path=str(out))
        print(f"Đã upload lên dataset PRIVATE: {args.upload}")


if __name__ == "__main__":
    main()
