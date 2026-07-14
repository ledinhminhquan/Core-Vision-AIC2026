"""Build EXTRA Vietnamese caption parquets (KTVIC, UIT-ViIC) for the LiT fine-tune (GPU).

    python scripts/12_build_public_datasets.py [--datasets ktvic uit_viic] [--overwrite]

Downloads the public caption datasets from Hugging Face, embeds their images
with the frozen image tower (``embedding.model``, default SigLIP-2 — must
match the tower used for the corpus keyframes), and writes
``artifacts/train_data/public/extra_{name}.parquet``. Resumable. scripts/11
and the Colab training notebook auto-merge every parquet found in that folder.
"""

import argparse

from _bootstrap import init

from cvf.training.public_datasets import PUBLIC_DATASETS, build_all_public_parquets


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--settings", default=None)
    ap.add_argument("--datasets", nargs="+", choices=sorted(PUBLIC_DATASETS),
                    default=sorted(PUBLIC_DATASETS))
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    settings = init(args.settings)
    paths = build_all_public_parquets(names=args.datasets, settings=settings,
                                      batch_size=args.batch_size, overwrite=args.overwrite)
    for p in paths:
        print(f"Extra training parquet: {p}")


if __name__ == "__main__":
    main()
