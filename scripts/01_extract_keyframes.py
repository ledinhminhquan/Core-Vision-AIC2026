"""Self-extract keyframes + map-keyframes for raw videos without organiser
keyframes (K-batches). TransNetV2 → PySceneDetect → fixed-window fallback."""

import argparse

from _bootstrap import init

from cvf.data.extraction import extract_missing


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--settings", default=None)
    args = ap.parse_args()

    settings = init(args.settings)
    n = extract_missing(settings)
    print(f"Extracted keyframes for {n} videos. Now re-run 00_build_catalog.py --force")


if __name__ == "__main__":
    main()
