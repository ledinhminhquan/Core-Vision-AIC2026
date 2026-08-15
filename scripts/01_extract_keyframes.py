"""Self-extract keyframes + map-keyframes for raw videos without organiser
keyframes (K-batches). TransNetV2 → PySceneDetect → fixed-window fallback."""

import argparse

from _bootstrap import init

from cvp.data.extraction import extract_missing


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--settings", default=None)
    ap.add_argument("--overwrite", action="store_true",
                    help="force re-extraction of EVERY video (e.g. after changing "
                         "CVP_EXTRACTION__SHOT_POSITIONS for denser keyframes)")
    args = ap.parse_args()

    settings = init(args.settings)
    n = extract_missing(settings, overwrite=args.overwrite)
    print(f"Extracted keyframes for {n} videos. Now re-run 00_build_catalog.py --force")


if __name__ == "__main__":
    main()
