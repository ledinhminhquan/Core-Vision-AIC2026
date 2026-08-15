"""Self-extract keyframes + map-keyframes for raw videos without organiser
keyframes (K-batches). TransNetV2 → PySceneDetect → fixed-window fallback."""

import argparse

from _bootstrap import init

from cvp.data.extraction import extract_missing


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--settings", default=None)
    ap.add_argument("--overwrite", action="store_true",
                    help="force re-extraction (e.g. after changing "
                         "CVP_EXTRACTION__SHOT_POSITIONS for denser keyframes). "
                         "Organiser keyframes/map csvs stay protected — scope with "
                         "--video, and add --force-organiser only if you REALLY "
                         "mean to destroy official data")
    ap.add_argument("--video", action="append", default=None, metavar="VIDEO_ID",
                    help="limit to these video ids (repeatable) — ALWAYS use this "
                         "with --overwrite")
    ap.add_argument("--force-organiser", action="store_true",
                    help="with --overwrite: allow replacing ORGANISER keyframes/"
                         "map csvs with approximate self-extracted output "
                         "(desyncs official clip-features/objects — last resort)")
    args = ap.parse_args()

    settings = init(args.settings)
    n = extract_missing(settings, overwrite=args.overwrite, only=args.video,
                        force_organiser=args.force_organiser)
    print(f"Extracted keyframes for {n} videos. Now re-run 00_build_catalog.py --force")


if __name__ == "__main__":
    main()
