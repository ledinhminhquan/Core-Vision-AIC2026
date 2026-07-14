"""Scan keyframes + map-keyframes → artifacts/catalog/manifest.parquet."""

import argparse

from _bootstrap import init

from cvp.data.catalog import KeyframeCatalog


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--settings", default=None, help="path to settings.yaml")
    ap.add_argument("--force", action="store_true", help="rebuild even if up-to-date")
    args = ap.parse_args()

    settings = init(args.settings)
    catalog = KeyframeCatalog(settings)
    df = catalog.build(force=args.force)
    print(
        f"Catalog: {len(df)} keyframes, {df['video_id'].nunique()} videos, "
        f"{int(df['has_map'].sum())} frames with map-keyframes"
    )


if __name__ == "__main__":
    main()
