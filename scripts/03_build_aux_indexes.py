"""Build auxiliary text indexes: OCR (keyframes), ASR (raw videos), captions.

All resumable per video — safe to interrupt and re-run. After the aux
artifacts are written this also persists the BM25 text index
(``artifacts/text_index/`` — see cvp.index.text_store) so query-time BM25 is
candidate-restricted instead of rescanning artifacts on every engine start.
"""

import argparse

from _bootstrap import init

from cvp.config import Settings
from cvp.data.catalog import KeyframeCatalog


def build_text_index(settings: Settings, catalog: KeyframeCatalog, force: bool = False) -> list[str]:
    """Persist BM25 stats for every field whose artifacts exist (incl. metadata).

    Idempotent: skipped when the stored meta signature already matches the
    catalog, unless ``force``. Returns per-field summaries ([] when skipped).
    """
    from cvp.index.text_store import TextIndexStore
    from cvp.search.text_signals import ALL_FIELDS, collect_field_documents

    signature = catalog.signature()
    if not force and TextIndexStore.signature_matches(settings.paths.artifacts_root, signature):
        return []
    built: list[str] = []
    for field in ALL_FIELDS:
        collected = collect_field_documents(settings, catalog, field)
        if collected is None:  # artifacts absent — in-memory fallback stays available
            continue
        keys, texts = collected
        f = TextIndexStore.build(field, keys, texts, settings.paths.artifacts_root, signature)
        built.append(f"{field}={f.n_docs} docs")
    return built


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--settings", default=None)
    ap.add_argument("--ocr", action="store_true")
    ap.add_argument("--asr", action="store_true")
    ap.add_argument("--captions", action="store_true")
    ap.add_argument("--caption-stride", type=int, default=1, help="caption every Nth keyframe")
    ap.add_argument("--videos", nargs="*", default=None, help="restrict to these video ids")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--text-index", action="store_true",
                    help="(re)build only the persisted BM25 text index")
    ap.add_argument("--force-text-index", action="store_true",
                    help="rebuild the text index even when its signature matches")
    ap.add_argument("--objects-index", action="store_true",
                    help="fold per-keyframe objects JSONs into one parquet "
                         "(ObjectBooster auto-prefers it; big win on Drive)")
    args = ap.parse_args()

    settings = init(args.settings)
    catalog = KeyframeCatalog(settings)
    catalog.load()

    if not (args.ocr or args.asr or args.captions or args.text_index
            or args.force_text_index or args.objects_index):
        ap.error("choose at least one of --ocr --asr --captions --text-index --objects-index")

    processed = 0
    if args.ocr:
        from cvp.auxindex.ocr import ocr_all_keyframes

        n = ocr_all_keyframes(settings, catalog, videos=args.videos, overwrite=args.overwrite)
        processed += n
        print(f"OCR: processed {n} videos")
    if args.asr:
        from cvp.auxindex.asr import asr_all_videos

        n = asr_all_videos(settings, catalog, videos=args.videos, overwrite=args.overwrite)
        processed += n
        print(f"ASR: processed {n} videos")
    if args.captions:
        from cvp.auxindex.captioner import caption_all_keyframes

        n = caption_all_keyframes(
            settings, catalog, videos=args.videos, stride=args.caption_stride, overwrite=args.overwrite
        )
        processed += n
        print(f"Captions: processed {n} videos")

    if args.objects_index:
        from cvp.data.objects_compact import build_objects_index

        out = build_objects_index(settings, catalog, overwrite=args.overwrite)
        print(f"Objects index: {out}")

    # Text index: refresh whenever this run changed artifacts (processed > 0)
    # or on --force-text-index; otherwise skip when the signature matches.
    built = build_text_index(settings, catalog, force=args.force_text_index or processed > 0)
    if built:
        print("Text index: built [" + ", ".join(built) + "]")
    else:
        print("Text index: up-to-date (signature match) — use --force-text-index to rebuild")


if __name__ == "__main__":
    main()
