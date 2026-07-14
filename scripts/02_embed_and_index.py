"""Embed all keyframes with the configured model(s) and build FAISS indexes.

Uses the config's embedding.model (or every ensemble member with --all-members).
For `provided_clip32`, the organiser's precomputed features are ingested
directly — no GPU needed.
"""

import argparse

from _bootstrap import init

from cvp.data.catalog import KeyframeCatalog
from cvp.index.embedder import embed_all_keyframes, ingest_provided_features
from cvp.index.store import IndexStore
from cvp.models.registry import build_model, index_key_for


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--settings", default=None)
    ap.add_argument("--model", default=None, help="single backend name (default: config)")
    ap.add_argument("--all-members", action="store_true", help="process every ensemble member")
    ap.add_argument("--overwrite", action="store_true", help="re-embed everything")
    ap.add_argument("--force-index", action="store_true", help="rebuild FAISS even if fresh")
    args = ap.parse_args()

    settings = init(args.settings)
    catalog = KeyframeCatalog(settings)
    catalog.build()

    if args.all_members or (args.model or settings.embedding.model) == "ensemble":
        raw = settings.embedding.ensemble_members
    else:
        raw = [args.model or settings.embedding.model]
    # map to embedding spaces (finetuned → siglip2) and dedupe, keeping order
    names: list[str] = []
    for n in raw:
        k = index_key_for(n)
        if k not in names:
            names.append(k)
    for name in names:
        model_tag = None
        if name == "provided_clip32":
            ingest_provided_features(settings, catalog)
        else:
            model = build_model(settings, name)
            embed_all_keyframes(model, settings, catalog, overwrite=args.overwrite)
            model_tag = getattr(model, "model_tag", None) or model.key
            del model
        store = IndexStore(settings, name)
        store.build(catalog, force=args.force_index or args.overwrite, model_tag=model_tag)
        print(f"[{name}] index ready: {store.count()} vectors, dim {store.dim()}")


if __name__ == "__main__":
    main()
