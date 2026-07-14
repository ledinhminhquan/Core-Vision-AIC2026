"""Compare the fine-tuned Vietnamese tower vs the zero-shot baseline on the
validation split (text→image R@1/5/10, MRR, MedR)."""

import argparse

from _bootstrap import init


def _eval(settings, name: str) -> dict[str, float]:
    # Heavy imports live here so `--help` works on the documented torch-less
    # [search,dev] laptop install (round-3 fix L-R3-4 — same convention as
    # scripts/23_dump_signals.py).
    import numpy as np

    from cvp.eval.metrics import retrieval_metrics
    from cvp.models.registry import build_model
    from cvp.training.datamodule import TextImageEmbedDataset

    ds = TextImageEmbedDataset(settings, "val", word_dropout=0.0)
    model = build_model(settings, name)
    text_vecs = model.encode_text(ds.captions)
    img = ds.embeds / np.maximum(np.linalg.norm(ds.embeds, axis=1, keepdims=True), 1e-9)
    return retrieval_metrics(text_vecs, img)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--settings", default=None)
    ap.add_argument("--baseline", default="siglip2")
    ap.add_argument("--candidate", default="finetuned")
    args = ap.parse_args()

    settings = init(args.settings)
    base = _eval(settings, args.baseline)
    cand = _eval(settings, args.candidate)
    print(f"{'metric':8} {'baseline':>10} {'finetuned':>10} {'delta':>8}")
    for k in base:
        print(f"{k:8} {base[k]:>10.4f} {cand[k]:>10.4f} {cand[k] - base[k]:>+8.4f}")


if __name__ == "__main__":
    main()
