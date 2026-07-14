"""Assemble the Vietnamese caption↔embedding training set (train_data/)."""

import argparse

from _bootstrap import init

from cvf.training.build_dataset import build_training_set


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--settings", default=None)
    ap.add_argument("--model-key", default="siglip2", help="embedding store to pair captions with")
    args = ap.parse_args()

    settings = init(args.settings)
    n_train, n_val = build_training_set(settings, model_key=args.model_key)
    print(f"Training set ready: {n_train} train / {n_val} val pairs")


if __name__ == "__main__":
    main()
