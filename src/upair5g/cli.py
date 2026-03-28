from __future__ import annotations

import argparse

from .config import load_config
from .evaluation import evaluate_model
from .plotting import make_all_plots
from .training import train_model


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="UPAIR-5G pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name in ["train", "eval", "full", "smoke"]:
        sub = subparsers.add_parser(name)
        sub.add_argument("--config", required=True, type=str)
        if name == "eval":
            sub.add_argument("--checkpoint", default=None, type=str)

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    cfg = load_config(args.config)

    if args.command == "train":
        result = train_model(cfg)
        print(result)
        return

    if args.command == "eval":
        eval_result = evaluate_model(cfg, checkpoint_path=args.checkpoint)
        plot_result = make_all_plots(cfg)
        print({"eval": eval_result, "plots": plot_result})
        return

    if args.command in {"full", "smoke"}:
        train_result = train_model(cfg)
        eval_result = evaluate_model(cfg, checkpoint_path=train_result["checkpoint_path"])
        plot_result = make_all_plots(cfg)
        print({"train": train_result, "eval": eval_result, "plots": plot_result})
        return

    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
