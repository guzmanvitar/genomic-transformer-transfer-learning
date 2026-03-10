"""Command-line entry points for the bootstrap scaffold."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .baselines import BASELINE_EVALUATION_STAGE
from .config import describe_experiment, load_experiment_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="jaguar-geo-assign")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in ("pretrain", "fine-tune", "evaluate", "baseline-evaluate", "report"):
        stage = subparsers.add_parser(command, help=f"Scaffold the {command} stage.")
        stage.add_argument("--config", type=Path, help="Optional path to a TOML config.")

    validate = subparsers.add_parser("validate-config", help="Validate a bootstrap TOML config.")
    validate.add_argument("config", type=Path)

    describe = subparsers.add_parser("describe-experiment", help="Summarize a bootstrap TOML config.")
    describe.add_argument("config", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.command == "validate-config":
        config = load_experiment_config(args.config)
        print(f"Config '{config.name}' is valid for the bootstrap scaffold.")
        return 0

    if args.command == "describe-experiment":
        print(describe_experiment(args.config))
        return 0

    config_name = None
    if args.config is not None:
        config = load_experiment_config(args.config)
        config_name = config.name

    print(f"{args.command} entry point scaffold is available.")
    if config_name:
        print(f"Loaded config: {config_name}")
        if args.command == "baseline-evaluate":
            print(
                "Deferred baseline stage is reserved for "
                f"{BASELINE_EVALUATION_STAGE} without enabling legacy execution."
            )
    print("Detailed pipeline implementation is intentionally deferred to later tasks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())