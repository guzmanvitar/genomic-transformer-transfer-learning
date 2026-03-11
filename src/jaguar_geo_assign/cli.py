"""Command-line entry points for the bootstrap scaffold."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .baselines import BASELINE_EVALUATION_STAGE
from .config import (
    check_feline_pipeline_runtime,
    describe_experiment,
    describe_feline_pipeline,
    load_experiment_config,
    load_feline_pipeline_config,
)
from .pretrain import format_feline_pretrain_result, run_feline_pretrain_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="jaguar-geo-assign")
    subparsers = parser.add_subparsers(dest="command", required=True)

    pretrain = subparsers.add_parser("pretrain", help="Run the feline genomics pretraining data pipeline.")
    pretrain.add_argument("--config", type=Path, required=True, help="Path to the feline pipeline TOML config.")
    pretrain.add_argument(
        "--bcftools-executable",
        default="bcftools",
        help="Override the bcftools executable for smoke tests or custom installations.",
    )

    for command in ("fine-tune", "evaluate", "baseline-evaluate", "report"):
        stage = subparsers.add_parser(command, help=f"Scaffold the {command} stage.")
        stage.add_argument("--config", type=Path, help="Optional path to a TOML config.")

    validate = subparsers.add_parser("validate-config", help="Validate a bootstrap TOML config.")
    validate.add_argument("config", type=Path)

    describe = subparsers.add_parser("describe-experiment", help="Summarize a bootstrap TOML config.")
    describe.add_argument("config", type=Path)

    validate_pipeline = subparsers.add_parser(
        "validate-feline-config",
        help="Validate the feline genomics pipeline TOML contract.",
    )
    validate_pipeline.add_argument("config", type=Path)

    describe_pipeline = subparsers.add_parser(
        "describe-feline-config",
        help="Summarize the feline genomics pipeline TOML contract.",
    )
    describe_pipeline.add_argument("config", type=Path)

    runtime = subparsers.add_parser(
        "check-feline-runtime",
        help="Check external runtime dependencies for the feline genomics pipeline.",
    )
    runtime.add_argument("config", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.command == "pretrain":
        try:
            result = run_feline_pretrain_pipeline(
                args.config,
                bcftools_executable=args.bcftools_executable,
            )
        except (RuntimeError, ValueError) as error:
            print(str(error))
            return 1
        print(format_feline_pretrain_result(result))
        return 0

    if args.command == "validate-config":
        config = load_experiment_config(args.config)
        print(f"Config '{config.name}' is valid for the bootstrap scaffold.")
        return 0

    if args.command == "describe-experiment":
        print(describe_experiment(args.config))
        return 0

    if args.command == "validate-feline-config":
        config = load_feline_pipeline_config(args.config)
        print(f"Feline pipeline config '{config.name}' matches the approved contract.")
        return 0

    if args.command == "describe-feline-config":
        print(describe_feline_pipeline(args.config))
        return 0

    if args.command == "check-feline-runtime":
        try:
            config = check_feline_pipeline_runtime(args.config)
        except RuntimeError as error:
            print(str(error))
            return 1
        print(
            f"Runtime contract satisfied for '{config.name}': "
            f"{', '.join(config.runtime.external_tools)}"
        )
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