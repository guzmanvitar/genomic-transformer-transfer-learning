"""Command-line entry points for the jaguar-geo-assign toolkit.

This module wires every user-facing CLI sub-command to its backend
implementation.  ``build_parser`` defines the argument grammar while
``main`` dispatches to the correct pipeline stage.

**Fragility flag – dispatch pattern:**  ``main`` uses an explicit
``if``-chain rather than a command→callable mapping dict.  This is
intentional: each branch carries bespoke argument unpacking and
error-handling that would be obscured by a generic dispatcher.
Do **not** refactor into a lookup table without first auditing every
branch's error contract.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .baselines import BASELINE_EVALUATION_STAGE
from .config import (
    check_feline_pipeline_runtime,
    check_felid_foundation_pipeline_runtime,
    describe_experiment,
    describe_feline_pipeline,
    describe_felid_foundation_config,
    load_experiment_config,
    load_feline_pipeline_config,
    load_felid_foundation_pipeline_config,
)
from .pretrain import (
    acquire_felid_foundation_assemblies,
    format_feline_pretrain_result,
    format_felid_foundation_pretrain_result,
    run_feline_pretrain_pipeline,
    run_felid_foundation_pretrain,
)


def build_parser() -> argparse.ArgumentParser:
    """Construct the top-level argument parser with all sub-commands.

    The parser exposes the following sub-commands:

    * ``pretrain`` – run the feline genomics pretraining data pipeline.
    * ``fine-tune``, ``evaluate``, ``baseline-evaluate``, ``report`` –
      scaffold placeholders for later pipeline stages.
    * ``validate-config`` – validate a bootstrap TOML config.
    * ``describe-experiment`` – summarise a bootstrap TOML config.
    * ``validate-feline-config`` – validate a feline pipeline TOML contract.
    * ``describe-feline-config`` – summarise a feline pipeline TOML contract.
    * ``check-feline-runtime`` – check external runtime dependencies.
    * ``felid-foundation-pretrain`` – run the felid foundation pretraining pipeline.
    * ``acquire-felid-foundation-assemblies`` – download felid reference FASTAs.
    * ``validate-felid-foundation-config`` – validate a felid foundation config.
    * ``describe-felid-foundation-config`` – summarise a felid foundation config.
    * ``check-felid-foundation-runtime`` – check felid foundation runtime dependencies.

    Returns:
        A fully-configured :class:`argparse.ArgumentParser` ready for
        ``parse_args``.
    """
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

    # Felid foundation pipeline subcommands
    felid_pretrain = subparsers.add_parser(
        "felid-foundation-pretrain",
        help="Run the multi-species felid foundation pretraining pipeline.",
    )
    felid_pretrain.add_argument("config", type=Path, help="Path to the felid foundation pipeline TOML config.")

    felid_acquire = subparsers.add_parser(
        "acquire-felid-foundation-assemblies",
        help="Download all six approved felid reference FASTAs with integrity checks.",
    )
    felid_acquire.add_argument("config", type=Path, help="Path to the felid foundation pipeline TOML config.")

    felid_validate = subparsers.add_parser(
        "validate-felid-foundation-config",
        help="Validate the felid foundation pipeline TOML contract.",
    )
    felid_validate.add_argument("config", type=Path)

    felid_describe = subparsers.add_parser(
        "describe-felid-foundation-config",
        help="Summarize the felid foundation pipeline TOML contract.",
    )
    felid_describe.add_argument("config", type=Path)

    felid_runtime = subparsers.add_parser(
        "check-felid-foundation-runtime",
        help="Check external runtime dependencies for the felid foundation pipeline.",
    )
    felid_runtime.add_argument("config", type=Path)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse CLI arguments and dispatch to the appropriate pipeline stage.

    Each sub-command branch performs its own argument unpacking and
    error handling.  The explicit ``if``-chain is deliberate – see the
    module-level fragility note before refactoring.

    Args:
        argv: Command-line tokens to parse.  When *None* (the default),
            ``sys.argv[1:]`` is used via :mod:`argparse`.

    Returns:
        Exit code: ``0`` on success, ``1`` on handled errors.
    """
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
        try:
            config = load_feline_pipeline_config(args.config)
        except ValueError as error:
            print(str(error))
            return 1
        print(f"Feline pipeline config '{config.name}' matches the approved contract.")
        return 0

    if args.command == "describe-feline-config":
        try:
            print(describe_feline_pipeline(args.config))
        except ValueError as error:
            print(str(error))
            return 1
        return 0

    if args.command == "check-feline-runtime":
        try:
            config = check_feline_pipeline_runtime(args.config)
        except (RuntimeError, ValueError) as error:
            print(str(error))
            return 1
        print(
            f"Runtime contract satisfied for '{config.name}': "
            f"{', '.join(config.runtime.external_tools)}"
        )
        return 0

    if args.command == "felid-foundation-pretrain":
        try:
            result = run_felid_foundation_pretrain(args.config)
        except (RuntimeError, ValueError) as error:
            print(str(error))
            return 1
        print(format_felid_foundation_pretrain_result(result))
        return 0

    if args.command == "acquire-felid-foundation-assemblies":
        try:
            config = load_felid_foundation_pipeline_config(args.config)
            summary = acquire_felid_foundation_assemblies(config)
        except (RuntimeError, ValueError) as error:
            print(str(error))
            return 1
        print("Felid foundation assembly acquisition summary:")
        print(f"  Total bytes written: {summary.total_bytes_written}")
        print(f"  Skipped (checksum match): {summary.skipped_count}")
        print(f"  Redownloaded (checksum mismatch): {summary.redownloaded_count}")
        return 0

    if args.command == "validate-felid-foundation-config":
        try:
            config = load_felid_foundation_pipeline_config(args.config)
        except ValueError as error:
            print(str(error))
            return 1
        print(f"Felid foundation pipeline config '{config.name}' matches the approved contract.")
        return 0

    if args.command == "describe-felid-foundation-config":
        try:
            print(describe_felid_foundation_config(args.config))
        except ValueError as error:
            print(str(error))
            return 1
        return 0

    if args.command == "check-felid-foundation-runtime":
        try:
            config = check_felid_foundation_pipeline_runtime(args.config)
        except (RuntimeError, ValueError) as error:
            print(str(error))
            return 1
        print(
            f"Runtime contract satisfied for '{config.name}': "
            f"{', '.join(config.runtime.external_tools) if config.runtime.external_tools else 'no external tools required'}"
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