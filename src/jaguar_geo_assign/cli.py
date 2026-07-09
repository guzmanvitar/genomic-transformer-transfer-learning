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
import logging
import traceback
from collections.abc import Sequence
from pathlib import Path

from .config import (
    check_felid_foundation_pipeline_runtime,
    describe_felid_foundation_config,
    load_felid_foundation_pipeline_config,
)
from .pretrain import (
    acquire_felid_foundation_assemblies,
    format_felid_foundation_pretrain_result,
    run_felid_foundation_pretrain,
)


def build_parser() -> argparse.ArgumentParser:
    """Construct the top-level argument parser with all sub-commands.

    Returns:
        A fully-configured :class:`argparse.ArgumentParser` ready for
        ``parse_args``.
    """
    parser = argparse.ArgumentParser(prog="jaguar-geo-assign")
    subparsers = parser.add_subparsers(dest="command", required=True)

    genotype_finetune = subparsers.add_parser(
        "genotype-finetune",
        help="Train genotype MLP with VES-based transfer learning (Option A).",
    )
    genotype_finetune.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the genotype fine-tuning TOML config.",
    )

    # Felid foundation pipeline subcommands
    felid_pretrain = subparsers.add_parser(
        "felid-foundation-pretrain",
        help="Run the multi-species felid foundation pretraining pipeline.",
    )
    felid_pretrain.add_argument(
        "config", type=Path, help="Path to the felid foundation pipeline TOML config."
    )

    felid_acquire = subparsers.add_parser(
        "acquire-felid-foundation-assemblies",
        help="Download all six approved felid reference FASTAs with integrity checks.",
    )
    felid_acquire.add_argument(
        "config", type=Path, help="Path to the felid foundation pipeline TOML config."
    )

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

    train_foundation = subparsers.add_parser(
        "train-felid-foundation",
        help=(
            "Train DNABERT-2 on the felid foundation tokenized corpus with continued pre-training."
        ),
    )
    train_foundation.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the felid foundation training TOML config.",
    )
    train_foundation.add_argument(
        "--integration-test",
        action="store_true",
        default=False,
        help="Run integration test mode (no HF Hub access needed for tiny model).",
    )

    jaguar_acquire = subparsers.add_parser(
        "acquire-jaguar-raw-data",
        help="Download jaguar VCF and location CSV to data/raw/ (or --output-dir).",
    )
    jaguar_acquire.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw"),
        help="Destination directory for the downloaded files (default: data/raw/).",
    )

    extract_windows = subparsers.add_parser(
        "extract-finetune-windows",
        help="Extract 512 bp locus-centered windows from jaguar VCF files.",
    )
    extract_windows.add_argument(
        "--reference-fasta",
        type=Path,
        required=True,
        help="Path to the DNA Zoo Panthera onca HiC reference FASTA.",
    )
    extract_windows.add_argument(
        "--vcf",
        type=Path,
        required=True,
        help="Path to the (possibly multi-sample) VCF file.",
    )
    extract_windows.add_argument(
        "--metadata-csv",
        type=Path,
        required=True,
        help="CSV with a sample_id column identifying samples to extract.",
    )
    extract_windows.add_argument(
        "--output-jsonl",
        type=Path,
        required=True,
        help="Output JSONL path for extracted FinetuneWindow records.",
    )

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
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

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
        tools_summary = (
            ", ".join(config.runtime.external_tools)
            if config.runtime.external_tools
            else "no external tools required"
        )
        print(f"Runtime contract satisfied for '{config.name}': {tools_summary}")
        return 0

    if args.command == "train-felid-foundation":
        # Lazy import to keep CLI startup fast and avoid importing torch/transformers
        # for unrelated subcommands
        from .pretrain.foundation_training import integration_test, run_felid_foundation_training

        try:
            if args.integration_test:
                integration_test(use_real_model=False)
            else:
                run_felid_foundation_training(args.config)
        except (RuntimeError, ValueError):
            traceback.print_exc()
            return 1
        return 0

    if args.command == "genotype-finetune":
        from .fine_tune.genotype_trainer import format_genotype_train_result, run_genotype_training

        try:
            result = run_genotype_training(args.config)
            print(format_genotype_train_result(result))
        except (RuntimeError, ValueError) as error:
            print(str(error))
            return 1
        return 0

    if args.command == "acquire-jaguar-raw-data":
        from .data.jaguar_raw_acquisition import JaguarRawAcquisitionError, acquire_jaguar_raw_data

        try:
            summary = acquire_jaguar_raw_data(args.output_dir)
        except JaguarRawAcquisitionError as error:
            print(str(error))
            return 1
        print("Jaguar raw data acquisition summary:")
        print(f"  Total bytes written: {summary.total_bytes_written}")
        print(f"  Skipped (already present): {summary.skipped_count}")
        return 0

    if args.command == "extract-finetune-windows":
        from .data.finetune_windows import extract_windows_for_samples

        try:
            result = extract_windows_for_samples(
                reference_fasta=args.reference_fasta,
                vcf=args.vcf,
                metadata_csv=args.metadata_csv,
                output_jsonl=args.output_jsonl,
            )
        except (RuntimeError, ValueError) as error:
            print(str(error))
            return 1
        print("Window extraction complete.")
        print(f"  Samples processed: {result.samples_processed}")
        print(f"  Samples skipped: {result.samples_skipped}")
        print(f"  Total windows written: {result.total_windows}")
        print(f"  Output: {result.output_path}")
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
