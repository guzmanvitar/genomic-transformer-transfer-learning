"""Bootstrap config loading and validation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib

from .baselines import (
    BASELINE_EVALUATION_STAGE,
    DEFERRED_BASELINE_PROVIDER,
    SHARED_BASELINE_EXTENSION_POINT,
)
from .data.contracts import JAGUAR_METADATA_FIELDS
from .data.pipeline_contract import (
    APPROVED_BIOPROJECT_ACCESSION,
    APPROVED_REFERENCE_ASSEMBLY,
    DNABERT2_TOKENIZER_ID,
    DNABERT2_TOKENIZER_REVISION,
    EXPLICIT_CONSENSUS_POLICIES,
    GLOBAL_LOCUS_SPLIT_STRATEGY,
    POST_CONSENSUS_ALLOWED_ALPHABET,
    PRE_WINDOW_ASSIGNMENT_STAGE,
    REFERENCE_BASELINE_POLICY,
    REQUIRED_EXTERNAL_TOOLS,
    assert_external_tools_available,
)

REQUIRED_STAGES = ("evaluate", BASELINE_EVALUATION_STAGE, "report")


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    description: str
    requires_private_data: bool
    primary_task: str
    primary_metric: str
    split_unit: str
    jaguar_metadata_fields: tuple[str, ...]
    stages: tuple[str, ...]
    baseline_stage: str
    baseline_provider: str
    baseline_enabled: bool
    baseline_extension_point: str


@dataclass(frozen=True)
class PipelinePathsConfig:
    reference_fasta: Path
    sample_manifest: Path
    source_vcf: Path
    raw_dir: Path
    processed_dir: Path
    baseline_dir: Path
    artifact_dir: Path
    report_dir: Path


@dataclass(frozen=True)
class ConsensusConfig:
    assembly: str
    require_assembly_match: bool
    require_contig_match: bool
    mask_symbol: str
    homozygous_reference: str
    homozygous_alternate: str
    heterozygous: str
    multiallelic: str
    filtered: str
    missing: str
    indel: str


@dataclass(frozen=True)
class WindowingConfig:
    context_window: int
    window_overlap: int
    max_ambiguous_fraction: float
    drop_short_sequences: bool


@dataclass(frozen=True)
class SplitConfig:
    strategy: str
    locus_key_fields: tuple[str, ...]
    locus_block_size: int
    assignment_stage: str
    evaluation_target: str
    baseline_policy: str


@dataclass(frozen=True)
class TokenizerConfig:
    identifier: str
    revision: str
    allowed_alphabet: tuple[str, ...]
    unsupported_symbol_policy: str
    max_position_embeddings: int


@dataclass(frozen=True)
class ExportConfig:
    format: str
    access_pattern: str
    row_group_size: int
    deterministic_partition_keys: tuple[str, ...]
    preserve_raw_windows: bool
    preserve_sequence_hashes: bool
    preserve_coordinates: bool
    sequence_hash_algorithm: str


@dataclass(frozen=True)
class RuntimeConfig:
    external_tools: tuple[str, ...]


@dataclass(frozen=True)
class FelinePipelineConfig:
    name: str
    description: str
    project_accession: str
    paths: PipelinePathsConfig
    consensus: ConsensusConfig
    windowing: WindowingConfig
    split: SplitConfig
    tokenizer: TokenizerConfig
    export: ExportConfig
    runtime: RuntimeConfig


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    raw = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    experiment = raw["experiment"]
    data = raw["data"]
    primary_task = raw["tasks"]["primary"]
    stages = tuple(raw["stages"]["order"])
    baseline = raw["baseline"]
    metadata_fields = tuple(data["jaguar_metadata_fields"])
    baseline_stage = baseline["stage"]

    if metadata_fields != JAGUAR_METADATA_FIELDS:
        raise ValueError(
            "jaguar_metadata_fields must exactly match the bootstrap metadata contract"
        )
    if primary_task["kind"] != "coordinate_regression":
        raise ValueError("bootstrap configs must use coordinate_regression as the primary task")
    if primary_task["primary_metric"] != "median_geodesic_error_km":
        raise ValueError("bootstrap configs must use median_geodesic_error_km as the primary metric")
    if data["split_unit"] not in {"sample_id", "individual_id"}:
        raise ValueError("split_unit must be sample_id or individual_id")
    if len(stages) != len(set(stages)) or not stages:
        raise ValueError("stages.order must be non-empty and contain unique stage names")
    if any(stage not in stages for stage in REQUIRED_STAGES):
        raise ValueError("stages.order must include evaluate, baseline_evaluate, and report")
    if baseline_stage != BASELINE_EVALUATION_STAGE:
        raise ValueError("bootstrap baseline stage must remain baseline_evaluate")
    if baseline["provider"] != DEFERRED_BASELINE_PROVIDER:
        raise ValueError("bootstrap baseline provider must stay on the deferred legacy extension")
    if baseline["enabled"]:
        raise ValueError("bootstrap baseline execution must remain disabled")
    if baseline["extension_point"] != SHARED_BASELINE_EXTENSION_POINT:
        raise ValueError("bootstrap baseline extension point must remain shared_split_metric_report_contract")

    return ExperimentConfig(
        name=experiment["name"],
        description=experiment["description"],
        requires_private_data=bool(experiment.get("requires_private_data", False)),
        primary_task=primary_task["kind"],
        primary_metric=primary_task["primary_metric"],
        split_unit=data["split_unit"],
        jaguar_metadata_fields=metadata_fields,
        stages=stages,
        baseline_stage=baseline_stage,
        baseline_provider=baseline["provider"],
        baseline_enabled=bool(baseline["enabled"]),
        baseline_extension_point=baseline["extension_point"],
    )


def load_feline_pipeline_config(path: str | Path) -> FelinePipelineConfig:
    raw = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    required_sections = (
        "pipeline",
        "paths",
        "consensus",
        "windowing",
        "split",
        "tokenizer",
        "export",
        "runtime",
    )
    missing_sections = [section for section in required_sections if section not in raw]
    if missing_sections:
        raise ValueError(
            "Feline pipeline config is missing required sections: " + ", ".join(missing_sections)
        )

    try:
        pipeline = raw["pipeline"]
        paths = raw["paths"]
        consensus = raw["consensus"]
        windowing = raw["windowing"]
        split = raw["split"]
        tokenizer = raw["tokenizer"]
        export = raw["export"]
        runtime = raw["runtime"]

        if pipeline["project_accession"] != APPROVED_BIOPROJECT_ACCESSION:
            raise ValueError(
                f"pipeline.project_accession must remain {APPROVED_BIOPROJECT_ACCESSION}"
            )
        if consensus["assembly"] != APPROVED_REFERENCE_ASSEMBLY:
            raise ValueError(f"consensus.assembly must remain {APPROVED_REFERENCE_ASSEMBLY}")
        if not consensus["require_assembly_match"] or not consensus["require_contig_match"]:
            raise ValueError("consensus must fail fast on assembly and contig mismatches")
        if consensus["mask_symbol"] != "N":
            raise ValueError("consensus.mask_symbol must remain N")

        policy_fields = (
            consensus["homozygous_reference"],
            consensus["homozygous_alternate"],
            consensus["heterozygous"],
            consensus["multiallelic"],
            consensus["filtered"],
            consensus["missing"],
            consensus["indel"],
        )
        if any(policy not in EXPLICIT_CONSENSUS_POLICIES for policy in policy_fields):
            raise ValueError("consensus policies must use the approved explicit policy values")

        context_window = int(windowing["context_window"])
        window_overlap = int(windowing["window_overlap"])
        if context_window <= 0:
            raise ValueError("windowing.context_window must be positive")
        if window_overlap < 0 or window_overlap >= context_window:
            raise ValueError("windowing.window_overlap must be >= 0 and smaller than context_window")
        max_ambiguous_fraction = float(windowing["max_ambiguous_fraction"])
        if not 0 <= max_ambiguous_fraction <= 1:
            raise ValueError("windowing.max_ambiguous_fraction must be between 0 and 1")

        locus_key_fields = tuple(split["locus_key_fields"])
        if split["strategy"] != GLOBAL_LOCUS_SPLIT_STRATEGY:
            raise ValueError("split.strategy must use the global locus-safe contract")
        if locus_key_fields != ("contig", "block_id"):
            raise ValueError("split.locus_key_fields must be ['contig', 'block_id']")
        locus_block_size = int(split["locus_block_size"])
        if locus_block_size < context_window:
            raise ValueError("split.locus_block_size must be >= windowing.context_window")
        if split["assignment_stage"] != PRE_WINDOW_ASSIGNMENT_STAGE:
            raise ValueError("split.assignment_stage must assign loci before windowing")
        if split["baseline_policy"] != REFERENCE_BASELINE_POLICY:
            raise ValueError("split.baseline_policy must reuse locus assignments for the baseline corpus")

        allowed_alphabet = tuple(tokenizer["allowed_alphabet"])
        if tokenizer["identifier"] != DNABERT2_TOKENIZER_ID:
            raise ValueError("tokenizer.identifier must pin zhihan1996/DNABERT-2-117M")
        if tokenizer["revision"] != DNABERT2_TOKENIZER_REVISION:
            raise ValueError("tokenizer.revision must pin the approved immutable DNABERT-2 revision")
        if allowed_alphabet != POST_CONSENSUS_ALLOWED_ALPHABET:
            raise ValueError("tokenizer.allowed_alphabet must exactly match the post-consensus contract")
        if tokenizer["unsupported_symbol_policy"] not in {"reject", "normalize_to_n"}:
            raise ValueError("tokenizer.unsupported_symbol_policy must be reject or normalize_to_n")
        max_position_embeddings = int(tokenizer["max_position_embeddings"])
        if max_position_embeddings < context_window:
            raise ValueError("tokenizer.max_position_embeddings must be >= windowing.context_window")

        if export["format"] != "parquet":
            raise ValueError("export.format must remain parquet for the approved v1 contract")
        if int(export["row_group_size"]) <= 0:
            raise ValueError("export.row_group_size must be positive")
        if not export["preserve_coordinates"]:
            raise ValueError("export.preserve_coordinates must remain enabled for auditability")
        if not export["preserve_raw_windows"] and not export["preserve_sequence_hashes"]:
            raise ValueError("export must preserve raw windows or immutable sequence hashes")
        if export["sequence_hash_algorithm"] != "sha256":
            raise ValueError("export.sequence_hash_algorithm must remain sha256")

        external_tools = tuple(runtime["external_tools"])
        if external_tools != REQUIRED_EXTERNAL_TOOLS:
            raise ValueError("runtime.external_tools must explicitly require bcftools")
    except KeyError as exc:
        raise ValueError(f"Feline pipeline config is missing required field: {exc.args[0]}") from exc

    return FelinePipelineConfig(
        name=pipeline["name"],
        description=pipeline["description"],
        project_accession=pipeline["project_accession"],
        paths=PipelinePathsConfig(
            reference_fasta=Path(paths["reference_fasta"]),
            sample_manifest=Path(paths["sample_manifest"]),
            source_vcf=Path(paths["source_vcf"]),
            raw_dir=Path(paths["raw_dir"]),
            processed_dir=Path(paths["processed_dir"]),
            baseline_dir=Path(paths["baseline_dir"]),
            artifact_dir=Path(paths["artifact_dir"]),
            report_dir=Path(paths["report_dir"]),
        ),
        consensus=ConsensusConfig(
            assembly=consensus["assembly"],
            require_assembly_match=bool(consensus["require_assembly_match"]),
            require_contig_match=bool(consensus["require_contig_match"]),
            mask_symbol=consensus["mask_symbol"],
            homozygous_reference=consensus["homozygous_reference"],
            homozygous_alternate=consensus["homozygous_alternate"],
            heterozygous=consensus["heterozygous"],
            multiallelic=consensus["multiallelic"],
            filtered=consensus["filtered"],
            missing=consensus["missing"],
            indel=consensus["indel"],
        ),
        windowing=WindowingConfig(
            context_window=context_window,
            window_overlap=window_overlap,
            max_ambiguous_fraction=max_ambiguous_fraction,
            drop_short_sequences=bool(windowing["drop_short_sequences"]),
        ),
        split=SplitConfig(
            strategy=split["strategy"],
            locus_key_fields=locus_key_fields,
            locus_block_size=locus_block_size,
            assignment_stage=split["assignment_stage"],
            evaluation_target=split["evaluation_target"],
            baseline_policy=split["baseline_policy"],
        ),
        tokenizer=TokenizerConfig(
            identifier=tokenizer["identifier"],
            revision=tokenizer["revision"],
            allowed_alphabet=allowed_alphabet,
            unsupported_symbol_policy=tokenizer["unsupported_symbol_policy"],
            max_position_embeddings=max_position_embeddings,
        ),
        export=ExportConfig(
            format=export["format"],
            access_pattern=export["access_pattern"],
            row_group_size=int(export["row_group_size"]),
            deterministic_partition_keys=tuple(export["deterministic_partition_keys"]),
            preserve_raw_windows=bool(export["preserve_raw_windows"]),
            preserve_sequence_hashes=bool(export["preserve_sequence_hashes"]),
            preserve_coordinates=bool(export["preserve_coordinates"]),
            sequence_hash_algorithm=export["sequence_hash_algorithm"],
        ),
        runtime=RuntimeConfig(external_tools=external_tools),
    )


def check_feline_pipeline_runtime(path: str | Path) -> FelinePipelineConfig:
    config = load_feline_pipeline_config(path)
    assert_external_tools_available(config.runtime.external_tools)
    return config


def describe_experiment(path: str | Path) -> str:
    config = load_experiment_config(path)
    return "\n".join(
        [
            f"Experiment: {config.name}",
            f"Description: {config.description}",
            f"Primary task: {config.primary_task}",
            f"Primary metric: {config.primary_metric}",
            f"Split unit: {config.split_unit}",
            f"Stages: {' -> '.join(config.stages)}",
            (
                "Deferred baseline: "
                f"{config.baseline_stage} -> {config.baseline_provider} "
                f"({config.baseline_extension_point})"
            ),
            f"Requires private data: {config.requires_private_data}",
        ]
    )


def describe_feline_pipeline(path: str | Path) -> str:
    config = load_feline_pipeline_config(path)
    return "\n".join(
        [
            f"Pipeline: {config.name}",
            f"Description: {config.description}",
            f"Project accession: {config.project_accession}",
            f"Consensus assembly: {config.consensus.assembly}",
            (
                "Split contract: "
                f"{config.split.strategy} via {', '.join(config.split.locus_key_fields)} "
                f"block_size={config.split.locus_block_size} "
                f"({config.split.assignment_stage}; baseline={config.split.baseline_policy})"
            ),
            (
                "Tokenizer: "
                f"{config.tokenizer.identifier}@{config.tokenizer.revision} "
                f"alphabet={''.join(config.tokenizer.allowed_alphabet)}"
            ),
            (
                "Export: "
                f"{config.export.format} row_group_size={config.export.row_group_size} "
                f"raw_windows={config.export.preserve_raw_windows} "
                f"sequence_hashes={config.export.preserve_sequence_hashes}"
            ),
            f"Runtime tools: {', '.join(config.runtime.external_tools)}",
        ]
    )