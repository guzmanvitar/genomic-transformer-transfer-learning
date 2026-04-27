"""Bootstrap config loading and validation helpers.

This module owns the contract-enforcement boundary between raw TOML
configuration files and the typed, frozen dataclass configs consumed by
every downstream pipeline stage.  Every loader function (**load_experiment_config**,
**load_feline_pipeline_config**) must reject configs that violate the
approved scientific or engineering contracts *before* any pipeline work
begins.

Design invariants:

* All dataclasses are ``frozen=True`` — once loaded and validated, a config
  object is immutable and safe to share across threads or stages.
* Boolean fields are guarded by :func:`_require_boolean_field`, which uses
  ``type(value) is not bool`` (identity check against the ``bool`` type)
  rather than ``isinstance``.  This is intentional: TOML distinguishes
  ``true``/``false`` from integers ``1``/``0``, but Python's ``isinstance``
  treats ``bool`` as a subclass of ``int``.  The strict ``type()`` check
  ensures TOML-spec compliance so that ``1`` is never silently accepted
  where ``true`` is required.
* Contract constants (approved assemblies, tokenizer pins, split strategies,
  etc.) are imported from ``data.pipeline_contract`` and ``baselines``; this
  module never hard-codes those values inline.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

from .baselines import (
    BASELINE_EVALUATION_STAGE,
    DEFERRED_BASELINE_PROVIDER,
    SHARED_BASELINE_EXTENSION_POINT,
)
from .data.contracts import JAGUAR_METADATA_FIELDS
from .data.felid_assemblies import APPROVED_FELID_ASSEMBLIES, FelidAssembly
from .data.pipeline_contract import (
    APPROVED_BIOPROJECT_ACCESSION,
    APPROVED_FELID_ACCESSIONS,
    APPROVED_REFERENCE_ASSEMBLY,
    DNABERT2_TOKENIZER_ID,
    DNABERT2_TOKENIZER_REVISION,
    DNABERT2_TRUST_REMOTE_CODE,
    EXPLICIT_CONSENSUS_POLICIES,
    GLOBAL_LOCUS_SPLIT_STRATEGY,
    POST_CONSENSUS_ALLOWED_ALPHABET,
    PRE_WINDOW_ASSIGNMENT_STAGE,
    REFERENCE_BASELINE_POLICY,
    REQUIRED_EXTERNAL_TOOLS,
    REQUIRED_FELID_FOUNDATION_SPECIES_COUNT,
    assert_external_tools_available,
)

REQUIRED_STAGES = ("evaluate", BASELINE_EVALUATION_STAGE, "report")


@dataclass(frozen=True)
class ExperimentConfig:
    """Immutable, validated snapshot of a bootstrap experiment TOML.

    Every field has already passed the contract checks enforced by
    :func:`load_experiment_config`; downstream stages may rely on the
    values without re-validation.

    Attributes:
        name: Human-readable experiment identifier.
        description: Free-text purpose or hypothesis.
        requires_private_data: Whether the experiment depends on data that
            should not be committed to the public repository.
        primary_task: Task kind name.
        primary_metric: Evaluation metric name.
        split_unit: Granularity of train/test splitting (``sample_id``
            or ``individual_id``).
        jaguar_metadata_fields: Ordered tuple of metadata field names.
        stages: Ordered pipeline stage names.
        baseline_stage: Pipeline stage at which baseline evaluation runs.
        baseline_provider: Identifier for the baseline provider
            implementation.
        baseline_enabled: Whether baseline execution is enabled for this
            run.
        baseline_extension_point: Name of the shared baseline extension
            point.
    """

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
    """Filesystem paths required by the feline genome pipeline.

    All paths are stored as :class:`pathlib.Path` instances.  The loader
    does **not** verify that the paths exist on disk — that responsibility
    belongs to the stage that first accesses them.

    Attributes:
        reference_fasta: Reference genome FASTA file.
        sample_manifest: CSV/TSV manifest mapping sample IDs to metadata.
        source_vcf: Multi-sample VCF used as the variant source.
        raw_dir: Root directory for raw (pre-consensus) intermediate files.
        processed_dir: Root directory for post-consensus processed outputs.
        baseline_dir: Directory for baseline model artifacts.
        artifact_dir: General artifact storage (models, checkpoints).
        report_dir: Directory where final evaluation reports are written.
    """

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
    """VCF-to-consensus-sequence conversion parameters.

    The loader enforces that ``assembly`` matches the approved reference,
    that both mismatch-guard booleans are ``True``, and that every genotype
    policy string belongs to :pydata:`pipeline_contract.EXPLICIT_CONSENSUS_POLICIES`.

    Attributes:
        assembly: Reference assembly identifier (e.g. ``felCat9``).
        require_assembly_match: Fail-fast guard against assembly
            mismatches between the VCF and the reference.
        require_contig_match: Fail-fast guard against contig name
            mismatches.
        mask_symbol: Character used for ambiguous or masked positions.
        homozygous_reference: Consensus policy for 0/0 genotypes.
        homozygous_alternate: Consensus policy for 1/1 genotypes.
        heterozygous: Consensus policy for 0/1 genotypes.
        multiallelic: Consensus policy for multi-allelic sites.
        filtered: Consensus policy for filtered sites.
        missing: Consensus policy for missing genotypes (``./.``).
        indel: Consensus policy for insertion/deletion variants.
    """

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
    """Sliding-window parameters for slicing consensus sequences.

    Attributes:
        context_window: Window width in base pairs.
        window_overlap: Number of overlapping base pairs between
            consecutive windows.
        max_ambiguous_fraction: Maximum fraction of ambiguous (``N``)
            bases allowed per window before the window is discarded.
        drop_short_sequences: Whether to discard windows shorter than
            ``context_window`` (e.g. at contig boundaries).
    """

    context_window: int
    window_overlap: int
    max_ambiguous_fraction: float
    drop_short_sequences: bool


@dataclass(frozen=True)
class SplitConfig:
    """Train/test split strategy enforcing locus-safe data leakage prevention.

    The loader validates that the strategy matches
    :pydata:`pipeline_contract.GLOBAL_LOCUS_SPLIT_STRATEGY`, key fields are
    ``('contig', 'block_id')``, and block size is at least as large as the
    context window.

    Attributes:
        strategy: Split algorithm name.
        locus_key_fields: Fields that define a unique genomic locus for
            split assignment.
        locus_block_size: Size of contiguous genomic blocks assigned to
            a single split.
        assignment_stage: Pipeline stage at which locus split assignments
            are computed.
        evaluation_target: Which split partition is used for evaluation.
        baseline_policy: How the baseline corpus reuses locus assignments.
    """

    strategy: str
    locus_key_fields: tuple[str, ...]
    locus_block_size: int
    assignment_stage: str
    evaluation_target: str
    baseline_policy: str


@dataclass(frozen=True)
class TokenizerConfig:
    """DNABERT-2 tokenizer pinning and alphabet contract.

    The loader pins the tokenizer to an exact HuggingFace model ID and
    immutable revision, and validates that the allowed alphabet matches
    the post-consensus contract.

    Attributes:
        identifier: HuggingFace model identifier on the HuggingFace Hub.
        revision: Immutable Git revision hash for reproducibility.
        allowed_alphabet: Tuple of single-character strings representing
            valid nucleotide symbols after consensus building.
        unsupported_symbol_policy: Action when an out-of-alphabet symbol
            is encountered (``reject`` or ``normalize_to_n``).
        max_position_embeddings: Maximum sequence length the tokenizer
            supports.
        trust_remote_code: Whether to allow execution of model-hub code.
    """

    identifier: str
    revision: str
    allowed_alphabet: tuple[str, ...]
    unsupported_symbol_policy: str
    max_position_embeddings: int
    trust_remote_code: bool


@dataclass(frozen=True)
class ExportConfig:
    """Parquet export settings with auditability guarantees.

    The loader enforces that the format is ``parquet``, coordinates are
    preserved, and at least one of raw windows or sequence hashes is
    retained for reproducibility auditing.

    Attributes:
        format: Serialisation format.
        access_pattern: Hint for downstream readers (e.g. ``row_group``).
        row_group_size: Number of rows per Parquet row group.
        deterministic_partition_keys: Column names used for deterministic
            partitioning of output files.
        preserve_raw_windows: Whether raw consensus windows are stored
            alongside tokenized outputs.
        preserve_sequence_hashes: Whether immutable SHA-256 hashes of
            each window are stored for integrity verification.
        preserve_coordinates: Whether exported rows carry their genomic
            coordinates.
        sequence_hash_algorithm: Hash algorithm for sequence integrity
            checks.
    """

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
    """Runtime dependency declarations.

    Attributes:
        external_tools: Tuple of CLI tool names (e.g. ``bcftools``)
            required on ``$PATH`` before the pipeline starts.
    """

    external_tools: tuple[str, ...]


@dataclass(frozen=True)
class FelinePipelineConfig:
    """Top-level validated configuration for the feline genome pipeline.

    Composed of section-specific frozen dataclasses, each independently
    validated by :func:`load_feline_pipeline_config`.  This object is the
    single source of truth passed to every pipeline stage.

    Attributes:
        name: Human-readable pipeline name.
        description: Free-text description of the pipeline run.
        project_accession: NCBI BioProject accession.
        paths: Filesystem paths (:class:`PipelinePathsConfig`).
        consensus: VCF-to-consensus parameters (:class:`ConsensusConfig`).
        windowing: Sliding-window parameters (:class:`WindowingConfig`).
        split: Train/test split strategy (:class:`SplitConfig`).
        tokenizer: DNABERT-2 tokenizer pinning (:class:`TokenizerConfig`).
        export: Parquet export settings (:class:`ExportConfig`).
        runtime: External tool requirements (:class:`RuntimeConfig`).
    """

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


def _require_boolean_field(
    value: object,
    *,
    field_name: str,
    contract_description: str | None = None,
) -> bool:
    """Validate that *value* is a native Python ``bool``, not a truthy int.

    This function deliberately uses ``type(value) is not bool`` instead of
    ``isinstance(value, bool)``.  The reason is TOML-spec compliance:
    TOML distinguishes ``true``/``false`` from integers ``1``/``0``, but
    in Python ``bool`` is a subclass of ``int``, so ``isinstance(1, bool)``
    returns ``False`` while ``isinstance(True, int)`` returns ``True``.
    Using the identity check against the ``bool`` type ensures that an
    integer ``1`` parsed from a TOML value is never silently accepted
    where a boolean ``true`` is required.

    Args:
        value: The raw value parsed from the TOML config file.
        field_name: Dotted config key name used in error messages
            (e.g. ``"consensus.require_assembly_match"``).
        contract_description: Optional human-readable description of the
            contract being enforced, appended to the error message.

    Returns:
        The validated ``bool`` value, unchanged.

    Raises:
        ValueError: If *value* is not exactly of type ``bool``.
    """
    if type(value) is not bool:
        contract_suffix = ""
        if contract_description is not None:
            contract_suffix = f" matching {contract_description}"
        raise ValueError(
            f"{field_name} must be a TOML boolean true/false{contract_suffix}; "
            f"got {value!r} ({type(value).__name__})"
        )
    return value


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load and validate a bootstrap experiment TOML file.

    Enforces every contract defined by the bootstrap specification:

    * ``jaguar_metadata_fields`` must exactly match the canonical list.
    * Primary task must be ``coordinate_regression`` with metric
      ``median_geodesic_error_km``.
    * ``split_unit`` must be ``sample_id`` or ``individual_id``.
    * Stage ordering must be non-empty, unique, and include all
      :pydata:`REQUIRED_STAGES`.
    * Baseline settings must match the deferred-legacy extension contract.

    Args:
        path: Filesystem path to a TOML experiment config file.

    Returns:
        A fully validated, frozen :class:`ExperimentConfig`.

    Raises:
        ValueError: If any contract check fails.
        KeyError: If a required TOML section or key is missing.
    """
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
        raise ValueError(
            "bootstrap configs must use median_geodesic_error_km as the primary metric"
        )
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
    baseline_enabled = _require_boolean_field(
        baseline["enabled"],
        field_name="baseline.enabled",
        contract_description="the bootstrap baseline disable contract",
    )
    if baseline_enabled:
        raise ValueError("bootstrap baseline execution must remain disabled")
    if baseline["extension_point"] != SHARED_BASELINE_EXTENSION_POINT:
        raise ValueError(
            "bootstrap baseline extension point must remain shared_split_metric_report_contract"
        )
    requires_private_data = _require_boolean_field(
        experiment.get("requires_private_data", False),
        field_name="experiment.requires_private_data",
        contract_description="the bootstrap private-data flag contract",
    )

    return ExperimentConfig(
        name=experiment["name"],
        description=experiment["description"],
        requires_private_data=requires_private_data,
        primary_task=primary_task["kind"],
        primary_metric=primary_task["primary_metric"],
        split_unit=data["split_unit"],
        jaguar_metadata_fields=metadata_fields,
        stages=stages,
        baseline_stage=baseline_stage,
        baseline_provider=baseline["provider"],
        baseline_enabled=baseline_enabled,
        baseline_extension_point=baseline["extension_point"],
    )


def load_feline_pipeline_config(path: str | Path) -> FelinePipelineConfig:
    """Load and validate a feline genome pipeline TOML config.

    Performs exhaustive contract enforcement across all eight required
    sections (``pipeline``, ``paths``, ``consensus``, ``windowing``,
    ``split``, ``tokenizer``, ``export``, ``runtime``).  Key validations
    include:

    * BioProject accession and reference assembly pinning.
    * Consensus mismatch-guard booleans (via :func:`_require_boolean_field`).
    * Genotype policy strings against the explicit consensus policy set.
    * Windowing arithmetic (positive window, valid overlap, ambiguity
      fraction in ``[0, 1]``).
    * Locus-safe split strategy and block-size >= context-window guard.
    * DNABERT-2 tokenizer identity, revision, and alphabet pinning.
    * Parquet export auditability (coordinates preserved, hash algorithm).
    * External tool manifest matching the required set.

    Args:
        path: Filesystem path to a TOML pipeline config file.

    Returns:
        A fully validated, frozen :class:`FelinePipelineConfig`.

    Raises:
        ValueError: If any contract check fails or a required section /
            field is missing.
    """
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
        require_assembly_match = _require_boolean_field(
            consensus["require_assembly_match"],
            field_name="consensus.require_assembly_match",
            contract_description="the approved consensus mismatch-guard contract",
        )
        require_contig_match = _require_boolean_field(
            consensus["require_contig_match"],
            field_name="consensus.require_contig_match",
            contract_description="the approved consensus mismatch-guard contract",
        )
        if not require_assembly_match or not require_contig_match:
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
            raise ValueError(
                "windowing.window_overlap must be >= 0 and smaller than context_window"
            )
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
            raise ValueError(
                "split.baseline_policy must reuse locus assignments for the baseline corpus"
            )

        allowed_alphabet = tuple(tokenizer["allowed_alphabet"])
        if tokenizer["identifier"] != DNABERT2_TOKENIZER_ID:
            raise ValueError("tokenizer.identifier must pin zhihan1996/DNABERT-2-117M")
        if tokenizer["revision"] != DNABERT2_TOKENIZER_REVISION:
            raise ValueError(
                "tokenizer.revision must pin the approved immutable DNABERT-2 revision"
            )
        if allowed_alphabet != POST_CONSENSUS_ALLOWED_ALPHABET:
            raise ValueError(
                "tokenizer.allowed_alphabet must exactly match the post-consensus contract"
            )
        if tokenizer["unsupported_symbol_policy"] not in {"reject", "normalize_to_n"}:
            raise ValueError("tokenizer.unsupported_symbol_policy must be reject or normalize_to_n")
        max_position_embeddings = int(tokenizer["max_position_embeddings"])
        if max_position_embeddings < context_window:
            raise ValueError(
                "tokenizer.max_position_embeddings must be >= windowing.context_window"
            )
        trust_remote_code = _require_boolean_field(
            tokenizer["trust_remote_code"],
            field_name="tokenizer.trust_remote_code",
            contract_description="the approved DNABERT-2 contract",
        )
        if trust_remote_code is not DNABERT2_TRUST_REMOTE_CODE:
            raise ValueError(
                "tokenizer.trust_remote_code must remain "
                f"{DNABERT2_TRUST_REMOTE_CODE} for the approved DNABERT-2 contract"
            )

        drop_short_sequences = _require_boolean_field(
            windowing["drop_short_sequences"],
            field_name="windowing.drop_short_sequences",
            contract_description="the feline windowing contract",
        )

        if export["format"] != "parquet":
            raise ValueError("export.format must remain parquet for the approved v1 contract")
        if int(export["row_group_size"]) <= 0:
            raise ValueError("export.row_group_size must be positive")
        preserve_coordinates = _require_boolean_field(
            export["preserve_coordinates"],
            field_name="export.preserve_coordinates",
            contract_description="the feline export auditability contract",
        )
        preserve_raw_windows = _require_boolean_field(
            export["preserve_raw_windows"],
            field_name="export.preserve_raw_windows",
            contract_description="the feline export preservation contract",
        )
        preserve_sequence_hashes = _require_boolean_field(
            export["preserve_sequence_hashes"],
            field_name="export.preserve_sequence_hashes",
            contract_description="the feline export preservation contract",
        )
        if not preserve_coordinates:
            raise ValueError("export.preserve_coordinates must remain enabled for auditability")
        if not preserve_raw_windows and not preserve_sequence_hashes:
            raise ValueError("export must preserve raw windows or immutable sequence hashes")
        if export["sequence_hash_algorithm"] != "sha256":
            raise ValueError("export.sequence_hash_algorithm must remain sha256")

        external_tools = tuple(runtime["external_tools"])
        if external_tools != REQUIRED_EXTERNAL_TOOLS:
            raise ValueError("runtime.external_tools must explicitly require bcftools")
    except KeyError as exc:
        raise ValueError(
            f"Feline pipeline config is missing required field: {exc.args[0]}"
        ) from exc

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
            require_assembly_match=require_assembly_match,
            require_contig_match=require_contig_match,
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
            drop_short_sequences=drop_short_sequences,
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
            trust_remote_code=trust_remote_code,
        ),
        export=ExportConfig(
            format=export["format"],
            access_pattern=export["access_pattern"],
            row_group_size=int(export["row_group_size"]),
            deterministic_partition_keys=tuple(export["deterministic_partition_keys"]),
            preserve_raw_windows=preserve_raw_windows,
            preserve_sequence_hashes=preserve_sequence_hashes,
            preserve_coordinates=preserve_coordinates,
            sequence_hash_algorithm=export["sequence_hash_algorithm"],
        ),
        runtime=RuntimeConfig(external_tools=external_tools),
    )


@dataclass(frozen=True)
class FelidSpeciesEntry:
    """One approved species pinned to its RefSeq accession and assembly name.

    The felid-foundation corpus mixes six species, and the per-species
    FASTA filename is deterministically derived from the accession + assembly
    name. Freezing the triple in a typed entry lets every downstream stage
    (path resolution, logging, run-summary keys) share the same species slug
    without re-deriving it from the Latin binomial each time.

    Attributes:
        species: Latin binomial as written in ``APPROVED_FELID_ASSEMBLIES``
            (e.g. ``"Panthera leo"``).
        accession: RefSeq accession prefixed with ``GCF_``.
        assembly_name: NCBI assembly name (e.g. ``"P.leo_Ple1_pat1.1"``).
        species_slug: Lowercase underscored identifier derived from ``species``
            (e.g. ``"panthera_leo"``). Used as ``individual_id`` on every
            emitted ``WindowRecord`` and as the keying identifier in the
            per-species run-summary map.
    """

    species: str
    accession: str
    assembly_name: str
    species_slug: str


@dataclass(frozen=True)
class FelidFoundationPathsConfig:
    """Filesystem paths required by the felid-foundation pretraining pipeline.

    Unlike :class:`PipelinePathsConfig`, this path block carries *no*
    ``sample_manifest`` or ``source_vcf`` — the foundation corpus is
    reference-FASTA-only and never invokes consensus calling. The loader
    does not verify that paths exist; that responsibility belongs to the
    stage that first accesses them so we can still validate configs on a
    machine without the reference directory materialised.

    Attributes:
        reference_dir: Directory containing one ``<ACC>_<ASM>.fna.gz`` file
            per approved felid accession.
        processed_dir: Root directory for tokenized Parquet output.
        artifact_dir: General artifact storage (run summaries, checkpoints).
        report_dir: Directory where descriptive reports are written.
    """

    reference_dir: Path
    processed_dir: Path
    artifact_dir: Path
    report_dir: Path


@dataclass(frozen=True)
class FelidFoundationPipelineConfig:
    """Top-level validated configuration for the felid-foundation pipeline.

    Distinct from :class:`FelinePipelineConfig` because the foundation path
    uses a multi-species FASTA-only input (no VCF, no consensus, no
    BioProject pinning) and shares only the windowing / tokenizer / split /
    export contract helpers with the legacy path.

    Attributes:
        name: Human-readable pipeline name.
        description: Free-text description of the pipeline run.
        species: Ordered tuple of :class:`FelidSpeciesEntry`, deduplicated
            by accession and validated against ``APPROVED_FELID_ACCESSIONS``.
        paths: Filesystem paths (:class:`FelidFoundationPathsConfig`).
        windowing: Sliding-window parameters (:class:`WindowingConfig`).
        split: Train/test split strategy (:class:`SplitConfig`).
        tokenizer: DNABERT-2 tokenizer pinning (:class:`TokenizerConfig`).
        export: Parquet export settings (:class:`ExportConfig`).
        runtime: External tool requirements (:class:`RuntimeConfig`); the
            approved foundation contract uses an *empty* tool list because
            the path never invokes bcftools.
    """

    name: str
    description: str
    species: tuple[FelidSpeciesEntry, ...]
    paths: FelidFoundationPathsConfig
    windowing: WindowingConfig
    split: SplitConfig
    tokenizer: TokenizerConfig
    export: ExportConfig
    runtime: RuntimeConfig


def _slugify_species(latin_binomial: str) -> str:
    """Convert a Latin binomial to the canonical species slug.

    The slug is used as ``individual_id`` on every emitted window
    and as the run-summary key, so it must be deterministic and stable.
    We normalise whitespace to a single underscore and lowercase the
    result; validation against the approved registry is the loader's job.
    """
    return "_".join(latin_binomial.strip().lower().split())


def check_feline_pipeline_runtime(path: str | Path) -> FelinePipelineConfig:
    """Load, validate, and verify runtime dependencies for the feline pipeline.

    This is the preferred entry point when you need both config validation
    **and** confirmation that all required external tools (e.g. ``bcftools``)
    are available on ``$PATH``.

    Args:
        path: Filesystem path to a TOML pipeline config file.

    Returns:
        A fully validated :class:`FelinePipelineConfig` (same as
        :func:`load_feline_pipeline_config`).

    Raises:
        ValueError: If any config contract check fails.
        RuntimeError: If a required external tool is not found on
            ``$PATH``.
    """
    config = load_feline_pipeline_config(path)
    assert_external_tools_available(config.runtime.external_tools)
    return config


def describe_experiment(path: str | Path) -> str:
    """Return a human-readable multi-line summary of an experiment config.

    Loads and validates the config via :func:`load_experiment_config`,
    then formats the key fields into a newline-separated string suitable
    for logging or CLI output.

    Args:
        path: Filesystem path to a TOML experiment config file.

    Returns:
        Multi-line string summarising experiment name, task, metric,
        stages, and baseline settings.
    """
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
    """Return a human-readable multi-line summary of a feline pipeline config.

    Loads and validates the config via :func:`load_feline_pipeline_config`,
    then formats the key fields — accession, consensus assembly, split
    contract, tokenizer pin, export settings, and runtime tools — into a
    newline-separated string suitable for logging or CLI output.

    Args:
        path: Filesystem path to a TOML pipeline config file.

    Returns:
        Multi-line string summarising the pipeline configuration.
    """
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


def _validate_felid_species_entries(
    raw_species: object,
) -> tuple[FelidSpeciesEntry, ...]:
    """Validate the ``[[species]]`` list against the approved felid registry.

    Prevents corpus drift. The foundation contract pins the exact
    set of six approved accessions in :data:`APPROVED_FELID_ACCESSIONS`;
    any config that references an unknown accession, duplicates an
    accession, or declares a species/accession pair that does not match
    the pinned :data:`APPROVED_FELID_ASSEMBLIES` registry is rejected.

    Args:
        raw_species: The raw value parsed from the TOML ``[[species]]`` list.

    Returns:
        A tuple of validated :class:`FelidSpeciesEntry` in registry order
        (sorted by accession for determinism).

    Raises:
        ValueError: If the list is not a non-empty sequence, shorter than
            the required species count, contains duplicate accessions, or
            references a species/accession pair that is not in the approved
            registry.
    """
    if not isinstance(raw_species, list) or not raw_species:
        raise ValueError("species must be a non-empty list of {species, accession} entries")
    if len(raw_species) < REQUIRED_FELID_FOUNDATION_SPECIES_COUNT:
        raise ValueError(
            "species list must include all "
            f"{REQUIRED_FELID_FOUNDATION_SPECIES_COUNT} approved felid accessions; "
            f"got {len(raw_species)}"
        )

    registry_by_accession: dict[str, FelidAssembly] = {
        assembly.accession: assembly for assembly in APPROVED_FELID_ASSEMBLIES
    }

    seen_accessions: set[str] = set()
    entries: list[FelidSpeciesEntry] = []
    for index, item in enumerate(raw_species):
        if not isinstance(item, dict):
            raise ValueError(
                f"species[{index}] must be a table with 'species' and 'accession' keys"
            )
        try:
            species_name = item["species"]
            accession = item["accession"]
        except KeyError as exc:
            raise ValueError(f"species[{index}] is missing required field: {exc.args[0]}") from exc
        if accession in seen_accessions:
            raise ValueError(f"species list contains duplicate accession {accession!r}")
        if accession not in APPROVED_FELID_ACCESSIONS:
            raise ValueError(
                f"species[{index}].accession {accession!r} is not an approved felid accession"
            )
        registry_entry = registry_by_accession[accession]
        if species_name != registry_entry.species:
            raise ValueError(
                f"species[{index}].species {species_name!r} does not match "
                f"the approved registry entry {registry_entry.species!r} "
                f"for accession {accession}"
            )
        seen_accessions.add(accession)
        entries.append(
            FelidSpeciesEntry(
                species=registry_entry.species,
                accession=registry_entry.accession,
                assembly_name=registry_entry.assembly_name,
                species_slug=_slugify_species(registry_entry.species),
            )
        )

    entries.sort(key=lambda entry: entry.accession)
    return tuple(entries)


def load_felid_foundation_pipeline_config(
    path: str | Path,
) -> FelidFoundationPipelineConfig:
    """Load and validate a felid-foundation pretraining TOML config.

    Performs exhaustive contract enforcement across all eight required
    sections (``pipeline``, ``paths``, ``species``, ``windowing``,
    ``split``, ``tokenizer``, ``export``, ``runtime``). Key validations:

    * ``[[species]]`` list against :data:`APPROVED_FELID_ACCESSIONS`,
      rejecting unknown, duplicate, or mislabeled entries, and requiring
      at least :data:`REQUIRED_FELID_FOUNDATION_SPECIES_COUNT` species.
    * Windowing arithmetic (positive window, valid overlap, ambiguity
      fraction in ``[0, 1]``).
    * Locus-safe split strategy and block-size >= context-window guard.
    * DNABERT-2 tokenizer identity, revision, and alphabet pinning.
    * Parquet export auditability (coordinates preserved, hash algorithm).
    * ``runtime.external_tools`` must be an explicit empty list: the
      foundation path does not call ``bcftools`` and the empty declaration
      makes that intentional rather than accidental.

    Args:
        path: Filesystem path to a TOML pipeline config file.

    Returns:
        A fully validated, frozen :class:`FelidFoundationPipelineConfig`.

    Raises:
        ValueError: If any contract check fails or a required section /
            field is missing.
    """
    raw = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    required_sections = (
        "pipeline",
        "paths",
        "species",
        "windowing",
        "split",
        "tokenizer",
        "export",
        "runtime",
    )
    missing_sections = [section for section in required_sections if section not in raw]
    if missing_sections:
        raise ValueError(
            "Felid foundation pipeline config is missing required sections: "
            + ", ".join(missing_sections)
        )

    try:
        pipeline = raw["pipeline"]
        paths = raw["paths"]
        windowing = raw["windowing"]
        split = raw["split"]
        tokenizer = raw["tokenizer"]
        export = raw["export"]
        runtime = raw["runtime"]

        species_entries = _validate_felid_species_entries(raw["species"])

        if "reference_dir" not in paths:
            raise ValueError("paths.reference_dir is required for the felid foundation pipeline")
        for forbidden in ("sample_manifest", "source_vcf", "reference_fasta"):
            if forbidden in paths:
                raise ValueError(
                    f"paths.{forbidden} is not allowed in the felid foundation config; "
                    "the foundation pipeline is reference-FASTA-only"
                )

        context_window = int(windowing["context_window"])
        window_overlap = int(windowing["window_overlap"])
        if context_window <= 0:
            raise ValueError("windowing.context_window must be positive")
        if window_overlap < 0 or window_overlap >= context_window:
            raise ValueError(
                "windowing.window_overlap must be >= 0 and smaller than context_window"
            )
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
            raise ValueError(
                "split.baseline_policy must reuse locus assignments for the baseline corpus"
            )

        allowed_alphabet = tuple(tokenizer["allowed_alphabet"])
        if tokenizer["identifier"] != DNABERT2_TOKENIZER_ID:
            raise ValueError("tokenizer.identifier must pin zhihan1996/DNABERT-2-117M")
        if tokenizer["revision"] != DNABERT2_TOKENIZER_REVISION:
            raise ValueError(
                "tokenizer.revision must pin the approved immutable DNABERT-2 revision"
            )
        if allowed_alphabet != POST_CONSENSUS_ALLOWED_ALPHABET:
            raise ValueError(
                "tokenizer.allowed_alphabet must exactly match the post-consensus contract"
            )
        if tokenizer["unsupported_symbol_policy"] not in {"reject", "normalize_to_n"}:
            raise ValueError("tokenizer.unsupported_symbol_policy must be reject or normalize_to_n")
        max_position_embeddings = int(tokenizer["max_position_embeddings"])
        if max_position_embeddings < context_window:
            raise ValueError(
                "tokenizer.max_position_embeddings must be >= windowing.context_window"
            )
        trust_remote_code = _require_boolean_field(
            tokenizer["trust_remote_code"],
            field_name="tokenizer.trust_remote_code",
            contract_description="the approved DNABERT-2 contract",
        )
        if trust_remote_code is not DNABERT2_TRUST_REMOTE_CODE:
            raise ValueError(
                "tokenizer.trust_remote_code must remain "
                f"{DNABERT2_TRUST_REMOTE_CODE} for the approved DNABERT-2 contract"
            )

        drop_short_sequences = _require_boolean_field(
            windowing["drop_short_sequences"],
            field_name="windowing.drop_short_sequences",
            contract_description="the felid foundation windowing contract",
        )

        if export["format"] != "parquet":
            raise ValueError("export.format must remain parquet for the approved v1 contract")
        if int(export["row_group_size"]) <= 0:
            raise ValueError("export.row_group_size must be positive")
        preserve_coordinates = _require_boolean_field(
            export["preserve_coordinates"],
            field_name="export.preserve_coordinates",
            contract_description="the felid foundation export auditability contract",
        )
        preserve_raw_windows = _require_boolean_field(
            export["preserve_raw_windows"],
            field_name="export.preserve_raw_windows",
            contract_description="the felid foundation export preservation contract",
        )
        preserve_sequence_hashes = _require_boolean_field(
            export["preserve_sequence_hashes"],
            field_name="export.preserve_sequence_hashes",
            contract_description="the felid foundation export preservation contract",
        )
        if not preserve_coordinates:
            raise ValueError("export.preserve_coordinates must remain enabled for auditability")
        if not preserve_raw_windows and not preserve_sequence_hashes:
            raise ValueError("export must preserve raw windows or immutable sequence hashes")
        if export["sequence_hash_algorithm"] != "sha256":
            raise ValueError("export.sequence_hash_algorithm must remain sha256")

        external_tools = tuple(runtime["external_tools"])
        if external_tools != ():
            raise ValueError(
                "runtime.external_tools must be an empty list for the felid foundation pipeline "
                "(the reference-FASTA-only path does not invoke bcftools)"
            )
    except KeyError as exc:
        raise ValueError(
            f"Felid foundation pipeline config is missing required field: {exc.args[0]}"
        ) from exc

    return FelidFoundationPipelineConfig(
        name=pipeline["name"],
        description=pipeline["description"],
        species=species_entries,
        paths=FelidFoundationPathsConfig(
            reference_dir=Path(paths["reference_dir"]),
            processed_dir=Path(paths["processed_dir"]),
            artifact_dir=Path(paths["artifact_dir"]),
            report_dir=Path(paths["report_dir"]),
        ),
        windowing=WindowingConfig(
            context_window=context_window,
            window_overlap=window_overlap,
            max_ambiguous_fraction=max_ambiguous_fraction,
            drop_short_sequences=drop_short_sequences,
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
            trust_remote_code=trust_remote_code,
        ),
        export=ExportConfig(
            format=export["format"],
            access_pattern=export["access_pattern"],
            row_group_size=int(export["row_group_size"]),
            deterministic_partition_keys=tuple(export["deterministic_partition_keys"]),
            preserve_raw_windows=preserve_raw_windows,
            preserve_sequence_hashes=preserve_sequence_hashes,
            preserve_coordinates=preserve_coordinates,
            sequence_hash_algorithm=export["sequence_hash_algorithm"],
        ),
        runtime=RuntimeConfig(external_tools=external_tools),
    )


def check_felid_foundation_pipeline_runtime(
    path: str | Path,
) -> FelidFoundationPipelineConfig:
    """Load, validate, and verify runtime dependencies for the felid foundation pipeline.

    The foundation contract declares an empty external-tool list, so this
    function reduces to "load + validate + no-op external tool check".
    Exposing it in parallel with :func:`check_feline_pipeline_runtime`
    keeps the CLI UX consistent: every pipeline has a
    ``check-*-runtime`` subcommand that returns a validated config.

    Args:
        path: Filesystem path to a TOML pipeline config file.

    Returns:
        A fully validated :class:`FelidFoundationPipelineConfig`.

    Raises:
        ValueError: If any config contract check fails.
        RuntimeError: If a required external tool is not found on ``$PATH``
            (currently unreachable under the approved empty-tool contract,
            but we still delegate to the shared helper so a future opt-in
            tool is honoured automatically).
    """
    config = load_felid_foundation_pipeline_config(path)
    assert_external_tools_available(config.runtime.external_tools)
    return config


def describe_felid_foundation_config(path: str | Path) -> str:
    """Return a human-readable multi-line summary of a felid foundation config.

    Loads and validates the config via
    :func:`load_felid_foundation_pipeline_config`, then formats the key
    fields — species roster, split contract, tokenizer pin, export
    settings, and runtime tools — into a newline-separated string suitable
    for logging or CLI output.

    Args:
        path: Filesystem path to a TOML pipeline config file.

    Returns:
        Multi-line string summarising the pipeline configuration.
    """
    config = load_felid_foundation_pipeline_config(path)
    species_lines = [
        f"  - {entry.species} ({entry.accession} / {entry.assembly_name})"
        for entry in config.species
    ]
    external_tools_display = (
        ", ".join(config.runtime.external_tools) if config.runtime.external_tools else "(none)"
    )
    return "\n".join(
        [
            f"Pipeline: {config.name}",
            f"Description: {config.description}",
            f"Species ({len(config.species)}):",
            *species_lines,
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
            f"Runtime tools: {external_tools_display}",
        ]
    )


@dataclass(frozen=True)
class FoundationTrainingConfig:
    """Immutable, validated configuration for felid foundation continued pre-training.

    Composed of required corpus metadata path, model identifier, and step counts,
    plus defaulted hyperparameters for training (batch size, learning rate, warmup,
    checkpointing) and evaluation. Every field is validated by
    :func:`load_foundation_training_config` before construction.

    Attributes:
        corpus_metadata_path: Path to metadata.json produced by TokenizedCorpusWriter.
        model_identifier: HuggingFace model ID (pinned to zhihan1996/DNABERT-2-117M).
        model_revision: Git revision hash for exact model reproducibility.
        max_steps: Total training steps (required; no automatic epoch-based scaling).
        output_dir: Root directory for checkpoints and logs.
        max_seq_length: Maximum sequence length (truncation in reader).
        mlm_probability: Masking probability for dynamic masking (0.15 standard).
        per_device_train_batch_size: Batch size per GPU/worker (before accumulation).
        per_device_eval_batch_size: Batch size per GPU/worker for validation.
        gradient_accumulation_steps: Accumulate gradients before optimizer step.
        learning_rate: Initial learning rate for AdamW.
        weight_decay: L2 regularization coefficient.
        warmup_steps: Linear warmup steps before cosine decay.
        eval_every: Validation frequency (steps).
        eval_max_steps: Max validation steps per epoch (auto-derived if None).
        save_every: Checkpoint save frequency (steps).
        log_every: Logging frequency (steps).
        gradient_clip: Gradient norm clipping value.
        seed: Random seed for reproducibility.
        num_workers: DataLoader worker processes.
        shuffle_buffer_size: Row-level shuffle buffer size.
        tensorboard_subdir: Subdirectory under output_dir for TensorBoard logs.
        pad_token_fallback: Strategy if tokenizer lacks pad_token (eos/unk/add_pad).
    """

    corpus_metadata_path: Path
    model_identifier: str
    model_revision: str
    max_steps: int
    output_dir: Path = Path("models/foundation_felid")
    max_seq_length: int = 512
    mlm_probability: float = 0.15
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    eval_every: int = 500
    eval_max_steps: int | None = None
    save_every: int = 1000
    log_every: int = 10
    gradient_clip: float = 1.0
    seed: int = 42
    num_workers: int = 4
    shuffle_buffer_size: int = 8192
    tensorboard_subdir: str = "tensorboard"
    pad_token_fallback: str = "eos"


def load_foundation_training_config(path: str | Path) -> FoundationTrainingConfig:
    """Load and validate a felid foundation training TOML config.

    Enforces that required fields (corpus_metadata_path, model_identifier,
    model_revision, max_steps) are present and valid, and that optional
    hyperparameters fall within sensible ranges. The model_identifier is
    pinned to zhihan1996/DNABERT-2-117M by contract; other fields are
    lightly validated (e.g., learning_rate > 0, batch sizes > 0).

    Args:
        path: Filesystem path to a TOML training config file.

    Returns:
        A fully validated, frozen :class:`FoundationTrainingConfig`.

    Raises:
        ValueError: If any contract check fails or a required field is missing.
        KeyError: If a required TOML section or key is missing.
    """
    raw = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    try:
        training = raw["training"]
        corpus_metadata_path = Path(training["corpus_metadata_path"])
        model_identifier = training["model_identifier"]
        model_revision = training["model_revision"]
        max_steps = int(training["max_steps"])

        # Defaults and optional fields
        output_dir = Path(training.get("output_dir", "models/foundation_felid"))
        max_seq_length = int(training.get("max_seq_length", 512))
        mlm_probability = float(training.get("mlm_probability", 0.15))
        per_device_train_batch_size = int(training.get("per_device_train_batch_size", 8))
        per_device_eval_batch_size = int(training.get("per_device_eval_batch_size", 8))
        gradient_accumulation_steps = int(training.get("gradient_accumulation_steps", 1))
        learning_rate = float(training.get("learning_rate", 1e-4))
        weight_decay = float(training.get("weight_decay", 0.01))
        warmup_steps = int(training.get("warmup_steps", 1000))
        eval_every = int(training.get("eval_every", 500))
        eval_max_steps_raw = training.get("eval_max_steps")
        eval_max_steps = int(eval_max_steps_raw) if eval_max_steps_raw is not None else None
        save_every = int(training.get("save_every", 1000))
        log_every = int(training.get("log_every", 10))
        gradient_clip = float(training.get("gradient_clip", 1.0))
        seed = int(training.get("seed", 42))
        num_workers = int(training.get("num_workers", 4))
        shuffle_buffer_size = int(training.get("shuffle_buffer_size", 8192))
        tensorboard_subdir = training.get("tensorboard_subdir", "tensorboard")
        pad_token_fallback = training.get("pad_token_fallback", "eos")

        # Validation
        if not corpus_metadata_path.is_absolute():
            raise ValueError(
                "training.corpus_metadata_path must be an absolute path "
                "or resolvable relative to the config directory"
            )
        if model_identifier != "zhihan1996/DNABERT-2-117M":
            raise ValueError(
                "training.model_identifier must be pinned to zhihan1996/DNABERT-2-117M"
            )
        if max_steps <= 0:
            raise ValueError("training.max_steps must be positive")
        if learning_rate <= 0:
            raise ValueError("training.learning_rate must be positive")
        if per_device_train_batch_size <= 0:
            raise ValueError("training.per_device_train_batch_size must be positive")
        if per_device_eval_batch_size <= 0:
            raise ValueError("training.per_device_eval_batch_size must be positive")
        if pad_token_fallback not in {"eos", "unk", "add_pad"}:
            raise ValueError("training.pad_token_fallback must be one of: eos, unk, add_pad")

    except KeyError as exc:
        msg = f"Foundation training config is missing required field: {exc.args[0]}"
        raise ValueError(msg) from exc

    return FoundationTrainingConfig(
        corpus_metadata_path=corpus_metadata_path,
        model_identifier=model_identifier,
        model_revision=model_revision,
        max_steps=max_steps,
        output_dir=output_dir,
        max_seq_length=max_seq_length,
        mlm_probability=mlm_probability,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        eval_every=eval_every,
        eval_max_steps=eval_max_steps,
        save_every=save_every,
        log_every=log_every,
        gradient_clip=gradient_clip,
        seed=seed,
        num_workers=num_workers,
        shuffle_buffer_size=shuffle_buffer_size,
        tensorboard_subdir=tensorboard_subdir,
        pad_token_fallback=pad_token_fallback,
    )
