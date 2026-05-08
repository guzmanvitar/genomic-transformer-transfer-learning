"""Bootstrap config loading and validation helpers.

This module owns the contract-enforcement boundary between raw TOML
configuration files and the typed, frozen dataclass configs consumed by
every downstream pipeline stage. Active loaders must reject configs that
violate the approved scientific or engineering contracts *before* any
pipeline work begins.

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
    APPROVED_FELID_IDENTIFIERS,
    DNABERT2_TOKENIZER_ID,
    DNABERT2_TOKENIZER_REVISION,
    DNABERT2_TRUST_REMOTE_CODE,
    GLOBAL_LOCUS_SPLIT_STRATEGY,
    POST_CONSENSUS_ALLOWED_ALPHABET,
    PRE_WINDOW_ASSIGNMENT_STAGE,
    REFERENCE_BASELINE_POLICY,
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


@dataclass(frozen=True)
class FelidSpeciesEntry:
    """One approved species pinned to its RefSeq identifier and assembly name.

    The felid-foundation corpus mixes six species, and the per-species
    FASTA filename is deterministically derived from the identifier + assembly
    name. Freezing the triple in a typed entry lets every downstream stage
    (path resolution, logging, run-summary keys) share the same species slug
    without re-deriving it from the Latin binomial each time.

    Attributes:
        species: Latin binomial as written in ``APPROVED_FELID_ASSEMBLIES``
            (e.g. ``"Panthera leo"``).
        identifier: RefSeq identifier prefixed with ``GCF_`` or DNA Zoo ID.
        assembly_name: NCBI assembly name (e.g. ``"P.leo_Ple1_pat1.1"``).
        species_slug: Lowercase underscored identifier derived from ``species``
            (e.g. ``"panthera_leo"``). Used as ``individual_id`` on every
            emitted ``WindowRecord`` and as the keying identifier in the
            per-species run-summary map.
    """

    species: str
    identifier: str
    assembly_name: str
    species_slug: str


@dataclass(frozen=True)
class FelidFoundationPathsConfig:
    """Filesystem paths required by the felid-foundation pretraining pipeline.

    The foundation corpus is reference-FASTA-only and never invokes VCF
    consensus calling. The loader does not verify that paths exist; that
    responsibility belongs to the stage that first accesses them so we can
    still validate configs on a machine without the reference directory
    materialised.

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

    The foundation path uses a multi-species FASTA-only input (no VCF, no
    consensus, no BioProject pinning) and shares the windowing / tokenizer /
    split / export helpers across active training workflows.

    Attributes:
        name: Human-readable pipeline name.
        description: Free-text description of the pipeline run.
        species: Ordered tuple of :class:`FelidSpeciesEntry`, deduplicated
            by identifier and validated against ``APPROVED_FELID_IDENTIFIERS``.
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


def _validate_felid_species_entries(
    raw_species: object,
) -> tuple[FelidSpeciesEntry, ...]:
    """Validate the ``[[species]]`` list against the approved felid registry.

    Prevents corpus drift. The foundation contract pins the exact
    set of six approved identifiers in :data:`APPROVED_FELID_IDENTIFIERS`;
    any config that references an unknown identifier, duplicates an
    identifier, or declares a species/identifier pair that does not match
    the pinned :data:`APPROVED_FELID_ASSEMBLIES` registry is rejected.

    Args:
        raw_species: The raw value parsed from the TOML ``[[species]]`` list.

    Returns:
        A tuple of validated :class:`FelidSpeciesEntry` in registry order
        (sorted by identifier for determinism).

    Raises:
        ValueError: If the list is not a non-empty sequence, shorter than
            the required species count, contains duplicate identifiers, or
            references a species/identifier pair that is not in the approved
            registry.
    """
    if not isinstance(raw_species, list) or not raw_species:
        raise ValueError("species must be a non-empty list of {species, identifier} entries")
    if len(raw_species) < REQUIRED_FELID_FOUNDATION_SPECIES_COUNT:
        raise ValueError(
            "species list must include all "
            f"{REQUIRED_FELID_FOUNDATION_SPECIES_COUNT} approved felid identifiers; "
            f"got {len(raw_species)}"
        )

    registry_by_identifier: dict[str, FelidAssembly] = {
        assembly.identifier: assembly for assembly in APPROVED_FELID_ASSEMBLIES
    }

    seen_identifiers: set[str] = set()
    entries: list[FelidSpeciesEntry] = []
    for index, item in enumerate(raw_species):
        if not isinstance(item, dict):
            raise ValueError(
                f"species[{index}] must be a table with 'species' and 'identifier' keys"
            )
        if "accession" in item:
            raise ValueError(
                "TOML key 'accession' was renamed to 'identifier'. "
                "Update your config file to use 'identifier = ...' instead."
            )
        try:
            species_name = item["species"]
            identifier = item["identifier"]
        except KeyError as exc:
            raise ValueError(f"species[{index}] is missing required field: {exc.args[0]}") from exc
        if identifier in seen_identifiers:
            raise ValueError(f"species list contains duplicate identifier {identifier!r}")
        if identifier not in APPROVED_FELID_IDENTIFIERS:
            raise ValueError(
                f"species[{index}].identifier {identifier!r} is not an approved felid identifier"
            )
        registry_entry = registry_by_identifier[identifier]
        if species_name != registry_entry.species:
            raise ValueError(
                f"species[{index}].species {species_name!r} does not match "
                f"the approved registry entry {registry_entry.species!r} "
                f"for identifier {identifier}"
            )
        seen_identifiers.add(identifier)
        entries.append(
            FelidSpeciesEntry(
                species=registry_entry.species,
                identifier=registry_entry.identifier,
                assembly_name=registry_entry.assembly_name,
                species_slug=_slugify_species(registry_entry.species),
            )
        )

    entries.sort(key=lambda entry: entry.identifier)
    return tuple(entries)


def load_felid_foundation_pipeline_config(
    path: str | Path,
) -> FelidFoundationPipelineConfig:
    """Load and validate a felid-foundation pretraining TOML config.

    Performs exhaustive contract enforcement across all eight required
    sections (``pipeline``, ``paths``, ``species``, ``windowing``,
    ``split``, ``tokenizer``, ``export``, ``runtime``). Key validations:

    * ``[[species]]`` list against :data:`APPROVED_FELID_IDENTIFIERS`,
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
    It remains as the CLI entry point for validating the runtime contract.

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
        f"  - {entry.species} ({entry.identifier} / {entry.assembly_name})"
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
            raise ValueError("training.corpus_metadata_path must be an absolute path")
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
        if gradient_clip < 0.0:
            raise ValueError("training.gradient_clip must be non-negative")
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


@dataclass(frozen=True)
class MtlFinetuneConfig:
    """Immutable, validated configuration for DNABERT-2 jaguar multi-task fine-tuning.

    Mirrors :class:`FoundationTrainingConfig` in structure: a small set of
    required filesystem paths plus defaulted hyperparameters for a two-phase
    training run. All fields are validated by
    :func:`load_mtl_finetune_config` before construction.

    Attributes:
        backbone_path: Local path to a pretrained DNABERT-2 checkpoint
            (e.g. ``models/foundation_felid/best/hf_model``).
        windows_jsonl: JSONL of :class:`~jaguar_geo_assign.data.finetune_windows.FinetuneWindow`
            records produced by :func:`finetune_windows.write_locus_windows_jsonl`.
        metadata_csv: CSV containing jaguar-level metadata (sample_id,
            individual_id, biome label, latitude, longitude).
        output_dir: Root directory for checkpoints and normalisation artefacts.
        n_folds: Number of StratifiedGroupKFold splits.
        fold_index: Zero-based index of the active fold.
        pooling_strategy: Pooled representation to feed the heads ("cls" or "mean").
        n_biomes: Number of biome classes (pinned to 5 by contract).
        phase1_steps: Training steps in the heads-only warm-up phase.
        phase2_steps: Training steps in the joint backbone+heads phase.
        unfreeze_layers: Number of final transformer layers to unfreeze in phase 2.
        lr_heads_phase1: Learning rate for task heads during phase 1.
        lr_backbone_phase2: Learning rate for the backbone during phase 2.
        lr_heads_phase2: Learning rate for task heads during phase 2.
        cls_loss_weight: Weight for the biome classification loss term.
        reg_loss_weight: Weight for the coordinate regression loss term.
        huber_delta: Delta parameter for the Huber regression loss.
        per_device_train_batch_size: Training batch size per device.
        per_device_eval_batch_size: Evaluation batch size per device.
        gradient_accumulation_steps: Gradient accumulation steps before an optimiser step.
        warmup_fraction: Fraction of total steps used for linear LR warmup.
        gradient_clip: Maximum gradient norm (0 disables clipping).
        seed: Random seed for reproducibility.
        num_workers: DataLoader worker processes.
        weight_decay: AdamW weight-decay coefficient.
        log_every: Logging frequency in steps.
        eval_every: Evaluation frequency in steps.
        save_every: Checkpoint save frequency in steps.
        tensorboard_subdir: Subdirectory under output_dir for TensorBoard logs.
        dropout: Dropout probability applied in the MTL heads.
    """

    backbone_path: Path
    windows_jsonl: Path
    metadata_csv: Path
    output_dir: Path
    n_folds: int = 5
    fold_index: int = 0
    pooling_strategy: str = "cls"
    n_biomes: int = 5
    phase1_steps: int = 1000
    phase2_steps: int = 3000
    unfreeze_layers: int = 2
    lr_heads_phase1: float = 1e-4
    lr_backbone_phase2: float = 1e-5
    lr_heads_phase2: float = 1e-4
    cls_loss_weight: float = 1.0
    reg_loss_weight: float = 0.1
    huber_delta: float = 1.0
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 32
    gradient_accumulation_steps: int = 4
    warmup_fraction: float = 0.1
    gradient_clip: float = 1.0
    seed: int = 42
    num_workers: int = 0
    weight_decay: float = 0.01
    log_every: int = 10
    eval_every: int = 100
    save_every: int = 500
    tensorboard_subdir: str = "tensorboard"
    dropout: float = 0.1


def load_mtl_finetune_config(path: str | Path) -> MtlFinetuneConfig:
    """Load and validate a DNABERT-2 jaguar multi-task fine-tuning TOML config.

    Follows the same loader pattern as :func:`load_foundation_training_config`.
    Hyperparameters are read from the ``[training]`` section and validated
    against the fine-tuning contracts defined in :class:`MtlFinetuneConfig`.

    Raises:
        ValueError: If any contract check fails or a required field is missing.
    """

    raw = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    try:
        training = raw["training"]

        backbone_path = Path(training["backbone_path"])
        windows_jsonl = Path(training["windows_jsonl"])
        metadata_csv = Path(training["metadata_csv"])
        output_dir = Path(training["output_dir"])

        n_folds = int(training.get("n_folds", 5))
        fold_index = int(training.get("fold_index", 0))
        pooling_strategy = str(training.get("pooling_strategy", "cls"))
        n_biomes = int(training.get("n_biomes", 5))
        phase1_steps = int(training.get("phase1_steps", 1000))
        phase2_steps = int(training.get("phase2_steps", 3000))
        unfreeze_layers = int(training.get("unfreeze_layers", 2))
        lr_heads_phase1 = float(training.get("lr_heads_phase1", 1e-4))
        lr_backbone_phase2 = float(training.get("lr_backbone_phase2", 1e-5))
        lr_heads_phase2 = float(training.get("lr_heads_phase2", 1e-4))
        cls_loss_weight = float(training.get("cls_loss_weight", 1.0))
        reg_loss_weight = float(training.get("reg_loss_weight", 0.1))
        huber_delta = float(training.get("huber_delta", 1.0))
        per_device_train_batch_size = int(training.get("per_device_train_batch_size", 16))
        per_device_eval_batch_size = int(training.get("per_device_eval_batch_size", 32))
        gradient_accumulation_steps = int(training.get("gradient_accumulation_steps", 4))
        warmup_fraction = float(training.get("warmup_fraction", 0.1))
        gradient_clip = float(training.get("gradient_clip", 1.0))
        seed = int(training.get("seed", 42))
        num_workers = int(training.get("num_workers", 0))
        weight_decay = float(training.get("weight_decay", 0.01))
        log_every = int(training.get("log_every", 10))
        eval_every = int(training.get("eval_every", 100))
        save_every = int(training.get("save_every", 500))
        tensorboard_subdir = str(training.get("tensorboard_subdir", "tensorboard"))
        dropout = float(training.get("dropout", 0.1))

        # Loader contract enforcement
        if pooling_strategy not in {"cls", "mean"}:
            raise ValueError("training.pooling_strategy must be 'cls' or 'mean'")
        if n_biomes != 5:
            raise ValueError("training.n_biomes must be exactly 5 for the current contract")
        if unfreeze_layers not in {2, 3}:
            raise ValueError("training.unfreeze_layers must be 2 or 3")
        if not 0 <= fold_index < n_folds:
            raise ValueError("training.fold_index must satisfy 0 <= fold_index < n_folds")
        if not 0.0 < warmup_fraction < 1.0:
            raise ValueError("training.warmup_fraction must be in the open interval (0, 1)")
        if cls_loss_weight <= 0:
            raise ValueError("training.cls_loss_weight must be positive")
        if reg_loss_weight <= 0:
            raise ValueError("training.reg_loss_weight must be positive")
        if huber_delta <= 0:
            raise ValueError("training.huber_delta must be positive")
        if phase1_steps <= 0 or phase2_steps <= 0:
            raise ValueError("training.phase1_steps and training.phase2_steps must be positive")
        if weight_decay < 0:
            raise ValueError("training.weight_decay must be non-negative")
        if gradient_clip < 0.0:
            raise ValueError("training.gradient_clip must be non-negative")

    except KeyError as exc:
        msg = f"MTL fine-tune config is missing required field: {exc.args[0]}"
        raise ValueError(msg) from exc

    return MtlFinetuneConfig(
        backbone_path=backbone_path,
        windows_jsonl=windows_jsonl,
        metadata_csv=metadata_csv,
        output_dir=output_dir,
        n_folds=n_folds,
        fold_index=fold_index,
        pooling_strategy=pooling_strategy,
        n_biomes=n_biomes,
        phase1_steps=phase1_steps,
        phase2_steps=phase2_steps,
        unfreeze_layers=unfreeze_layers,
        lr_heads_phase1=lr_heads_phase1,
        lr_backbone_phase2=lr_backbone_phase2,
        lr_heads_phase2=lr_heads_phase2,
        cls_loss_weight=cls_loss_weight,
        reg_loss_weight=reg_loss_weight,
        huber_delta=huber_delta,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_fraction=warmup_fraction,
        gradient_clip=gradient_clip,
        seed=seed,
        num_workers=num_workers,
        weight_decay=weight_decay,
        log_every=log_every,
        eval_every=eval_every,
        save_every=save_every,
        tensorboard_subdir=tensorboard_subdir,
        dropout=dropout,
    )
