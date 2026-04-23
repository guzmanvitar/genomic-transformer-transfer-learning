"""Shared architecture contracts for the feline genomics pipeline.

This module defines the single-source-of-truth constants, enumerations, and
runtime checks that every stage of the jaguar geographic-assignment pipeline
must respect.  By centralising these values here, downstream modules can
import them instead of duplicating magic strings, preventing silent
contract drift between data ingestion, consensus calling, windowing, and
model training.
"""

from __future__ import annotations

from dataclasses import dataclass
import shutil

# Data-provenance contracts

# NCBI BioProject accession that all input VCFs must originate from.
APPROVED_BIOPROJECT_ACCESSION = "PRJNA308208"

# NCBI reference assembly used for variant calling and coordinate mapping.
APPROVED_REFERENCE_ASSEMBLY = "Felis_catus_9.0"

# DNABERT-2 tokenizer pinned coordinates

# Hugging Face model identifier for the pre-trained DNABERT-2 tokenizer.
DNABERT2_TOKENIZER_ID = "zhihan1996/DNABERT-2-117M"

# Exact Git revision hash pinned for reproducible tokenizer loading.
DNABERT2_TOKENIZER_REVISION = "7bce263b15377fc15361f52cfab88f8b586abda0"

# DNABERT-2 requires custom tokenizer code; this flag authorises it.
DNABERT2_TRUST_REMOTE_CODE = True

# Consensus-sequence contracts

# Only these five characters may appear after consensus resolution.
# Any base outside this alphabet indicates a pipeline bug.
POST_CONSENSUS_ALLOWED_ALPHABET = ("A", "C", "G", "T", "N")

# Train / validation / test split contracts

# Split strategy that assigns entire locus blocks to a single fold,
# preventing data leakage from overlapping genomic windows.
GLOBAL_LOCUS_SPLIT_STRATEGY = "global_locus_block"

# Sentinel value indicating that split assignment happens before the
# windowing stage, so every window inherits its parent locus's fold.
PRE_WINDOW_ASSIGNMENT_STAGE = "before_windowing"

# Policy for the reference-genome baseline: reuse the locus-level
# fold assignments rather than re-splitting at the window level.
REFERENCE_BASELINE_POLICY = "reuse_locus_assignments"

# External-tool and manifest contracts

# CLI tools that must be on $PATH before any pipeline stage executes.
REQUIRED_EXTERNAL_TOOLS = ("bcftools",)

# Mandatory columns in the sample manifest CSV/TSV.
REQUIRED_SAMPLE_MANIFEST_FIELDS = ("sample_id", "individual_id", "vcf_path")

# Felid foundation corpus contracts.
#
# The felid-foundation pretraining path mixes six reference assemblies
# into a single tokenized corpus. Pinning the *set* of approved accessions at
# contract level (rather than taking the list directly from
# ``felid_assemblies.APPROVED_FELID_ASSEMBLIES``) gives the config loader an
# O(1) membership check and makes drift visible in PR diffs: adding a species
# requires updating both the registry and this frozenset, exactly like the
# feline BioProject pinning above.
from .felid_assemblies import APPROVED_FELID_ASSEMBLIES as _APPROVED_FELID_ASSEMBLIES

APPROVED_FELID_ACCESSIONS: frozenset[str] = frozenset(
    assembly.accession for assembly in _APPROVED_FELID_ASSEMBLIES
)
"""Frozen set of the six pinned RefSeq accessions approved for the felid foundation corpus."""

REQUIRED_FELID_FOUNDATION_SPECIES_COUNT = len(_APPROVED_FELID_ASSEMBLIES)
"""The felid foundation contract requires exactly this many approved species."""

# Consensus-calling policy whitelist

# Exhaustive set of allowed policies for resolving heterozygous or
# multi-allelic sites during consensus-sequence construction.
EXPLICIT_CONSENSUS_POLICIES = {
    "emit_reference_if_callable",
    "apply_alternate_allele",
    "mask_and_report",
}


@dataclass(frozen=True)
class RuntimeToolRequirement:
    """Immutable descriptor for an external CLI tool required at runtime.

    Attributes:
        name: Executable name expected on ``$PATH`` (e.g. ``"bcftools"``).
        install_hint: Human-readable installation instruction shown when the
            tool is not found.
    """

    name: str
    install_hint: str


def get_runtime_tool_requirements(
    tool_names: tuple[str, ...] = REQUIRED_EXTERNAL_TOOLS,
) -> tuple[RuntimeToolRequirement, ...]:
    """Build ``RuntimeToolRequirement`` objects for the requested tools.

    Args:
        tool_names: Executable names to wrap.  Defaults to
            ``REQUIRED_EXTERNAL_TOOLS``.

    Returns:
        A tuple of ``RuntimeToolRequirement`` instances, one per tool name,
        each carrying a default install hint.
    """
    return tuple(
        RuntimeToolRequirement(
            name=tool_name,
            install_hint=f"Install {tool_name} and ensure it is available on PATH.",
        )
        for tool_name in tool_names
    )


def assert_external_tools_available(tool_names: tuple[str, ...] = REQUIRED_EXTERNAL_TOOLS) -> None:
    """Verify that every required CLI tool is reachable on ``$PATH``.

    This guard should be called at the entry-point of any pipeline stage
    that shells out to external programs, failing fast with actionable
    install instructions rather than producing a cryptic downstream error.

    Args:
        tool_names: Executable names to check.  Defaults to
            ``REQUIRED_EXTERNAL_TOOLS``.

    Raises:
        RuntimeError: If one or more tools cannot be found via
            ``shutil.which``.
    """
    missing = [
        requirement
        for requirement in get_runtime_tool_requirements(tool_names)
        if shutil.which(requirement.name) is None
    ]
    if missing:
        details = "; ".join(
            f"{requirement.name}: {requirement.install_hint}" for requirement in missing
        )
        raise RuntimeError(f"Missing required external tools: {details}")