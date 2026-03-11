"""Shared architecture contracts for the feline genomics pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import shutil

DNABERT2_TOKENIZER_ID = "zhihan1996/DNABERT-2-117M"
DNABERT2_TOKENIZER_REVISION = "7bce263b15377fc15361f52cfab88f8b586abda0"
POST_CONSENSUS_ALLOWED_ALPHABET = ("A", "C", "G", "T", "N")
GLOBAL_LOCUS_SPLIT_STRATEGY = "global_locus_block"
PRE_WINDOW_ASSIGNMENT_STAGE = "before_windowing"
REFERENCE_BASELINE_POLICY = "reuse_locus_assignments"
REQUIRED_EXTERNAL_TOOLS = ("bcftools",)
REQUIRED_SAMPLE_MANIFEST_FIELDS = ("sample_id", "individual_id", "vcf_path")
PYARROW_INSTALL_HINT = 'uv add "pyarrow>=16,<20"'
EXPLICIT_CONSENSUS_POLICIES = {
    "emit_reference_if_callable",
    "apply_alternate_allele",
    "mask_and_report",
}


@dataclass(frozen=True)
class RuntimeToolRequirement:
    name: str
    install_hint: str


def get_runtime_tool_requirements(
    tool_names: tuple[str, ...] = REQUIRED_EXTERNAL_TOOLS,
) -> tuple[RuntimeToolRequirement, ...]:
    return tuple(
        RuntimeToolRequirement(
            name=tool_name,
            install_hint=f"Install {tool_name} and ensure it is available on PATH.",
        )
        for tool_name in tool_names
    )


def assert_external_tools_available(tool_names: tuple[str, ...] = REQUIRED_EXTERNAL_TOOLS) -> None:
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