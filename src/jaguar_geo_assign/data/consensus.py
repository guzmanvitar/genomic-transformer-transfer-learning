"""Shared VCF validation helpers for jaguar fine-tuning workflows.

Only the pure-Python parsing and reference-validation utilities that are
still imported by ``data.finetune_windows`` remain in this module.
"""

from __future__ import annotations

import gzip
import re
from collections.abc import Sequence
from pathlib import Path

from .acquisition import AcquisitionError

EXPECTED_REFERENCE_TOKENS = ("GCF_000181335.3", "Felis_catus_9.0")
PASSING_FILTER_VALUES = frozenset({"PASS", "."})


class ReferenceMismatchError(AcquisitionError):
    """The FASTA reference and VCF ``##reference`` header disagree on build.

    Raised during consensus preparation when either the reference FASTA file
    name or the VCF ``##reference`` metadata line does not contain the
    expected canonical build tokens (e.g. ``GCF_000181335.3``,
    ``Felis_catus_9.0``).  This prevents silent genome-build mismatches from
    corrupting consensus sequences.
    """


class ContigMismatchError(AcquisitionError):
    """A VCF contig identifier is absent from the reference FASTA.

    Raised when a ``##contig=<ID=…>`` header or a data-record chromosome
    field references a contig that does not appear in the reference FASTA
    header lines.  This typically indicates that the VCF was called against
    a different assembly or that contig naming conventions differ.
    """


class MalformedGenotypeError(AcquisitionError):
    """A VCF ``GT`` field contains non-numeric, non-missing allele tokens.

    Raised by :func:`_validated_gt_tokens` when allele tokens (after
    splitting on ``/`` or ``|``) are neither valid digit strings nor the
    VCF-standard ``.`` no-call marker.  The error message includes the
    sample ID, locus, and VCF path for rapid triage.
    """


def _normalize_alt_alleles(alts: Sequence[str]) -> tuple[str, ...]:
    """Normalise VCF ALT alleles, collapsing the monomorphic ``"."`` sentinel.

    The VCF specification uses a single ``"."`` in the ALT column to
    indicate a monomorphic reference site (no alternate alleles).  This
    function converts that sentinel into an empty tuple so that downstream
    length checks (e.g. ``len(normalized_alts) > 1`` for multi-allelic
    detection) work uniformly without special-casing.

    .. warning::

       Only a **single-element** list containing exactly ``"."`` is
       collapsed.  A multi-element list like ``["A", "."]`` is passed
       through unchanged — the ``"."`` would be treated as a literal
       allele string, which is intentional: such records are malformed
       and will be caught by downstream allele-index validation.

    Args:
        alts: Raw ALT column values, already split on ``,``.

    Returns:
        An empty tuple for monomorphic sites, or the input converted to a
        tuple otherwise.
    """
    if len(alts) == 1 and alts[0] == ".":
        return ()
    return tuple(alts)


def _validated_gt_tokens(
    genotype: str | None,
    *,
    sample_id: str | None,
    contig: str | None,
    position: int | None,
    vcf_path: str | Path | None,
) -> list[str] | None:
    """Parse and validate a VCF ``GT`` field into numeric allele tokens.

    Splits the genotype string on ``/`` (unphased) or ``|`` (phased).
    The separator detection is order-dependent: ``/`` is checked first,
    so a genotype like ``"0/1"`` is always treated as unphased even if
    ``|`` appears elsewhere.  This matches the VCF 4.x specification
    where ``/`` and ``|`` are mutually exclusive separators within a
    single GT field.

    The function distinguishes three outcomes:

    * **None** — returned when the genotype is empty/falsy, or contains
      any ``.`` no-call token (VCF-standard missing allele marker).
    * **list[str]** — returned when all tokens are valid digit strings.
    * **MalformedGenotypeError** — raised when tokens are non-numeric
      *and* non-missing, which indicates a corrupted or non-standard VCF.

    Args:
        genotype: Raw GT field value (e.g. ``"0/1"``, ``"1|1"``,
            ``"./."``) or ``None``.
        sample_id: Optional sample ID for diagnostic error messages.
        contig: Optional contig name for diagnostic error messages.
        position: Optional 1-based position for diagnostic error messages.
        vcf_path: Optional VCF path for diagnostic error messages.

    Returns:
        A list of digit-string allele tokens, or ``None`` for missing /
        no-call genotypes.

    Raises:
        MalformedGenotypeError: If any allele token is neither a digit
            string nor the ``"."`` no-call marker.
    """
    if not genotype:
        return None
    separator = "/" if "/" in genotype else "|"
    allele_tokens = genotype.split(separator)
    if not allele_tokens or any(token == "." for token in allele_tokens):
        return None

    malformed_tokens = sorted({token for token in allele_tokens if not token.isdigit()})
    if malformed_tokens:
        sample_fragment = f" for sample '{sample_id}'" if sample_id else ""
        locus_fragment = f" at {contig}:{position}" if contig and position is not None else ""
        vcf_fragment = f" in VCF {vcf_path}" if vcf_path is not None else ""
        raise MalformedGenotypeError(
            "Malformed non-numeric GT token(s) "
            f"{malformed_tokens} in GT='{genotype}'"
            f"{sample_fragment}{locus_fragment}{vcf_fragment}; "
            "expected numeric allele indices or '.' no-call markers."
        )
    return allele_tokens


def _raise_malformed_vcf_record(
    *,
    sample_vcf: Path,
    sample_id: str,
    line_number: int,
    raw_record: str,
    observed_columns: int,
    expected_min_columns: int,
) -> None:
    """Raise a diagnostic :class:`AcquisitionError` for a malformed VCF record.

    Constructs a human-readable error message that includes the file path,
    sample ID, line number, expected vs. observed column counts, and a
    truncated preview of the offending record (capped at 200 characters).

    This helper is shared by VCF parsing code paths so malformed rows fail
    loudly with actionable context instead of being dropped silently.

    Args:
        sample_vcf: Path to the VCF file being parsed.
        sample_id: Sample identifier for the error message.
        line_number: 1-based line number of the malformed record.
        raw_record: The raw tab-delimited line (stripped of newline).
        observed_columns: Actual number of tab-delimited fields found.
        expected_min_columns: Minimum columns required (at least 9, or
            the sample column index + 1).

    Raises:
        AcquisitionError: Always raised with a descriptive message.
    """
    record_preview = raw_record if raw_record else "<blank line>"
    if len(record_preview) > 200:
        record_preview = f"{record_preview[:197]}..."
    raise AcquisitionError(
        "Malformed VCF record "
        f"in {sample_vcf} for sample '{sample_id}' at line {line_number}: expected at least "
        f"{expected_min_columns} tab-delimited columns, found {observed_columns}; "
        f"record={record_preview!r}"
    )


def _read_fasta_headers(reference_fasta: Path) -> dict[str, str]:
    """Extract contig names and full header lines from a FASTA file.

    Reads only the ``>``-prefixed header lines (not sequence data).
    The first whitespace-delimited token of each header is used as the
    contig key.

    Args:
        reference_fasta: Path to the FASTA file (plain or gzipped).

    Returns:
        Dict mapping contig name → full header string (without the
        leading ``>``).

    Raises:
        AcquisitionError: If the file contains no contig headers.
    """
    headers: dict[str, str] = {}
    with _open_maybe_gzip(reference_fasta) as handle:
        for line in handle:
            if line.startswith(">"):
                full_header = line[1:].strip()
                headers[full_header.split()[0]] = full_header
    if not headers:
        raise AcquisitionError(
            f"Reference FASTA {reference_fasta} did not contain any contig headers"
        )
    return headers


def _matches_expected_reference_build(
    evidence: str, positive_reference_tokens: Sequence[str]
) -> bool:
    """Check whether *evidence* contains all expected reference build tokens.

    Both the evidence string and each token are canonicalized (lowercased,
    non-alphanumeric characters replaced with ``_``) before substring
    matching.

    Args:
        evidence: Free-form string to search (e.g. a FASTA filename
            concatenated with contig headers).
        positive_reference_tokens: Canonical token strings that must all
            appear in *evidence*.

    Returns:
        ``True`` if every token is found in the canonicalized evidence.
    """
    canonical_evidence = _canonicalize_reference_evidence(evidence)
    return all(
        _canonicalize_reference_evidence(token) in canonical_evidence
        for token in positive_reference_tokens
    )


def _canonicalize_reference_evidence(evidence: str) -> str:
    """Lowercase and normalise a string for case-insensitive token matching.

    Replaces all non-alphanumeric character runs with ``_`` and strips
    leading/trailing underscores.

    Args:
        evidence: Raw string to canonicalize.

    Returns:
        Normalised string suitable for substring containment checks.
    """
    return re.sub(r"[^a-z0-9]+", "_", evidence.lower()).strip("_")


def _open_maybe_gzip(path: Path):
    """Open a file for text reading, decompressing if the suffix is ``.gz``.

    Args:
        path: Path to the file.

    Returns:
        A text-mode file handle (UTF-8).
    """
    return (
        gzip.open(path, "rt", encoding="utf-8")
        if path.suffix == ".gz"
        else path.open("r", encoding="utf-8")
    )


canonicalize_reference_evidence = _canonicalize_reference_evidence
