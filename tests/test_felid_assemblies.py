"""Tests for the felid multi-species registry and reference-FASTA manifest builder.

These tests guard the closed-registry contract for the foundation pretraining corpus:
the six approved GCF accessions, their pinned MD5 checksums, the deterministic NCBI
FTP URL shape, and the in-code assertion that refuses to return any download asset
with an empty checksum. No network calls are made — the point is to catch silent drift
in the pinned registry before any real download is attempted.
"""

from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path

import pytest

from jaguar_geo_assign.data.felid_assemblies import (
    APPROVED_FELID_ASSEMBLIES,
    FelidAssembly,
    build_felid_reference_manifest,
    build_refseq_fasta_url,
)

_EXPECTED_SPECIES = {
    "Felis catus",
    "Panthera leo",
    "Panthera tigris",
    "Panthera onca",
    "Puma concolor",
    "Panthera pardus",
}
_MD5_HEX = re.compile(r"^[0-9a-f]{32}$")


def test_approved_registry_shape() -> None:
    """The registry is closed at exactly six species; adding one is a code change."""
    assert len(APPROVED_FELID_ASSEMBLIES) == 6
    assert {a.species for a in APPROVED_FELID_ASSEMBLIES} == _EXPECTED_SPECIES


def test_every_pinned_md5_is_lowercase_hex() -> None:
    """Guards against typos or uppercase drift that would break hashlib comparison."""
    for assembly in APPROVED_FELID_ASSEMBLIES:
        assert _MD5_HEX.match(assembly.expected_checksum), (
            f"{assembly.identifier} MD5 {assembly.expected_checksum!r} is not lowercase 32-char hex"
        )


@pytest.mark.parametrize("assembly", APPROVED_FELID_ASSEMBLIES, ids=lambda a: a.identifier)
def test_build_refseq_fasta_url_round_trip(assembly: FelidAssembly) -> None:
    """Each pinned identifier produces the canonical NCBI FTP URL shape."""
    url = build_refseq_fasta_url(assembly.identifier, assembly.assembly_name)
    stem = f"{assembly.identifier}_{assembly.assembly_name}"
    numeric = assembly.identifier.split("_", 1)[1].split(".", 1)[0]
    expected = (
        "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/"
        f"{numeric[0:3]}/{numeric[3:6]}/{numeric[6:9]}/{stem}/{stem}_genomic.fna.gz"
    )
    assert url == expected


def test_build_refseq_fasta_url_rejects_non_gcf_prefix() -> None:
    """GCA (GenBank) and other prefixes are out of contract for RefSeq FASTAs."""
    with pytest.raises(ValueError):
        build_refseq_fasta_url("GCA_000181335.3", "Felis_catus_9.0")


@pytest.mark.parametrize(
    "bad_identifier",
    ["GCF_00018133.3", "GCF_0001813355.3", "GCF_ABCDEFGHI.3", "GCF_000181335"],
)
def test_build_refseq_fasta_url_rejects_malformed_numeric_part(bad_identifier: str) -> None:
    """The numeric part must be exactly nine digits in 3/3/3 groupings with a version."""
    with pytest.raises(ValueError):
        build_refseq_fasta_url(bad_identifier, "Felis_catus_9.0")


def test_build_refseq_fasta_url_returns_override() -> None:
    """Bypasses NCBI construction if url_override is set."""
    override = "https://example.com/custom.fna.gz"
    # Even if identifier is invalid, override takes precedence
    url = build_refseq_fasta_url("INVALID_ID", "ASM", url_override=override)
    assert url == override


def test_manifest_returns_six_sorted_reference_assets(tmp_path: Path) -> None:
    """Default manifest is deterministic: six assets, sorted by identifier, kind=reference."""
    manifest = build_felid_reference_manifest(tmp_path)
    assert len(manifest) == 6
    identifiers = [asset.url.rsplit("/", 1)[1].split("_genomic")[0] for asset in manifest]
    assert identifiers == sorted(identifiers)
    for asset in manifest:
        assert asset.kind == "reference"
        assert asset.checksum_name == "md5"
        assert asset.destination.parent == tmp_path / "reference"
        assert asset.destination.suffix == ".gz"


def test_manifest_threads_pinned_checksum_into_asset(tmp_path: Path) -> None:
    """Every asset's checksum matches the registry's ``expected_checksum`` by identifier."""
    manifest = build_felid_reference_manifest(tmp_path)
    by_url: dict[str, str] = {}
    for asset in manifest:
        assert asset.checksum, "checksum must be non-empty for every felid reference"
        by_url[asset.url] = asset.checksum or ""
    for assembly in APPROVED_FELID_ASSEMBLIES:
        url = build_refseq_fasta_url(
            assembly.identifier, assembly.assembly_name, assembly.url_override
        )
        assert by_url[url] == assembly.expected_checksum


def test_manifest_honours_checksum_override(tmp_path: Path) -> None:
    """Per-identifier overrides let tests stub checksums without mutating the registry."""
    target = APPROVED_FELID_ASSEMBLIES[0]
    override_hash = "a" * 32
    manifest = build_felid_reference_manifest(
        tmp_path, checksum_override={target.identifier: override_hash}
    )
    overridden = [
        a
        for a in manifest
        if a.url.endswith(f"{target.identifier}_{target.assembly_name}_genomic.fna.gz")
    ]
    assert len(overridden) == 1
    assert overridden[0].checksum == override_hash


def test_manifest_rejects_empty_checksum_clone(tmp_path: Path) -> None:
    """The in-code assertion fires when any assembly carries an empty ``expected_checksum``."""
    bad_clone = replace(APPROVED_FELID_ASSEMBLIES[0], expected_checksum="")
    with pytest.raises(AssertionError):
        build_felid_reference_manifest(tmp_path, assemblies=(bad_clone,))
