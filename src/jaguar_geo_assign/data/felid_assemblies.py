"""Typed registry and manifest builder for the six approved felid RefSeq assemblies.

This module is the single source of truth for the multi-species felid reference-FASTA
corpus consumed by the foundation pretraining path. It pins each assembly's canonical
NCBI RefSeq accession, assembly name, tax ID, and expected MD5 of the ``_genomic.fna.gz``
file so that ``download_with_retry`` can verify integrity and refuse silently truncated
or stale payloads.

Pinning is deliberate and closed — adding a species requires a code + test change,
which prevents drift from turning the foundation corpus into an untracked mixture.

The MD5s are pinned manually from each assembly's ``md5checksums.txt``. This avoids
an online dependency at registry-construction time while still catching on-disk
corruption when the manifest is consumed.

The module intentionally does not call NCBI Entrez or execute any downloads; it only
produces typed plans. Actual transfer is handled by consumers via the existing
``download_with_retry`` primitive in :mod:`jaguar_geo_assign.data.acquisition`.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from jaguar_geo_assign.data.acquisition import DownloadAsset

_REFSEQ_FTP_ROOT = "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF"
_ACCESSION_PATTERN = re.compile(r"^GCF_(\d{3})(\d{3})(\d{3})\.\d+$")


@dataclass(frozen=True)
class FelidAssembly:
    """Immutable registry entry for one approved felid RefSeq assembly.

    The ``expected_md5`` field is populated from the NCBI ``md5checksums.txt`` file
    inside each assembly's FTP directory at registry build time. It is threaded into
    :class:`DownloadAsset.checksum` by :func:`build_felid_reference_manifest` so that
    any post-download integrity check fails loudly on a truncated or stale file.

    Attributes:
        species: Latin binomial (e.g. ``"Panthera onca"``).
        common_name: Human-readable common name for logging and run summaries.
        identifier: RefSeq assembly accession prefixed with ``GCF_`` or DNA Zoo ID.
        assembly_name: NCBI assembly name used in the canonical FTP path.
        tax_id: NCBI taxonomy identifier for the species.
        expected_checksum: Checksum of the ``_genomic.fna.gz`` file.
        checksum_name: Name of the hash algorithm (default ``"md5"``).
        url_override: Optional direct URL to bypass NCBI FTP construction.
        mirror_url: Optional secondary URL to try if primary fails.
        expected_size: Optional expected file size in bytes.
    """

    species: str
    common_name: str
    identifier: str
    assembly_name: str
    tax_id: int
    expected_checksum: str
    checksum_name: str = "md5"
    url_override: str | None = None
    mirror_url: str | None = None
    expected_size: int | None = None


# Provenance: MD5s pinned manually from
# ``https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/<AAA>/<BBB>/<CCC>/<ACC>_<ASM>/md5checksums.txt``
# on 2026-04-22. Any FASTA whose live MD5 disagrees must either (a) update this pin
# after manual review, or (b) be rejected by the download layer.
APPROVED_FELID_ASSEMBLIES: tuple[FelidAssembly, ...] = (
    FelidAssembly(
        species="Felis catus",
        common_name="Domestic cat",
        identifier="GCF_000181335.3",
        assembly_name="Felis_catus_9.0",
        tax_id=9685,
        expected_checksum="55c8869801266af1b1fb13ffa7464209",
        checksum_name="md5",
        url_override=None,
        mirror_url=None,
        expected_size=None,
    ),
    FelidAssembly(
        species="Panthera leo",
        common_name="Lion",
        identifier="GCF_018350215.1",
        assembly_name="P.leo_Ple1_pat1.1",
        tax_id=9689,
        expected_checksum="33cb54c850de6090ff6364f3823606ac",
        checksum_name="md5",
        url_override=None,
        mirror_url=None,
        expected_size=None,
    ),
    FelidAssembly(
        species="Panthera tigris",
        common_name="Amur tiger",
        identifier="GCF_000464555.1",
        assembly_name="PanTig1.0",
        tax_id=74533,
        expected_checksum="7d373b516c4ae82a1dc9c18bcc1ca389",
        checksum_name="md5",
        url_override=None,
        mirror_url=None,
        expected_size=None,
    ),
    FelidAssembly(
        species="Panthera onca",
        common_name="Jaguar",
        identifier="DNAZOO_Panthera_onca_HiC",
        assembly_name="Panthera_onca_HiC",
        tax_id=9690,
        expected_checksum="3b3811dff68a704075cfdceb5fb3f1f4a869d6deda6dced6dd159b6498919229",
        checksum_name="sha256",
        url_override="https://dnazoo.s3.wasabisys.com/Panthera_onca/Panthera_onca_HiC.fasta.gz",
        mirror_url="https://huggingface.co/datasets/coding-racoon/jaguar-reference-dnazoo/resolve/470b6aae8eabeaf7a6bf9c841583be4b271519ca/Panthera_onca_HiC.fasta.gz",
        expected_size=745_951_926,
    ),
    FelidAssembly(
        species="Puma concolor",
        common_name="Puma",
        identifier="GCF_003327715.1",
        assembly_name="PumCon1.0",
        tax_id=9696,
        expected_checksum="0bbbfb230e807f8601d1017449f71ea5",
        checksum_name="md5",
        url_override=None,
        mirror_url=None,
        expected_size=None,
    ),
    FelidAssembly(
        species="Panthera pardus",
        common_name="Leopard",
        identifier="GCF_001857705.1",
        assembly_name="PanPar1.0",
        tax_id=9691,
        expected_checksum="e3e5fa056f33374224d43aeddb673b2d",
        checksum_name="md5",
        url_override=None,
        mirror_url=None,
        expected_size=None,
    ),
)


def build_refseq_fasta_url(
    identifier: str, assembly_name: str, url_override: str | None = None
) -> str:
    """Return the canonical RefSeq FASTA URL for a pinned ``GCF_*`` accession.

    The URL is derived deterministically by splitting the nine digits of the
    identifier into three-digit groups. We validate the shape up front so that a
    typo in a pinned identifier fails loudly at URL-construction time rather than
    producing a 404 during download.
    """
    if url_override is not None:
        return url_override

    match = _ACCESSION_PATTERN.match(identifier)
    if not match:
        raise ValueError(
            f"identifier {identifier!r} must match GCF_<9 digits grouped 3/3/3>.<version>"
        )
    aaa, bbb, ccc = match.group(1), match.group(2), match.group(3)
    stem = f"{identifier}_{assembly_name}"
    return f"{_REFSEQ_FTP_ROOT}/{aaa}/{bbb}/{ccc}/{stem}/{stem}_genomic.fna.gz"


def build_felid_reference_manifest(
    output_dir: str | Path,
    *,
    assemblies: Sequence[FelidAssembly] = APPROVED_FELID_ASSEMBLIES,
    checksum_override: Mapping[str, str] | None = None,
) -> tuple[DownloadAsset, ...]:
    """Build a deterministic per-species download plan for the felid corpus.

    Assets are sorted by identifier so manifest order is reproducible across runs
    regardless of the input sequence order. The ``expected_checksum`` on each assembly
    is threaded into :attr:`DownloadAsset.checksum` with ``checksum_name="md5"``;
    an in-code assertion refuses to return any asset with an empty checksum so a
    future edit cannot silently drop the integrity guard.
    """
    reference_dir = Path(output_dir) / "reference"
    override = dict(checksum_override or {})
    ordered = sorted(assemblies, key=lambda a: a.identifier)
    assets: list[DownloadAsset] = []
    for assembly in ordered:
        checksum = override.get(assembly.identifier, assembly.expected_checksum)
        assets.append(
            DownloadAsset(
                url=build_refseq_fasta_url(
                    assembly.identifier, assembly.assembly_name, assembly.url_override
                ),
                destination=reference_dir / f"{assembly.identifier}.fna.gz",
                checksum=checksum,
                checksum_name=assembly.checksum_name,
                mirror_url=assembly.mirror_url,
                expected_size=assembly.expected_size,
                kind="reference",
            )
        )
    assert all(asset.checksum for asset in assets), (
        "build_felid_reference_manifest produced an asset with empty checksum; "
        "every felid reference download must be verified"
    )
    return tuple(assets)
