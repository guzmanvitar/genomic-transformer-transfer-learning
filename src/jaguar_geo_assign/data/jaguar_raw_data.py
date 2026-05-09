"""Typed registry and manifest builder for the two jaguar raw input files.

This module is the single source of truth for the raw jaguar data files
consumed by the fine-tuning path:

* A multi-sample VCF with hard-filtered, MAF/LD/HWE-cleaned SNPs.
* A location CSV mapping each sample to its geographic coordinates and
  biome label.

Both files are hosted on HuggingFace (``coding-racoon/jaguar-raw-data``) as
public datasets and can be downloaded without credentials. The VCF was
originally published on DataDryad (doi:10.5061/dryad.4tmpg4fkm, CC0 license);
the SHA-256 pin was cross-validated against DataDryad's API metadata digest.

This module intentionally does not execute any downloads; it only produces
typed plans consumed by :mod:`jaguar_geo_assign.data.jaguar_raw_acquisition`
via the :func:`jaguar_geo_assign.data.acquisition.download_with_retry`
primitive.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from jaguar_geo_assign.data.acquisition import DownloadAsset

_HF_BASE = "https://huggingface.co/datasets/coding-racoon/jaguar-raw-data/resolve/main"


@dataclass(frozen=True)
class JaguarRawAsset:
    """Immutable registry entry for one jaguar raw input file.

    Attributes:
        name: Short human-readable label used in log messages.
        destination_name: Filename written under the caller-supplied output
            directory.
        url: Fully-qualified HTTPS URL for the remote resource.
        checksum: Expected lowercase hex SHA-256 digest.
        expected_size: Optional expected file size in bytes.
    """

    name: str
    destination_name: str
    url: str
    checksum: str
    expected_size: int | None = None


# Provenance:
#   VCF  – doi:10.5061/dryad.4tmpg4fkm (CC0); SHA-256 cross-validated against
#           DataDryad API digest (digestType: sha-256) on 2026-05-09.
#   CSV  – SHA-256 computed locally from the canonical copy on 2026-05-09.
JAGUAR_RAW_ASSETS: tuple[JaguarRawAsset, ...] = (
    JaguarRawAsset(
        name="jaguar-vcf",
        destination_name=(
            "jaguar.57samples.allChr.snps.hardFilter.bi.maf.ld.masked.hwe.recode.vcf"
        ),
        url=(f"{_HF_BASE}/jaguar.57samples.allChr.snps.hardFilter.bi.maf.ld.masked.hwe.recode.vcf"),
        checksum="8bf9083a5708b99c7935f1ac7688a43d3561237397cd70121455009a2d93308b",
        expected_size=147_354_171,
    ),
    JaguarRawAsset(
        name="jaguar-location-csv",
        destination_name="jaguar_location.csv",
        url=f"{_HF_BASE}/jaguar_location.csv",
        checksum="684831526787193fbe7ab8885165b72cc2a07cf4b83b3eb12de24a6f4cfbb691",
    ),
)


def build_jaguar_raw_manifest(output_dir: str | Path) -> tuple[DownloadAsset, ...]:
    """Return one :class:`DownloadAsset` per jaguar raw file, rooted at *output_dir*.

    Assets are ordered identically to :data:`JAGUAR_RAW_ASSETS` so callers
    can zip results back to registry entries when needed. The in-code
    assertion refuses to return any asset with an empty checksum so a
    future edit cannot silently drop the integrity guard.
    """
    dest = Path(output_dir)
    assets = tuple(
        DownloadAsset(
            url=asset.url,
            destination=dest / asset.destination_name,
            checksum=asset.checksum,
            checksum_name="sha256",
            kind="jaguar-raw",
            expected_size=asset.expected_size,
        )
        for asset in JAGUAR_RAW_ASSETS
    )
    assert all(a.checksum for a in assets), (
        "build_jaguar_raw_manifest produced an asset with empty checksum; "
        "every jaguar raw download must be integrity-verified"
    )
    return assets
