"""Tests for the jaguar raw data registry and manifest builder.

Guards the pinned registry contract for the two jaguar raw input files: the
VCF and location CSV, their SHA-256 checksums, and the HuggingFace URLs. No
network calls are made — the point is to catch silent drift in the pinned
registry before any real download is attempted.
"""

from __future__ import annotations

import re
from pathlib import Path

from jaguar_geo_assign.data.jaguar_raw_data import JAGUAR_RAW_ASSETS, build_jaguar_raw_manifest

_SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")
_HF_PREFIX = "https://huggingface.co/datasets/coding-racoon/jaguar-raw-data/"

_EXPECTED_NAMES = {"jaguar-vcf", "jaguar-location-csv"}
_EXPECTED_FILENAMES = {
    "jaguar.57samples.allChr.snps.hardFilter.bi.maf.ld.masked.hwe.recode.vcf",
    "jaguar_location.csv",
}


def test_registry_has_exactly_two_assets() -> None:
    assert len(JAGUAR_RAW_ASSETS) == 2
    assert {a.name for a in JAGUAR_RAW_ASSETS} == _EXPECTED_NAMES


def test_destination_names_are_expected() -> None:
    assert {a.destination_name for a in JAGUAR_RAW_ASSETS} == _EXPECTED_FILENAMES


def test_urls_are_huggingface_https() -> None:
    for asset in JAGUAR_RAW_ASSETS:
        assert asset.url.startswith(_HF_PREFIX), (
            f"Asset {asset.name!r} URL {asset.url!r} does not start with "
            f"the expected HuggingFace prefix"
        )


def test_checksums_are_lowercase_sha256_hex() -> None:
    for asset in JAGUAR_RAW_ASSETS:
        assert _SHA256_HEX.match(asset.checksum), (
            f"Asset {asset.name!r} checksum {asset.checksum!r} is not lowercase 64-char hex"
        )


def test_vcf_has_expected_size_pinned() -> None:
    vcf = next(a for a in JAGUAR_RAW_ASSETS if a.name == "jaguar-vcf")
    assert vcf.expected_size == 147_354_171


def test_build_manifest_returns_two_assets(tmp_path: Path) -> None:
    manifest = build_jaguar_raw_manifest(tmp_path)
    assert len(manifest) == 2


def test_build_manifest_destinations_are_under_output_dir(tmp_path: Path) -> None:
    manifest = build_jaguar_raw_manifest(tmp_path)
    for asset in manifest:
        assert asset.destination.parent == tmp_path


def test_build_manifest_destination_names_match_registry(tmp_path: Path) -> None:
    manifest = build_jaguar_raw_manifest(tmp_path)
    actual = {a.destination.name for a in manifest}
    assert actual == _EXPECTED_FILENAMES


def test_build_manifest_checksums_are_populated(tmp_path: Path) -> None:
    manifest = build_jaguar_raw_manifest(tmp_path)
    for asset in manifest:
        assert asset.checksum, f"Asset at {asset.destination} has empty checksum"


def test_build_manifest_kind_is_jaguar_raw(tmp_path: Path) -> None:
    manifest = build_jaguar_raw_manifest(tmp_path)
    for asset in manifest:
        assert asset.kind == "jaguar-raw"
