"""Tests for experiment and feline pipeline configuration loaders.

These tests defend the scientific and supply-chain contracts encoded in
project TOML configs: jaguar metadata fields must match the bootstrap
contract, feline pretraining must stay pinned to the approved BioProject,
reference assembly, DNABERT-2 revision, and ``trust_remote_code`` policy,
and every boolean-valued safety guard must be a real TOML boolean rather
than a truthy-coercible string or integer. Together they ensure misconfigured
runs fail loudly at load time instead of silently producing invalid science.
"""

from pathlib import Path

import pytest

from jaguar_geo_assign.config import (
    _validate_felid_species_entries,
    load_experiment_config,
    load_felid_foundation_pipeline_config,
    load_feline_pipeline_config,
)
from jaguar_geo_assign.data.contracts import JAGUAR_METADATA_FIELDS
from jaguar_geo_assign.data.pipeline_contract import (
    APPROVED_BIOPROJECT_ACCESSION,
    APPROVED_REFERENCE_ASSEMBLY,
    DNABERT2_TRUST_REMOTE_CODE,
)


def test_load_experiment_config_preserves_metadata_contract() -> None:
    """An experiment config round-trips the jaguar metadata fields and baseline stage wiring
    intact."""
    config = load_experiment_config("configs/examples/regression_transfer.toml")

    assert config.jaguar_metadata_fields == JAGUAR_METADATA_FIELDS
    assert config.split_unit == "individual_id"
    assert config.baseline_stage == "baseline_evaluate"
    assert "baseline_evaluate" in config.stages


def test_load_experiment_config_rejects_extra_metadata_fields(tmp_path: Path) -> None:
    """Adding fields outside the bootstrap metadata contract fails config loading."""
    invalid_config = tmp_path / "invalid.toml"
    invalid_config.write_text(
        Path("configs/examples/fine_tune.toml")
        .read_text(encoding="utf-8")
        .replace(
            '  "longitude",\n]',
            '  "longitude",\n  "coordinate_uncertainty_meters",\n]',
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="bootstrap metadata contract"):
        load_experiment_config(invalid_config)


def test_load_feline_pipeline_config_preserves_scientific_contracts() -> None:
    """The feline pipeline config exposes the approved accession, assembly, tokenizer revision,
    and split strategy."""
    config = load_feline_pipeline_config("configs/examples/feline_pretrain.toml")

    assert config.project_accession == APPROVED_BIOPROJECT_ACCESSION
    assert config.consensus.assembly == APPROVED_REFERENCE_ASSEMBLY
    assert config.split.strategy == "global_locus_block"
    assert config.split.locus_key_fields == ("contig", "block_id")
    assert config.split.locus_block_size == 2048
    assert config.tokenizer.revision == "7bce263b15377fc15361f52cfab88f8b586abda0"
    assert config.tokenizer.trust_remote_code is DNABERT2_TRUST_REMOTE_CODE
    assert config.runtime.external_tools == ("bcftools",)


def test_load_feline_pipeline_config_rejects_unpinned_tokenizer_revision(tmp_path: Path) -> None:
    """A mutable tokenizer revision (e.g. ``main``) is rejected to prevent silent upstream drift."""
    invalid_config = tmp_path / "invalid_feline.toml"
    invalid_config.write_text(
        Path("configs/examples/feline_pretrain.toml")
        .read_text(encoding="utf-8")
        .replace(
            'revision = "7bce263b15377fc15361f52cfab88f8b586abda0"',
            'revision = "main"',
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="approved immutable DNABERT-2 revision"):
        load_feline_pipeline_config(invalid_config)


def test_load_feline_pipeline_config_reports_missing_required_sections(tmp_path: Path) -> None:
    """Omitting required top-level sections produces an explicit error listing what is missing."""
    invalid_config = tmp_path / "invalid_sections.toml"
    invalid_config.write_text("[experiment]\nname = 'wrong'\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing required sections"):
        load_feline_pipeline_config(invalid_config)


def test_load_feline_pipeline_config_requires_explicit_trust_remote_code(tmp_path: Path) -> None:
    """The security-critical ``trust_remote_code`` field must be declared, not defaulted."""
    invalid_config = tmp_path / "missing_trust_remote_code.toml"
    invalid_config.write_text(
        Path("configs/examples/feline_pretrain.toml")
        .read_text(encoding="utf-8")
        .replace("trust_remote_code = true\n", ""),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing required field: trust_remote_code"):
        load_feline_pipeline_config(invalid_config)


def test_load_feline_pipeline_config_rejects_trust_remote_code_mismatch(tmp_path: Path) -> None:
    """Loader refuses any ``trust_remote_code`` value that diverges from the approved policy."""
    invalid_config = tmp_path / "invalid_trust_remote_code.toml"
    invalid_config.write_text(
        Path("configs/examples/feline_pretrain.toml")
        .read_text(encoding="utf-8")
        .replace(
            "trust_remote_code = true",
            "trust_remote_code = false",
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="tokenizer.trust_remote_code must remain True"):
        load_feline_pipeline_config(invalid_config)


@pytest.mark.parametrize(
    ("replacement", "expected_fragment"),
    [("1", "1 (int)"), ('"true"', "'true' (str)")],
)
def test_load_feline_pipeline_config_rejects_non_boolean_trust_remote_code(
    tmp_path: Path, replacement: str, expected_fragment: str
) -> None:
    """Truthy strings/ints cannot stand in for a TOML boolean on ``trust_remote_code``."""
    invalid_config = tmp_path / "invalid_trust_remote_code_type.toml"
    invalid_config.write_text(
        Path("configs/examples/feline_pretrain.toml")
        .read_text(encoding="utf-8")
        .replace(
            "trust_remote_code = true",
            f"trust_remote_code = {replacement}",
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError, match="tokenizer.trust_remote_code must be a TOML boolean true/false"
    ) as exc_info:
        load_feline_pipeline_config(invalid_config)

    assert expected_fragment in str(exc_info.value)


@pytest.mark.parametrize("field_name", ["require_assembly_match", "require_contig_match"])
@pytest.mark.parametrize(
    ("replacement", "expected_fragment"),
    [("1", "1 (int)"), ('"true"', "'true' (str)")],
)
def test_load_feline_pipeline_config_rejects_non_boolean_consensus_mismatch_guards(
    tmp_path: Path,
    field_name: str,
    replacement: str,
    expected_fragment: str,
) -> None:
    """Consensus mismatch guards (assembly/contig) must be real booleans to prevent silent
    bypass."""
    invalid_config = tmp_path / f"invalid_{field_name}_type.toml"
    invalid_config.write_text(
        Path("configs/examples/feline_pretrain.toml")
        .read_text(encoding="utf-8")
        .replace(
            f"{field_name} = true",
            f"{field_name} = {replacement}",
            1,
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=rf"consensus\.{field_name} must be a TOML boolean true/false",
    ) as exc_info:
        load_feline_pipeline_config(invalid_config)

    assert expected_fragment in str(exc_info.value)


@pytest.mark.parametrize(
    ("field_line", "section_field", "original_value"),
    [
        ("drop_short_sequences = true", "windowing.drop_short_sequences", "true"),
        ("preserve_raw_windows = false", "export.preserve_raw_windows", "false"),
        ("preserve_sequence_hashes = true", "export.preserve_sequence_hashes", "true"),
        ("preserve_coordinates = true", "export.preserve_coordinates", "true"),
    ],
)
@pytest.mark.parametrize(
    ("replacement", "expected_fragment"),
    [("1", "1 (int)"), ('"true"', "'true' (str)"), ("0", "0 (int)")],
)
def test_load_feline_pipeline_config_rejects_non_boolean_preservation_flags(
    tmp_path: Path,
    field_line: str,
    section_field: str,
    original_value: str,
    replacement: str,
    expected_fragment: str,
) -> None:
    """Windowing/export boolean guards must reject truthy-coercible scalars.

    These flags gate short-sequence dropping and the
    auditability guarantees of the export format. If Python's
    ``if value`` coercion let a ``1`` or ``"true"`` stand in for
    ``True``, a misconfigured TOML could silently disable audit
    trails. Parametrising across every strictly-validated boolean
    in the feline export/windowing sections gives us regression
    coverage for each individual ``_require_boolean_field`` call
    site without hand-written duplication.
    """
    field_name = field_line.split(" = ")[0]
    invalid_config = tmp_path / f"invalid_{field_name}_{replacement}.toml"
    invalid_config.write_text(
        Path("configs/examples/feline_pretrain.toml")
        .read_text(encoding="utf-8")
        .replace(
            field_line,
            f"{field_name} = {replacement}",
            1,
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=rf"{section_field} must be a TOML boolean true/false",
    ) as exc_info:
        load_feline_pipeline_config(invalid_config)

    assert expected_fragment in str(exc_info.value)


@pytest.mark.parametrize(
    ("field_line", "section_field"),
    [
        ("drop_short_sequences = true", "windowing.drop_short_sequences"),
        ("preserve_raw_windows = false", "export.preserve_raw_windows"),
        ("preserve_sequence_hashes = true", "export.preserve_sequence_hashes"),
        ("preserve_coordinates = true", "export.preserve_coordinates"),
    ],
)
@pytest.mark.parametrize(
    ("replacement", "expected_fragment"),
    [("1", "1 (int)"), ('"true"', "'true' (str)"), ("0", "0 (int)")],
)
def test_load_felid_foundation_pipeline_config_rejects_non_boolean_flags(
    tmp_path: Path,
    field_line: str,
    section_field: str,
    replacement: str,
    expected_fragment: str,
) -> None:
    """Felid foundation loader rejects truthy-coercible scalars on every bool flag.

    The felid foundation pretraining contract shares the same
    ``_require_boolean_field`` helper as the feline loader, but has
    its own call sites. Parametrising over each strictly-validated
    boolean field in ``configs/examples/felid_foundation_pretrain.toml``
    guards against future regressions where a new boolean gets added
    to the felid loader without the strict-validation guard, silently
    letting an integer ``1`` or the string ``"true"`` masquerade as a
    TOML boolean.
    """
    field_name = field_line.split(" = ")[0]
    invalid_config = tmp_path / f"invalid_felid_{field_name}_{replacement}.toml"
    invalid_config.write_text(
        Path("configs/examples/felid_foundation_pretrain.toml")
        .read_text(encoding="utf-8")
        .replace(field_line, f"{field_name} = {replacement}", 1),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=rf"{section_field} must be a TOML boolean true/false",
    ) as exc_info:
        load_felid_foundation_pipeline_config(invalid_config)

    assert expected_fragment in str(exc_info.value)


@pytest.mark.parametrize("bool_literal", ["true", "false"])
def test_load_felid_foundation_pipeline_config_accepts_real_booleans(
    tmp_path: Path, bool_literal: str
) -> None:
    """Real TOML booleans on ``drop_short_sequences`` round-trip through the loader.

    The strict-bool guard must still admit the two values a
    legitimate config can express (``true`` / ``false``) so the
    validator does not become a trap door for valid configurations.
    ``drop_short_sequences`` is picked because, unlike
    ``preserve_coordinates``, it is not gated by a downstream
    contract-specific equality check, so both ``true`` and ``false``
    are accepted by the loader.
    """
    rebuilt = tmp_path / f"bool_{bool_literal}.toml"
    rebuilt.write_text(
        Path("configs/examples/felid_foundation_pretrain.toml")
        .read_text(encoding="utf-8")
        .replace(
            "drop_short_sequences = true",
            f"drop_short_sequences = {bool_literal}",
            1,
        ),
        encoding="utf-8",
    )

    config = load_felid_foundation_pipeline_config(rebuilt)
    assert config.windowing.drop_short_sequences is (bool_literal == "true")


def test_validate_felid_species_entries_rejects_legacy_accession_key() -> None:
    """The loader provides a clear migration path when encountering the old accession key."""
    raw_species = [
        {"species": "Felis catus", "identifier": "GCF_000181335.3"},
        {"species": "Panthera leo", "identifier": "GCF_018350215.1"},
        {"species": "Panthera tigris", "identifier": "GCF_000464555.1"},
        {"species": "Puma concolor", "identifier": "GCF_003327715.1"},
        {"species": "Panthera pardus", "identifier": "GCF_001857705.1"},
        {"species": "Panthera onca", "accession": "DNAZOO_Panthera_onca_HiC"},
    ]
    with pytest.raises(ValueError, match="renamed to 'identifier'"):
        _validate_felid_species_entries(raw_species)
