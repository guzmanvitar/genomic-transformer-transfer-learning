"""Tests for the active configuration loaders.

The legacy feline pretraining config loader has been removed, so this module
now focuses on the still-supported bootstrap experiment config and the felid
foundation pretraining config.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from jaguar_geo_assign.config import (
    _find_project_root,
    _validate_felid_species_entries,
    load_experiment_config,
    load_felid_foundation_pipeline_config,
    load_foundation_training_config,
)
from jaguar_geo_assign.data.contracts import JAGUAR_METADATA_FIELDS
from jaguar_geo_assign.data.pipeline_contract import (
    DNABERT2_TOKENIZER_REVISION,
    DNABERT2_TRUST_REMOTE_CODE,
)


def test_load_experiment_config_preserves_metadata_contract() -> None:
    """Bootstrap experiment configs should retain the canonical jaguar metadata contract."""
    config = load_experiment_config("configs/examples/regression_transfer.toml")

    assert config.jaguar_metadata_fields == JAGUAR_METADATA_FIELDS
    assert config.split_unit == "individual_id"
    assert config.baseline_stage == "baseline_evaluate"
    assert "baseline_evaluate" in config.stages


def test_load_experiment_config_rejects_extra_metadata_fields(tmp_path: Path) -> None:
    """Adding undeclared jaguar metadata columns should fail loudly at load time."""
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


def test_load_felid_foundation_pipeline_config_preserves_contracts() -> None:
    """The active felid foundation config should round-trip its pinned tokenizer and roster."""
    config = load_felid_foundation_pipeline_config(
        "configs/examples/felid_foundation_pretrain.toml"
    )

    assert config.name == "felid_foundation_pretrain_contract"
    assert len(config.species) == 6
    assert config.split.strategy == "global_locus_block"
    assert config.split.locus_key_fields == ("contig", "block_id")
    assert config.tokenizer.revision == "7bce263b15377fc15361f52cfab88f8b586abda0"
    assert config.tokenizer.trust_remote_code is DNABERT2_TRUST_REMOTE_CODE
    assert config.pipeline.chunk_size == 10_000
    assert config.pipeline.num_workers == 6
    assert config.pipeline.queue_maxsize_factor == 2
    assert config.pipeline.sigterm_timeout == pytest.approx(30.0)
    assert config.runtime.external_tools == ()


@pytest.mark.parametrize(
    ("field_line", "replacement", "field_name"),
    [
        ("chunk_size = 10000", "chunk_size = 0", "pipeline.chunk_size"),
        ("num_workers = 6", "num_workers = 0", "pipeline.num_workers"),
        (
            "queue_maxsize_factor = 2",
            "queue_maxsize_factor = 0",
            "pipeline.queue_maxsize_factor",
        ),
        ("sigterm_timeout = 30.0", "sigterm_timeout = 0.0", "pipeline.sigterm_timeout"),
    ],
)
def test_load_felid_foundation_pipeline_config_rejects_non_positive_runtime_knobs(
    tmp_path: Path,
    field_line: str,
    replacement: str,
    field_name: str,
) -> None:
    """Execution knobs in ``[pipeline]`` must reject zero or negative sentinel values."""
    key = field_line.split(" = ", maxsplit=1)[0]
    invalid_config = tmp_path / f"invalid_{key}.toml"
    invalid_config.write_text(
        Path("configs/examples/felid_foundation_pretrain.toml")
        .read_text(encoding="utf-8")
        .replace(field_line, replacement, 1),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=rf"{re.escape(field_name)} must be positive"):
        load_felid_foundation_pipeline_config(invalid_config)


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
    """Foundation loader should reject truthy-coercible scalars on strict boolean fields."""
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
    tmp_path: Path,
    bool_literal: str,
) -> None:
    """Strict boolean validation must still accept the two legal TOML boolean values."""
    rebuilt = tmp_path / f"bool_{bool_literal}.toml"
    rebuilt.write_text(
        Path("configs/examples/felid_foundation_pretrain.toml")
        .read_text(encoding="utf-8")
        .replace("drop_short_sequences = true", f"drop_short_sequences = {bool_literal}", 1),
        encoding="utf-8",
    )

    config = load_felid_foundation_pipeline_config(rebuilt)
    assert config.windowing.drop_short_sequences is (bool_literal == "true")


def test_load_foundation_training_config_rejects_unpinned_model_revision(
    tmp_path: Path,
) -> None:
    """Foundation training configs must pin the warm-start revision to the approved hash."""
    config_path = tmp_path / "invalid_foundation_training.toml"
    config_path.write_text(
        (
            "[training]\n"
            'corpus_metadata_path = "/tmp/corpus/metadata.json"\n'
            'model_identifier = "zhihan1996/DNABERT-2-117M"\n'
            'model_revision = "main"\n'
            "max_steps = 100\n"
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=rf"training\.model_revision must be pinned to {DNABERT2_TOKENIZER_REVISION}",
    ):
        load_foundation_training_config(config_path)


def test_find_project_root_walks_up(tmp_path: Path) -> None:
    """_find_project_root walks parent directories to find pyproject.toml."""
    (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")
    nested = tmp_path / "a" / "b" / "c"
    nested.mkdir(parents=True)
    assert _find_project_root(nested) == tmp_path


def test_find_project_root_raises_when_not_found(tmp_path: Path) -> None:
    """_find_project_root raises ValueError when no pyproject.toml exists."""
    with pytest.raises(ValueError, match="pyproject.toml"):
        _find_project_root(tmp_path)


def test_validate_felid_species_entries_rejects_legacy_accession_key() -> None:
    """Loader diagnostics should explain the historical ``accession`` → ``identifier`` rename."""
    raw_species = [
        {"species": "Felis catus", "identifier": "GCF_000181335.3"},
        {"species": "Panthera leo", "identifier": "GCF_018350215.1"},
        {"species": "Panthera tigris", "identifier": "GCF_000464555.1"},
        {"species": "Puma concolor", "identifier": "GCF_003327715.1"},
        {"species": "Panthera pardus", "identifier": "GCF_001857705.1"},
        {"species": "Panthera onca", "accession": "DNAZOO_Panthera_onca_HiC"},
    ]

    with pytest.raises(ValueError, match="renamed to 'identifier'") as exc_info:
        _validate_felid_species_entries(raw_species)

    assert not re.search(r"<[A-Z][A-Z-]*>", str(exc_info.value))
