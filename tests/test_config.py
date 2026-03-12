from pathlib import Path

import pytest

from jaguar_geo_assign.config import load_experiment_config, load_feline_pipeline_config
from jaguar_geo_assign.data.contracts import JAGUAR_METADATA_FIELDS
from jaguar_geo_assign.data.pipeline_contract import (
    APPROVED_BIOPROJECT_ACCESSION,
    APPROVED_REFERENCE_ASSEMBLY,
    DNABERT2_TRUST_REMOTE_CODE,
)


def test_load_experiment_config_preserves_metadata_contract() -> None:
    config = load_experiment_config("configs/examples/regression_transfer.toml")

    assert config.jaguar_metadata_fields == JAGUAR_METADATA_FIELDS
    assert config.split_unit == "individual_id"
    assert config.baseline_stage == "baseline_evaluate"
    assert "baseline_evaluate" in config.stages


def test_load_experiment_config_rejects_extra_metadata_fields(tmp_path: Path) -> None:
    invalid_config = tmp_path / "invalid.toml"
    invalid_config.write_text(
        Path("configs/examples/fine_tune.toml").read_text(encoding="utf-8").replace(
            '  "longitude",\n]',
            '  "longitude",\n  "coordinate_uncertainty_meters",\n]',
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="bootstrap metadata contract"):
        load_experiment_config(invalid_config)


def test_load_feline_pipeline_config_preserves_scientific_contracts() -> None:
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
    invalid_config = tmp_path / "invalid_feline.toml"
    invalid_config.write_text(
        Path("configs/examples/feline_pretrain.toml").read_text(encoding="utf-8").replace(
            'revision = "7bce263b15377fc15361f52cfab88f8b586abda0"',
            'revision = "main"',
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="approved immutable DNABERT-2 revision"):
        load_feline_pipeline_config(invalid_config)


def test_load_feline_pipeline_config_reports_missing_required_sections(tmp_path: Path) -> None:
    invalid_config = tmp_path / "invalid_sections.toml"
    invalid_config.write_text("[experiment]\nname = 'wrong'\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing required sections"):
        load_feline_pipeline_config(invalid_config)


def test_load_feline_pipeline_config_requires_explicit_trust_remote_code(tmp_path: Path) -> None:
    invalid_config = tmp_path / "missing_trust_remote_code.toml"
    invalid_config.write_text(
        Path("configs/examples/feline_pretrain.toml").read_text(encoding="utf-8").replace(
            "trust_remote_code = true\n", ""
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing required field: trust_remote_code"):
        load_feline_pipeline_config(invalid_config)


def test_load_feline_pipeline_config_rejects_trust_remote_code_mismatch(tmp_path: Path) -> None:
    invalid_config = tmp_path / "invalid_trust_remote_code.toml"
    invalid_config.write_text(
        Path("configs/examples/feline_pretrain.toml").read_text(encoding="utf-8").replace(
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
    invalid_config = tmp_path / "invalid_trust_remote_code_type.toml"
    invalid_config.write_text(
        Path("configs/examples/feline_pretrain.toml").read_text(encoding="utf-8").replace(
            "trust_remote_code = true",
            f"trust_remote_code = {replacement}",
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="tokenizer.trust_remote_code must be a TOML boolean true/false") as exc_info:
        load_feline_pipeline_config(invalid_config)

    assert expected_fragment in str(exc_info.value)