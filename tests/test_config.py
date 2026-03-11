from pathlib import Path

import pytest

from jaguar_geo_assign.config import load_experiment_config, load_feline_pipeline_config
from jaguar_geo_assign.data.contracts import JAGUAR_METADATA_FIELDS


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

    assert config.project_accession == "PRJNA308208"
    assert config.split.strategy == "global_locus_block"
    assert config.split.locus_key_fields == ("contig", "block_id")
    assert config.tokenizer.revision == "7bce263b15377fc15361f52cfab88f8b586abda0"
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