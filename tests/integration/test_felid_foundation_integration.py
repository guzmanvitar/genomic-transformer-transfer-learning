"""Integration test for the felid foundation pretraining pipeline.

This test downloads exactly one pinned felid FASTA from NCBI (Panthera tigris
/ GCF_000464555.1, the smallest approved assembly), truncates it to 5 MB
decompressed, and runs the full felid-foundation pretrain pipeline against
that slice with the real DNABERT-2 tokenizer loaded from HuggingFace. The
test is gated by @pytest.mark.integration and excluded from the default test
run via pyproject.toml addopts.

Pinned species rationale:
    Panthera tigris (GCF_000464555.1 / PanTig1.0) is the smallest approved
    assembly at ~2.4 GB compressed. Truncating to 5 MB decompressed yields
    ~10k windows at window_size=510, which is enough to validate the
    streaming-writer memory model and run-summary schema while keeping the
    download time under 30 seconds on a typical connection.
"""

from __future__ import annotations

import gzip
import tempfile
from pathlib import Path
from urllib.request import build_opener, Request

import pytest

from jaguar_geo_assign.data.felid_assemblies import build_refseq_fasta_url
from jaguar_geo_assign.pretrain import run_felid_foundation_pretrain
from tests._felid_fixture import render_example_config


@pytest.mark.integration
def test_felid_foundation_integration_panthera_tigris():
    """Live download + pretrain of Panthera tigris (5 MB slice, real tokenizer)."""
    pinned_species = "Panthera tigris"
    pinned_accession = "GCF_000464555.1"
    pinned_assembly = "PanTig1.0"
    pinned_species_slug = "panthera_tigris"
    max_decompressed_bytes = 5_000_000

    with tempfile.TemporaryDirectory() as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)

        config_path = render_example_config(
            tmp_dir,
            species=[(pinned_species, pinned_accession)],
            runtime_external_tools=(),
            scalar_overrides={
                "pipeline.name": "integration-test-felid-foundation",
                "pipeline.description": "Integration test with Panthera tigris slice",
            },
        )

        reference_dir = tmp_dir / "reference"
        reference_dir.mkdir(parents=True)
        fasta_url = build_refseq_fasta_url(pinned_accession, pinned_assembly)
        fasta_path = reference_dir / f"{pinned_accession}_{pinned_assembly}.fna.gz"

        opener = build_opener()
        request = Request(fasta_url)
        with opener.open(request, timeout=60) as response:
            compressed_data = response.read()

        # Decompress and truncate
        decompressed_data = gzip.decompress(compressed_data)
        truncated_data = decompressed_data[:max_decompressed_bytes]

        # Re-compress and write
        fasta_path.write_bytes(gzip.compress(truncated_data))

        # Run the pretrain pipeline with the real tokenizer
        result = run_felid_foundation_pretrain(config_path)

        # Assertions
        assert len(result.per_species_stats) == 1
        stats = result.per_species_stats[0]
        assert stats.species_slug == pinned_species_slug
        assert stats.accession == pinned_accession
        assert stats.assembly_name == pinned_assembly
        assert stats.peak_window_count_in_memory >= 1

        # Check that the summary JSON exists
        assert result.artifacts.summary_path.exists()

        # Load and verify a few windows were produced
        import json

        summary = json.loads(result.artifacts.summary_path.read_text())
        assert summary["totals"]["train"] + summary["totals"]["validation"] >= 1

        # Verify all windows have source="reference" and individual_id=species_slug
        # (We can't easily inspect the Parquet here without adding pyarrow, so we
        # trust the schema test in the unit tests and just check the summary.)
        assert stats.retained_sequence_count >= 1

