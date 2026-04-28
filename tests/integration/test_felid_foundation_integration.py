"""Integration test for the felid foundation pretraining pipeline.

This test downloads exactly one pinned felid FASTA from NCBI (Panthera tigris
/ GCF_000464555.1, the smallest approved assembly), truncates it to 5 MB
decompressed, and runs the full felid-foundation pretrain pipeline against
that slice with the real DNABERT-2 tokenizer loaded from HuggingFace. The
test is gated by @pytest.mark.integration and excluded from the default test
run via pyproject.toml addopts.

The config loader enforces the full six-species approved roster
(``APPROVED_FELID_ASSEMBLIES``), so the test populates the other five
species on disk as short placeholder FASTAs (see
:func:`tests._felid_fixture.write_placeholder_fastas`) whose 128 bp contig
produces zero windows and therefore does not perturb the tigris-focused
assertions below.

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
from urllib.request import Request, build_opener

import pytest

from jaguar_geo_assign.data.felid_assemblies import (
    APPROVED_FELID_ASSEMBLIES,
    build_refseq_fasta_url,
)
from jaguar_geo_assign.pretrain import run_felid_foundation_pretrain
from tests._felid_fixture import render_example_config, write_placeholder_fastas


@pytest.mark.integration
def test_felid_foundation_integration_panthera_tigris():
    """Live download + pretrain of Panthera tigris (5 MB slice, real tokenizer).

    Exercises the full felid-foundation pipeline end-to-end against a
    real DNABERT-2 tokenizer and a real (truncated) RefSeq assembly while
    honouring the six-species loader contract via placeholder FASTAs for the
    non-tigris roster.
    """
    pinned_identifier = "GCF_000464555.1"
    pinned_assembly = "PanTig1.0"
    pinned_species_slug = "panthera_tigris"
    # TRADE-OFF: ``max_decompressed_bytes`` is a
    # byte count applied via a ``bytes`` slice below. FASTA bytes are ASCII
    # (1 byte per char) so the byte-level truncation is semantically
    # equivalent to a char-level truncation here, but the naming/semantics
    # gap is real for any caller who generalises this fixture to non-ASCII
    # payloads.
    max_decompressed_bytes = 5_000_000

    with tempfile.TemporaryDirectory() as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)

        # Full six-species roster is required by the loader; the test only
        # exercises tigris in depth and relies on placeholder FASTAs (below)
        # for the other five species to keep their ``retained_sequence_count``
        # at zero without tripping ``MissingFelidReferenceError``.
        species_roster = [
            (assembly.species, assembly.identifier) for assembly in APPROVED_FELID_ASSEMBLIES
        ]

        config_path = render_example_config(
            tmp_dir,
            species=species_roster,
            runtime_external_tools=(),
            scalar_overrides={
                "pipeline.name": "integration-test-felid-foundation",
                "pipeline.description": "Integration test with Panthera tigris slice",
            },
        )

        reference_dir = tmp_dir / "reference"
        reference_dir.mkdir(parents=True)

        padded_identifiers = [
            assembly.identifier
            for assembly in APPROVED_FELID_ASSEMBLIES
            if assembly.identifier != pinned_identifier
        ]
        write_placeholder_fastas(reference_dir, padded_identifiers)

        fasta_url = build_refseq_fasta_url(pinned_identifier, pinned_assembly)
        fasta_path = reference_dir / f"{pinned_identifier}.fna.gz"

        opener = build_opener()
        request = Request(fasta_url)
        with opener.open(request, timeout=60) as response:
            compressed_data = response.read()

        decompressed_data = gzip.decompress(compressed_data)
        truncated_data = decompressed_data[:max_decompressed_bytes]
        fasta_path.write_bytes(gzip.compress(truncated_data))

        result = run_felid_foundation_pretrain(config_path)

        assert len(result.per_species_stats) == len(APPROVED_FELID_ASSEMBLIES)
        tigris_stats = next(
            stats for stats in result.per_species_stats if stats.species_slug == pinned_species_slug
        )
        assert tigris_stats.identifier == pinned_identifier
        assert tigris_stats.assembly_name == pinned_assembly
        assert tigris_stats.peak_window_count_in_memory >= 1
        assert tigris_stats.retained_sequence_count >= 1

        assert result.artifacts.summary_path.exists()

        import json

        summary = json.loads(result.artifacts.summary_path.read_text())
        assert summary["totals"]["train"] + summary["totals"]["validation"] >= 1

        # Placeholder species must not contribute any windows; total windows
        # therefore come entirely from the tigris slice.
        for stats in result.per_species_stats:
            if stats.species_slug == pinned_species_slug:
                continue
            assert sum(stats.window_counts_by_split.values()) == 0, (
                f"placeholder species {stats.species_slug} unexpectedly produced windows"
            )
