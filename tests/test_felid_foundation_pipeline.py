"""Tests for the multi-species felid foundation pretraining pipeline.

Guards the contract that the felid-foundation path processes six reference
FASTAs one at a time (streaming-writer memory model), emits exactly the
run-summary JSON schema specified in the task note, validates species-slug
derivation, detects cross-species contig collisions, and produces diagnostic
errors (missing FASTA, acquisition failure) that name both the problem and
the fix. No network calls or bcftools invocations — all tests use fixture
FASTAs and fake tokenizers.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import jaguar_geo_assign.pretrain.felid_foundation_pipeline as felid_foundation_pipeline
from jaguar_geo_assign.config import load_felid_foundation_pipeline_config
from jaguar_geo_assign.data.acquisition import DownloadAsset, DownloadResult
from jaguar_geo_assign.data.preprocessor import TokenizedWindow
from jaguar_geo_assign.pretrain import (
    FelidAcquisitionError,
    MissingFelidReferenceError,
    acquire_felid_foundation_assemblies,
    run_felid_foundation_pretrain,
)
from jaguar_geo_assign.pretrain._shared import normalize_ru_maxrss_to_bytes
from tests._felid_fixture import build_fixture_fasta as _build_fixture_fasta
from tests._felid_fixture import load_example_config_dict, render_example_config
from tests._felid_fixture import pad_species_to_full_roster as _pad_to_six_species
from tests._felid_fixture import placeholder_fasta_checksum as _placeholder_fasta_checksum
from tests._felid_fixture import write_placeholder_fastas as _write_placeholder_fastas


def test_species_slug_derivation():
    """Species Latin binomials slugify correctly."""
    from jaguar_geo_assign.config import _slugify_species

    assert _slugify_species("Felis catus") == "felis_catus"
    assert _slugify_species("Panthera leo") == "panthera_leo"
    assert _slugify_species("Panthera onca") == "panthera_onca"


def test_normalize_ru_maxrss_linux():
    """Linux reports KB; normalization multiplies by 1024."""
    assert normalize_ru_maxrss_to_bytes(1024, "linux") == 1_048_576
    assert normalize_ru_maxrss_to_bytes(1, "linux") == 1024


def test_normalize_ru_maxrss_darwin():
    """macOS reports bytes; normalization is identity."""
    assert normalize_ru_maxrss_to_bytes(1024, "darwin") == 1024
    assert normalize_ru_maxrss_to_bytes(1_048_576, "darwin") == 1_048_576


def test_normalize_ru_maxrss_unknown_platform():
    """Unknown platform raises ValueError."""
    with pytest.raises(ValueError, match="Unsupported platform"):
        normalize_ru_maxrss_to_bytes(1024, "win32")


# Fixture helpers (``_build_fixture_fasta``, ``_ALL_APPROVED_FELIDS``,
# ``_pad_to_six_species``, ``_placeholder_fasta_filename``,
# ``_placeholder_fasta_checksum``, ``_write_placeholder_fastas``) are imported
# from :mod:`tests._felid_fixture` to keep the integration test and this
# unit-test module on a single source of truth (no copy-paste).


def _build_full_mock_manifest(
    reference_dir: Path,
    *,
    overrides: dict[str, str] | None = None,
) -> list[DownloadAsset]:
    """Build a 6-asset mock manifest matching APPROVED_FELID_ASSEMBLIES.

    Padded-species checksums default to the placeholder FASTA MD5 so
    those entries skip naturally; pass ``overrides`` to inject a
    test-specific checksum for a particular identifier.
    """
    from jaguar_geo_assign.data.felid_assemblies import APPROVED_FELID_ASSEMBLIES

    overrides = overrides or {}
    assets: list[DownloadAsset] = []
    for assembly in APPROVED_FELID_ASSEMBLIES:
        filename = f"{assembly.identifier}.fna.gz"
        checksum = overrides.get(
            assembly.identifier, _placeholder_fasta_checksum(assembly.identifier)
        )
        assets.append(
            DownloadAsset(
                url=f"http://fake.ncbi.nlm.nih.gov/{filename}",
                destination=reference_dir / filename,
                checksum=checksum,
                checksum_name=assembly.checksum_name,
                kind="reference",
                mirror_url=assembly.mirror_url,
                expected_size=assembly.expected_size,
            )
        )
    return assets


def _build_fixture_config(
    tmp_path: Path,
    species_subset: list[tuple[str, str]],
    *,
    pad: bool = True,
    write_placeholder_fastas: bool = True,
    scalar_overrides: dict[str, object] | None = None,
) -> Path:
    """Build a felid-foundation config TOML in *tmp_path* derived from the example.

    The canonical TOML contract lives in
    ``configs/examples/felid_foundation_pretrain.toml``; this helper
    round-trips that file through
    :func:`tests._felid_fixture.render_example_config` so every test
    fixture stays byte-compatible with the loader schema. Any drift
    between the example and the loader contract now surfaces as a
    loader ``ValueError`` at test time instead of being masked by a
    hand-authored literal.

    Args:
        tmp_path: Pytest tmp_path fixture.
        species_subset: List of (species, identifier) tuples to include.
        pad: When True (default), pad ``species_subset`` with remaining
            approved felids so the config has six entries (the contract).
            Pass ``False`` for tests that exercise the "too-few" validation.
        write_placeholder_fastas: When True (default), write a minimal
            unique FASTA for each padded species under
            ``tmp_path / 'reference'`` so run tests can reach the logic
            under test without hitting :class:`MissingFelidReferenceError`
            on a padded entry.
        scalar_overrides: Optional dotted-key overrides applied on top
            of the canonical example (e.g. ``{"windowing.context_window":
            510}``). The dictionary must only reference sections/fields
            that already exist in the example.

    Returns:
        Path to the written config.toml.
    """
    final_species = _pad_to_six_species(species_subset) if pad else list(species_subset)
    if write_placeholder_fastas and pad:
        explicit_identifiers = {identifier for _, identifier in species_subset}
        padded_only = [
            identifier for _, identifier in final_species if identifier not in explicit_identifiers
        ]
        _write_placeholder_fastas(tmp_path / "reference", padded_only)
    overrides: dict[str, object] = {
        "windowing.context_window": 510,
        "windowing.window_overlap": 255,
        "windowing.max_ambiguous_fraction": 0.5,
        "pipeline.name": "test-felid-foundation",
        "pipeline.description": "Fixture config",
    }
    if scalar_overrides:
        overrides.update(scalar_overrides)
    return render_example_config(
        tmp_path,
        species=final_species,
        runtime_external_tools=(),
        scalar_overrides=overrides,
    )


def test_config_loader_happy_path(tmp_path):
    """Config loader accepts the full six-species contract."""
    config_path = _build_fixture_config(
        tmp_path,
        [
            ("Felis catus", "GCF_000181335.3"),
            ("Panthera leo", "GCF_018350215.1"),
        ],
    )
    config = load_felid_foundation_pipeline_config(config_path)
    assert config.name == "test-felid-foundation"
    assert len(config.species) == 6
    slugs = {entry.species_slug for entry in config.species}
    assert {"felis_catus", "panthera_leo"}.issubset(slugs)


def test_example_config_is_loader_valid():
    """Canary: the canonical example TOML always round-trips through the loader.

    Any divergence between
    ``configs/examples/felid_foundation_pretrain.toml`` and the felid
    foundation loader contract must surface as a test failure here
    before it silently propagates into every downstream fixture that
    derives from the example via :func:`render_example_config`.
    """
    from tests._felid_fixture import EXAMPLE_FELID_FOUNDATION_CONFIG_PATH

    config = load_felid_foundation_pipeline_config(EXAMPLE_FELID_FOUNDATION_CONFIG_PATH)
    assert config.name == "felid_foundation_pretrain_contract"
    assert len(config.species) == 6


def test_config_loader_rejects_unknown_identifier(tmp_path):
    """Config loader rejects an identifier not in APPROVED_FELID_IDENTIFIERS."""
    config_path = _build_fixture_config(
        tmp_path,
        [
            ("Felis catus", "GCF_000181335.3"),
            ("Unknown species", "GCF_999999999.9"),
        ],
    )
    with pytest.raises(ValueError, match="not an approved felid identifier"):
        load_felid_foundation_pipeline_config(config_path)


def test_config_loader_rejects_duplicate_identifier(tmp_path):
    """Config loader rejects duplicate identifiers."""
    config_path = _build_fixture_config(
        tmp_path,
        [
            ("Felis catus", "GCF_000181335.3"),
            ("Felis catus", "GCF_000181335.3"),
        ],
    )
    with pytest.raises(ValueError, match="duplicate identifier"):
        load_felid_foundation_pipeline_config(config_path)


def test_config_loader_rejects_missing_reference_dir(tmp_path):
    """Config loader requires paths.reference_dir.

    Drops the required ``reference_dir`` key from an otherwise
    canonical example config and asserts the loader surfaces the exact
    missing-field diagnostic, so future edits to ``[paths]`` can't
    silently make ``reference_dir`` optional.
    """
    import tomli_w

    config_dict = load_example_config_dict()
    config_dict["paths"].pop("reference_dir", None)
    config_path = tmp_path / "config.toml"
    with config_path.open("wb") as handle:
        tomli_w.dump(config_dict, handle)

    with pytest.raises(ValueError, match="paths.reference_dir is required"):
        load_felid_foundation_pipeline_config(config_path)


def test_run_pretrain_missing_fasta(tmp_path):
    """run_felid_foundation_pretrain raises when a FASTA is missing."""
    config_path = _build_fixture_config(
        tmp_path,
        [("Felis catus", "GCF_000181335.3")],
    )
    # Don't create the FASTA
    with pytest.raises(MissingFelidReferenceError, match="acquire-felid-foundation-assemblies"):
        run_felid_foundation_pretrain(config_path)


def test_run_pretrain_source_is_reference(tmp_path):
    """Every emitted TokenizedWindow has source='reference'."""
    config_path = _build_fixture_config(
        tmp_path,
        [("Felis catus", "GCF_000181335.3")],
    )
    reference_dir = tmp_path / "reference"
    reference_dir.mkdir(exist_ok=True)
    fasta_path = reference_dir / "GCF_000181335.3.fna.gz"
    fasta_path.write_bytes(
        _build_fixture_fasta(
            {
                "NC_018723.3": "A" * 1000,
            }
        )
    )

    # Use a fake tokenizer that captures windows
    captured_windows = []

    def fake_tokenizer_loader(provenance):
        def fake_tokenizer(sequence, **kwargs):
            n = min(max(1, len(sequence) // 6), provenance.max_position_embeddings)
            return {"input_ids": list(range(n)), "attention_mask": [1] * n}

        return fake_tokenizer, provenance

    def fake_export_writer(*args, **kwargs):
        writer = MagicMock()
        writer.__enter__ = MagicMock(return_value=writer)
        writer.__exit__ = MagicMock(return_value=False)

        def capture_batch(windows):
            captured_windows.extend(windows)

        writer.write_batch = capture_batch
        return writer

    run_felid_foundation_pretrain(
        config_path,
        tokenizer_loader=fake_tokenizer_loader,
        export_writer=fake_export_writer,
    )

    assert len(captured_windows) > 0
    for window in captured_windows:
        assert isinstance(window, TokenizedWindow)
        assert window.window.source == "reference"


def test_run_pretrain_individual_id_is_species_slug(tmp_path):
    """Every emitted TokenizedWindow has individual_id=species_slug."""
    config_path = _build_fixture_config(
        tmp_path,
        [("Panthera leo", "GCF_018350215.1")],
    )
    reference_dir = tmp_path / "reference"
    reference_dir.mkdir(exist_ok=True)
    fasta_path = reference_dir / "GCF_018350215.1.fna.gz"
    fasta_path.write_bytes(
        _build_fixture_fasta(
            {
                "NC_fake_1": "C" * 1000,
            }
        )
    )

    captured_windows = []

    def fake_tokenizer_loader(provenance):
        def fake_tokenizer(sequence, **kwargs):
            n = min(max(1, len(sequence) // 6), provenance.max_position_embeddings)
            return {"input_ids": list(range(n)), "attention_mask": [1] * n}

        return fake_tokenizer, provenance

    def fake_export_writer(*args, **kwargs):
        writer = MagicMock()
        writer.__enter__ = MagicMock(return_value=writer)
        writer.__exit__ = MagicMock(return_value=False)

        def capture_batch(windows):
            captured_windows.extend(windows)

        writer.write_batch = capture_batch
        return writer

    run_felid_foundation_pretrain(
        config_path,
        tokenizer_loader=fake_tokenizer_loader,
        export_writer=fake_export_writer,
    )

    assert len(captured_windows) > 0
    for window in captured_windows:
        assert window.window.individual_id == "panthera_leo"


def test_run_pretrain_contig_collision_raises(tmp_path):
    """Cross-species contig-name collision raises with both species named."""
    config_path = _build_fixture_config(
        tmp_path,
        [
            ("Felis catus", "GCF_000181335.3"),
            ("Panthera leo", "GCF_018350215.1"),
        ],
    )
    reference_dir = tmp_path / "reference"
    reference_dir.mkdir(exist_ok=True)

    # Both FASTAs declare the same contig ID
    shared_contig = "chr1"
    (reference_dir / "GCF_000181335.3.fna.gz").write_bytes(
        _build_fixture_fasta({shared_contig: "A" * 1000})
    )
    (reference_dir / "GCF_018350215.1.fna.gz").write_bytes(
        _build_fixture_fasta({shared_contig: "C" * 1000})
    )

    def fake_tokenizer_loader(provenance):
        def fake_tokenizer(sequence, **kwargs):
            n = min(max(1, len(sequence) // 6), provenance.max_position_embeddings)
            return {"input_ids": list(range(n)), "attention_mask": [1] * n}

        return fake_tokenizer, provenance

    def fake_export_writer(*args, **kwargs):
        writer = MagicMock()
        writer.__enter__ = MagicMock(return_value=writer)
        writer.__exit__ = MagicMock(return_value=False)
        writer.write_batch = MagicMock()
        return writer

    with pytest.raises(RuntimeError, match="chr1.*felis_catus.*panthera_leo"):
        run_felid_foundation_pretrain(
            config_path,
            tokenizer_loader=fake_tokenizer_loader,
            export_writer=fake_export_writer,
        )


def test_pipeline_zero_windows_leaves_no_artifacts(tmp_path):
    """Zero-window pipeline run raises and leaves no corpus directory.

    When every species FASTA yields zero windows post-filter,
    the pipeline must abort inside the writer's ``with`` block so the
    writer's exception cleanup path runs before the ``RuntimeError``
    surfaces. Beyond the artefact-level guarantees (no ``metadata.json``,
    unlinked Parquet files, no SQLite sidecar) the run must also leave
    no *directory* on disk — a stray empty ``felid_foundation_tokens/``
    tree would otherwise surface to downstream autodiscovery tooling
    as a phantom corpus. Guards against the prior order-of-operations
    bug where the check ran after ``__exit__`` and a zero-manifest
    ``metadata.json`` was emitted into the output tree.
    """
    config_path = _build_fixture_config(
        tmp_path,
        [("Felis catus", "GCF_000181335.3")],
    )
    reference_dir = tmp_path / "reference"
    reference_dir.mkdir(exist_ok=True)
    (reference_dir / "GCF_000181335.3.fna.gz").write_bytes(
        _build_fixture_fasta({"NC_short": "A" * 100})
    )

    def fake_tokenizer_loader(provenance):
        def fake_tokenizer(sequence, **kwargs):
            n = min(max(1, len(sequence) // 6), provenance.max_position_embeddings)
            return {"input_ids": list(range(n)), "attention_mask": [1] * n}

        return fake_tokenizer, provenance

    with pytest.raises(RuntimeError, match="zero tokenized windows"):
        run_felid_foundation_pretrain(
            config_path,
            tokenizer_loader=fake_tokenizer_loader,
        )

    corpus_dir = tmp_path / "processed" / "felid_foundation_tokens"
    assert not corpus_dir.exists(), (
        "zero-window aborts must leave no corpus directory behind; "
        f"found {sorted(corpus_dir.rglob('*')) if corpus_dir.exists() else 'n/a'}"
    )


def test_run_pretrain_streaming_memory_model(tmp_path):
    """Tokenizer sees at most one species' records concurrently."""
    config_path = _build_fixture_config(
        tmp_path,
        [
            ("Felis catus", "GCF_000181335.3"),
            ("Panthera leo", "GCF_018350215.1"),
        ],
    )
    reference_dir = tmp_path / "reference"
    reference_dir.mkdir(exist_ok=True)
    (reference_dir / "GCF_000181335.3.fna.gz").write_bytes(
        _build_fixture_fasta({"NC_A": "A" * 1000})
    )
    (reference_dir / "GCF_018350215.1.fna.gz").write_bytes(
        _build_fixture_fasta({"NC_B": "C" * 1000})
    )

    # Track which species the tokenizer has seen
    species_seen_at_once = set()
    max_concurrent_species = 0

    def fake_tokenizer_loader(provenance):
        def fake_tokenizer(sequence, **kwargs):
            n = min(max(1, len(sequence) // 6), provenance.max_position_embeddings)
            return {"input_ids": list(range(n)), "attention_mask": [1] * n}

        return fake_tokenizer, provenance

    def fake_export_writer(*args, **kwargs):
        writer = MagicMock()
        writer.__enter__ = MagicMock(return_value=writer)
        writer.__exit__ = MagicMock(return_value=False)

        def capture_batch(windows):
            nonlocal max_concurrent_species
            # Extract unique species from this batch
            batch_species = {w.window.individual_id for w in windows}
            species_seen_at_once.update(batch_species)
            max_concurrent_species = max(max_concurrent_species, len(batch_species))
            # After capturing, clear for next species
            if len(batch_species) > 0:
                # This batch should only have one species
                assert len(batch_species) == 1

        writer.write_batch = capture_batch
        return writer

    run_felid_foundation_pretrain(
        config_path,
        tokenizer_loader=fake_tokenizer_loader,
        export_writer=fake_export_writer,
    )

    # The tokenizer should have seen both species, but only one at a time
    assert species_seen_at_once == {"felis_catus", "panthera_leo"}
    assert max_concurrent_species == 1


def test_run_pretrain_streams_tokenized_windows_in_chunks(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    """Single-species writes are chunked so tokenized windows never fully materialize."""
    monkeypatch.setattr(
        felid_foundation_pipeline,
        "_TOKENIZED_WINDOW_CHUNK_SIZE",
        2,
    )

    config_path = _build_fixture_config(
        tmp_path,
        [("Felis catus", "GCF_000181335.3")],
    )
    reference_dir = tmp_path / "reference"
    reference_dir.mkdir(exist_ok=True)
    (reference_dir / "GCF_000181335.3.fna.gz").write_bytes(
        _build_fixture_fasta({"NC_chunked": "A" * 1600})
    )

    batch_sizes: list[int] = []

    def fake_tokenizer_loader(provenance):
        def fake_tokenizer(sequence, **kwargs):
            n = min(max(1, len(sequence) // 6), provenance.max_position_embeddings)
            return {"input_ids": list(range(n)), "attention_mask": [1] * n}

        return fake_tokenizer, provenance

    def fake_export_writer(*args, **kwargs):
        writer = MagicMock()
        writer.__enter__ = MagicMock(return_value=writer)
        writer.__exit__ = MagicMock(return_value=False)

        def capture_batch(windows):
            batch_sizes.append(len(tuple(windows)))

        writer.write_batch = capture_batch
        return writer

    result = run_felid_foundation_pretrain(
        config_path,
        tokenizer_loader=fake_tokenizer_loader,
        export_writer=fake_export_writer,
    )

    assert len(batch_sizes) >= 2
    assert max(batch_sizes) == 2
    assert all(size == 2 for size in batch_sizes[:-1])
    assert batch_sizes[-1] <= 2
    felis_stats = next(
        stats for stats in result.per_species_stats if stats.species_slug == "felis_catus"
    )
    assert sum(felis_stats.window_counts_by_split.values()) == sum(batch_sizes)
    assert felis_stats.peak_window_count_in_memory == 2


def test_run_summary_schema_exact_keys(tmp_path):
    """Run-summary JSON has exactly the documented key set."""
    config_path = _build_fixture_config(
        tmp_path,
        [("Felis catus", "GCF_000181335.3")],
    )
    reference_dir = tmp_path / "reference"
    reference_dir.mkdir(exist_ok=True)
    (reference_dir / "GCF_000181335.3.fna.gz").write_bytes(
        _build_fixture_fasta({"NC_test": "A" * 1000})
    )

    def fake_tokenizer_loader(provenance):
        def fake_tokenizer(sequence, **kwargs):
            n = min(max(1, len(sequence) // 6), provenance.max_position_embeddings)
            return {"input_ids": list(range(n)), "attention_mask": [1] * n}

        return fake_tokenizer, provenance

    def fake_export_writer(*args, **kwargs):
        writer = MagicMock()
        writer.__enter__ = MagicMock(return_value=writer)
        writer.__exit__ = MagicMock(return_value=False)
        writer.write_batch = MagicMock()
        return writer

    result = run_felid_foundation_pretrain(
        config_path,
        tokenizer_loader=fake_tokenizer_loader,
        export_writer=fake_export_writer,
    )

    # Load the run-summary JSON
    summary = json.loads(result.artifacts.summary_path.read_text())

    # Top-level keys
    assert set(summary.keys()) == {
        "config_name",
        "tokenizer_identifier",
        "tokenizer_revision",
        "species",
        "per_species",
        "totals",
    }

    # per_species entry keys
    for _species_slug, species_data in summary["per_species"].items():
        assert set(species_data.keys()) == {
            "identifier",
            "assembly_name",
            "contig_count",
            "retained_sequence_count",
            "filtered_short_count",
            "filtered_high_ambiguity_count",
            "window_counts_by_split",
            "peak_window_count_in_memory",
            "peak_rss_bytes",
            "bytes_tokenized",
            "export_path",
        }
        assert set(species_data["window_counts_by_split"].keys()) == {"train", "validation"}

    # totals keys
    assert set(summary["totals"].keys()) == {"train", "validation"}


def test_ambiguity_threshold_boundary_retained(tmp_path):
    """A window whose ambiguity_fraction == max_ambiguity_fraction exactly is retained."""
    config_path = _build_fixture_config(
        tmp_path,
        [("Felis catus", "GCF_000181335.3")],
    )
    reference_dir = tmp_path / "reference"
    reference_dir.mkdir(exist_ok=True)

    # Create a sequence with exactly 50% ambiguity (255 Ns out of 510)
    sequence = "A" * 255 + "N" * 255
    (reference_dir / "GCF_000181335.3.fna.gz").write_bytes(
        _build_fixture_fasta({"NC_boundary": sequence})
    )

    captured_windows = []

    def fake_tokenizer_loader(provenance):
        def fake_tokenizer(sequence, **kwargs):
            n = min(max(1, len(sequence) // 6), provenance.max_position_embeddings)
            return {"input_ids": list(range(n)), "attention_mask": [1] * n}

        return fake_tokenizer, provenance

    def fake_export_writer(*args, **kwargs):
        writer = MagicMock()
        writer.__enter__ = MagicMock(return_value=writer)
        writer.__exit__ = MagicMock(return_value=False)

        def capture_batch(windows):
            captured_windows.extend(windows)

        writer.write_batch = capture_batch
        return writer

    run_felid_foundation_pretrain(
        config_path,
        tokenizer_loader=fake_tokenizer_loader,
        export_writer=fake_export_writer,
    )

    # The window should be retained (ambiguity_fraction = 0.5, max = 0.5)
    assert len(captured_windows) >= 1


def test_acquisition_happy_path_skip(tmp_path):
    """acquire_felid_foundation_assemblies skips files with matching MD5."""
    config_path = _build_fixture_config(
        tmp_path,
        [("Felis catus", "GCF_000181335.3")],
    )
    config = load_felid_foundation_pipeline_config(config_path)

    reference_dir = tmp_path / "reference"
    reference_dir.mkdir(exist_ok=True)

    # Pre-place the explicit Felis catus FASTA with a known MD5; padded
    # species already have placeholder FASTAs written by the fixture.
    from jaguar_geo_assign.data.felid_assemblies import APPROVED_FELID_ASSEMBLIES

    felis_assembly = next(a for a in APPROVED_FELID_ASSEMBLIES if a.identifier == "GCF_000181335.3")
    fasta_content = _build_fixture_fasta({"NC_test": "A" * 100})
    fasta_path = reference_dir / f"{felis_assembly.identifier}.fna.gz"
    fasta_path.write_bytes(fasta_content)
    computed_md5 = hashlib.md5(fasta_content).hexdigest()

    def fake_opener():
        # Should not be called if checksum matches
        raise AssertionError("Opener should not be called for checksum-matched files")

    from unittest.mock import patch

    with patch(
        "jaguar_geo_assign.data.felid_acquisition.build_felid_reference_manifest"
    ) as mock_manifest:
        mock_manifest.return_value = _build_full_mock_manifest(
            reference_dir,
            overrides={"GCF_000181335.3": computed_md5},
        )

        summary = acquire_felid_foundation_assemblies(config, opener=fake_opener)

        # All six species have matching checksums on disk and therefore skip.
        assert summary.skipped_count == 6
        assert summary.redownloaded_count == 0
        assert summary.total_bytes_written == 0


def test_acquisition_checksum_mismatch_redownload(tmp_path):
    """acquire_felid_foundation_assemblies deletes and redownloads on MD5 mismatch."""
    config_path = _build_fixture_config(
        tmp_path,
        [("Felis catus", "GCF_000181335.3")],
    )
    config = load_felid_foundation_pipeline_config(config_path)

    reference_dir = tmp_path / "reference"
    reference_dir.mkdir(exist_ok=True)

    from jaguar_geo_assign.data.felid_assemblies import APPROVED_FELID_ASSEMBLIES

    felis_assembly = next(a for a in APPROVED_FELID_ASSEMBLIES if a.identifier == "GCF_000181335.3")
    fasta_path = reference_dir / f"{felis_assembly.identifier}.fna.gz"

    # Pre-place a file with the WRONG MD5
    wrong_content = b"wrong content"
    fasta_path.write_bytes(wrong_content)
    assert fasta_path.exists()

    # The correct content
    correct_content = _build_fixture_fasta({"NC_test": "A" * 100})
    correct_md5 = hashlib.md5(correct_content).hexdigest()

    from unittest.mock import patch

    with patch(
        "jaguar_geo_assign.data.felid_acquisition.build_felid_reference_manifest"
    ) as mock_manifest:
        mock_manifest.return_value = _build_full_mock_manifest(
            reference_dir,
            overrides={"GCF_000181335.3": correct_md5},
        )

        with patch("jaguar_geo_assign.data.felid_acquisition.download_with_retry") as mock_download:
            # Mock download_with_retry to write the correct content
            def fake_download(asset, *args, **kwargs):
                asset.destination.write_bytes(correct_content)
                return DownloadResult(
                    path=asset.destination,
                    attempts=1,
                    resumed=False,
                    skipped_existing=False,
                    bytes_written=len(correct_content),
                )

            mock_download.side_effect = fake_download

            summary = acquire_felid_foundation_assemblies(config)

            # Felis catus triggers re-download (checksum mismatch); the
            # five padded species skip on matching placeholder MD5s.
            assert summary.redownloaded_count == 1
            assert summary.skipped_count == 5
            assert summary.total_bytes_written == len(correct_content)


def test_acquisition_failure_preserves_root_cause(tmp_path):
    """Acquisition failure includes root-cause exception class and message."""
    config_path = _build_fixture_config(
        tmp_path,
        [("Felis catus", "GCF_000181335.3")],
    )
    config = load_felid_foundation_pipeline_config(config_path)

    reference_dir = tmp_path / "reference"
    reference_dir.mkdir(exist_ok=True)

    from jaguar_geo_assign.data.felid_assemblies import APPROVED_FELID_ASSEMBLIES

    felis_assembly = next(a for a in APPROVED_FELID_ASSEMBLIES if a.identifier == "GCF_000181335.3")
    _fasta_path = reference_dir / f"{felis_assembly.identifier}.fna.gz"

    from unittest.mock import patch

    with patch(
        "jaguar_geo_assign.data.felid_acquisition.build_felid_reference_manifest"
    ) as mock_manifest:
        # Full six-species manifest; Felis catus (explicit, no on-disk
        # file) is iterated first, so the download attempt fires before
        # padded-species skips matter.
        mock_manifest.return_value = _build_full_mock_manifest(
            reference_dir,
            overrides={"GCF_000181335.3": "fakechecksum"},
        )

        with patch("jaguar_geo_assign.data.felid_acquisition.download_with_retry") as mock_download:
            # The acquisition loop catches AcquisitionError and preserves
            # its ``__cause__`` in the wrapped FelidAcquisitionError, so the
            # fake must mirror that contract: raise AcquisitionError with
            # ConnectionResetError as the root cause.
            from jaguar_geo_assign.data.acquisition import AcquisitionError

            def raise_wrapped(*args, **kwargs):
                try:
                    raise ConnectionResetError("kaboom")
                except ConnectionResetError as exc:
                    raise AcquisitionError("download failed") from exc

            mock_download.side_effect = raise_wrapped

            with pytest.raises(FelidAcquisitionError) as exc_info:
                acquire_felid_foundation_assemblies(config)

            error_message = str(exc_info.value)
            assert "ConnectionResetError" in error_message
            assert "kaboom" in error_message
            assert "GCF_000181335.3" in error_message


def test_acquisition_forwards_mirror_and_size(tmp_path: Path) -> None:
    """rebases paths but forwards mirror_url and expected_size."""
    config_path = _build_fixture_config(
        tmp_path,
        [("Panthera onca", "DNAZOO_Panthera_onca_HiC")],
    )
    config = load_felid_foundation_pipeline_config(config_path)
    reference_dir = tmp_path / "reference"

    def fake_opener():
        pass

    from unittest.mock import patch

    with patch(
        "jaguar_geo_assign.data.felid_acquisition.build_felid_reference_manifest"
    ) as mock_manifest:
        # Mock manifest sets mirror_url and expected_size for jaguar.
        mock_manifest.return_value = _build_full_mock_manifest(
            reference_dir,
            overrides={"DNAZOO_Panthera_onca_HiC": "mocked_checksum"},
        )

        with patch("jaguar_geo_assign.data.felid_acquisition.download_with_retry") as mock_download:
            # We mock download_with_retry so it succeeds
            def fake_download(asset, *args, **kwargs):
                asset.destination.write_bytes(b"content")
                return DownloadResult(
                    path=asset.destination,
                    attempts=1,
                    resumed=False,
                    skipped_existing=False,
                    bytes_written=7,
                )

            mock_download.side_effect = fake_download

            acquire_felid_foundation_assemblies(config, opener=fake_opener)

            # Assert that download_with_retry was called with the jaguar asset containing
            # mirror_url and expected_size
            jaguar_call = None
            for call in mock_download.call_args_list:
                asset = call.args[0]
                if asset.destination.name.startswith("DNAZOO_Panthera_onca_HiC"):
                    jaguar_call = asset
                    break

            assert jaguar_call is not None, "Jaguar asset was not downloaded"
            assert "huggingface.co" in (jaguar_call.mirror_url or ""), "mirror_url was dropped"
            assert jaguar_call.expected_size == 745951926, "expected_size was dropped"


def test_build_full_mock_manifest_jaguar_checksum_name():
    """Mock manifest for jaguar has checksum_name='sha256'."""
    manifest = _build_full_mock_manifest(Path("/fake"))
    jaguar_asset = next(a for a in manifest if "Panthera_onca" in a.destination.name)
    assert jaguar_asset.checksum_name == "sha256"
