"""Shared test helper for materialising felid foundation pretrain configs.

Prevents drift between the canonical example config
(``configs/examples/felid_foundation_pretrain.toml``) and the TOML
fixtures synthesised per-test. Every test that previously hand-wrote a
``config_content = f\"\"\"...\"\"\"`` literal now routes through
:func:`render_example_config`, which reads the example, applies a
minimal set of targeted overrides (``paths.*``, ``species`` list,
``runtime.external_tools``, and an arbitrary scalar override map),
re-serialises via ``tomli_w`` and returns the path to the temporary
TOML file.

The round-trip guarantees that any schema drift between the canonical
example and what the loader expects surfaces as a loader ``ValueError``
at test time rather than a silent divergence masked by a bespoke
fixture.
"""

from __future__ import annotations

import gzip
import hashlib
import tomllib
from pathlib import Path
from typing import Any

import tomli_w

EXAMPLE_FELID_FOUNDATION_CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "configs" / "examples" / "felid_foundation_pretrain.toml"
)


# All six approved felid species in (species, identifier) form, kept here so
# both the unit-test suite and the integration test derive their fixture
# roster from a single source of truth. Mirrors the order published by
# :data:`jaguar_geo_assign.data.felid_assemblies.APPROVED_FELID_ASSEMBLIES`.
ALL_APPROVED_FELIDS: tuple[tuple[str, str], ...] = (
    ("Felis catus", "GCF_000181335.3"),
    ("Panthera leo", "GCF_018350215.1"),
    ("Panthera tigris", "GCF_000464555.1"),
    ("Panthera onca", "DNAZOO_Panthera_onca_HiC"),
    ("Puma concolor", "GCF_003327715.1"),
    ("Panthera pardus", "GCF_001857705.1"),
)


def build_fixture_fasta(contigs: dict[str, str]) -> bytes:
    """Build a gzipped FASTA fixture from a ``{contig_id: sequence}`` dict.

    Keeps the exact byte layout deterministic so the placeholder
    MD5 helper below stays in sync with what
    :func:`write_placeholder_fastas` actually writes to disk.
    """
    lines = []
    for contig_id, sequence in contigs.items():
        lines.append(f">{contig_id}")
        lines.append(sequence)
    fasta_text = "\n".join(lines) + "\n"
    return gzip.compress(fasta_text.encode("ascii"))


def placeholder_fasta_filename(identifier: str) -> str:
    """Return the fixture FASTA filename for *identifier*.

    Mirrors the ``<identifier>.fna.gz`` filename convention enforced
    by :func:`build_felid_reference_manifest`. Padded species need a real
    on-disk fixture FASTA so tests reach the logic under test instead of
    failing with :class:`MissingFelidReferenceError` on a padded entry.
    """
    from jaguar_geo_assign.data.felid_assemblies import APPROVED_FELID_ASSEMBLIES

    for assembly in APPROVED_FELID_ASSEMBLIES:
        if assembly.identifier == identifier:
            return f"{identifier}.fna.gz"
    raise AssertionError(f"Unknown identifier in fixture: {identifier}")


def write_placeholder_fastas(reference_dir: Path, padded_identifiers: list[str]) -> None:
    """Write a minimal unique FASTA for each padded species.

    Placeholder FASTAs use the identifier as the contig ID so they
    cannot collide with user-authored fixture contigs. The 128 bp sequence
    is short enough that windowing yields zero windows, keeping padded
    species invisible to window-count assertions.
    """
    reference_dir.mkdir(parents=True, exist_ok=True)
    for identifier in padded_identifiers:
        path = reference_dir / placeholder_fasta_filename(identifier)
        if path.exists():
            continue
        if "Panthera_onca" in identifier:
            contigs = {identifier: "A" * 128, "HiC_scaffold_1": "A" * 128}
        else:
            contigs = {identifier: "A" * 128}
        path.write_bytes(build_fixture_fasta(contigs))


def placeholder_fasta_checksum(identifier: str) -> str:
    """Return the MD5 of the placeholder FASTA written by :func:`write_placeholder_fastas`."""
    if "Panthera_onca" in identifier:
        contigs = {identifier: "A" * 128, "HiC_scaffold_1": "A" * 128}
    else:
        contigs = {identifier: "A" * 128}
    return hashlib.md5(build_fixture_fasta(contigs)).hexdigest()


def pad_species_to_full_roster(
    subset: list[tuple[str, str]],
) -> list[tuple[str, str]]:
    """Pad *subset* with remaining approved felids so the result has six entries.

    The config loader rejects species lists shorter than six, but
    most tests only care about behaviour on a smaller subset. Padding keeps
    the tests focused while respecting the contract.
    """
    seen = {acc for _, acc in subset}
    padding = [entry for entry in ALL_APPROVED_FELIDS if entry[1] not in seen]
    return list(subset) + padding[: max(0, 6 - len(subset))]


def load_example_config_dict() -> dict[str, Any]:
    """Return the parsed example config as a mutable dict.

    Callers that want to apply structured edits before writing
    can do so on a fresh copy of the canonical contract rather than
    re-inventing it per test.
    """
    with EXAMPLE_FELID_FOUNDATION_CONFIG_PATH.open("rb") as handle:
        return tomllib.load(handle)


def render_example_config(
    tmp_path: Path,
    *,
    paths_overrides: dict[str, Path] | None = None,
    species: list[tuple[str, str]] | None = None,
    runtime_external_tools: tuple[str, ...] | None = None,
    scalar_overrides: dict[str, Any] | None = None,
    filename: str = "config.toml",
) -> Path:
    """Render a test-scoped TOML by overriding the canonical example.

    Args:
        tmp_path: Pytest tmp_path into which the rendered TOML is
            written.
        paths_overrides: Optional mapping of ``paths.*`` keys (e.g.
            ``reference_dir``) to concrete filesystem paths. If
            ``None``, every entry in ``[paths]`` is rewritten to point
            under ``tmp_path``.
        species: Optional replacement for the ``[[species]]`` list as
            a list of ``(species, identifier)`` tuples. When ``None``
            the canonical six-entry roster is retained.
        runtime_external_tools: Optional replacement for
            ``runtime.external_tools``. Unit tests typically pass
            ``()`` to suppress bcftools requirements.
        scalar_overrides: Optional dotted-key overrides (e.g.
            ``{"windowing.max_ambiguous_fraction": 0.5}``) applied
            after structured overrides.
        filename: Name for the rendered TOML inside ``tmp_path``.

    Returns:
        Path to the written TOML file.

    Raises:
        KeyError: If a dotted scalar override references a section or
            key that does not exist in the canonical example.
    """
    config = load_example_config_dict()

    if paths_overrides is None:
        paths_overrides = {
            "reference_dir": tmp_path / "reference",
            "processed_dir": tmp_path / "processed",
            "artifact_dir": tmp_path / "artifacts",
            "report_dir": tmp_path / "reports",
        }
    for key, value in paths_overrides.items():
        config["paths"][key] = str(value)

    if species is not None:
        config["species"] = [
            {"species": sp, "identifier": identifier} for sp, identifier in species
        ]

    if runtime_external_tools is not None:
        config["runtime"]["external_tools"] = list(runtime_external_tools)

    if scalar_overrides:
        for dotted_key, value in scalar_overrides.items():
            section, _, field = dotted_key.partition(".")
            if not field:
                raise KeyError(
                    f"scalar_overrides key {dotted_key!r} must be dotted (section.field)"
                )
            if section not in config:
                raise KeyError(f"scalar_overrides section {section!r} not in example config")
            if field not in config[section]:
                raise KeyError(f"scalar_overrides field {dotted_key!r} not in example config")
            config[section][field] = value

    target = tmp_path / filename
    with target.open("wb") as handle:
        tomli_w.dump(config, handle)
    return target
