"""Shared test helper for materialising felid foundation pretrain configs.

Intent: prevent drift between the canonical example config
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

from pathlib import Path
import tomllib
from typing import Any

import tomli_w

EXAMPLE_FELID_FOUNDATION_CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "examples"
    / "felid_foundation_pretrain.toml"
)


def load_example_config_dict() -> dict[str, Any]:
    """Return the parsed example config as a mutable dict.

    Intent: callers that want to apply structured edits before writing
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
            a list of ``(species, accession)`` tuples. When ``None``
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
            {"species": sp, "accession": acc} for sp, acc in species
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
                raise KeyError(
                    f"scalar_overrides section {section!r} not in example config"
                )
            if field not in config[section]:
                raise KeyError(
                    f"scalar_overrides field {dotted_key!r} not in example config"
                )
            config[section][field] = value

    target = tmp_path / filename
    with target.open("wb") as handle:
        tomli_w.dump(config, handle)
    return target

