"""Tests for the active DNABERT-2 ``trust_remote_code`` policy.

The approved tokenizer provenance is shared by the felid foundation corpus
builder and the continued pre-training stack. These tests keep that security-
critical contract pinned after the legacy feline pipeline removal.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from jaguar_geo_assign.config import load_felid_foundation_pipeline_config
from jaguar_geo_assign.data.pipeline_contract import DNABERT2_TRUST_REMOTE_CODE
from jaguar_geo_assign.data.preprocessor import (
    DNABERT2_TOKENIZER_PROVENANCE,
    TokenizerContractError,
    load_dnabert2_tokenizer,
)
from jaguar_geo_assign.pretrain._shared import _assert_tokenizer_matches_config


def test_load_dnabert2_tokenizer_uses_explicit_trust_remote_code_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tokenizer loading should forward the pinned boolean policy into transformers."""
    captured: dict[str, object] = {}
    fake_tokenizer = object()

    class FakeAutoTokenizer:
        """Small stand-in used to capture ``from_pretrained`` kwargs."""

        @staticmethod
        def from_pretrained(identifier: str, *, revision: str, trust_remote_code: bool) -> object:
            captured.update(
                {
                    "identifier": identifier,
                    "revision": revision,
                    "trust_remote_code": trust_remote_code,
                }
            )
            return fake_tokenizer

    monkeypatch.setattr("jaguar_geo_assign.data.preprocessor.AutoTokenizer", FakeAutoTokenizer)

    tokenizer, provenance = load_dnabert2_tokenizer(DNABERT2_TOKENIZER_PROVENANCE)

    assert tokenizer is fake_tokenizer
    assert provenance == DNABERT2_TOKENIZER_PROVENANCE
    assert captured == {
        "identifier": DNABERT2_TOKENIZER_PROVENANCE.identifier,
        "revision": DNABERT2_TOKENIZER_PROVENANCE.revision,
        "trust_remote_code": DNABERT2_TRUST_REMOTE_CODE,
    }


def test_load_dnabert2_tokenizer_rejects_unapproved_trust_remote_code_policy() -> None:
    """Any divergence from the approved remote-code policy should raise immediately."""
    with pytest.raises(TokenizerContractError, match="trust_remote_code policy mismatch"):
        load_dnabert2_tokenizer(
            replace(
                DNABERT2_TOKENIZER_PROVENANCE,
                trust_remote_code=not DNABERT2_TRUST_REMOTE_CODE,
            )
        )


@pytest.mark.parametrize(
    ("invalid_value", "expected_fragment"),
    [(1, "1 (int)"), ("true", "'true' (str)")],
)
def test_tokenizer_provenance_requires_boolean_trust_remote_code(
    invalid_value: object,
    expected_fragment: str,
) -> None:
    """Non-boolean provenance values should fail before any model code executes."""
    with pytest.raises(ValueError, match="actual boolean") as exc_info:
        replace(DNABERT2_TOKENIZER_PROVENANCE, trust_remote_code=invalid_value)

    assert expected_fragment in str(exc_info.value)


def test_assert_tokenizer_matches_config_rejects_trust_remote_code_mismatch() -> None:
    """Foundation tokenizer provenance must match the config's approved boolean exactly."""
    config = load_felid_foundation_pipeline_config("configs/examples/felid_foundation_pretrain.toml")

    with pytest.raises(RuntimeError, match="trust_remote_code"):
        _assert_tokenizer_matches_config(
            config,
            replace(
                DNABERT2_TOKENIZER_PROVENANCE,
                max_position_embeddings=config.tokenizer.max_position_embeddings,
                unsupported_symbol_policy=config.tokenizer.unsupported_symbol_policy,
                trust_remote_code=not config.tokenizer.trust_remote_code,
            ),
        )


def test_assert_tokenizer_matches_config_rejects_non_boolean_trust_remote_code() -> None:
    """Runtime validation should reject truthy non-bools even when equality would pass."""
    config = load_felid_foundation_pipeline_config("configs/examples/felid_foundation_pretrain.toml")

    with pytest.raises(RuntimeError, match="actual boolean") as exc_info:
        _assert_tokenizer_matches_config(
            config,
            SimpleNamespace(
                identifier=config.tokenizer.identifier,
                revision=config.tokenizer.revision,
                allowed_alphabet=config.tokenizer.allowed_alphabet,
                unsupported_symbol_policy=config.tokenizer.unsupported_symbol_policy,
                max_position_embeddings=config.tokenizer.max_position_embeddings,
                trust_remote_code=1,
            ),
        )

    assert "1 (int)" in str(exc_info.value)
