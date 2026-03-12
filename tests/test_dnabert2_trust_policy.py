from dataclasses import replace
import sys
from types import SimpleNamespace

import pytest

from jaguar_geo_assign.config import load_feline_pipeline_config
from jaguar_geo_assign.data.pipeline_contract import DNABERT2_TRUST_REMOTE_CODE
from jaguar_geo_assign.data.preprocessor import (
    DNABERT2_TOKENIZER_PROVENANCE,
    TokenizerContractError,
    load_dnabert2_tokenizer,
)
from jaguar_geo_assign.pretrain import pipeline as pretrain_pipeline


def test_load_dnabert2_tokenizer_uses_explicit_trust_remote_code_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    fake_tokenizer = object()

    class FakeAutoTokenizer:
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

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoTokenizer=FakeAutoTokenizer))

    tokenizer, provenance = load_dnabert2_tokenizer(DNABERT2_TOKENIZER_PROVENANCE)

    assert tokenizer is fake_tokenizer
    assert provenance == DNABERT2_TOKENIZER_PROVENANCE
    assert captured == {
        "identifier": DNABERT2_TOKENIZER_PROVENANCE.identifier,
        "revision": DNABERT2_TOKENIZER_PROVENANCE.revision,
        "trust_remote_code": DNABERT2_TRUST_REMOTE_CODE,
    }


def test_load_dnabert2_tokenizer_rejects_unapproved_trust_remote_code_policy() -> None:
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
    invalid_value: object, expected_fragment: str
) -> None:
    with pytest.raises(ValueError, match="actual boolean") as exc_info:
        replace(DNABERT2_TOKENIZER_PROVENANCE, trust_remote_code=invalid_value)

    assert expected_fragment in str(exc_info.value)


def test_assert_tokenizer_matches_config_rejects_trust_remote_code_mismatch() -> None:
    config = load_feline_pipeline_config("configs/examples/feline_pretrain.toml")

    with pytest.raises(RuntimeError, match="trust_remote_code"):
        pretrain_pipeline._assert_tokenizer_matches_config(
            config,
            replace(
                DNABERT2_TOKENIZER_PROVENANCE,
                max_position_embeddings=config.tokenizer.max_position_embeddings,
                unsupported_symbol_policy=config.tokenizer.unsupported_symbol_policy,
                trust_remote_code=not config.tokenizer.trust_remote_code,
            ),
        )


def test_assert_tokenizer_matches_config_rejects_non_boolean_trust_remote_code() -> None:
    config = load_feline_pipeline_config("configs/examples/feline_pretrain.toml")

    with pytest.raises(RuntimeError, match="actual boolean") as exc_info:
        pretrain_pipeline._assert_tokenizer_matches_config(
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