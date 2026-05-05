"""Shared test fixtures and helpers for foundation training tests.

Reduces boilerplate by centralizing:
- CPU accelerator setup and cleanup
- Tiny BERT model creation
- Training config file writing
- Dummy data loader builders for DDP integration tests
"""

from pathlib import Path

import pytest
import torch
from transformers import AutoModelForMaskedLM, BertConfig

try:
    from accelerate import Accelerator
    from accelerate.state import AcceleratorState, PartialState
except ImportError:
    Accelerator = None
    AcceleratorState = None
    PartialState = None


@pytest.fixture(autouse=True)
def reset_accelerate_state():
    """Reset AcceleratorState before and after each test to prevent cross-test pollution."""
    if AcceleratorState is not None:
        AcceleratorState._reset_state()
        PartialState._reset_state()
    yield
    if AcceleratorState is not None:
        AcceleratorState._reset_state()
        PartialState._reset_state()


@pytest.fixture
def cpu_accelerator(monkeypatch):
    """Force accelerate to CPU mode for the duration of the test.

    Replaces manual os.environ["ACCELERATE_USE_CPU"] = "true" patterns.
    The reset_accelerate_state fixture (autouse) handles state cleanup.
    """
    monkeypatch.setenv("ACCELERATE_USE_CPU", "true")
    yield


@pytest.fixture
def tiny_bert_model():
    """Create a 1-layer 2-head BERT MLM model for DDP tests.

    Config: 1 hidden layer, 2 attention heads, hidden_size=32, vocab_size=30522.
    Used across all integration tests to avoid repeated BertConfig boilerplate.
    """
    config = BertConfig(
        num_hidden_layers=1,
        num_attention_heads=2,
        hidden_size=32,
        vocab_size=30522,
    )
    return AutoModelForMaskedLM.from_config(config)


class DummyDataset:
    """Minimal dataset with set_epoch support for DDP training tests.

    Attributes:
        record_count: Number of records for eval loop termination (default 1).
    """

    def __init__(self, record_count: int = 1):
        """Initialize with configurable record count.

        Args:
            record_count: Number of records (used by eval to compute max steps).
        """
        self.record_count = record_count

    def set_epoch(self, epoch: int) -> None:
        """Placeholder for epoch-based RNG seeding (unused in unit tests)."""
        pass


class DummyLoader:
    """Data loader yielding fixed batches for integration tests.

    Attributes:
        dataset: DummyDataset instance with set_epoch method and record_count.
        num_batches: Number of batches to yield per __iter__.
        masked_label_count: Number of non-masked labels per batch.
    """

    def __init__(self, num_batches: int = 1, masked_label_count: int = 1):
        """Initialize loader with configurable batch count and label pattern.

        Args:
            num_batches: Number of batches to yield in __iter__.
            masked_label_count: Number of non-masked labels in the batch.
        """
        self.dataset = DummyDataset(record_count=num_batches)
        self.num_batches = num_batches
        self.masked_label_count = masked_label_count

    def __iter__(self):
        """Yield num_batches of fixed-shape batches on the accelerator device."""
        if Accelerator is None:
            device = torch.device("cpu")
        else:
            device = Accelerator().device

        for _ in range(self.num_batches):
            yield {
                "input_ids": torch.tensor([[101, 200, 102, 0]], device=device),
                "attention_mask": torch.tensor([[1, 1, 1, 0]], device=device),
                "labels": torch.tensor([[-100, 200, -100, -100]], device=device),
            }

    def __len__(self) -> int:
        """Return the number of batches."""
        return self.num_batches


def make_dummy_loader(
    num_batches: int = 1,
    masked_label_count: int = 1,
    with_eval: bool = False,
) -> DummyLoader | tuple:
    """Create a DummyLoader (or tuple of train/eval loaders) for DDP tests.

    Args:
        num_batches: Number of batches per epoch.
        masked_label_count: Number of non-masked labels per batch.
        with_eval: If True, return (train_loader, eval_loader); else just train_loader.

    Returns:
        DummyLoader if with_eval=False, else (DummyLoader, DummyLoader) tuple.
    """
    train_loader = DummyLoader(num_batches, masked_label_count)
    if with_eval:
        eval_loader = DummyLoader(num_batches, masked_label_count)
        return (train_loader, eval_loader)
    return train_loader


def write_train_config(
    tmp_path: Path,
    metadata_path: Path,
    **overrides,
) -> Path:
    """Write a training config TOML file with sensible defaults and overrides.

    Args:
        tmp_path: Directory to write the config to.
        metadata_path: Path to the corpus metadata.json.
        **overrides: Keyword arguments to override defaults (e.g., max_steps=5).

    Returns:
        Path to the written config file.

    Defaults (applied if not overridden):
        - model_identifier: zhihan1996/DNABERT-2-117M
        - model_revision: main
        - output_dir: {tmp_path}/out
        - max_steps: 1
        - learning_rate: 1e-4
        - seed: 42
        - per_device_train_batch_size: 1
        - gradient_accumulation_steps: 1
        - log_every: 1
    """
    defaults = {
        "model_identifier": "zhihan1996/DNABERT-2-117M",
        "model_revision": "main",
        "output_dir": str(tmp_path / "out"),
        "max_steps": 1,
        "learning_rate": 1e-4,
        "seed": 42,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "log_every": 1,
    }
    defaults.update(overrides)

    lines = ["[training]", f'corpus_metadata_path = "{metadata_path}"']
    for key, val in defaults.items():
        if isinstance(val, str):
            lines.append(f'{key} = "{val}"')
        else:
            lines.append(f"{key} = {val}")

    config_file = tmp_path / "train_config.toml"
    config_file.write_text("\n".join(lines))
    return config_file
