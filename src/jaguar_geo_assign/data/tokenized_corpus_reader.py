"""Streaming corpus reader for tokenized felid foundation Parquet datasets.

This module provides a distributed-aware dataset reader that consumes
the metadata.json + Parquet corpus produced by TokenizedCorpusWriter.
The reader handles file-level and row-level shuffling for training,
forced deterministic ordering for validation, and multi-GPU/multi-worker
sharding without inter-rank deadlock.

Key design choices:
- IterableDataset is used (not Map-style) because Parquet columnar access
  is slow for random seeks; streaming with iter_batches() bounds memory.
- Global worker ID = process_index * num_workers + worker_id is computed
  via accelerate.state.PartialState or os.environ fallback.
- Validation splits disable shuffling and yield rows in canonical order
  so eval loss is reproducible across runs.
"""

from __future__ import annotations

import json
import logging
import os
import random
from collections import deque
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import IterableDataset

logger = logging.getLogger(__name__)


class CorpusReaderError(RuntimeError):
    """Raised when metadata.json is missing, incomplete, or invalid."""

    pass


class CorpusSchemaError(ValueError):
    """Raised when Parquet schema does not match the expected contract."""

    pass


@dataclass
class _PartialStateEstimate:
    """Estimate of distributed state when accelerate.state.PartialState is unavailable."""

    process_index: int = 0
    num_processes: int = 1


def _get_distributed_state() -> tuple[int, int, int, int]:
    """Retrieve distributed state: (process_index, num_processes, worker_id, num_workers).

    Attempts to use accelerate.state.PartialState; falls back to environment
    variables (RANK, WORLD_SIZE, LOCAL_RANK, LOCAL_WORLD_SIZE) and PyTorch
    worker info. This ensures the reader works in single-GPU, multi-GPU/DDP,
    and multi-worker DataLoader contexts.

    # TRADE-OFF: environment variable fallback is used when accelerate is not
    # available or in early initialization. Ensures compatibility across training
    # frameworks.
    """
    process_index = 0
    num_processes = 1

    try:
        from accelerate.state import PartialState

        state = PartialState()
        process_index = state.process_index
        num_processes = state.num_processes
    except (ImportError, RuntimeError):
        # Fallback to environment variables set by torchrun / accelerate launch
        process_index = int(os.environ.get("RANK", 0))
        num_processes = int(os.environ.get("WORLD_SIZE", 1))

    # PyTorch DataLoader worker_id (0-indexed within current process)
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        worker_id = worker_info.id
        num_workers = worker_info.num_workers
    else:
        worker_id = 0
        num_workers = 1

    return process_index, num_processes, worker_id, num_workers


class TokenizedCorpusReader(IterableDataset):
    """Distributed streaming reader for tokenized felid corpus Parquet datasets.

    Consumes metadata.json produced by TokenizedCorpusWriter and yields rows
    from the corresponding Parquet files with optional shuffling (disabled for
    validation). Handles multi-GPU (DDP) and multi-worker DataLoader sharding
    transparently.

    Attributes:
        metadata_path (Path): Path to metadata.json sidecar.
        split (str): Split name (e.g., "train", "validation").
        max_seq_length (int): Maximum sequence length; rows are truncated.
        file_shuffle (bool): Enable file-level shuffling (False for validation).
        shuffle_buffer_size (int): Row-level shuffle buffer size.
        seed (int): Random seed for reproducibility.
        drop_last (bool): Drop incomplete final batch (unused for IterableDataset).
    """

    def __init__(
        self,
        metadata_path: str | Path,
        split: str,
        *,
        max_seq_length: int = 512,
        file_shuffle: bool = True,
        shuffle_buffer_size: int = 8192,
        seed: int = 42,
        drop_last: bool = False,
        world_size: int = 1,
        num_workers: int = 0,
    ) -> None:
        """Initialize the corpus reader.

        Args:
            metadata_path: Path to metadata.json produced by TokenizedCorpusWriter.
            split: Split name to read (e.g., "train" or "validation").
            max_seq_length: Truncate each row's input_ids to this length.
            file_shuffle: Enable file-level shuffling. Forced off for validation.
            shuffle_buffer_size: Size of row-level shuffle buffer.
            seed: Random seed for reproducibility.
            drop_last: Unused but accepted for DataLoader compatibility.

        Raises:
            CorpusReaderError: If metadata.json is missing, split key absent,
                files list empty, or record_count is zero.
            CorpusSchemaError: If Parquet schema lacks required columns.
        """
        self.metadata_path = Path(metadata_path)
        self.split = split
        self.max_seq_length = max_seq_length
        self.shuffle_buffer_size = shuffle_buffer_size
        self.seed = seed
        self.corpus_root = self.metadata_path.parent
        self._epoch = 0  # Fix #18: Track epoch for deterministic multi-epoch shuffling

        # Validation split disables shuffling
        self.file_shuffle = file_shuffle and (split != "validation")

        # Cold-start guards: validate metadata and files
        self._validate_metadata_and_schema(world_size, num_workers)

    def _validate_metadata_and_schema(self, world_size: int, num_workers: int) -> None:
        """Validate metadata.json exists, split is present, and Parquet schema is sound.

        Raises:
            CorpusReaderError: If metadata or split is missing/empty.
            CorpusSchemaError: If Parquet schema is invalid.
        """
        if not self.metadata_path.exists():
            raise CorpusReaderError(
                f"metadata.json not found at {self.metadata_path}. "
                "Run `jaguar-geo-assign felid-foundation-pretrain <config>` first."
            )

        try:
            metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise CorpusReaderError(f"Failed to parse metadata.json: {exc}") from exc

        splits = metadata.get("splits", {})
        if self.split not in splits:
            raise CorpusReaderError(
                f"Split '{self.split}' not found in metadata.json. "
                f"Available splits: {list(splits.keys())}. "
                "Run `jaguar-geo-assign felid-foundation-pretrain <config>` first."
            )

        split_info = splits[self.split]
        files = split_info.get("files", [])
        record_count = split_info.get("record_count", 0)

        if not files:
            raise CorpusReaderError(
                f"No files found for split '{self.split}'. "
                "Run `jaguar-geo-assign felid-foundation-pretrain <config>` first."
            )

        if record_count == 0:
            raise CorpusReaderError(
                f"record_count is zero for split '{self.split}'. "
                "Run `jaguar-geo-assign felid-foundation-pretrain <config>` first."
            )

        self._files = [self.corpus_root / f for f in files]
        self._record_count = record_count

        global_workers = max(1, world_size) * max(1, num_workers)
        if len(self._files) < global_workers:
            # TRADE-OFF: Empty-shard deadlock guard catches configurations that would
            # result in a trailing rank receiving 0 batches, which deadlocks DDP.
            raise CorpusReaderError(
                f"Found {len(self._files)} files in {self.corpus_root} but DDP needs at least "
                f"{global_workers} (world_size={world_size} x num_workers={num_workers}). "
                f"Either reduce world_size/num_workers or add more shards to the corpus. "
                f"Producer: `uv run jaguar-geo-assign felid-foundation-pretrain ...`"
            )

        # Probe first Parquet file for schema
        if self._files:
            self._probe_parquet_schema(self._files[0])

    def _probe_parquet_schema(self, parquet_path: Path) -> None:
        """Verify that the first Parquet file contains required columns.

        Args:
            parquet_path: Path to a Parquet file.

        Raises:
            CorpusSchemaError: If input_ids or attention_mask columns are missing.
        """
        try:
            import pyarrow.parquet
        except ImportError as exc:
            raise CorpusSchemaError("pyarrow is required for Parquet schema validation") from exc

        try:
            parquet_file = pyarrow.parquet.ParquetFile(parquet_path)
            schema_names = set(parquet_file.schema.names)
        except Exception as exc:
            msg = f"Failed to read Parquet schema from {parquet_path}: {exc}"
            raise CorpusSchemaError(msg) from exc

        required = {"input_ids", "attention_mask"}
        missing = required - schema_names
        if missing:
            raise CorpusSchemaError(
                f"Parquet file {parquet_path} is missing required columns: {missing}. "
                f"Found columns: {sorted(schema_names)}"
            )

    @property
    def record_count(self) -> int:
        """Return the total number of records in the split.

        Used by training loop to compute max_steps and eval_steps.
        """
        return self._record_count

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch number for reproducible multi-epoch training.

        Fix #18: Mirrors torch.utils.data.distributed.DistributedSampler.set_epoch.
        The epoch is XORed with the random seed inside __iter__ so that the file
        permutation (and row-level RNG) differs across epochs, ensuring data diversity
        in multi-epoch runs. Without this, the same shuffle permutation is reused,
        reducing effective training diversity.

        Args:
            epoch: Epoch number (0-indexed). Should be called by the trainer at the
                start of each epoch before iterating the dataloader.
        """
        self._epoch = epoch

    def __iter__(self) -> Generator[dict[str, Any], None, None]:
        """Iterate over rows with file-level and row-level shuffling.

        Implements two-level shuffle:
        1. File-level: shuffle the file list (epoch-dependent seed), then
           distribute files across global workers using global_worker_id.
        2. Row-level: fill a shuffle buffer from the worker's file stream,
           randomly pop rows, and refill as the stream advances.

        For validation split, shuffling is forced off and rows are yielded
        in canonical metadata.json order.

        Yields:
            Dictionary with at least 'input_ids' and 'attention_mask' keys,
            truncated to max_seq_length.
        """
        try:
            import pyarrow.parquet
        except ImportError as exc:
            raise CorpusSchemaError("pyarrow is required for reading Parquet files") from exc

        # Compute distributed state
        process_index, num_processes, worker_id, num_workers = _get_distributed_state()

        # # TRADE-OFF: global_worker_id formula ensures each worker reads a disjoint
        # subset of files without overlap or gaps. Used for DDP + multi-worker setup.
        global_worker_id = process_index * num_workers + worker_id
        global_world_size = num_processes * num_workers

        if global_world_size > 1:
            logger.info(
                f"Distributed reader: rank={process_index}/{num_processes}, "
                f"worker={worker_id}/{num_workers}, "
                f"global_id={global_worker_id}/{global_world_size}"
            )

        # File-level shuffle: use epoch-dependent seed for multi-epoch diversity
        # Fix #18: Use self._epoch (set by trainer via set_epoch) instead of hardcoded 0
        files = list(self._files)
        if self.file_shuffle:
            # Shuffle with epoch-dependent seed for reproducibility across restarts
            rng = random.Random(self.seed ^ self._epoch)
            rng.shuffle(files)

        # Distribute files across global workers (each worker gets disjoint subset)
        assigned_files = [
            f for i, f in enumerate(files) if i % global_world_size == global_worker_id
        ]

        if not assigned_files:
            logger.warning(
                f"Worker {global_worker_id}/{global_world_size} has no assigned files "
                f"for split '{self.split}'"
            )
            return

        logger.info(
            f"Worker {global_worker_id}/{global_world_size} assigned {len(assigned_files)} files"
        )

        # Row-level shuffle buffer
        shuffle_buffer: deque[dict[str, Any]] = deque(maxlen=self.shuffle_buffer_size)
        rng = random.Random(self.seed ^ self._epoch ^ global_worker_id)

        # Iterate through assigned files
        for file_path in assigned_files:
            try:
                parquet_file = pyarrow.parquet.ParquetFile(file_path)
            except Exception as exc:
                logger.error(f"Failed to open Parquet file {file_path}: {exc}")
                continue

            # Stream batches from Parquet (keeps memory bounded)
            for batch in parquet_file.iter_batches(batch_size=256):
                batch_dict = batch.to_pydict()
                for row_idx in range(len(batch)):
                    row = {k: v[row_idx] for k, v in batch_dict.items()}

                    # Truncate to max_seq_length
                    if "input_ids" in row:
                        row["input_ids"] = row["input_ids"][: self.max_seq_length]
                    if "attention_mask" in row:
                        row["attention_mask"] = row["attention_mask"][: self.max_seq_length]

                    # Fill shuffle buffer (validation: no shuffle, just yield in order)
                    if self.file_shuffle:
                        shuffle_buffer.append(row)
                        if len(shuffle_buffer) == shuffle_buffer.maxlen:
                            # Buffer full; randomly pop one
                            idx = rng.randint(0, len(shuffle_buffer) - 1)
                            yield shuffle_buffer[idx]
                            shuffle_buffer[idx] = shuffle_buffer[-1]
                            shuffle_buffer.pop()
                    else:
                        # Validation: canonical order, no shuffle
                        yield row

        # Flush remaining shuffle buffer
        if self.file_shuffle:
            while shuffle_buffer:
                idx = rng.randint(0, len(shuffle_buffer) - 1)
                yield shuffle_buffer[idx]
                shuffle_buffer[idx] = shuffle_buffer[-1]
                shuffle_buffer.pop()
