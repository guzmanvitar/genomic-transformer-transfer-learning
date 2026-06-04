"""Offline DNABERT-2 embedding extraction for jaguar MIL training.

.. deprecated::
    This module produces per-window CLS embeddings consumed by the MIL
    pipeline (``mil_trainer.py``), which was found to produce results
    indistinguishable from a random baseline for geographic assignment.
    See ``dev_docs/pipeline_diagnosis_and_plan.md`` for the root-cause
    analysis.

    The module is preserved for the E5 comparison experiment and as
    reference code. The ``_load_tokenizer`` and ``_disable_flash_attention``
    helpers are still imported by ``variant_scoring.py`` for VES computation.

This module freezes a pretrained DNABERT-2 backbone, encodes all window
sequences for each jaguar individual, and writes per-individual embedding
shards to disk. The output contract is intentionally simple and torch-native:

* ``{output_dir}/{individual_id}.pt`` stores one individual's full bag.
* ``manifest.jsonl`` records shard-level metadata for downstream training.
* ``contig_rank.json`` preserves the canonical lexicographic contig ordering.

The extraction path is a one-way materialization step: it does not compute
losses, gradients, or checkpoints, and it must never fine-tune the backbone.
"""

from __future__ import annotations

import json
import logging
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

from jaguar_geo_assign.config import load_embedding_extraction_config
from jaguar_geo_assign.data.finetune_windows import WINDOW_SIZE
from jaguar_geo_assign.fine_tune.dataset import _load_metadata_csv, _load_windows_jsonl
from jaguar_geo_assign.fine_tune.trainer import _ensure_custom_code

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExtractionResult:
    """Summary of one completed offline embedding-extraction run.

    Attributes:
        n_individuals: Number of per-individual shards written.
        n_windows_extracted: Total number of joined windows encoded.
        n_windows_dropped: Windows skipped because their ``sample_id`` had no
            matching metadata row.
        manifest_path: Path to the written manifest JSONL file.
    """

    n_individuals: int
    n_windows_extracted: int
    n_windows_dropped: int
    manifest_path: Path


def _resolve_device(device_name: str) -> torch.device:
    """Resolve the runtime device from the config contract.

    ``auto`` prefers CUDA when available and otherwise falls back to CPU. The
    contract intentionally omits MPS here because the production requirement was
    scoped to ``cpu``/``cuda``/``auto`` only.
    """

    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("extraction.device='cuda' was requested but CUDA is unavailable")
        return torch.device("cuda")
    if device_name == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unsupported device: {device_name!r}")


def _load_tokenizer(backbone_path: Path, backbone: nn.Module | None = None) -> Any:
    """Load a DNABERT-2-compatible tokenizer from the local backbone directory.

    This mirrors the fine-tuning tokenizer fallback semantics so offline
    extraction and downstream training cannot disagree about how padding is
    represented.
    """

    tokenizer_dir = backbone_path.parent / "tokenizer"
    tokenizer_src = str(tokenizer_dir) if tokenizer_dir.is_dir() else str(backbone_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_src, trust_remote_code=True)

    pad_added = False
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            pad_added = True

    if tokenizer.pad_token is None:
        raise RuntimeError(
            "_load_tokenizer: failed to set pad_token; tokenizer has no eos/unk token and "
            "add_special_tokens did not register a pad token."
        )

    if pad_added and backbone is not None:
        backbone.resize_token_embeddings(len(tokenizer))

    return tokenizer


def _disable_flash_attention_if_present() -> None:
    """Disable DNABERT-2's Triton flash-attention hook when it is incompatible.

    The local custom-code path may import a Triton kernel that uses APIs removed
    in newer Triton builds. The MTL trainer already applies this defensive patch;
    extraction must do the same so both paths can load the same backbone.
    """

    patched_any = False
    for module_name, module in list(sys.modules.items()):
        if not module_name.endswith(".bert_layers") or module is None:
            continue
        if getattr(module, "flash_attn_qkvpacked_func", None) is None:
            continue
        module.flash_attn_qkvpacked_func = None
        patched_any = True
    if patched_any:
        logger.info("Disabled DNABERT-2 Triton flash attention; using PyTorch fallback.")


def _pool_hidden_states(
    last_hidden_state: torch.Tensor,
    attention_mask: torch.Tensor | None,
    pooling_strategy: str,
) -> torch.Tensor:
    """Pool one embedding per sequence from backbone hidden states.

    The implementation mirrors :meth:`JaguarMTLModel._pool` so the offline
    materialization path and the original online fine-tuning path remain
    semantically aligned.
    """

    if pooling_strategy == "cls":
        return last_hidden_state[:, 0, :]
    if pooling_strategy != "mean":
        raise ValueError(f"Unsupported pooling strategy: {pooling_strategy!r}")
    if attention_mask is None:
        return last_hidden_state.mean(dim=1)

    mask = attention_mask.to(device=last_hidden_state.device, dtype=last_hidden_state.dtype)
    lengths = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    masked_sum = (last_hidden_state * mask.unsqueeze(-1)).sum(dim=1)
    return masked_sum / lengths


def _join_windows_with_metadata(
    windows_jsonl: Path,
    metadata_csv: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    """Load windows and metadata, then inner-join on ``sample_id``.

    Returns the full raw window list, the joined records used for extraction,
    and the dropped-window count.
    """

    windows = _load_windows_jsonl(Path(windows_jsonl))
    metadata_by_sample = _load_metadata_csv(Path(metadata_csv))
    joined: list[dict[str, Any]] = []
    dropped = 0
    for window in windows:
        sample_id = window.get("sample_id")
        if not sample_id or sample_id not in metadata_by_sample:
            dropped += 1
            continue
        joined.append({**window, **metadata_by_sample[sample_id]})

    if dropped > 0:
        logger.warning(
            "Dropped %d windows during embedding extraction because sample_id was missing "
            "from metadata_csv.",
            dropped,
        )
    if not joined:
        raise ValueError("No windows remained after joining windows_jsonl with metadata_csv")
    return windows, joined, dropped


def _build_contig_rank(windows: list[dict[str, Any]]) -> dict[str, int]:
    """Build the canonical lexicographic contig rank map for stable sorting."""

    unique_contigs = {
        str(window["contig"]) for window in windows if window.get("contig") is not None
    }
    if not unique_contigs:
        raise ValueError("windows_jsonl contained no contig names; cannot build contig rank map")
    return {name: idx for idx, name in enumerate(sorted(unique_contigs))}


def _validate_group_metadata(individual_id: str, records: list[dict[str, Any]]) -> SimpleNamespace:
    """Validate shard-level metadata consistency for one individual.

    The manifest stores one ``sample_id``/latitude/longitude/biome row per
    individual. If the input data would collapse conflicting metadata into a
    single shard, fail loudly instead of silently writing an ambiguous artefact.
    """

    sample_ids = {str(record["sample_id"]) for record in records}
    latitudes = {float(record["latitude"]) for record in records}
    longitudes = {float(record["longitude"]) for record in records}
    biome_labels = {str(record["biome_population_label"]) for record in records}
    if len(sample_ids) != 1:
        raise ValueError(
            f"Individual {individual_id!r} maps to multiple sample_ids {sorted(sample_ids)}; "
            "manifest requires a single sample_id per individual shard"
        )
    if len(latitudes) != 1 or len(longitudes) != 1 or len(biome_labels) != 1:
        raise ValueError(
            f"Individual {individual_id!r} has inconsistent metadata rows; refuse to write "
            "an ambiguous shard"
        )
    return SimpleNamespace(
        sample_id=next(iter(sample_ids)),
        latitude=next(iter(latitudes)),
        longitude=next(iter(longitudes)),
        biome_population_label=next(iter(biome_labels)),
    )


def _record_sort_key(record: dict[str, Any], contig_to_rank: dict[str, int]) -> tuple[Any, ...]:
    """Return the deterministic per-window ordering key within one individual."""

    contig = str(record["contig"])
    return (
        contig_to_rank[contig],
        int(record["locus_pos"]),
        int(record.get("window_start", 0)),
        int(record.get("window_end", 0)),
        str(record.get("alt_allele", "")),
        str(record.get("ref_allele", "")),
    )


def run_embedding_extraction(config_path: str | Path) -> ExtractionResult:
    """Materialize frozen DNABERT-2 embeddings to disk for all jaguar individuals.

    The phase order is intentionally fixed:

    1. Load and validate config.
    2. Load windows + metadata and construct the canonical contig rank map.
    3. Load tokenizer + backbone, freeze all parameters, and switch to eval mode.
    4. Group joined windows by individual and sort each bag deterministically.
    5. Tokenize/encode windows in batches, pool one embedding per window, and
       accumulate CPU-side float32 outputs.
    6. Write per-individual ``.pt`` shards, ``manifest.jsonl``, and
       ``contig_rank.json``.
    """

    config = load_embedding_extraction_config(config_path)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    windows, joined_records, dropped_windows = _join_windows_with_metadata(
        Path(config.windows_jsonl), Path(config.metadata_csv)
    )
    contig_to_rank = _build_contig_rank(joined_records)
    contig_rank_path = output_dir / "contig_rank.json"
    contig_rank_path.write_text(
        json.dumps(contig_to_rank, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    _ensure_custom_code(Path(config.backbone_path))
    backbone = AutoModel.from_pretrained(str(config.backbone_path), trust_remote_code=True)
    _disable_flash_attention_if_present()
    tokenizer = _load_tokenizer(Path(config.backbone_path), backbone=backbone)

    hidden_size = getattr(getattr(backbone, "config", None), "hidden_size", None)
    if hidden_size is None:
        raise RuntimeError("backbone must expose config.hidden_size for embedding extraction")

    for parameter in backbone.parameters():
        parameter.requires_grad = False
    backbone.eval()

    device = _resolve_device(config.device)
    backbone = backbone.to(device)

    records_by_individual: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in joined_records:
        records_by_individual[str(record["individual_id"])].append(record)

    manifest_path = output_dir / "manifest.jsonl"
    n_windows_extracted = 0
    with manifest_path.open("w", encoding="utf-8") as manifest_handle:
        for individual_id in sorted(records_by_individual):
            records = sorted(
                records_by_individual[individual_id],
                key=lambda record: _record_sort_key(record, contig_to_rank),
            )
            metadata = _validate_group_metadata(individual_id, records)

            embeddings_batches: list[torch.Tensor] = []
            bp_positions: list[float] = []
            contigs: list[str] = []

            with torch.inference_mode():
                for start in range(0, len(records), config.extraction_batch_size):
                    batch_records = records[start : start + config.extraction_batch_size]
                    sequences = [str(record["sequence"]) for record in batch_records]
                    encoded = tokenizer(
                        sequences,
                        padding="max_length",
                        truncation=True,
                        max_length=WINDOW_SIZE,
                        return_tensors="pt",
                    )
                    encoded_on_device = {
                        name: tensor.to(device)
                        for name, tensor in encoded.items()
                        if isinstance(tensor, torch.Tensor)
                    }
                    outputs = backbone(**encoded_on_device)
                    if isinstance(outputs, tuple):
                        last_hidden_state = outputs[0]
                    else:
                        last_hidden_state = outputs.last_hidden_state
                    pooled = _pool_hidden_states(
                        last_hidden_state=last_hidden_state,
                        attention_mask=encoded_on_device.get("attention_mask"),
                        pooling_strategy=config.pooling_strategy,
                    )
                    embeddings_batches.append(pooled.detach().to(dtype=torch.float32).cpu())
                    bp_positions.extend(float(record["locus_pos"]) for record in batch_records)
                    contigs.extend(str(record["contig"]) for record in batch_records)

            shard = {
                "embeddings": torch.cat(embeddings_batches, dim=0),
                "bp_positions": torch.tensor(bp_positions, dtype=torch.float32),
                "contigs": contigs,
            }
            shard_path = output_dir / f"{individual_id}.pt"
            torch.save(shard, shard_path)
            n_windows_extracted += len(records)

            manifest_record = {
                "individual_id": individual_id,
                "shard_path": shard_path.name,
                "n_windows": len(records),
                "sample_id": metadata.sample_id,
                "latitude": metadata.latitude,
                "longitude": metadata.longitude,
                "biome_population_label": metadata.biome_population_label,
            }
            manifest_handle.write(json.dumps(manifest_record) + "\n")

    return ExtractionResult(
        n_individuals=len(records_by_individual),
        n_windows_extracted=n_windows_extracted,
        n_windows_dropped=dropped_windows,
        manifest_path=manifest_path,
    )


def format_extraction_result(result: ExtractionResult) -> str:
    """Format an :class:`ExtractionResult` for human-readable CLI output."""

    lines = [
        "Embedding extraction complete.",
        f"  Individuals processed: {result.n_individuals}",
        f"  Windows extracted: {result.n_windows_extracted}",
        f"  Windows dropped: {result.n_windows_dropped}",
        f"  Manifest: {result.manifest_path}",
    ]
    return "\n".join(lines)


__all__ = ["ExtractionResult", "format_extraction_result", "run_embedding_extraction"]
