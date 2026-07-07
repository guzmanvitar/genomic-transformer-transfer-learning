"""Variant Effect Scoring (VES) via masked DNABERT-2 prediction.

Computes per-SNP functional importance scores by exploiting the felid-
pretrained DNABERT-2's learned genomic context. For each locus, the center
nucleotide is masked and the model predicts probabilities for all possible
bases. The VES is:

    VES = log P(alt | context) - log P(ref | context)

Negative VES indicates the alternate allele is unexpected in the felid
genomic context (functionally constrained region). This provides a
label-free alternative to population-label-based SNP selection for geographic
assignment.
"""

from __future__ import annotations

import json
import logging
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import hf_hub_download
from torch import Tensor, nn
from transformers import AutoModelForMaskedLM, AutoTokenizer

from jaguar_geo_assign.data.finetune_windows import (
    UPSTREAM_BASES,
    WINDOW_SIZE,
    extract_fasta_window,
    load_reference_index,
)
from jaguar_geo_assign.data.pipeline_contract import (
    DNABERT2_TOKENIZER_ID,
    DNABERT2_TOKENIZER_REVISION,
)

logger = logging.getLogger(__name__)


def _ensure_custom_code(backbone_path: Path) -> None:
    """Ensure custom Python files referenced by ``auto_map`` exist in the model directory.

    Locally-saved checkpoints may be missing the custom ``.py`` files that
    DNABERT-2 needs when loaded with ``trust_remote_code=True``.  If any are
    absent we download them from the pinned Hub revision.
    """
    config_path = backbone_path / "config.json"
    if not config_path.exists():
        return

    with open(config_path) as f:
        cfg = json.load(f)

    auto_map = cfg.get("auto_map", {})
    if not auto_map:
        return

    py_files = {v.split(".")[0] + ".py" for v in auto_map.values()}
    missing = [f for f in py_files if not (backbone_path / f).exists()]
    if not missing:
        return

    for filename in missing:
        try:
            cached = hf_hub_download(
                DNABERT2_TOKENIZER_ID,
                filename,
                revision=DNABERT2_TOKENIZER_REVISION,
            )
            shutil.copy2(cached, backbone_path / filename)
            logger.info("Downloaded missing custom code file: %s", filename)
        except Exception:
            logger.warning("Could not download %s from Hub", filename, exc_info=True)

    for extra in ("bert_padding.py", "flash_attn_triton.py"):
        if (backbone_path / extra).exists():
            continue
        try:
            cached = hf_hub_download(
                DNABERT2_TOKENIZER_ID,
                extra,
                revision=DNABERT2_TOKENIZER_REVISION,
            )
            shutil.copy2(cached, backbone_path / extra)
            logger.info("Downloaded missing dependency file: %s", extra)
        except Exception:
            pass


def _load_tokenizer(backbone_path: Path, backbone: nn.Module | None = None) -> Any:
    """Load a DNABERT-2-compatible tokenizer from the local backbone directory.

    To guarantee that ``pad_token_id`` is defined, a three-step fallback is
    applied if the pretrained tokenizer is missing a pad token:

    1. Reuse ``eos_token`` as pad when present.
    2. Otherwise reuse ``unk_token``.
    3. Otherwise add a new ``[PAD]`` token.  When a backbone is supplied, its
       embedding matrix is resized so that model and tokenizer remain aligned.
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
    in newer Triton builds. This defensive patch ensures the backbone falls back
    to PyTorch's native attention implementation.
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


@dataclass(frozen=True)
class VESResult:
    """Output of variant effect scoring for a panel of SNP loci.

    Attributes:
        scores: Float32 tensor of shape ``(n_loci,)``, one VES per SNP.
            Positive values indicate the alternate allele is *more likely*
            than the reference in the learned genomic context; negative values
            indicate functional constraint.
        n_scored: Number of loci that received a computed VES.
        n_failed: Number of loci where VES could not be computed (assigned 0.0).
        score_mean: Arithmetic mean of all VES values (including 0.0 placeholders).
        score_std: Population standard deviation of all VES values.
        score_min: Minimum VES value across all loci.
        score_max: Maximum VES value across all loci.
    """

    scores: Tensor  # Float32[n_loci]
    n_scored: int
    n_failed: int
    score_mean: float
    score_std: float
    score_min: float
    score_max: float


def _resolve_device(device_name: str) -> torch.device:
    """Resolve the runtime device from a string specification.

    Args:
        device_name: One of ``"auto"``, ``"cuda"``, or ``"cpu"``.

    Returns:
        Resolved :class:`torch.device`.

    Raises:
        RuntimeError: If ``"cuda"`` is requested but unavailable.
        ValueError: If *device_name* is not a recognized option.
    """
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("device='cuda' was requested but CUDA is unavailable")
        return torch.device("cuda")
    if device_name == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unsupported device: {device_name!r}")


def _find_center_token_index(
    offset_mapping: list[tuple[int, int]],
    center_char_idx: int,
) -> int | None:
    """Find the token index whose character span covers the center position.

    The offset mapping produced by ``tokenizer(..., return_offsets_mapping=True)``
    gives ``(start, end)`` character ranges for each token. Special tokens
    (CLS, SEP, PAD) typically have ``(0, 0)``; we skip those.

    Args:
        offset_mapping: List of ``(start_char, end_char)`` tuples per token.
        center_char_idx: 0-based character index of the center nucleotide.

    Returns:
        Token index (into the tokenized sequence) covering
        ``center_char_idx``, or ``None`` if no token spans it.
    """
    for token_idx, (start, end) in enumerate(offset_mapping):
        if start == end:
            # Skip special tokens with zero-width spans.
            continue
        if start <= center_char_idx < end:
            return token_idx
    return None


def compute_variant_effect_scores(
    locus_info: list,  # list[LocusInfo] — imported at runtime to avoid circular dep
    reference_fasta: str | Path,
    backbone_path: str | Path,
    *,
    batch_size: int = 128,
    device: str = "auto",
) -> VESResult:
    """Compute VES for each locus using masked prediction.

    The algorithm proceeds as follows:

    1. Load reference FASTA (reuse ``load_reference_index``).
    2. Load and freeze DNABERT-2 backbone in MLM mode.
    3. For each locus in batches:

       a. Extract a 512bp window with the reference allele at center.
       b. Tokenize the window with offset mapping.
       c. Find the token covering the center position (character index 256).
       d. Replace that token with the mask token ID.
       e. Forward pass through frozen backbone to obtain MLM logits.
       f. Compute ``log_softmax`` at the masked position.
       g. Map ref and alt alleles to their token IDs.
       h. ``VES = log_prob[alt_token] - log_prob[ref_token]``.

    4. For loci where VES cannot be computed (boundary effects,
       tokenization issues): assign ``VES = 0.0`` and increment the
       ``n_failed`` counter.

    Args:
        locus_info: List of ``LocusInfo`` instances, each describing one
            SNP with ``contig``, ``pos``, ``ref``, ``alt``, and ``locus_key``
            fields.
        reference_fasta: Path to the reference FASTA used for window
            extraction.
        backbone_path: Path to the felid-pretrained DNABERT-2 checkpoint
            directory.
        batch_size: Number of loci to process per forward pass. All
            windows are 512bp so tokenized lengths are similar.
        device: Compute device — ``"auto"``, ``"cuda"``, or ``"cpu"``.

    Returns:
        :class:`VESResult` containing the per-locus score tensor and
        summary statistics.

    Raises:
        RuntimeError: If the backbone does not expose a mask token or the
            model cannot be loaded.
    """
    resolved_device = _resolve_device(device)
    backbone_path = Path(backbone_path)

    # Load reference.
    logger.info("Loading reference FASTA from %s", reference_fasta)
    ref_index = load_reference_index(reference_fasta)

    # Load and freeze DNABERT-2 in MLM mode.
    logger.info("Loading DNABERT-2 MLM backbone from %s", backbone_path)
    _ensure_custom_code(backbone_path)
    model = AutoModelForMaskedLM.from_pretrained(str(backbone_path), trust_remote_code=True)
    _disable_flash_attention_if_present()
    tokenizer = _load_tokenizer(backbone_path, backbone=model)

    if tokenizer.mask_token_id is None:
        raise RuntimeError(
            f"Tokenizer loaded from {backbone_path} does not define a mask token; "
            "variant effect scoring requires a masked language model tokenizer."
        )

    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    model = model.to(resolved_device)

    n_loci = len(locus_info)
    scores = torch.zeros(n_loci, dtype=torch.float32)
    n_scored = 0
    n_failed = 0

    logger.info(
        "Scoring %d loci (batch_size=%d, device=%s)",
        n_loci,
        batch_size,
        resolved_device,
    )

    with torch.inference_mode():
        for batch_start in range(0, n_loci, batch_size):
            batch_end = min(batch_start + batch_size, n_loci)
            batch_loci = locus_info[batch_start:batch_end]

            # Prepare per-locus data for this batch.
            batch_sequences: list[str] = []
            batch_ref_alleles: list[str] = []
            batch_alt_alleles: list[str] = []
            batch_valid_mask: list[bool] = []

            for locus in batch_loci:
                contig = locus.contig
                pos = locus.pos
                ref_allele = locus.ref
                alt_allele = locus.alt

                if contig not in ref_index.contig_sequences:
                    logger.debug(
                        "Contig %r not found in reference for locus %s; assigning VES=0.0",
                        contig,
                        locus.locus_key,
                    )
                    batch_sequences.append("")
                    batch_ref_alleles.append(ref_allele)
                    batch_alt_alleles.append(alt_allele)
                    batch_valid_mask.append(False)
                    continue

                result = extract_fasta_window(
                    contig_sequence=ref_index.contig_sequences[contig],
                    locus_pos=pos,
                    allele=ref_allele,
                )
                if result is None:
                    logger.debug(
                        "Window extraction returned None for locus %s (boundary); "
                        "assigning VES=0.0",
                        locus.locus_key,
                    )
                    batch_sequences.append("")
                    batch_ref_alleles.append(ref_allele)
                    batch_alt_alleles.append(alt_allele)
                    batch_valid_mask.append(False)
                    continue

                sequence, *_ = result
                batch_sequences.append(sequence)
                batch_ref_alleles.append(ref_allele)
                batch_alt_alleles.append(alt_allele)
                batch_valid_mask.append(True)

            # Collect valid indices for batched forward pass.
            valid_indices = [i for i, valid in enumerate(batch_valid_mask) if valid]

            if not valid_indices:
                n_failed += len(batch_loci)
                continue

            # Tokenize all valid sequences together for batched inference.
            valid_sequences = [batch_sequences[i] for i in valid_indices]
            encoded = tokenizer(
                valid_sequences,
                return_tensors="pt",
                return_offsets_mapping=True,
                padding=True,
                truncation=True,
                max_length=WINDOW_SIZE,
            )
            offset_mappings = encoded.pop("offset_mapping")  # (n_valid, seq_len, 2)

            # Find center token index for each valid sequence and prepare
            # masked input_ids.
            center_char_idx = UPSTREAM_BASES  # 256, 0-indexed center of 512bp window
            input_ids = encoded["input_ids"].clone()
            mask_token_id = tokenizer.mask_token_id

            per_sequence_center_idx: list[int | None] = []
            for seq_idx in range(len(valid_indices)):
                offsets = offset_mappings[seq_idx].tolist()
                center_idx = _find_center_token_index(offsets, center_char_idx)
                per_sequence_center_idx.append(center_idx)
                if center_idx is not None:
                    input_ids[seq_idx, center_idx] = mask_token_id

            # Move tensors to device.
            model_inputs: dict[str, Tensor] = {}
            for key, value in encoded.items():
                if isinstance(value, Tensor):
                    model_inputs[key] = value.to(resolved_device)
            model_inputs["input_ids"] = input_ids.to(resolved_device)

            # Forward pass.
            outputs = model(**model_inputs)
            logits = outputs.logits  # (n_valid, seq_len, vocab_size)

            # Extract VES for each valid sequence.
            for local_idx, global_batch_idx in enumerate(valid_indices):
                abs_idx = batch_start + global_batch_idx
                center_idx = per_sequence_center_idx[local_idx]
                ref_allele = batch_ref_alleles[global_batch_idx]
                alt_allele = batch_alt_alleles[global_batch_idx]

                if center_idx is None:
                    logger.debug(
                        "Could not locate center token for locus %s; assigning VES=0.0",
                        locus_info[abs_idx].locus_key,
                    )
                    n_failed += 1
                    continue

                # Map alleles to token IDs.
                ref_token_ids = tokenizer.encode(ref_allele, add_special_tokens=False)
                alt_token_ids = tokenizer.encode(alt_allele, add_special_tokens=False)
                if not ref_token_ids or not alt_token_ids:
                    logger.debug(
                        "Could not encode alleles ref=%r alt=%r for locus %s; assigning VES=0.0",
                        ref_allele,
                        alt_allele,
                        locus_info[abs_idx].locus_key,
                    )
                    n_failed += 1
                    continue

                ref_token_id = ref_token_ids[0]
                alt_token_id = alt_token_ids[0]

                masked_logits = logits[local_idx, center_idx, :]
                log_probs = torch.log_softmax(masked_logits.float(), dim=-1)

                ves = log_probs[alt_token_id].item() - log_probs[ref_token_id].item()
                scores[abs_idx] = ves
                n_scored += 1

            # Count failures from invalid sequences in this batch.
            n_invalid_in_batch = len(batch_loci) - len(valid_indices)
            n_failed += n_invalid_in_batch

            if batch_end % (batch_size * 10) == 0 or batch_end == n_loci:
                logger.info(
                    "VES progress: %d / %d loci scored (%d failed so far)",
                    n_scored,
                    n_loci,
                    n_failed,
                )

    # Compute summary statistics.
    score_mean = float(scores.mean().item())
    score_std = float(scores.std(correction=0).item())
    score_min = float(scores.min().item())
    score_max = float(scores.max().item())

    logger.info(
        "VES computation complete: %d scored, %d failed, mean=%.4f, std=%.4f, min=%.4f, max=%.4f",
        n_scored,
        n_failed,
        score_mean,
        score_std,
        score_min,
        score_max,
    )

    return VESResult(
        scores=scores,
        n_scored=n_scored,
        n_failed=n_failed,
        score_mean=score_mean,
        score_std=score_std,
        score_min=score_min,
        score_max=score_max,
    )


def save_ves_scores(result: VESResult, output_path: str | Path) -> None:
    """Save VES scores tensor and metadata to disk.

    Creates two files at the given path:

    * ``{output_path}.pt`` — the raw score tensor.
    * ``{output_path}.json`` — summary metadata (counts, statistics).

    Args:
        result: The :class:`VESResult` to persist.
        output_path: Base path (without extension). Both ``.pt`` and
            ``.json`` suffixes are appended automatically.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tensor_path = output_path.with_suffix(".pt")
    torch.save(result.scores, tensor_path)

    metadata = {
        "n_loci": int(result.scores.shape[0]),
        "n_scored": result.n_scored,
        "n_failed": result.n_failed,
        "score_mean": result.score_mean,
        "score_std": result.score_std,
        "score_min": result.score_min,
        "score_max": result.score_max,
    }
    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    logger.info(
        "Saved VES scores: tensor=%s, metadata=%s",
        tensor_path,
        metadata_path,
    )


def load_ves_scores(output_path: str | Path) -> VESResult:
    """Load previously computed VES scores from disk.

    Reads the tensor and metadata files created by :func:`save_ves_scores`
    and reconstructs a :class:`VESResult`.

    Args:
        output_path: Base path (without extension) used when saving.
            Expects ``{output_path}.pt`` and ``{output_path}.json`` to exist.

    Returns:
        Reconstructed :class:`VESResult`.

    Raises:
        FileNotFoundError: If either the ``.pt`` or ``.json`` file is missing.
    """
    output_path = Path(output_path)

    tensor_path = output_path.with_suffix(".pt")
    if not tensor_path.exists():
        raise FileNotFoundError(
            f"VES score tensor not found at {tensor_path}; "
            "expected a .pt file created by save_ves_scores()."
        )

    metadata_path = output_path.with_suffix(".json")
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"VES metadata not found at {metadata_path}; "
            "expected a .json file created by save_ves_scores()."
        )

    scores = torch.load(tensor_path, map_location="cpu", weights_only=True)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    return VESResult(
        scores=scores,
        n_scored=int(metadata["n_scored"]),
        n_failed=int(metadata["n_failed"]),
        score_mean=float(metadata["score_mean"]),
        score_std=float(metadata["score_std"]),
        score_min=float(metadata["score_min"]),
        score_max=float(metadata["score_max"]),
    )


__all__ = [
    "VESResult",
    "compute_variant_effect_scores",
    "load_ves_scores",
    "save_ves_scores",
]
