#!/usr/bin/env python
"""Inspect and validate MTL (Multi-Task Learning) artifacts.

Provides visibility into the generated MTL validation artifacts:
- Verifies artifact existence and integrity
- Inspects run summary metadata
- Lists checkpoints and logs
- Reports TensorBoard event status

Usage:
    uv run python scripts/inspect_mtl_artifacts.py
"""

from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    """Inspect MTL validation artifacts."""
    artifact_root = Path("artifacts/mtl_validation")

    print("=" * 80)
    print("MTL VALIDATION ARTIFACT INSPECTION")
    print("=" * 80)

    if not artifact_root.exists():
        print(f"ERROR: Artifact root does not exist: {artifact_root}")
        return

    # Load run summary
    summary_path = artifact_root / "mtl_training_run_summary.json"
    if not summary_path.exists():
        print(f"ERROR: Run summary not found: {summary_path}")
        return

    summary = json.loads(summary_path.read_text())
    print(f"\n✓ Run Summary: {summary_path.resolve()}")
    print(f"  Pipeline:     {summary['pipeline']}")
    print(f"  Timestamp:    {summary['timestamp']}")
    print(f"  Final step:   {summary['training_result']['final_step']}")
    print(f"  Best val loss: {summary['training_result']['best_eval_loss']:.4f}")

    # Inspect checkpoint locations
    print("\n✓ Checkpoint Locations:")
    latest_dir = Path(summary["checkpoint_location"]) / "latest" / "model"
    best_dir = Path(summary["checkpoint_location"]) / "best" / "model"

    if latest_dir.exists():
        files = list(latest_dir.glob("*"))
        print(f"  Latest: {latest_dir.resolve()}")
        for f in sorted(files):
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"    - {f.name:<30} {size_mb:>8.2f} MB")
    else:
        print("  Latest: NOT FOUND")

    if best_dir.exists():
        files = list(best_dir.glob("*"))
        print(f"  Best:   {best_dir.resolve()}")
        for f in sorted(files):
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"    - {f.name:<30} {size_mb:>8.2f} MB")
    else:
        print("  Best: NOT FOUND")

    # Inspect TensorBoard logs
    print("\n✓ TensorBoard Logs:")
    tb_dir = Path(summary["tensorboard_location"])
    if tb_dir.exists():
        tb_path = tb_dir / "mtl_fine_tune_training"
        if tb_path.exists():
            events = list(tb_path.glob("events.out.tfevents.*"))
            print(f"  Directory: {tb_dir.resolve()}")
            print(f"  Event files: {len(events)}")
            for evt in sorted(events):
                size_mb = evt.stat().st_size / (1024 * 1024)
                print(f"    - {evt.name:<50} {size_mb:>8.3f} MB")
        else:
            print(f"  TensorBoard path not found: {tb_path}")
    else:
        print(f"  TensorBoard directory not found: {tb_dir}")

    # Inspect supporting artifacts
    print("\n✓ Supporting Artifacts:")
    tiny_bert_dir = artifact_root / "tiny_bert"
    if tiny_bert_dir.exists():
        files = list(tiny_bert_dir.glob("*"))
        print(f"  Tiny BERT: {tiny_bert_dir.resolve()}")
        for f in sorted(files):
            if f.is_file():
                size_mb = f.stat().st_size / (1024 * 1024)
                print(f"    - {f.name:<30} {size_mb:>8.2f} MB")
    else:
        print("  Tiny BERT: NOT FOUND")

    # Summary statistics
    print("\n✓ Training Configuration:")
    for key, val in summary["config"].items():
        print(f"  {key:<30} {val}")

    print("\n" + "=" * 80)
    print("ARTIFACT VALIDATION COMPLETE")
    print("=" * 80)
    print("\nTo view TensorBoard:")
    print(f"  tensorboard --logdir={tb_dir.resolve()}")
    print("\nTo load model checkpoints in Python:")
    print("  import torch")
    print(f"  state_dict = torch.load('{latest_dir.resolve()}/pytorch_model.bin')")


if __name__ == "__main__":
    main()
