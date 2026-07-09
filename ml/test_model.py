"""Evaluate a trained InterfaceCorrectionCNN checkpoint on held-out CGNS data,
reporting the same masked-MSE / per-field RMSE+MAE statistics train.py prints.

Loads fields, grid_size, ring_width, and normalization stats from the
checkpoint (saved by train.py) rather than recomputing them here -- a test
set must be normalized with the *training* distribution's mean/std, not its
own, or the numbers aren't comparable to what the model was trained against.

Usage:
    python3 -m ml.test_model --root plot/plot_result
"""

from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from .dataset import DualBlockInterfaceDataset, list_cases
from .metrics import FieldErrorAccumulator, format_errs, masked_mse
from .model import InterfaceCorrectionCNN
from .train import CHECKPOINT_PATH


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=CHECKPOINT_PATH)
    ap.add_argument("--root", default="plot/plot_result")
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    fields = checkpoint["fields"]
    grid_size = checkpoint["grid_size"]
    ring_width = checkpoint["ring_width"]
    field_stats = checkpoint["field_stats"]

    print(f"Loaded checkpoint from {args.checkpoint}")
    print(f"  fields={fields}  grid_size={grid_size}  ring_width={ring_width}")
    print(f"  trained on cases: {checkpoint.get('train_cases')}")
    print(f"  normalization stats (mean, std): {field_stats}")

    test_ds = DualBlockInterfaceDataset(
        args.root,
        fields=fields,
        grid_size=grid_size,
        ring_width=ring_width,
        field_stats=field_stats,
    )
    print(f"Test set: {len(test_ds)} (case, timestep) samples across cases {list_cases(args.root)}")

    loader = DataLoader(test_ds, batch_size=min(args.batch_size, len(test_ds)), shuffle=False)

    model = InterfaceCorrectionCNN(n_fields=len(fields))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    total_loss = 0.0
    err_acc = FieldErrorAccumulator(field_stats, fields)
    with torch.no_grad():
        for batch in loader:
            b_state, a_state, mask = batch["input"], batch["target"], batch["mask"]
            pred_delta = model(b_state)
            target_delta = a_state - b_state
            loss = masked_mse(pred_delta, target_delta, mask)
            total_loss += loss.item()
            err_acc.update(pred_delta, b_state, a_state, mask)

    rmse, mae = err_acc.rmse(), err_acc.mae()
    print(f"test_masked_mse={total_loss / len(loader):.6f}")
    print(f"test=({format_errs(rmse, mae)})")


if __name__ == "__main__":
    main()
