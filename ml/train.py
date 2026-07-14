"""Training loop, masked to the interface ring.

Uses the paired train/test holdout recorded in runs/manifest.csv by
generate_cases.py: every non-held-out case is trained on directly, with no
validation split. Held-out ICs are never seen during training at any filter
width (see cases_in_split/cases_by_width in ml/dataset.py), so the per-width
test metrics from ml/test_model.py -- averaged across the 5 width partitions
-- already give an honest held-out generalization measure; an in-loop
validation split would just shrink the training set for no added benefit.

Falls back to training on every discovered case (no held-out test set) if no
manifest is found under --manifest.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .dataset import DualBlockInterfaceDataset, cases_in_split, compute_field_stats, list_cases, load_manifest
from .metrics import FieldErrorAccumulator, format_errs, masked_mse
from .model import InterfaceCorrectionCNN

CHECKPOINT_PATH = "ml/interface_correction_cnn.pt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="plot")
    ap.add_argument("--manifest", default="runs/manifest.csv")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--ring-width", type=int, default=4)
    args = ap.parse_args()

    present = set(list_cases(args.root))
    manifest_path = Path(args.manifest)
    if manifest_path.exists():
        manifest = load_manifest(manifest_path)
        train_cases = [c for c in cases_in_split(manifest, "train") if c in present]
        test_cases = [c for c in cases_in_split(manifest, "test") if c in present]
        print(
            f"Cases present: {len(present)}  (manifest split) -> "
            f"train {len(train_cases)}, test held out {len(test_cases)}"
        )
    else:
        import warnings

        warnings.warn(
            f"No manifest at {manifest_path}; training on every case found under {args.root}, "
            "with no held-out test set."
        )
        train_cases = sorted(present)
        test_cases = []
        print(f"Cases found: {len(present)} -> train {train_cases}")

    stats = compute_field_stats(args.root, include_cases=train_cases)
    print("Normalization stats (mean, std), computed on train cases only:", stats)

    train_ds = DualBlockInterfaceDataset(
        args.root, ring_width=args.ring_width, field_stats=stats, include_cases=train_cases
    )
    print(f"Train set: {len(train_ds)} (case, timestep) samples")

    loader = DataLoader(train_ds, batch_size=min(8, len(train_ds)), shuffle=True)

    model = InterfaceCorrectionCNN()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    def run_epoch(loader, train: bool):
        model.train(train)
        total_loss = 0.0
        err_acc = FieldErrorAccumulator(stats, train_ds.fields)
        with torch.set_grad_enabled(train):
            for batch in loader:
                b_state, a_state, mask = batch["input"], batch["target"], batch["mask"]
                pred_delta = model(b_state)
                target_delta = a_state - b_state
                loss = masked_mse(pred_delta, target_delta, mask)

                if train:
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                total_loss += loss.item()
                with torch.no_grad():
                    err_acc.update(pred_delta.detach(), b_state, a_state, mask)
        return total_loss / len(loader), err_acc.rmse(), err_acc.mae()

    for epoch in range(args.epochs):
        train_loss, train_rmse, train_mae = run_epoch(loader, train=True)
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print(
                f"epoch {epoch:4d}  train_masked_mse={train_loss:.6f}  "
                f"train=({format_errs(train_rmse, train_mae)})"
            )

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "fields": train_ds.fields,
        "grid_size": train_ds.grid_size,
        "ring_width": args.ring_width,
        "field_stats": stats,
        "train_cases": train_cases,
        "test_cases": test_cases,
    }
    torch.save(checkpoint, CHECKPOINT_PATH)
    print(f"Saved checkpoint (weights + normalization + config) to {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
