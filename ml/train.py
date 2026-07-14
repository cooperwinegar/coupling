"""Training loop, masked to the interface ring.

Splits by whole case (initial-condition sweep run), not by timestep -- the
thing we actually want to generalize to is unseen initial conditions, so
holding out timesteps from an already-trained-on case wouldn't test that.
With only one case's worth of data (no case_NNNN dirs under plot/ yet),
there's nothing to hold out, so this falls back to training on everything as
a pipeline smoke test.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import (
    DualBlockInterfaceDataset,
    cases_by_width,
    cases_in_split,
    compute_field_stats,
    list_cases,
    load_manifest,
    split_cases,
)
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
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    present = set(list_cases(args.root))
    manifest_path = Path(args.manifest)
    if manifest_path.exists():
        # Manifest-driven paired holdout: the test cases (held-out ICs at every
        # filter width) never enter training/validation. Validation is carved
        # from the training pool, stratified by filter width so every width is
        # represented in the val metrics.
        manifest = load_manifest(manifest_path)
        train_pool = [c for c in cases_in_split(manifest, "train") if c in present]
        test_cases = [c for c in cases_in_split(manifest, "test") if c in present]
        rng = np.random.default_rng(args.seed)
        train_cases: list[str] = []
        val_cases: list[str] = []
        for _w, cs in cases_by_width(manifest, train_pool).items():
            cs = list(cs)
            rng.shuffle(cs)
            n_val = max(1, round(len(cs) * args.val_fraction)) if len(cs) > 1 else 0
            val_cases += cs[:n_val]
            train_cases += cs[n_val:]
        train_cases.sort()
        val_cases.sort()
        print(
            f"Cases present: {len(present)}  (manifest split) -> "
            f"train {len(train_cases)}, val {len(val_cases)}, test held out {len(test_cases)}"
        )
    else:
        import warnings

        warnings.warn(
            f"No manifest at {manifest_path}; falling back to a random whole-case split "
            "with no filter-width-aware test set."
        )
        train_cases, val_cases = split_cases(args.root, args.val_fraction, args.seed)
        test_cases = []
        print(f"Cases found: {len(present)} -> train {train_cases}, val {val_cases}")

    stats = compute_field_stats(args.root, include_cases=train_cases)
    print("Normalization stats (mean, std), computed on train cases only:", stats)

    train_ds = DualBlockInterfaceDataset(
        args.root, ring_width=args.ring_width, field_stats=stats, include_cases=train_cases
    )
    print(f"Train set: {len(train_ds)} (case, timestep) samples")
    if not val_cases:
        print(
            "WARNING: no validation cases -- this run only verifies the pipeline "
            "runs end-to-end and cannot demonstrate generalization. Run "
            "ml/generate_cases.py + ml/run_sweep.py to add more cases."
        )
        val_ds = None
    else:
        val_ds = DualBlockInterfaceDataset(
            args.root, ring_width=args.ring_width, field_stats=stats, include_cases=val_cases
        )
        print(f"Val set: {len(val_ds)} (case, timestep) samples")

    loader = DataLoader(train_ds, batch_size=min(8, len(train_ds)), shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=min(8, len(val_ds)), shuffle=False) if val_ds else None

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
        val_msg = ""
        if val_loader is not None:
            val_loss, val_rmse, val_mae = run_epoch(val_loader, train=False)
            val_msg = f"  val_masked_mse={val_loss:.6f}  val=({format_errs(val_rmse, val_mae)})"
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print(
                f"epoch {epoch:4d}  train_masked_mse={train_loss:.6f}  "
                f"train=({format_errs(train_rmse, train_mae)}){val_msg}"
            )

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "fields": train_ds.fields,
        "grid_size": train_ds.grid_size,
        "ring_width": args.ring_width,
        "field_stats": stats,
        "train_cases": train_cases,
        "val_cases": val_cases,
        "test_cases": test_cases,
    }
    torch.save(checkpoint, CHECKPOINT_PATH)
    print(f"Saved checkpoint (weights + normalization + config) to {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
