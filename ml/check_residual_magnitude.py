"""Standalone diagnostic: how large is the true A-B residual over the
interface ring, directly from training data -- no model involved.

Answers the question "is the network predicting near-zero because of a
training/pipeline bug, or because the true residual really is tiny at the
current spectral_filter_width?" by computing the ground-truth target_delta
(= a_state - b_state, normalized, masked to the ring) that train.py's loss
actually regresses against, with no model in the loop at all.

The "predict zero baseline masked_mse" printed at the end is directly
comparable to the train_masked_mse train.py prints each epoch -- if a
trained model's loss lands close to that baseline, the model has learned
essentially nothing beyond "predict zero", which points at the residual
itself being too small to learn from (not a training bug).

Usage:
    python3 -m ml.check_residual_magnitude --root plot --ring-width 4
"""

from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from .dataset import DualBlockInterfaceDataset, compute_field_stats, list_cases
from .metrics import ResidualErrorAccumulator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="plot")
    ap.add_argument("--ring-width", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    cases = list_cases(args.root)
    print(f"Cases found under {args.root}: {len(cases)}")

    stats = compute_field_stats(args.root)
    print("Normalization stats (mean, std), pooled across all cases/fields:")
    for f, (mean, std) in stats.items():
        print(f"  {f}: mean={mean:.6g}  std={std:.6g}")

    ds = DualBlockInterfaceDataset(args.root, ring_width=args.ring_width, field_stats=stats)
    print(f"Samples: {len(ds)} (case, timestep) pairs\n")

    loader = DataLoader(ds, batch_size=min(args.batch_size, len(ds)), shuffle=False)

    res_acc = ResidualErrorAccumulator(stats, ds.fields)
    norm_abs_sum = {f: 0.0 for f in ds.fields}
    norm_sq_sum = {f: 0.0 for f in ds.fields}
    norm_max = {f: 0.0 for f in ds.fields}
    count = {f: 0 for f in ds.fields}
    pooled_sq_sum = 0.0
    pooled_count = 0

    for batch in loader:
        b_state, a_state, mask = batch["input"], batch["target"], batch["mask"]
        res_acc.update(b_state, a_state, mask)

        # Normalized units -- exactly the space train.py's loss operates in.
        target_delta = a_state - b_state
        for c, field in enumerate(ds.fields):
            diff = target_delta[:, c][mask]
            norm_abs_sum[field] += diff.abs().sum().item()
            norm_sq_sum[field] += (diff**2).sum().item()
            if diff.numel():
                norm_max[field] = max(norm_max[field], diff.abs().max().item())
            count[field] += diff.numel()

        # Pooled across all fields+cells, matching masked_mse's own flattening,
        # so this is directly comparable to train.py's printed train_masked_mse.
        pooled_diff = target_delta[mask.unsqueeze(1).expand_as(target_delta)]
        pooled_sq_sum += (pooled_diff**2).sum().item()
        pooled_count += pooled_diff.numel()

    print("True A-B residual over the interface ring, NO MODEL involved:\n")
    print("Normalized units (the space train.py's loss actually operates in):")
    for f in ds.fields:
        mae = norm_abs_sum[f] / count[f]
        rmse = (norm_sq_sum[f] / count[f]) ** 0.5
        print(f"  {f}: mae={mae:.6g}  rmse={rmse:.6g}  max_abs={norm_max[f]:.6g}")

    print("\nPhysical units (un-normalized, for intuition):")
    phys_mae = res_acc.mae()
    for f in ds.fields:
        print(f"  {f}: mae={phys_mae[f]:.6g}")

    zero_pred_masked_mse = pooled_sq_sum / pooled_count
    print(
        f"\n'Predict zero' baseline masked_mse: {zero_pred_masked_mse:.6f}"
        "\n(directly comparable to train.py's printed train_masked_mse -- "
        "if a trained model's loss is close to this, it learned essentially "
        "nothing beyond predicting zero)"
    )


if __name__ == "__main__":
    main()
