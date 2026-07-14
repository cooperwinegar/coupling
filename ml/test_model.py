"""Evaluate a trained InterfaceCorrectionCNN checkpoint on the held-out test
cases, reporting per-field RMSE/MAE broken down by spectral_filter_width.

The held-out cases are the paired IC holdout recorded by generate_cases.py
(split == "test" in the manifest) and saved into the checkpoint by train.py.
Because the ICs are shared across widths, these test ICs were never seen in
training at any width. Grouping the metrics by the case's spectral_filter_width
shows how correction accuracy changes as the filter gets wider.

Normalization uses the *training* distribution's stats (from the checkpoint),
not the test set's own -- otherwise the numbers aren't comparable to training.

Usage:
    python3 -m ml.test_model --root plot --manifest runs/manifest.csv
"""

from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from .dataset import DualBlockInterfaceDataset, cases_by_width, cases_in_split, list_cases, load_manifest
from .metrics import FieldErrorAccumulator, format_errs, masked_mse
from .model import InterfaceCorrectionCNN
from .train import CHECKPOINT_PATH


def evaluate(model, dataset, fields, field_stats, batch_size):
    """-> (masked_mse, {field: rmse}, {field: mae}) over the whole dataset."""
    loader = DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=False)
    total_loss = 0.0
    err_acc = FieldErrorAccumulator(field_stats, fields)
    with torch.no_grad():
        for batch in loader:
            b_state, a_state, mask = batch["input"], batch["target"], batch["mask"]
            pred_delta = model(b_state)
            total_loss += masked_mse(pred_delta, a_state - b_state, mask).item()
            err_acc.update(pred_delta, b_state, a_state, mask)
    return total_loss / len(loader), err_acc.rmse(), err_acc.mae()


def _build(root, cases, fields, grid_size, ring_width, field_stats):
    return DualBlockInterfaceDataset(
        root,
        fields=fields,
        grid_size=grid_size,
        ring_width=ring_width,
        field_stats=field_stats,
        include_cases=cases,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=CHECKPOINT_PATH)
    ap.add_argument("--root", default="plot")
    ap.add_argument("--manifest", default="runs/manifest.csv")
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    fields = checkpoint["fields"]
    grid_size = checkpoint["grid_size"]
    ring_width = checkpoint["ring_width"]
    field_stats = checkpoint["field_stats"]

    print(f"Loaded checkpoint from {args.checkpoint}")
    print(f"  fields={fields}  grid_size={grid_size}  ring_width={ring_width}")

    model = InterfaceCorrectionCNN(n_fields=len(fields))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    manifest = load_manifest(args.manifest)
    present = set(list_cases(args.root))
    # Prefer the exact held-out set saved in the checkpoint; fall back to the
    # manifest's test split if an older checkpoint didn't record it.
    test_cases = checkpoint.get("test_cases") or cases_in_split(manifest, "test")
    test_cases = [c for c in test_cases if c in present]
    if not test_cases:
        raise SystemExit(f"No held-out test cases from the manifest are present under {args.root}")
    by_width = cases_by_width(manifest, test_cases)
    print(f"Held-out test cases: {len(test_cases)} across filter widths {sorted(by_width)}\n")

    for width in sorted(by_width):
        cases_w = by_width[width]
        try:
            ds = _build(args.root, cases_w, fields, grid_size, ring_width, field_stats)
        except FileNotFoundError as e:
            print(f"[filter width {width}] no readable samples ({len(cases_w)} cases): {e}")
            continue
        loss, rmse, mae = evaluate(model, ds, fields, field_stats, args.batch_size)
        print(f"[filter width {width}] {len(ds)} samples over {len(cases_w)} cases  masked_mse={loss:.6f}")
        print(f"    {format_errs(rmse, mae)}")

    ds_all = _build(args.root, test_cases, fields, grid_size, ring_width, field_stats)
    loss, rmse, mae = evaluate(model, ds_all, fields, field_stats, args.batch_size)
    print(f"\n[all widths] {len(ds_all)} samples  masked_mse={loss:.6f}")
    print(f"    {format_errs(rmse, mae)}")


if __name__ == "__main__":
    main()
