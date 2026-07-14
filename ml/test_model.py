"""Evaluate a trained InterfaceCorrectionCNN checkpoint on the held-out test
cases, reporting per-field normalized RMSE and absolute (physical) MAE,
broken down by spectral_filter_width.

The held-out cases are the paired IC holdout recorded by generate_cases.py
(split == "test" in the manifest) and saved into the checkpoint by train.py.
Because the ICs are shared across widths, these test ICs were never seen in
training at any width. Grouping the metrics by the case's spectral_filter_width
shows how correction accuracy changes as the filter gets wider.

Normalized RMSE is computed directly on pred_delta vs target_delta (the
model's own training space, mean-0/std-1 per field), so it's comparable
across channels of very different physical magnitude. Absolute MAE uses the
*training* distribution's stats (from the checkpoint) to unnormalize back to
physical units -- not the test set's own stats, or the numbers wouldn't be
comparable to training.

Usage:
    python3 -m ml.test_model --root plot --manifest runs/manifest.csv
"""

from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from .dataset import DualBlockInterfaceDataset, cases_by_width, cases_in_split, list_cases, load_manifest
from .metrics import FieldErrorAccumulator, NormalizedFieldErrorAccumulator, masked_mse
from .model import InterfaceCorrectionCNN
from .train import CHECKPOINT_PATH


def _format_field_dict(d: dict[str, float]) -> str:
    return ", ".join(f"{f}={v:.4g}" for f, v in d.items())


def evaluate(model, dataset, fields, field_stats, batch_size):
    """-> (masked_mse, {field: mae}, {field: norm_rmse}) over the whole dataset.
    mae is physical/absolute units; norm_rmse is RMSE computed in normalized
    (mean-0/std-1 per field) units -- the space the model is actually trained
    in, and comparable across channels of very different physical magnitude
    (unlike physical-unit RMSE)."""
    loader = DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=False)
    total_loss = 0.0
    err_acc = FieldErrorAccumulator(field_stats, fields)
    norm_acc = NormalizedFieldErrorAccumulator(fields)
    with torch.no_grad():
        for batch in loader:
            b_state, a_state, mask = batch["input"], batch["target"], batch["mask"]
            pred_delta = model(b_state)
            target_delta = a_state - b_state
            total_loss += masked_mse(pred_delta, target_delta, mask).item()
            err_acc.update(pred_delta, b_state, a_state, mask)
            norm_acc.update(pred_delta, target_delta, mask)
    return total_loss / len(loader), err_acc.mae(), norm_acc.rmse()


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

    per_width = []  # [(loss, mae, norm_rmse), ...] -- one entry per filter width partition
    for width in sorted(by_width):
        cases_w = by_width[width]
        try:
            ds = _build(args.root, cases_w, fields, grid_size, ring_width, field_stats)
        except FileNotFoundError as e:
            print(f"[filter width {width}] no readable samples ({len(cases_w)} cases): {e}")
            continue
        loss, mae, norm_rmse = evaluate(model, ds, fields, field_stats, args.batch_size)
        print(f"[filter width {width}] {len(ds)} samples over {len(cases_w)} cases  masked_mse={loss:.6f}")
        print(f"    normalized_rmse: {_format_field_dict(norm_rmse)}")
        print(f"    absolute_mae:    {_format_field_dict(mae)}")
        per_width.append((loss, mae, norm_rmse))

    # Averaged across the width partitions (not sample-pooled): each width
    # contributes one number regardless of its case count, since widths don't
    # have exactly equal case counts (e.g. width 5 has 9 held-out cases vs 10
    # for the others) and pooling would silently overweight the larger ones.
    n = len(per_width)
    avg_loss = sum(l for l, _, _ in per_width) / n
    avg_mae = {f: sum(m[f] for _, m, _ in per_width) / n for f in fields}
    avg_norm_rmse = {f: sum(nr[f] for _, _, nr in per_width) / n for f in fields}
    print(f"\n[average across {n} width partitions] masked_mse={avg_loss:.6f}")
    print(f"    normalized_rmse: {_format_field_dict(avg_norm_rmse)}")
    print(f"    absolute_mae:    {_format_field_dict(avg_mae)}")


if __name__ == "__main__":
    main()
