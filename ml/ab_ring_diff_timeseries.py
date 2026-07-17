"""Per-timestep A-B discrepancy at the interface ring, averaged across cases.

For every (case, timestep) pair under --root, computes the raw block A minus
block B difference over the interface-ring cells, reduces it to a per-channel
mean absolute (and signed mean) value, then averages across all cases sharing
the same timestep. Prints one row per timestep per channel plus an overall
average across every timestep, and optionally writes the same numbers to CSV.

This is the raw physical discrepancy (no trained model involved) -- the same
quantity ml/inspect_ab_diff.py pools into a single aggregate, but broken down
as a time series so you can see how the correction magnitude evolves over a
trajectory.

Usage:
    python3 -m ml.ab_ring_diff_timeseries --root plot/plot_result
    python3 -m ml.ab_ring_diff_timeseries --root plot --csv ring_diff.csv
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict

import numpy as np

from .dataset import DualBlockInterfaceDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="plot/plot_result", help="directory of A/B CGNS trajectories")
    ap.add_argument("--ring-width", type=int, default=4)
    ap.add_argument("--case", default=None, help="Restrict to one case_id (e.g. case_0001)")
    ap.add_argument("--csv", default=None, help="Also write the per-step table to this CSV path")
    args = ap.parse_args()

    include_cases = {args.case} if args.case is not None else None

    # field_stats=None -> normalization is a no-op, values stay in physical units.
    ds = DualBlockInterfaceDataset(
        args.root, ring_width=args.ring_width, field_stats=None, include_cases=include_cases
    )
    ring_mask = ds.mask.numpy()

    # {step: {field: [per-case mean over ring cells, ...]}} -- each sample
    # contributes one scalar per field, so the across-case average at a step
    # weights every case equally (cases that blew up simply stop contributing
    # at later steps rather than dragging the average).
    abs_by_step: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    signed_by_step: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for idx in range(len(ds)):
        (_case_id, step), _files = ds.samples[idx]
        sample = ds[idx]
        diff = sample["target"].numpy() - sample["input"].numpy()  # A - B, (C, H, W)
        for c, field in enumerate(ds.fields):
            ring_vals = diff[c][ring_mask]
            abs_by_step[step][field].append(float(np.abs(ring_vals).mean()))
            signed_by_step[step][field].append(float(ring_vals.mean()))

    steps = sorted(abs_by_step)
    rows = []
    for step in steps:
        n_cases = len(abs_by_step[step][ds.fields[0]])
        for field in ds.fields:
            rows.append({
                "step": step,
                "field": field,
                "n_cases": n_cases,
                "mean_abs": float(np.mean(abs_by_step[step][field])),
                "mean": float(np.mean(signed_by_step[step][field])),
            })

    print(f"Per-timestep A-B ring discrepancy averaged across cases (ring width {args.ring_width}):")
    print(f"{'step':>8}  {'field':<20} {'n_cases':>7}  {'mean_abs':>12}  {'mean':>12}")
    for r in rows:
        print(
            f"{r['step']:>8}  {r['field']:<20} {r['n_cases']:>7}  "
            f"{r['mean_abs']:>12.6g}  {r['mean']:>12.6g}"
        )

    print(f"\nOverall average across all {len(steps)} timesteps:")
    for field in ds.fields:
        # Average of the per-step averages -- each timestep weighted equally,
        # matching how the per-width results tables aggregate.
        overall_abs = float(np.mean([np.mean(abs_by_step[s][field]) for s in steps]))
        overall_signed = float(np.mean([np.mean(signed_by_step[s][field]) for s in steps]))
        print(f"  {field:<20} mean_abs={overall_abs:.6g}  mean={overall_signed:.6g}")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["step", "field", "n_cases", "mean_abs", "mean"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote {len(rows)} rows to {args.csv}")


if __name__ == "__main__":
    main()
