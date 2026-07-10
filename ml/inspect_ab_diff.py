"""Visualize the raw block A vs block B discrepancy at the interface ring --
i.e. (true A) - (true B) per cell/field, with no trained model involved.

This is the actual physical quantity the CNN is trained to predict (as a
target delta): the flux-correction footprint left behind once B's embedded
high-fidelity inner box gets copied into A and the interface fluxes are
reconciled. It is NOT the model's prediction error -- see ml/test_model.py
and ml/metrics.py for that (predicted_A - true_A).

For each selected (case, step) sample, writes per field:
  <out-dir>/<case>_<step>_<field>.npy  -- (grid_size, grid_size) array, NaN
                                          outside the interface ring
  <out-dir>/<case>_<step>_<field>.png  -- heatmap, diverging colormap centered
                                          at 0, only over the interface ring

Usage:
    python3 -m ml.inspect_ab_diff --root plot/plot_result
    python3 -m ml.inspect_ab_diff --root plot --case case_0001 --step 000005
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .dataset import DualBlockInterfaceDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="plot/plot_result")
    ap.add_argument("--ring-width", type=int, default=3)
    ap.add_argument("--case", default=None, help="Restrict to one case_id (e.g. case_0001)")
    ap.add_argument("--step", default=None, help="Restrict to one step, zero-padded (e.g. 000005)")
    ap.add_argument("--out-dir", default="ml/ab_diff_maps")
    ap.add_argument("--no-plots", action="store_true", help="Skip PNG heatmaps, only save .npy arrays")
    args = ap.parse_args()

    # Restrict indexing/validation to the requested case up front -- otherwise
    # --case/--step would only filter *after* every (case, timestep) in the
    # whole root got fully read once for validation, which is what made this
    # take forever when --root pointed at a multi-case sweep directory.
    include_cases = {args.case} if args.case is not None else None

    # field_stats=None -> DualBlockInterfaceDataset's normalization is a no-op,
    # so "input"/"target" come back in raw physical units.
    ds = DualBlockInterfaceDataset(
        args.root, ring_width=args.ring_width, field_stats=None, include_cases=include_cases
    )
    ring_mask = ds.mask.numpy()  # (H, W) bool, shared across all samples

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.no_plots:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

    selected = 0
    for idx in range(len(ds)):
        (case_id, step), _files = ds.samples[idx]
        if args.case is not None and case_id != args.case:
            continue
        if args.step is not None and step != args.step:
            continue
        selected += 1

        sample = ds[idx]
        b_state = sample["input"].numpy()  # (C, H, W), raw physical units
        a_state = sample["target"].numpy()
        diff = a_state - b_state  # raw A - B discrepancy, per field

        label = f"{case_id or 'nocasedir'}_{step}"
        print(f"\n{label}:")
        for c, field in enumerate(ds.fields):
            ring_vals = diff[c][ring_mask]
            print(
                f"  {field}: min={ring_vals.min():.6g}  max={ring_vals.max():.6g}  "
                f"mean_abs={np.abs(ring_vals).mean():.6g}"
            )

            masked = np.where(ring_mask, diff[c], np.nan)
            np.save(out_dir / f"{label}_{field}.npy", masked)

            if not args.no_plots:
                vmax = np.abs(ring_vals).max() or 1.0
                fig, ax = plt.subplots()
                im = ax.imshow(masked.T, origin="lower", cmap="coolwarm", vmin=-vmax, vmax=vmax)
                ax.set_title(f"A - B: {field}\n{label}")
                ax.set_xlabel("i")
                ax.set_ylabel("j")
                fig.colorbar(im, ax=ax, label=field)
                fig.savefig(out_dir / f"{label}_{field}.png", dpi=150, bbox_inches="tight")
                plt.close(fig)

    if selected == 0:
        raise SystemExit(f"No samples matched --case={args.case} --step={args.step} under {args.root}")
    print(f"\nWrote arrays/plots for {selected} (case, step) sample(s) to {out_dir}/")


if __name__ == "__main__":
    main()
