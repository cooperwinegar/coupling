"""Generate real-data test vectors for test_ml_kernel.cu, using the ACTUAL
trained checkpoint and a real (block B pre-stage, block A) sample from the
training set -- unlike generate_kernel_test_vectors.py (random weights,
random input, a pure translation-correctness check), this validates whether
the CUDA kernel chain reproduces the real trained model's real predictions,
to isolate a deployment-side bug (weight loading / kernel execution) from
anything else in how DualBlocks/main.cu wires the correction into the
physics loop.

Writes the same three files test_ml_kernel.cu already expects, so no C++
changes are needed -- just point --vectors-dir at a new output directory:
  weights.bin          -- from the real checkpoint, via export_weights.py's
                          own export_checkpoint_to_binary (same function
                          ml/export_weights.py itself calls, so this is
                          exactly what main.cu's initML() would load)
  synthetic_input.bin   -- one real block B pre-stage sample, physical units
  expected_output.bin   -- predict_state() applied to that real sample via
                          the actual PyTorch model, physical units

Also prints the model's own prediction error against the real ground-truth
A at this sample, for a quick sanity read before even touching the GPU.

Usage:
    python3 -m ml.generate_real_kernel_test_vectors \
        --checkpoint ml/interface_correction_cnn.pt \
        --root plot --case case_0000 --step 000010 \
        --out-dir /path/on/gpu/machine
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from .cgns_io import GRID_SIZE, inner_box_mask, interface_ring_mask, read_block_stacked
from .export_weights import export_checkpoint_to_binary
from .model import InterfaceCorrectionCNN


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="ml/interface_correction_cnn.pt")
    ap.add_argument("--root", default="plot")
    ap.add_argument("--case", required=True, help='e.g. "case_0000"')
    ap.add_argument("--step", required=True, help='zero-padded step, e.g. "000010"')
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    fields = checkpoint["fields"]
    field_stats = checkpoint["field_stats"]

    model = InterfaceCorrectionCNN(n_fields=len(fields))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    export_checkpoint_to_binary(checkpoint, os.path.join(args.out_dir, "weights.bin"))

    case_dir = os.path.join(args.root, args.case)
    b_grid = os.path.join(case_dir, f"blockB_grid_2d_{args.step}.cgns")
    b_soln = os.path.join(case_dir, f"blockB_precopy_2d_{args.step}.cgns")
    a_grid = os.path.join(case_dir, f"blockA_grid_2d_{args.step}.cgns")
    a_soln = os.path.join(case_dir, f"blockA_2d_{args.step}.cgns")

    b_state_phys = read_block_stacked(b_grid, b_soln, fields, GRID_SIZE).astype(np.float32)
    a_state_phys = read_block_stacked(a_grid, a_soln, fields, GRID_SIZE).astype(np.float32)

    # Sanity check on the CGNS reading pipeline itself, independent of any
    # CUDA/C++ code: block B's precopy state and block A are proven (by the
    # copyToFine/writeIO ordering in DualBlocks.cu) to be bit-identical
    # inside the fine-region box at this exact snapshot point. If they don't
    # match here, the bug is in how (i, j) gets reconstructed from the CGNS
    # vertex data -- e.g. an i/j transpose -- not in the CUDA kernels, and
    # random-noise-based tests could never have caught it.
    box = inner_box_mask(GRID_SIZE)
    box_diff = np.abs(b_state_phys[:, box] - a_state_phys[:, box])
    print("Box-interior A vs B-precopy (should be ~0, bit-identical by construction):")
    for c, f in enumerate(fields):
        print(f"  {f}: max abs diff = {box_diff[c].max():.6g}  mean abs diff = {box_diff[c].mean():.6g}")

    # Cross-check reference for test_ml_kernel.cu's OWN reading of
    # synthetic_input.bin -- this only proves the CGNS pipeline is
    # self-consistent (A and B-precopy always go through the same reader),
    # not that C++'s (i, j) convention when loading the raw flat file lines
    # up with it. Print a ring-region statistic and a specific probe cell
    # here so a matching print added to test_ml_kernel.cu (before it runs
    # any CNN kernels) can be compared by eye against these exact numbers.
    ring = interface_ring_mask(ring_width=4, grid_size=GRID_SIZE)
    print(f"Ring-region input (B-precopy) stats, for cross-check against a matching C++ print:")
    for c, f in enumerate(fields):
        print(f"  {f}: ring mean = {b_state_phys[c][ring].mean():.6g}  n_ring_cells = {ring.sum()}")
    probe_i, probe_j = 18, 30
    print(f"Probe cell (i={probe_i}, j={probe_j}) [should be a ring cell]: "
          f"{[float(b_state_phys[c, probe_i, probe_j]) for c in range(len(fields))]}")

    b_state_phys.tofile(os.path.join(args.out_dir, "synthetic_input.bin"))

    mean = torch.tensor([field_stats[f][0] for f in fields], dtype=torch.float32).view(1, -1, 1, 1)
    std = torch.tensor([field_stats[f][1] for f in fields], dtype=torch.float32).view(1, -1, 1, 1)
    b_t = torch.from_numpy(b_state_phys).unsqueeze(0)
    with torch.no_grad():
        b_norm = (b_t - mean) / std
        a_pred_norm = model.predict_state(b_norm)
        a_pred_phys = (a_pred_norm * std + mean).squeeze(0).numpy()
    a_pred_phys.astype(np.float32).tofile(os.path.join(args.out_dir, "expected_output.bin"))

    print(f"Wrote weights.bin, synthetic_input.bin, expected_output.bin to {args.out_dir}")
    print(f"Sample: {args.case} step {args.step}")
    err = np.abs(a_pred_phys - a_state_phys)
    for c, f in enumerate(fields):
        print(f"  {f}: mean abs (Python model's pred vs true A) = {err[c].mean():.6g}")


if __name__ == "__main__":
    main()
