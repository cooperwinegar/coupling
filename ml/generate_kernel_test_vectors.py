"""Generate fixed test vectors for the on-GPU hand-written CNN kernels in
~/Desktop/euler-miniapp_AITraining/MLInterfaceCorrection.cu, to be checked by
the companion test_ml_kernel.cu hardware harness.

Writes three files (paths relative to --out-dir):
  weights.bin          -- ml/export_weights.py's binary format, from a
                           fixed-seed randomly initialized model (this is a
                           translation-correctness check, not an accuracy
                           check -- any weights work, as in
                           ml/verify_kernel_reference.py).
  synthetic_input.bin   -- (n_fields, grid_size, grid_size) float32, raw
                           physical-unit values, C-order (component-major,
                           then i, then j) -- same layout convention as
                           ml/infer_cli.py's flat buffer.
  expected_output.bin    -- predict_state() applied to synthetic_input via
                           the actual PyTorch model, same layout, so the CUDA
                           harness can diff its own on-device result against
                           ground truth for the *exact* same input/weights.

Usage:
    python3 -m ml.generate_kernel_test_vectors --out-dir /path/on/gpu/machine
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from .cgns_io import FIELDS, GRID_SIZE
from .export_weights import export_checkpoint_to_binary
from .model import InterfaceCorrectionCNN


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = InterfaceCorrectionCNN(n_fields=len(FIELDS))
    model.eval()

    field_stats = {
        f: (float(np.random.uniform(-2, 2)), float(np.random.uniform(0.5, 3))) for f in FIELDS
    }
    checkpoint = {"fields": FIELDS, "field_stats": field_stats, "model_state_dict": model.state_dict()}
    export_checkpoint_to_binary(checkpoint, os.path.join(args.out_dir, "weights.bin"))

    b_state_phys = (np.random.randn(len(FIELDS), GRID_SIZE, GRID_SIZE) * 2.0).astype(np.float32)
    b_state_phys.tofile(os.path.join(args.out_dir, "synthetic_input.bin"))

    mean = torch.tensor([field_stats[f][0] for f in FIELDS], dtype=torch.float32).view(1, -1, 1, 1)
    std = torch.tensor([field_stats[f][1] for f in FIELDS], dtype=torch.float32).view(1, -1, 1, 1)
    b_t = torch.from_numpy(b_state_phys).unsqueeze(0)
    with torch.no_grad():
        b_norm = (b_t - mean) / std
        a_pred_norm = model.predict_state(b_norm)
        a_pred_phys = (a_pred_norm * std + mean).squeeze(0).numpy()
    a_pred_phys.astype(np.float32).tofile(os.path.join(args.out_dir, "expected_output.bin"))

    print(f"Wrote weights.bin, synthetic_input.bin, expected_output.bin to {args.out_dir}")


if __name__ == "__main__":
    main()
