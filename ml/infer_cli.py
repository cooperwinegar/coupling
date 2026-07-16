"""Standalone CLI inference wrapper for DualBlocks::applyML.

Reads a raw flat float32 buffer (shape [n_comp, nx, ny], physical units,
component-major then i then j) from --input, runs the trained
interface-correction model, and writes the predicted full-domain state
(physical units, same flat layout) to --output.

Invoked as a subprocess per call by the C++ miniapp, rather than linking
LibTorch directly into DualBlocks.cu -- nvcc's device-code compiler doesn't
mix well with LibTorch's headers (see chat/commit history: it broke
unrelated device-code symbols elsewhere in that file). Since this runs as an
ordinary Python process rather than being embedded in C++, it loads the raw
training checkpoint directly -- no TorchScript export/tracing needed here.

Usage:
    python3 -m ml.infer_cli --input in.bin --output out.bin \
        --n-comp 4 --nx 63 --ny 63
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from .model import InterfaceCorrectionCNN
from .train import CHECKPOINT_PATH


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="path to input flat float32 buffer")
    ap.add_argument("--output", required=True, help="path to write the output flat float32 buffer")
    ap.add_argument("--n-comp", type=int, required=True)
    ap.add_argument("--nx", type=int, required=True)
    ap.add_argument("--ny", type=int, required=True)
    ap.add_argument("--checkpoint", default=CHECKPOINT_PATH)
    args = ap.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    fields = checkpoint["fields"]
    field_stats = checkpoint["field_stats"]

    if len(fields) != args.n_comp:
        raise ValueError(f"checkpoint has {len(fields)} fields but --n-comp={args.n_comp}")

    model = InterfaceCorrectionCNN(n_fields=len(fields))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    flat = np.fromfile(args.input, dtype=np.float32)
    expected = args.n_comp * args.nx * args.ny
    if flat.size != expected:
        raise ValueError(f"{args.input}: read {flat.size} floats, expected {expected}")
    b_state_phys = torch.from_numpy(flat.reshape(1, args.n_comp, args.nx, args.ny).copy())

    mean = torch.tensor([field_stats[f][0] for f in fields], dtype=torch.float32).view(1, -1, 1, 1)
    std = torch.tensor([field_stats[f][1] for f in fields], dtype=torch.float32).view(1, -1, 1, 1)

    with torch.no_grad():
        b_state_norm = (b_state_phys - mean) / std
        a_pred_norm = model.predict_state(b_state_norm)
        a_pred_phys = a_pred_norm * std + mean

    a_pred_phys.numpy().astype(np.float32).tofile(args.output)


if __name__ == "__main__":
    main()
