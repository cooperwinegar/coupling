"""Export a trained InterfaceCorrectionCNN checkpoint as a self-contained
TorchScript module the C++/CUDA miniapp (DualBlocks::applyML) can load
directly via torch::jit::load(), with no normalization logic duplicated on
the C++ side.

The exported module takes/returns *physical* units directly: raw block B
state in, predicted full block A state out. Normalization (using the
checkpoint's saved per-field mean/std), the model's forward pass, and
denormalization are all baked into one traced nn.Module -- so the C++ side
just feeds it a (1, n_fields, H, W) tensor of a_source's raw values and gets
back the predicted A state in the same units, and field_stats never has to
be hand-copied into C++ (and drift out of sync with retraining).

Usage:
    python3 -m ml.export_torchscript \
        --checkpoint ml/interface_correction_cnn.pt \
        --out ml/interface_correction_cnn_traced.pt
"""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn

from .model import InterfaceCorrectionCNN


class _PhysicalUnitsWrapper(nn.Module):
    """Wraps a trained InterfaceCorrectionCNN so forward() takes/returns
    physical units directly: normalize -> predict_state -> denormalize."""

    def __init__(self, model: InterfaceCorrectionCNN, mean: torch.Tensor, std: torch.Tensor):
        super().__init__()
        self.model = model
        # (1, C, 1, 1) so they broadcast against (B, C, H, W) with no loop.
        self.register_buffer("mean", mean.view(1, -1, 1, 1))
        self.register_buffer("std", std.view(1, -1, 1, 1))

    def forward(self, b_state_phys: torch.Tensor) -> torch.Tensor:
        b_state_norm = (b_state_phys - self.mean) / self.std
        a_pred_norm = self.model.predict_state(b_state_norm)
        return a_pred_norm * self.std + self.mean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="ml/interface_correction_cnn.pt")
    ap.add_argument("--out", default="ml/interface_correction_cnn_traced.pt")
    args = ap.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    fields = checkpoint["fields"]
    field_stats = checkpoint["field_stats"]
    grid_size = checkpoint["grid_size"]

    model = InterfaceCorrectionCNN(n_fields=len(fields))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    mean = torch.tensor([field_stats[f][0] for f in fields], dtype=torch.float32)
    std = torch.tensor([field_stats[f][1] for f in fields], dtype=torch.float32)
    wrapper = _PhysicalUnitsWrapper(model, mean, std)
    wrapper.eval()

    example = torch.zeros(1, len(fields), grid_size, grid_size)
    traced = torch.jit.trace(wrapper, example)
    traced.save(args.out)

    print(f"Traced module saved to {args.out}")
    print(f"  fields={fields}  grid_size={grid_size}  ring_width={checkpoint['ring_width']}")
    print(f"  field_stats (mean, std): {field_stats}")
    print("Input/output: (1, n_fields, grid_size, grid_size), PHYSICAL units, full domain.")
    print("Output is the predicted full-domain A state (already denormalized).")


if __name__ == "__main__":
    main()
