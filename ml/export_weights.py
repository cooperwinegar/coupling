"""Export a trained interface-correction CNN checkpoint to a flat binary file
a C++/CUDA loader can read directly -- no ONNX/TorchScript/protobuf, just
self-describing raw ints and float32 arrays, matching the flat-buffer style
already used elsewhere in this repo (see ml/infer_cli.py). This replaces
shelling out to ml/infer_cli.py at inference time: DualBlocks::applyML now
runs the network as hand-written CUDA kernels directly on GPU-resident data,
loading the weights this script exports once at startup.

Binary layout (all native-endian float32/int32, matching whatever machine
runs this -- the CUDA loader and this script are expected to run on the same
architecture):
    int32   n_fields
    int32   grid_size
    int32   n_layers
    per layer (n_layers times):
        int32   in_ch
        int32   out_ch
        float32 weight[out_ch * in_ch * 3 * 3]   -- PyTorch's native
                                                     (out_ch, in_ch, kh, kw)
                                                     row-major layout
        float32 bias[out_ch]
    float32 field_mean[n_fields]
    float32 field_std[n_fields]
    float32 inner_box_mask[grid_size * grid_size]   -- row-major (i, j),
                                                        matching ml/cgns_io.py

Usage:
    python3 -m ml.export_weights --checkpoint ml/interface_correction_cnn.pt \
        --output ml/interface_correction_cnn.bin
"""

from __future__ import annotations

import argparse
import struct

import numpy as np
import torch

from .cgns_io import FIELDS, GRID_SIZE, inner_box_mask


def export_checkpoint_to_binary(checkpoint: dict, out_path: str) -> None:
    fields = checkpoint["fields"]
    if tuple(fields) != tuple(FIELDS):
        raise ValueError(
            f"checkpoint fields {fields} don't match ml.cgns_io.FIELDS {FIELDS} -- "
            "the C++ side assumes this exact channel order (see DualBlocks.cu's own "
            "component-order comments), so exporting would silently mislabel channels"
        )
    field_stats = checkpoint["field_stats"]
    state_dict = checkpoint["model_state_dict"]

    # Conv2d layers land at the even indices of the net.<i> nn.Sequential;
    # GELU has no parameters, so it never appears in the state dict at all.
    layer_indices = sorted(
        {int(k.split(".")[1]) for k in state_dict if k.startswith("net.") and k.endswith(".weight")}
    )

    with open(out_path, "wb") as f:
        f.write(struct.pack("=iii", len(fields), GRID_SIZE, len(layer_indices)))

        for i in layer_indices:
            weight = state_dict[f"net.{i}.weight"].detach().cpu().numpy().astype(np.float32)
            bias = state_dict[f"net.{i}.bias"].detach().cpu().numpy().astype(np.float32)
            out_ch, in_ch, kh, kw = weight.shape
            if (kh, kw) != (3, 3):
                raise ValueError(f"net.{i}: expected a 3x3 kernel, got {kh}x{kw}")
            f.write(struct.pack("=ii", in_ch, out_ch))
            f.write(np.ascontiguousarray(weight).tobytes())
            f.write(np.ascontiguousarray(bias).tobytes())

        means = np.array([field_stats[fld][0] for fld in fields], dtype=np.float32)
        stds = np.array([field_stats[fld][1] for fld in fields], dtype=np.float32)
        f.write(means.tobytes())
        f.write(stds.tobytes())

        mask = np.ascontiguousarray(inner_box_mask(GRID_SIZE).astype(np.float32))
        f.write(mask.tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="ml/interface_correction_cnn.pt")
    ap.add_argument("--output", default="ml/interface_correction_cnn.bin")
    args = ap.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    export_checkpoint_to_binary(checkpoint, args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
