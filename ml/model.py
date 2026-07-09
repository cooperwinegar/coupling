"""Small fully-convolutional net that predicts the interface-ring correction.

Design choice: predict a residual (A - B) rather than A's absolute state.
This is well-motivated here specifically because B and A already agree
exactly inside the inner box and are close everywhere else (both trace back
to the same field, one just spectrally filtered) -- so "predict zero, add to
B" is a strong baseline the network only has to improve on near the
interface, rather than having to reconstruct the whole field from scratch.

The inner-box mask is concatenated as an extra input channel so the network
has an explicit spatial cue for where the fidelity boundary sits.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .cgns_io import FIELDS, GRID_SIZE, inner_box_mask


class InterfaceCorrectionCNN(nn.Module):
    def __init__(self, n_fields: int = len(FIELDS), hidden: int = 32, n_layers: int = 6):
        super().__init__()
        in_ch = n_fields + 1  # + inner-box mask channel
        layers: list[nn.Module] = [nn.Conv2d(in_ch, hidden, 3, padding=1), nn.GELU()]
        for _ in range(n_layers - 2):
            layers += [nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU()]
        layers += [nn.Conv2d(hidden, n_fields, 3, padding=1)]
        self.net = nn.Sequential(*layers)
        self.register_buffer("box_mask", torch.from_numpy(inner_box_mask()).float().unsqueeze(0))

    def forward(self, b_state: torch.Tensor) -> torch.Tensor:
        """b_state: (B, C, H, W) -> predicted delta (B, C, H, W), where
        A_pred = b_state + delta."""
        box = self.box_mask.expand(b_state.shape[0], -1, -1, -1)
        x = torch.cat([b_state, box], dim=1)
        return self.net(x)

    def predict_state(self, b_state: torch.Tensor) -> torch.Tensor:
        return b_state + self.forward(b_state)
