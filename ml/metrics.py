"""Loss and evaluation metrics shared by ml/train.py and ml/test_model.py."""

from __future__ import annotations

import torch


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # mask: (B, H, W) bool -> broadcast over channel dim
    m = mask.unsqueeze(1)
    diff2 = (pred - target) ** 2
    return diff2[m.expand_as(diff2)].mean()


def unnormalize(x: torch.Tensor, field_stats: dict, fields: tuple[str, ...]) -> torch.Tensor:
    out = x.clone()
    for c, field in enumerate(fields):
        mean, std = field_stats[field]
        out[:, c] = out[:, c] * std + mean
    return out


class FieldErrorAccumulator:
    """Physical-unit (un-normalized) per-field RMSE and MAE over the interface
    ring, accumulated across batches as running sums/count -- not an average
    of per-batch RMSEs (mathematically wrong: sqrt isn't linear, so
    mean(sqrt(x)) != sqrt(mean(x))); MAE's running mean is exact either way
    since it's already linear."""

    def __init__(self, field_stats: dict, fields: tuple[str, ...]):
        self.field_stats = field_stats
        self.fields = fields
        self.sq_sum = {f: 0.0 for f in fields}
        self.abs_sum = {f: 0.0 for f in fields}
        self.count = {f: 0 for f in fields}

    def update(self, pred_delta: torch.Tensor, b_state: torch.Tensor, a_state: torch.Tensor, mask: torch.Tensor):
        a_pred_norm = b_state + pred_delta
        a_pred_phys = unnormalize(a_pred_norm, self.field_stats, self.fields)
        a_true_phys = unnormalize(a_state, self.field_stats, self.fields)
        for c, field in enumerate(self.fields):
            diff = a_pred_phys[:, c][mask] - a_true_phys[:, c][mask]
            self.sq_sum[field] += (diff**2).sum().item()
            self.abs_sum[field] += diff.abs().sum().item()
            self.count[field] += diff.numel()

    def rmse(self) -> dict[str, float]:
        return {f: (self.sq_sum[f] / self.count[f]) ** 0.5 for f in self.fields}

    def mae(self) -> dict[str, float]:
        return {f: self.abs_sum[f] / self.count[f] for f in self.fields}


def format_errs(rmse: dict[str, float], mae: dict[str, float]) -> str:
    return ", ".join(f"{f}=(rmse={rmse[f]:.4g}, mae={mae[f]:.4g})" for f in rmse)


class ResidualErrorAccumulator:
    """Mean absolute per-field A-B residual over the interface ring, in
    physical units -- i.e. |true_A - true_B|, with no model involved. This is
    the actual magnitude of the flux-correction the model is trying to
    predict, so it's the right thing to compare the model's own physical-unit
    MAE (FieldErrorAccumulator.mae()) against: is the prediction error small
    relative to the size of the correction itself, or comparable to it?"""

    def __init__(self, field_stats: dict, fields: tuple[str, ...]):
        self.field_stats = field_stats
        self.fields = fields
        self.abs_sum = {f: 0.0 for f in fields}
        self.count = {f: 0 for f in fields}

    def update(self, b_state: torch.Tensor, a_state: torch.Tensor, mask: torch.Tensor):
        a_phys = unnormalize(a_state, self.field_stats, self.fields)
        b_phys = unnormalize(b_state, self.field_stats, self.fields)
        for c, field in enumerate(self.fields):
            diff = (a_phys[:, c] - b_phys[:, c])[mask]
            self.abs_sum[field] += diff.abs().sum().item()
            self.count[field] += diff.numel()

    def mae(self) -> dict[str, float]:
        return {f: self.abs_sum[f] / self.count[f] for f in self.fields}


class NormalizedFieldErrorAccumulator:
    """Per-field RMSE and MAE over the interface ring, in *normalized*
    (mean-0/std-1 per field) units -- i.e. directly on pred_delta vs
    target_delta, with no unnormalize() step. Unlike FieldErrorAccumulator's
    physical-unit numbers (not comparable across fields of very different
    magnitude, e.g. EnergyTotalDensity vs Density), these are on a common
    scale and so can be meaningfully averaged across channels."""

    def __init__(self, fields: tuple[str, ...]):
        self.fields = fields
        self.sq_sum = {f: 0.0 for f in fields}
        self.abs_sum = {f: 0.0 for f in fields}
        self.count = {f: 0 for f in fields}

    def update(self, pred_delta: torch.Tensor, target_delta: torch.Tensor, mask: torch.Tensor):
        for c, field in enumerate(self.fields):
            diff = pred_delta[:, c][mask] - target_delta[:, c][mask]
            self.sq_sum[field] += (diff**2).sum().item()
            self.abs_sum[field] += diff.abs().sum().item()
            self.count[field] += diff.numel()

    def rmse(self) -> dict[str, float]:
        return {f: (self.sq_sum[f] / self.count[f]) ** 0.5 for f in self.fields}

    def mae(self) -> dict[str, float]:
        return {f: self.abs_sum[f] / self.count[f] for f in self.fields}
