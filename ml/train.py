"""Training loop, masked to the interface ring.

Splits by whole case (initial-condition sweep run), not by timestep -- the
thing we actually want to generalize to is unseen initial conditions, so
holding out timesteps from an already-trained-on case wouldn't test that.
With only one case's worth of data (no case_NNNN dirs under plot/ yet),
there's nothing to hold out, so this falls back to training on everything as
a pipeline smoke test.
"""

from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from .dataset import DualBlockInterfaceDataset, compute_field_stats, list_cases, split_cases
from .model import InterfaceCorrectionCNN


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


class FieldRMSEAccumulator:
    """Physical-unit (un-normalized) per-field RMSE over the interface ring,
    accumulated across batches as running sum-of-squares/count (not an
    average of per-batch RMSEs, which would be mathematically wrong: sqrt is
    not linear, so mean(sqrt(x)) != sqrt(mean(x)))."""

    def __init__(self, field_stats: dict, fields: tuple[str, ...]):
        self.field_stats = field_stats
        self.fields = fields
        self.sq_sum = {f: 0.0 for f in fields}
        self.count = {f: 0 for f in fields}

    def update(self, pred_delta: torch.Tensor, b_state: torch.Tensor, a_state: torch.Tensor, mask: torch.Tensor):
        a_pred_norm = b_state + pred_delta
        a_pred_phys = unnormalize(a_pred_norm, self.field_stats, self.fields)
        a_true_phys = unnormalize(a_state, self.field_stats, self.fields)
        for c, field in enumerate(self.fields):
            diff2 = (a_pred_phys[:, c][mask] - a_true_phys[:, c][mask]) ** 2
            self.sq_sum[field] += diff2.sum().item()
            self.count[field] += diff2.numel()

    def rmse(self) -> dict[str, float]:
        return {f: (self.sq_sum[f] / self.count[f]) ** 0.5 for f in self.fields}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="plot")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--ring-width", type=int, default=3)
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cases = list_cases(args.root)
    train_cases, val_cases = split_cases(args.root, args.val_fraction, args.seed)
    print(f"Cases found: {len(cases)} -> train {train_cases}, val {val_cases}")

    stats = compute_field_stats(args.root, include_cases=train_cases)
    print("Normalization stats (mean, std), computed on train cases only:", stats)

    train_ds = DualBlockInterfaceDataset(
        args.root, ring_width=args.ring_width, field_stats=stats, include_cases=train_cases
    )
    print(f"Train set: {len(train_ds)} (case, timestep) samples")
    if len(cases) <= 1:
        print(
            "WARNING: only one case available -- there are no held-out initial "
            "conditions, so this run only verifies the pipeline runs end-to-end "
            "and cannot demonstrate generalization. Run ml/generate_cases.py + "
            "ml/run_sweep.py to add more cases."
        )
        val_ds = None
    else:
        val_ds = DualBlockInterfaceDataset(
            args.root, ring_width=args.ring_width, field_stats=stats, include_cases=val_cases
        )
        print(f"Val set: {len(val_ds)} (case, timestep) samples")

    loader = DataLoader(train_ds, batch_size=min(8, len(train_ds)), shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=min(8, len(val_ds)), shuffle=False) if val_ds else None

    model = InterfaceCorrectionCNN()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    def run_epoch(loader, train: bool):
        model.train(train)
        total_loss = 0.0
        rmse_acc = FieldRMSEAccumulator(stats, train_ds.fields)
        with torch.set_grad_enabled(train):
            for batch in loader:
                b_state, a_state, mask = batch["input"], batch["target"], batch["mask"]
                pred_delta = model(b_state)
                target_delta = a_state - b_state
                loss = masked_mse(pred_delta, target_delta, mask)

                if train:
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                total_loss += loss.item()
                with torch.no_grad():
                    rmse_acc.update(pred_delta.detach(), b_state, a_state, mask)
        return total_loss / len(loader), rmse_acc.rmse()

    def format_rmse(rmse: dict[str, float]) -> str:
        return ", ".join(f"{f}={v:.4g}" for f, v in rmse.items())

    for epoch in range(args.epochs):
        train_loss, train_rmse = run_epoch(loader, train=True)
        val_msg = ""
        if val_loader is not None:
            val_loss, val_rmse = run_epoch(val_loader, train=False)
            val_msg = f"  val_masked_mse={val_loss:.6f}  val_rmse=({format_rmse(val_rmse)})"
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print(
                f"epoch {epoch:4d}  train_masked_mse={train_loss:.6f}  "
                f"train_rmse=({format_rmse(train_rmse)}){val_msg}"
            )

    torch.save(model.state_dict(), "ml/interface_correction_cnn.pt")
    print("Saved weights to ml/interface_correction_cnn.pt")


if __name__ == "__main__":
    main()
