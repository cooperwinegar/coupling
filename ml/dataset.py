"""PyTorch Dataset over paired block A/B CGNS timesteps, across one or more
initial-condition "cases".

Scans a directory tree for
  block A solution : blockA_2d_{step}.cgns
  block A grid     : blockA_grid_2d_{step}.cgns
  block B solution : blockB_2d_{step}.cgns
  block B grid     : blockB_grid_2d_{step}.cgns
and yields one sample per (case, timestep) where all four files are present.

Timestep numbers reset per case (each run's plotting starts at 000000), so
samples are keyed by (case_id, step), not step alone -- otherwise different
cases' t=0 would collide. A file's case_id is the first `case_NNNN`-named
ancestor directory under `root` (as produced by run_sweep.py); anything not
under a `case_NNNN` directory (e.g. files dropped directly in plot/, or in
an arbitrarily-named folder from manual testing) is pooled into a single
implicit case "" so earlier ad hoc single-run data keeps working.

Sample = (block B full-domain state, block A full-domain state, interface
ring mask). The training loss should be computed only over cells selected
by the mask; block A's full state is returned (not just the ring) so a
model can also be evaluated/visualized over the whole domain.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .cgns_io import FIELDS, GRID_SIZE, interface_ring_mask, read_block_stacked

_SOLN_RE = re.compile(r"^block([AB])_2d_(\d+)\.cgns$")
_GRID_RE = re.compile(r"^block([AB])_grid_2d_(\d+)\.cgns$")
_CASE_RE = re.compile(r"^case_\d+$")

_Key = tuple[str, str]  # (case_id, step)


def _case_id_for(path: Path, root: Path) -> str:
    for part in path.relative_to(root).parts:
        if _CASE_RE.match(part):
            return part
    return ""


def _index_dir(root: Path) -> dict[_Key, dict[str, Path]]:
    """-> {(case_id, step): {"A_soln":..., "A_grid":..., "B_soln":..., "B_grid":...}}"""
    index: dict[_Key, dict[str, Path]] = {}
    for path in root.rglob("*.cgns"):
        case_id = _case_id_for(path, root)
        m = _SOLN_RE.match(path.name)
        if m:
            block, step = m.groups()
            index.setdefault((case_id, step), {})[f"{block}_soln"] = path
            continue
        m = _GRID_RE.match(path.name)
        if m:
            block, step = m.groups()
            index.setdefault((case_id, step), {})[f"{block}_grid"] = path
    return index


def list_cases(root: str | Path) -> list[str]:
    """All distinct case_ids found under root (sorted; "" included if present)."""
    index = _index_dir(Path(root))
    return sorted({case_id for case_id, _ in index})


def split_cases(
    root: str | Path, val_fraction: float = 0.2, seed: int = 0
) -> tuple[list[str], list[str]]:
    """Hold out whole cases (not individual timesteps) for validation -- the
    meaningful generalization test here is to unseen initial conditions, not
    to unseen timesteps of a case already trained on."""
    cases = list_cases(root)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(cases))
    n_val = max(1, round(len(cases) * val_fraction)) if len(cases) > 1 else 0
    val_idx = set(perm[:n_val].tolist())
    train_cases = [c for i, c in enumerate(cases) if i not in val_idx]
    val_cases = [c for i, c in enumerate(cases) if i in val_idx]
    return train_cases, val_cases


class DualBlockInterfaceDataset(Dataset):
    """Input: block B full state (C,H,W). Target: block A full state (C,H,W).
    Also returns a boolean interface-ring mask (H,W), shared across samples.
    """

    def __init__(
        self,
        root: str | Path,
        fields: tuple[str, ...] = FIELDS,
        grid_size: int = GRID_SIZE,
        ring_width: int = 3,
        field_stats: dict[str, tuple[float, float]] | None = None,
        include_cases: set[str] | list[str] | None = None,
    ):
        self.root = Path(root)
        self.fields = fields
        self.grid_size = grid_size
        self.mask = torch.from_numpy(interface_ring_mask(ring_width, grid_size))
        self.field_stats = field_stats  # optional {field: (mean, std)} for normalization

        required = {"A_soln", "A_grid", "B_soln", "B_grid"}
        index = _index_dir(self.root)
        allowed = set(include_cases) if include_cases is not None else None
        self.samples = sorted(
            (key, files)
            for key, files in index.items()
            if required.issubset(files) and (allowed is None or key[0] in allowed)
        )
        missing = {
            key: required - set(files)
            for key, files in index.items()
            if not required.issubset(files) and (allowed is None or key[0] in allowed)
        }
        if missing:
            import warnings

            warnings.warn(f"Skipping incomplete (case, timestep) pairs (missing files): {missing}")
        if not self.samples:
            raise FileNotFoundError(f"No complete block A/B timestep pairs found under {self.root}")

    def __len__(self) -> int:
        return len(self.samples)

    def _normalize(self, arr: np.ndarray) -> np.ndarray:
        if self.field_stats is None:
            return arr
        out = arr.copy()
        for c, field in enumerate(self.fields):
            mean, std = self.field_stats[field]
            out[c] = (out[c] - mean) / std
        return out

    def __getitem__(self, idx: int):
        (case_id, step), files = self.samples[idx]
        b_state = read_block_stacked(files["B_grid"], files["B_soln"], self.fields, self.grid_size)
        a_state = read_block_stacked(files["A_grid"], files["A_soln"], self.fields, self.grid_size)

        b_state = self._normalize(b_state)
        a_state = self._normalize(a_state)

        return {
            "case_id": case_id,
            "step": step,
            "input": torch.from_numpy(b_state).float(),
            "target": torch.from_numpy(a_state).float(),
            "mask": self.mask,
        }


def compute_field_stats(
    root: str | Path,
    fields: tuple[str, ...] = FIELDS,
    include_cases: set[str] | list[str] | None = None,
) -> dict[str, tuple[float, float]]:
    """Mean/std per field across every block A + B state found under root.
    Run once (ideally on the training cases only) and reuse (pass as
    `field_stats=`) so train/val use the same normalization."""
    index = _index_dir(Path(root))
    allowed = set(include_cases) if include_cases is not None else None
    values: dict[str, list[np.ndarray]] = {f: [] for f in fields}
    for (case_id, step), files in index.items():
        if allowed is not None and case_id not in allowed:
            continue
        for block in ("A", "B"):
            if f"{block}_soln" in files and f"{block}_grid" in files:
                d = read_block_stacked(files[f"{block}_grid"], files[f"{block}_soln"], fields)
                for c, field in enumerate(fields):
                    values[field].append(d[c])
    return {
        field: (float(np.mean(arrs)), float(np.std(arrs)))
        for field, arrs in values.items()
    }
