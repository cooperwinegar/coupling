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

from .cgns_io import FIELDS, GRID_SIZE, has_nonfinite, interface_ring_mask, read_block_stacked

_SOLN_RE = re.compile(r"^block([AB])_2d_(\d+)\.cgns$")
_GRID_RE = re.compile(r"^block([AB])_grid_2d_(\d+)\.cgns$")
_CASE_RE = re.compile(r"^case_\d+$")

# Held-out test data lives under plot/plot_result -- always excluded from
# training-side indexing (train.py's --root defaults to plot/, whose rglob
# would otherwise recurse straight into it), so a model can never accidentally
# train or validate on the same data ml/test_model.py evaluates it against.
# Passing --root plot/plot_result directly (test_model.py's default) is
# unaffected: this only excludes the name when it appears *below* root, not
# when it IS root.
RESERVED_TEST_DIR_NAME = "plot_result"

_Key = tuple[str, str]  # (case_id, step)


def _case_id_for(path: Path, root: Path) -> str:
    for part in path.relative_to(root).parts:
        if _CASE_RE.match(part):
            return part
    return ""


def _is_under_reserved_test_dir(path: Path, root: Path) -> bool:
    return RESERVED_TEST_DIR_NAME in path.relative_to(root).parts


def _index_dir(root: Path) -> dict[_Key, dict[str, Path]]:
    """-> {(case_id, step): {"A_soln":..., "A_grid":..., "B_soln":..., "B_grid":...}}"""
    index: dict[_Key, dict[str, Path]] = {}
    for path in root.rglob("*.cgns"):
        if _is_under_reserved_test_dir(path, root):
            continue
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


def _load_pair(files: dict[str, Path], fields: tuple[str, ...], grid_size: int):
    """Read (block B, block A) as stacked arrays for one (case, step). Raises on
    unreadable/inconsistent files; caller decides whether to also reject
    non-finite results (a genuine solver blowup, e.g. from an unstable IC --
    common enough in a parameter sweep that it shouldn't crash the whole run)."""
    b_state = read_block_stacked(files["B_grid"], files["B_soln"], fields, grid_size)
    a_state = read_block_stacked(files["A_grid"], files["A_soln"], fields, grid_size)
    return b_state, a_state


def _filter_readable(
    candidates: list[tuple[_Key, dict[str, Path]]],
    fields: tuple[str, ...],
    grid_size: int,
) -> list[tuple[_Key, dict[str, Path]]]:
    """Drop (case, step) pairs that are unreadable, and -- once a case goes
    non-finite (a genuine solver blowup) -- every later timestep of that same
    case too, even if it happens to reconstruct as finite. This is a
    time-marching solver: a step downstream of a diverged state is not
    trustworthy data just because its own values pass an isfinite check (a
    limiter/clamp could produce finite-but-nonphysical output after a blowup).

    An unreadable file does NOT cascade the same way -- that's treated as an
    isolated I/O/file problem (e.g. a truncated write), not evidence the
    simulation itself diverged, so later timesteps of that case are still
    considered on their own merits.

    Warns with the reason per dropped key."""
    by_case: dict[str, list[tuple[_Key, dict[str, Path]]]] = {}
    for key, files in candidates:
        by_case.setdefault(key[0], []).append((key, files))
    for items in by_case.values():
        items.sort(key=lambda kf: kf[0][1])  # steps are zero-padded -> lexicographic == numeric order

    kept = []
    dropped: dict[_Key, str] = {}
    for items in by_case.values():
        blown_up = False
        for key, files in items:
            if blown_up:
                dropped[key] = "downstream of an earlier solver blowup in this case"
                continue
            try:
                b_state, a_state = _load_pair(files, fields, grid_size)
            except Exception as e:  # noqa: BLE001 -- corrupt CGNS files are data, not bugs
                dropped[key] = f"unreadable: {type(e).__name__}: {e}"
                continue
            if has_nonfinite(b_state) or has_nonfinite(a_state):
                dropped[key] = "non-finite values (solver blowup)"
                blown_up = True
                continue
            kept.append((key, files))

    if dropped:
        import warnings

        warnings.warn(f"Dropping {len(dropped)} (case, timestep) pair(s): {dropped}")
    return kept


def list_cases(root: str | Path) -> list[str]:
    """All distinct case_ids found under root (sorted; "" included if present)."""
    index = _index_dir(Path(root))
    return sorted({case_id for case_id, _ in index})


def load_manifest(path: str | Path) -> dict[str, dict]:
    """Read runs/manifest.csv (written by generate_cases.py) into
    {case_id: {"ic_index": int, "spectral_filter_width": int, "split": str}}.

    Lets the training/eval side tag each case by its filter width and by the
    paired train/test holdout without re-deriving them."""
    import csv

    out: dict[str, dict] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            out[row["case_id"]] = {
                "ic_index": int(row["ic_index"]),
                "spectral_filter_width": int(row["spectral_filter_width"]),
                "split": row.get("split", "train"),
            }
    return out


def cases_in_split(manifest: dict[str, dict], split: str) -> list[str]:
    """Case ids in the given manifest split ("train" or "test"), sorted."""
    return sorted(c for c, m in manifest.items() if m["split"] == split)


def cases_by_width(
    manifest: dict[str, dict], cases: list[str] | set[str] | None = None
) -> dict[int, list[str]]:
    """{spectral_filter_width: [case_id, ...]} for the given cases (or all),
    sorted by width then case id -- for per-filter-width metrics."""
    sel = set(cases) if cases is not None else set(manifest)
    out: dict[int, list[str]] = {}
    for c, m in manifest.items():
        if c in sel:
            out.setdefault(m["spectral_filter_width"], []).append(c)
    return {w: sorted(out[w]) for w in sorted(out)}


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
        ring_width: int = 4,
        field_stats: dict[str, tuple[float, float]] | None = None,
        include_cases: set[str] | list[str] | None = None,
        validate: bool = True,
    ):
        self.root = Path(root)
        self.fields = fields
        self.grid_size = grid_size
        self.mask = torch.from_numpy(interface_ring_mask(ring_width, grid_size))
        self.field_stats = field_stats  # optional {field: (mean, std)} for normalization

        required = {"A_soln", "A_grid", "B_soln", "B_grid"}
        index = _index_dir(self.root)
        allowed = set(include_cases) if include_cases is not None else None
        candidates = sorted(
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

        self.samples = _filter_readable(candidates, fields, grid_size) if validate else candidates
        if not self.samples:
            raise FileNotFoundError(f"No complete, readable block A/B timestep pairs found under {self.root}")

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
        b_state, a_state = _load_pair(files, self.fields, self.grid_size)

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
    grid_size: int = GRID_SIZE,
    include_cases: set[str] | list[str] | None = None,
) -> dict[str, tuple[float, float]]:
    """Mean/std per field across every readable, finite block A + B state found
    under root (same filtering DualBlockInterfaceDataset applies, so a blown-up
    timestep can't poison normalization with NaN). Run once (ideally on the
    training cases only) and reuse (pass as `field_stats=`) so train/val use
    the same normalization."""
    required = {"A_soln", "A_grid", "B_soln", "B_grid"}
    index = _index_dir(Path(root))
    allowed = set(include_cases) if include_cases is not None else None
    candidates = [
        (key, files)
        for key, files in index.items()
        if required.issubset(files) and (allowed is None or key[0] in allowed)
    ]
    good = _filter_readable(candidates, fields, grid_size)

    values: dict[str, list[np.ndarray]] = {f: [] for f in fields}
    for _key, files in good:
        b_state, a_state = _load_pair(files, fields, grid_size)
        for c, field in enumerate(fields):
            values[field].append(b_state[c])
            values[field].append(a_state[c])
    return {
        field: (float(np.mean(arrs)), float(np.std(arrs)))
        for field, arrs in values.items()
    }
