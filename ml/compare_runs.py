"""Compare one block's state between two separate run directories, at each
matching timestep -- e.g. two runs that differ only in the interface-
correction CNN's arithmetic precision (float32 vs float64), to check whether
that choice produces a physically meaningful difference in the corrected
block.

This is a different question from ml/ab_ring_diff_timeseries.py, which
compares block A vs block B *within* a single run (same root). Here, both
--root-a and --root-b hold output from the *same* block (default: B) across
two independent runs, and the comparison is root-vs-root at each shared
timestep.

By default, only the interface ring is compared: outside the ring, the ML
correction never touches block B, so any real difference there would mean
something else diverged between the two runs (not just the correction
kernel's arithmetic), which is itself worth knowing -- pass --full-domain to
check that too.

Usage:
    python3 -m ml.compare_runs --root-a plot_fp32 --root-b plot_fp64 --block B
    python3 -m ml.compare_runs --root-a plot_fp32 --root-b plot_fp64 --full-domain
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

from .cgns_io import FIELDS, GRID_SIZE, interface_ring_mask, read_block_stacked

_SOLN_RE = re.compile(r"^block([AB])_2d_(\d+)\.cgns$")
_GRID_RE = re.compile(r"^block([AB])_grid_2d_(\d+)\.cgns$")


def _index_block(root: Path, block: str) -> dict[str, dict[str, Path]]:
    """{step: {"soln": path, "grid": path}} for one block letter under root."""
    index: dict[str, dict[str, Path]] = {}
    for path in root.rglob(f"block{block}_2d_*.cgns"):
        m = _SOLN_RE.match(path.name)
        if m and m.group(1) == block:
            index.setdefault(m.group(2), {})["soln"] = path
    for path in root.rglob(f"block{block}_grid_2d_*.cgns"):
        m = _GRID_RE.match(path.name)
        if m and m.group(1) == block:
            index.setdefault(m.group(2), {})["grid"] = path
    return index


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root-a", required=True, help="first run's plot directory")
    ap.add_argument("--root-b", required=True, help="second run's plot directory")
    ap.add_argument("--block", default="B", choices=["A", "B"], help="which block to compare")
    ap.add_argument("--ring-width", type=int, default=4)
    ap.add_argument(
        "--full-domain",
        action="store_true",
        help="compare every cell, not just the interface ring",
    )
    args = ap.parse_args()

    index_a = _index_block(Path(args.root_a), args.block)
    index_b = _index_block(Path(args.root_b), args.block)

    common_steps = sorted(set(index_a) & set(index_b))
    if not common_steps:
        raise SystemExit(f"No matching timesteps found for block {args.block} in both roots")

    only_a = sorted(set(index_a) - set(index_b))
    only_b = sorted(set(index_b) - set(index_a))
    if only_a or only_b:
        print(
            f"Note: {len(only_a)} step(s) only in --root-a, "
            f"{len(only_b)} only in --root-b (skipped)"
        )

    ring_mask = None if args.full_domain else interface_ring_mask(args.ring_width, GRID_SIZE)
    scope = "full domain" if args.full_domain else "interface ring only"

    print(f"Comparing block {args.block} across {len(common_steps)} matching timestep(s) ({scope}):")
    print(f"{'step':>8}  {'field':<20} {'mean_abs':>12}  {'max_abs':>12}")

    overall_mean_abs = {f: [] for f in FIELDS}
    n_compared = 0
    for step in common_steps:
        files_a = index_a[step]
        files_b = index_b[step]
        if not ({"soln", "grid"} <= files_a.keys() and {"soln", "grid"} <= files_b.keys()):
            continue
        n_compared += 1
        state_a = read_block_stacked(files_a["grid"], files_a["soln"], FIELDS, GRID_SIZE)
        state_b = read_block_stacked(files_b["grid"], files_b["soln"], FIELDS, GRID_SIZE)
        diff = state_a - state_b

        for c, field in enumerate(FIELDS):
            vals = diff[c][ring_mask] if ring_mask is not None else diff[c].ravel()
            mean_abs = float(np.abs(vals).mean())
            max_abs = float(np.abs(vals).max())
            overall_mean_abs[field].append(mean_abs)
            print(f"{step:>8}  {field:<20} {mean_abs:>12.6g}  {max_abs:>12.6g}")

    if n_compared == 0:
        raise SystemExit("No timestep had both grid+solution files present in both roots")

    print(f"\nOverall average across {n_compared} timestep(s):")
    for field in FIELDS:
        print(f"  {field:<20} mean_abs={np.mean(overall_mean_abs[field]):.6g}")


if __name__ == "__main__":
    main()
