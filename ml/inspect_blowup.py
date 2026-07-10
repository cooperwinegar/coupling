"""Diagnose exactly where a solver "blowup" first appears in a case, using the
same non-finite check ml/dataset.py's _filter_readable applies to drop training
samples -- so instead of guessing which timestep/field/cell tripped it, you can
go straight to the offending CGNS file.

Walks every timestep of the given --case in step order and, for both block A
and block B, reconstructs the full (field, i, j) state and checks np.isfinite
per field (matching ml/cgns_io.py's has_nonfinite, which flags NaN *or* Inf,
across the whole domain and all four fields -- not just the interface ring).

Prints one line per step ("clean" or "NON-FINITE VALUES FOUND"), and for the
first non-finite step, prints per field: NaN count, Inf count, and the (i, j)
location of up to --max-cells offending cells with their raw values.

Remember: ml.train's cascade rule (ml/dataset.py's _filter_readable) drops the
first non-finite step of a case AND every later step of that same case, even
if a later step reconstructs as finite on its own. So a step you inspect by
hand may look clean while still being excluded, because an earlier step in the
same case blew up. This script scans the whole case precisely to surface that.

Usage:
    python3 -m ml.inspect_blowup --list-cases --root plot
    python3 -m ml.inspect_blowup --root plot --case case_0007
    python3 -m ml.inspect_blowup --root plot --case case_0007 --max-cells 20
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .cgns_io import FIELDS, GRID_SIZE, read_block_stacked
from .dataset import _index_dir, list_cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="plot")
    ap.add_argument("--case", default=None, help='Case id to inspect, e.g. case_0007 (pass "" for ungrouped data)')
    ap.add_argument("--max-cells", type=int, default=10, help="Max non-finite cell locations to print per field")
    ap.add_argument("--list-cases", action="store_true", help="List available case ids under --root and exit")
    args = ap.parse_args()

    if args.list_cases:
        for c in list_cases(args.root):
            print(c or "(ungrouped)")
        return

    if args.case is None:
        raise SystemExit("--case is required (use --list-cases to see available ids under --root)")

    required = {"A_soln", "A_grid", "B_soln", "B_grid"}
    index = _index_dir(Path(args.root))
    items = sorted(
        (key, files) for key, files in index.items() if key[0] == args.case and required.issubset(files)
    )
    if not items:
        raise SystemExit(f"No complete (case, step) pairs found for case={args.case!r} under {args.root}")

    print(f"{len(items)} timestep(s) found for case {args.case!r}; scanning in order...\n")

    first_blowup_step = None
    for (case_id, step), files in items:
        try:
            b_state = read_block_stacked(files["B_grid"], files["B_soln"], FIELDS, GRID_SIZE)
            a_state = read_block_stacked(files["A_grid"], files["A_soln"], FIELDS, GRID_SIZE)
        except Exception as e:  # noqa: BLE001 -- corrupt/unreadable CGNS is data, not a bug here
            print(f"step {step}: UNREADABLE ({type(e).__name__}: {e})")
            continue

        problems = []
        for block_name, state in (("A", a_state), ("B", b_state)):
            for c, field in enumerate(FIELDS):
                arr = state[c]
                bad = ~np.isfinite(arr)
                if bad.any():
                    problems.append((block_name, field, arr, bad))

        if not problems:
            print(f"step {step}: clean")
            continue

        if first_blowup_step is None:
            first_blowup_step = step
            marker = "  <-- FIRST BLOWUP"
        else:
            marker = "  (cascades from first blowup above)"
        print(f"step {step}: NON-FINITE VALUES FOUND{marker}")
        for block_name, field, arr, bad in problems:
            n_nan = int(np.isnan(arr).sum())
            n_inf = int(np.isinf(arr).sum())
            locs = np.argwhere(bad)
            print(f"  block {block_name} / {field}: {n_nan} NaN, {n_inf} Inf cell(s)")
            for i, j in locs[: args.max_cells]:
                print(f"    (i={i}, j={j}) = {arr[i, j]}")
            if len(locs) > args.max_cells:
                print(f"    ... and {len(locs) - args.max_cells} more")

    print()
    if first_blowup_step is None:
        print(f"No non-finite values found in any of the {len(items)} timestep(s) for case {args.case!r}.")
        print("ml.train's blowup filter would NOT exclude this case.")
    else:
        print(
            f"First blowup at step {first_blowup_step}. ml.train's cascade rule (ml/dataset.py's "
            f"_filter_readable) drops this step and every later step of case {args.case!r}, even "
            f"though later steps may themselves reconstruct as finite."
        )


if __name__ == "__main__":
    main()
