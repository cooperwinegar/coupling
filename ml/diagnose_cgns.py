"""Diagnose a CGNS grid/solution file pair that fails (i, j) reconstruction
in cgns_io.read_block -- prints exactly which cells are duplicated/missing
and why, instead of just the "N cells missing" summary read_block raises.

Usage:
    python3 -m ml.diagnose_cgns plot/case_0001/blockB_grid_2d_000041.cgns \
                                 plot/case_0001/blockB_2d_000041.cgns \
        [--ref-grid plot/case_0001/blockB_grid_2d_000000.cgns]
"""

from __future__ import annotations

import argparse

import h5py
import numpy as np


def inspect(grid_path: str, soln_path: str | None):
    with h5py.File(grid_path, "r") as f:
        zone = f["Base/Zone_000001"]
        dims = zone[" data"][()].ravel()
        print(f"Zone dims [NVertex, NCell, NBoundVertex]: {dims}")

        conn = zone["RegularElements/ElementConnectivity/ data"][()]
        er = zone["RegularElements/ElementRange/ data"][()]
        node_ij = zone["StructuredNodeIndices/Indices/ data"][()]
        n_nodes = node_ij.shape[0]

        print(f"ElementConnectivity: {conn.shape}, ElementRange: {er}")
        print(f"StructuredNodeIndices/Indices: {node_ij.shape}, "
              f"range i=[{node_ij[:,0].min()},{node_ij[:,0].max()}] "
              f"j=[{node_ij[:,1].min()},{node_ij[:,1].max()}]")

        n_cells = er[1] - er[0] + 1
        conn = conn.reshape(n_cells, -1)
        print(f"Reshaped connectivity: {conn.shape} (n_cells={n_cells}, verts/cell={conn.shape[1]})")

        node_ids = conn - 1
        print(f"node id range referenced by connectivity: [{node_ids.min()}, {node_ids.max()}] "
              f"(n_nodes available: {n_nodes})")
        if node_ids.max() >= n_nodes or node_ids.min() < 0:
            print("  !! connectivity references node ids outside StructuredNodeIndices range")

        cell_ij = np.stack(
            [node_ij[node_ids, 0].mean(axis=1), node_ij[node_ids, 1].mean(axis=1)], axis=1
        )
        i_idx = np.round(cell_ij[:, 0] - 0.5).astype(np.int64)
        j_idx = np.round(cell_ij[:, 1] - 0.5).astype(np.int64)

        # how far each cell's corner-average is from a clean half-integer --
        # large values mean the 4 "corners" aren't a clean unit square in index space
        residual = np.abs(cell_ij - (np.stack([i_idx, j_idx], axis=1) + 0.5))
        bad = residual.max(axis=1) > 1e-9
        print(f"Cells whose corner-average isn't a clean half-integer: {bad.sum()} / {n_cells}")
        if bad.sum():
            worst = np.argsort(-residual.max(axis=1))[:10]
            for k in worst:
                print(f"  cell {k}: corners(i,j)={node_ij[node_ids[k]].tolist()} "
                      f"-> mean={cell_ij[k]} residual={residual[k]}")

        pairs = list(zip(i_idx.tolist(), j_idx.tolist()))
        uniq, counts = np.unique(pairs, axis=0, return_counts=True)
        dup = uniq[counts > 1]
        print(f"Distinct (i,j) produced: {len(uniq)} (expected {n_cells}); duplicated: {len(dup)}")
        if len(dup):
            print("  duplicated (i,j) pairs (first 10):", dup[:10].tolist())

        grid_size = int(max(i_idx.max(), j_idx.max())) + 1
        full = {(i, j) for i in range(grid_size) for j in range(grid_size)}
        present = set(pairs)
        missing = full - present
        print(f"Missing (i,j) grid cells: {len(missing)} (first 10): {sorted(missing)[:10]}")

    if soln_path:
        with h5py.File(soln_path, "r") as f:
            for field in ("Density", "MomentumX", "MomentumY", "EnergyTotalDensity"):
                flat = f[f"Base/Zone_000001/Solution/{field}/ data"][()]
                n_bad = (~np.isfinite(flat)).sum()
                print(f"{field}: n={flat.size} min={np.nanmin(flat):.6g} max={np.nanmax(flat):.6g} "
                      f"non-finite={n_bad}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("grid_path")
    ap.add_argument("soln_path", nargs="?", default=None)
    ap.add_argument("--ref-grid", default=None, help="a known-good grid file (e.g. step 000000) to diff against")
    args = ap.parse_args()

    print(f"=== {args.grid_path} ===")
    inspect(args.grid_path, args.soln_path)

    if args.ref_grid:
        print(f"\n=== reference: {args.ref_grid} ===")
        inspect(args.ref_grid, None)


if __name__ == "__main__":
    main()
