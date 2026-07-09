"""Reader for the DualBlocks CGNS output (main_gpu ShockTube 2D case).

Each timestep/block is written as a pair of HDF5-backed CGNS files:
  block{A,B}_2d_{step:06d}.cgns       -- FlowSolution (CellCenter), one flat
                                         array per field, in Bricks-library
                                         storage order (NOT row-major).
  block{A,B}_grid_2d_{step:06d}.cgns  -- GridCoordinates + RegularElements
                                         (quad connectivity) + StructuredNodeIndices
                                         (per-vertex (i,j) in the structured mesh).

The flat field arrays are ordered however the Bricks GPU layout put them, so
the only safe way to recover the (i, j) grid position of cell k is via its
corner vertex ids (RegularElements/ElementConnectivity) and each vertex's
(i, j) (StructuredNodeIndices/Indices) -- both live in the grid file. This
was verified against known analytic values (t=0 shock IC densities of
0.125 / 1.0 landing at the expected quadrant) before trusting it.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import h5py
import numpy as np

FIELDS = ("Density", "MomentumX", "MomentumY", "EnergyTotalDensity")

# Empirically located by diffing block A vs block B at t=1: this is the
# region where the two blocks are bit-identical, i.e. the embedded
# high-fidelity box that gets copied from B into A each step.
INNER_BOX_I = (20, 40)  # inclusive
INNER_BOX_J = (20, 40)  # inclusive
GRID_SIZE = 63


@lru_cache(maxsize=None)
def _read_grid_mapping(grid_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (i_idx, j_idx), each shape (n_cells,), giving the structured
    grid position of each entry in a same-timestep/block solution array."""
    with h5py.File(grid_path, "r") as f:
        zone = f["Base/Zone_000001"]
        conn = zone["RegularElements/ElementConnectivity/ data"][()].reshape(-1, 4)
        node_ij = zone["StructuredNodeIndices/Indices/ data"][()]  # (n_nodes, 2)

    node_ids = conn - 1  # CGNS connectivity is 1-indexed
    cell_ij = np.stack(
        [node_ij[node_ids, 0].mean(axis=1), node_ij[node_ids, 1].mean(axis=1)],
        axis=1,
    )
    i_idx = np.round(cell_ij[:, 0] - 0.5).astype(np.int64)
    j_idx = np.round(cell_ij[:, 1] - 0.5).astype(np.int64)
    return i_idx, j_idx


def read_block(
    grid_path: str | Path,
    soln_path: str | Path,
    fields: tuple[str, ...] = FIELDS,
    grid_size: int = GRID_SIZE,
) -> dict[str, np.ndarray]:
    """Reconstruct a dict of field_name -> (grid_size, grid_size) array."""
    i_idx, j_idx = _read_grid_mapping(str(grid_path))

    out: dict[str, np.ndarray] = {}
    with h5py.File(soln_path, "r") as f:
        for field in fields:
            flat = f[f"Base/Zone_000001/Solution/{field}/ data"][()]
            img = np.full((grid_size, grid_size), np.nan, dtype=flat.dtype)
            img[i_idx, j_idx] = flat
            if np.isnan(img).any():
                raise ValueError(
                    f"{soln_path}: incomplete reconstruction for {field} "
                    f"({np.isnan(img).sum()} missing cells)"
                )
            out[field] = img
    return out


def read_block_stacked(
    grid_path: str | Path,
    soln_path: str | Path,
    fields: tuple[str, ...] = FIELDS,
    grid_size: int = GRID_SIZE,
) -> np.ndarray:
    """Same as read_block but stacked into a (len(fields), grid_size, grid_size) array."""
    d = read_block(grid_path, soln_path, fields, grid_size)
    return np.stack([d[f] for f in fields], axis=0)


def inner_box_mask(grid_size: int = GRID_SIZE) -> np.ndarray:
    mask = np.zeros((grid_size, grid_size), dtype=bool)
    mask[INNER_BOX_I[0] : INNER_BOX_I[1] + 1, INNER_BOX_J[0] : INNER_BOX_J[1] + 1] = True
    return mask


def interface_ring_mask(ring_width: int = 3, grid_size: int = GRID_SIZE) -> np.ndarray:
    """Cells within `ring_width` cells of the inner box, excluding the box itself.

    NOTE: ring_width is a placeholder until it's validated against data --
    see README in this directory. PPM-type reconstructions typically need
    ~2 ghost cells, so 3 is a safe-ish starting guess for the flux-correction
    footprint, not a measured value.
    """
    inner = inner_box_mask(grid_size)
    i0, i1 = INNER_BOX_I
    j0, j1 = INNER_BOX_J
    outer = np.zeros((grid_size, grid_size), dtype=bool)
    lo_i, hi_i = max(i0 - ring_width, 0), min(i1 + ring_width, grid_size - 1)
    lo_j, hi_j = max(j0 - ring_width, 0), min(j1 + ring_width, grid_size - 1)
    outer[lo_i : hi_i + 1, lo_j : hi_j + 1] = True
    return outer & ~inner
