"""Generate a *paired* Latin-Hypercube sweep of ShockTube.input cases across
spectral filter widths.

Samples N initial conditions once (Latin-Hypercube over SHOCK.pressureLow and
SHOCK.pressureHigh; SHOCK.densityLow/High derived from pressure at the base
input's fixed density/pressure ratios), then runs that *same* set of ICs at
every spectral_filter_width in --filter-widths (default 1,2,3,4,5). This paired
design means any performance difference across widths is attributable to the
filter width, not to different ICs. Default 50 ICs x 5 widths = 250 cases.

Pressure is swept over [0.5x, 2x] of the canonical base values
(pressureLow=10000, pressureHigh=100000), giving the documented ranges
(Low 5000-20000, High 50000-200000); density follows at the fixed base ratio
(densityLow=0.125, densityHigh=1.0), so densityLow/pressureLow and
densityHigh/pressureHigh are constant across every case. The two pressure
ranges never overlap, so the shock is never inverted. These canonical values
are used regardless of what happens to be uncommented in the base .input file,
so the ranges are stable; pass --base-values-from-input to sample around the
file's own uncommented values instead.

A stratified *paired* holdout reserves --n-holdout-ic of the ICs (default 10) as
the test set. Because ICs are shared across widths, holding out an IC holds it
out at *every* width, so a test IC is never seen in training at any width. The
split (train/test) is recorded per case in the manifest; validation is carved
from the training cases later by train.py.

Writes one directory per case (runs/case_NNNN/ShockTube.input) and a manifest
(runs/manifest.csv) recording ic_index, spectral_filter_width, split, and the
sampled/derived SHOCK parameters -- for provenance and for tagging training data
(and per-filter-width test metrics) by case later.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np
from scipy.stats.qmc import LatinHypercube

SWEPT_PARAMS = ("SHOCK.densityLow", "SHOCK.densityHigh", "SHOCK.pressureLow", "SHOCK.pressureHigh")
SAMPLED_PARAMS = ("SHOCK.pressureLow", "SHOCK.pressureHigh")
FILTER_WIDTH_PARAM = "spectral_filter_width"
_RATIO_OF = {
    "SHOCK.densityLow": "SHOCK.pressureLow",
    "SHOCK.densityHigh": "SHOCK.pressureHigh",
}

# Canonical base values (the documented setup). Density is derived from pressure
# at these ratios; the sweep ranges are [0.5x, 2x] of the pressure bases. Used
# instead of whatever is uncommented in the base .input file so the ranges stay
# fixed at Low 5000-20000 / High 50000-200000 (see --base-values-from-input).
BASE_VALUES = {
    "SHOCK.pressureLow": 10000.0,
    "SHOCK.pressureHigh": 100000.0,
    "SHOCK.densityLow": 0.125,
    "SHOCK.densityHigh": 1.0,
}


def parse_base_input(path: Path) -> dict[str, float]:
    values = {}
    text = path.read_text()
    for param in SWEPT_PARAMS:
        m = re.search(rf"^\s*{re.escape(param)}\s*=\s*([-+0-9.eE]+)", text, re.MULTILINE)
        if not m:
            raise ValueError(f"Could not find '{param}' in {path}")
        values[param] = float(m.group(1))
    return values


def render_case_input(base_text: str, params: dict[str, float]) -> str:
    text = base_text
    for param, value in params.items():
        pattern = re.compile(rf"^(\s*{re.escape(param)}\s*=\s*)[-+0-9.eE]+", re.MULTILINE)
        text, n = pattern.subn(rf"\g<1>{value:.8g}", text)
        if n != 1:
            raise ValueError(f"Expected exactly one occurrence of '{param}', found {n}")
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-input", default="ShockTube.input")
    ap.add_argument("--out-dir", default="runs")
    ap.add_argument("--n-cases", type=int, default=50, help="number of ICs, shared across all widths")
    ap.add_argument("--filter-widths", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--n-holdout-ic", type=int, default=10, help="ICs reserved as the test set (at every width)")
    ap.add_argument("--seed", type=int, default=0, help="LHS sampling seed")
    ap.add_argument("--split-seed", type=int, default=0, help="seed for choosing the held-out test ICs")
    ap.add_argument(
        "--base-values-from-input",
        action="store_true",
        help="sample around the base .input's uncommented values instead of the canonical BASE_VALUES",
    )
    args = ap.parse_args()

    base_path = Path(args.base_input)
    base_text = base_path.read_text()
    base_values = parse_base_input(base_path) if args.base_values_from_input else dict(BASE_VALUES)
    print("Base values:", base_values)

    density_pressure_ratio = {
        density_param: base_values[density_param] / base_values[pressure_param]
        for density_param, pressure_param in _RATIO_OF.items()
    }
    print("Fixed density/pressure ratios (held constant across all cases):", density_pressure_ratio)

    bounds = {p: (0.5 * base_values[p], 2.0 * base_values[p]) for p in SAMPLED_PARAMS}
    print("Sample ranges:", bounds)

    # Sample the ICs once, then reuse them at every filter width (paired design).
    sampler = LatinHypercube(d=len(SAMPLED_PARAMS), seed=args.seed)
    unit_samples = sampler.random(n=args.n_cases)  # (n_cases, 2) in [0,1)
    ic_params: list[dict[str, float]] = []
    for unit_row in unit_samples:
        params: dict[str, float] = {}
        for param, u in zip(SAMPLED_PARAMS, unit_row):
            lo, hi = bounds[param]
            params[param] = lo + u * (hi - lo)
        for density_param, pressure_param in _RATIO_OF.items():
            params[density_param] = density_pressure_ratio[density_param] * params[pressure_param]
        ic_params.append(params)

    # Paired holdout: the same held-out ICs are the test set at every width.
    rng = np.random.default_rng(args.split_seed)
    holdout_ics = set(rng.permutation(args.n_cases)[: args.n_holdout_ic].tolist())
    print(f"Held-out test ICs ({len(holdout_ics)} of {args.n_cases}): {sorted(holdout_ics)}")

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    case_idx = 0
    for width in args.filter_widths:
        for ic_index, ic in enumerate(ic_params):
            case_id = f"case_{case_idx:04d}"
            params = {**ic, FILTER_WIDTH_PARAM: width}

            case_dir = out_root / case_id
            case_dir.mkdir(parents=True, exist_ok=True)
            (case_dir / "ShockTube.input").write_text(render_case_input(base_text, params))

            manifest_rows.append(
                {
                    "case_id": case_id,
                    "ic_index": ic_index,
                    "spectral_filter_width": width,
                    "split": "test" if ic_index in holdout_ics else "train",
                    **ic,
                }
            )
            case_idx += 1

    manifest_path = out_root / "manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["case_id", "ic_index", "spectral_filter_width", "split", *SWEPT_PARAMS]
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    n_test = sum(r["split"] == "test" for r in manifest_rows)
    print(
        f"Wrote {len(manifest_rows)} cases "
        f"({len(args.filter_widths)} widths x {args.n_cases} ICs) under {out_root}/"
    )
    print(f"  train: {len(manifest_rows) - n_test}   test (held out): {n_test}   manifest: {manifest_path}")


if __name__ == "__main__":
    main()
