"""Generate paired extrapolation-test cases: ICs sampled *outside* the
training sweep's [0.5x, 2x] pressure range on one side, to test whether the
model generalizes beyond the distribution it was trained on -- unlike
ml/generate_cases.py's held-out IC test, which is still within-distribution
(same [0.5x, 2x] range, just unseen ICs).

Config 1 ("low"): pressureLow/densityLow sampled from the next band below the
training range -- [0.125x, 0.5x) of the canonical base, i.e. 1250-5000 for
pressureLow -- while pressureHigh/densityHigh stay within the normal training
range ([0.5x, 2x], same as generate_cases.py).

Config 2 ("high"): the mirror image -- pressureHigh/densityHigh sampled from
the next band above the training range ((2x, 8x] of base, i.e. 200000-800000
for pressureHigh), while pressureLow/densityLow stay within the normal
training range.

Both extrapolation bands are the same 4x multiplicative spread as the
training range itself, placed immediately adjacent with no overlap. Density
is derived from pressure at the base input's fixed ratios, same as
generate_cases.py.

Each config gets --n-per-config paired ICs, each run at every filter width in
--filter-widths (same paired-across-widths design as generate_cases.py), so
any performance difference across widths is attributable to the width, not
different ICs. Default 10 ICs x 5 widths x 2 configs = 100 cases.

Writes one directory per case (--out-dir/case_NNNN/ShockTube.input) and a
manifest (--out-dir/manifest.csv) recording config, ic_index,
spectral_filter_width, and the sampled/derived SHOCK parameters.

Usage:
    python3 -m ml.generate_extrapolation_cases --out-dir runs_extrapolation
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from scipy.stats.qmc import LatinHypercube

from .generate_cases import BASE_VALUES, FILTER_WIDTH_PARAM, SWEPT_PARAMS, _RATIO_OF, render_case_input

# Next multiplicative band immediately adjacent to the training [0.5x, 2x]
# range, same 4x spread, with no overlap.
EXTRAP_LOW_BOUNDS = (0.125, 0.5)  # pressureLow in Config 1
EXTRAP_HIGH_BOUNDS = (2.0, 8.0)  # pressureHigh in Config 2
TRAIN_BOUNDS = (0.5, 2.0)  # the in-range side of each config, same as generate_cases.py


def _sample(n: int, bounds: tuple[float, float], base_value: float, seed: int):
    lo, hi = bounds[0] * base_value, bounds[1] * base_value
    unit = LatinHypercube(d=1, seed=seed).random(n=n)[:, 0]
    return lo + unit * (hi - lo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-input", default="ShockTube.input")
    ap.add_argument("--out-dir", default="runs_extrapolation")
    ap.add_argument("--n-per-config", type=int, default=10)
    ap.add_argument("--filter-widths", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    base_path = Path(args.base_input)
    base_text = base_path.read_text()
    base_values = dict(BASE_VALUES)
    print("Base values:", base_values)

    density_pressure_ratio = {
        density_param: base_values[density_param] / base_values[pressure_param]
        for density_param, pressure_param in _RATIO_OF.items()
    }

    config1_pressure_low = _sample(args.n_per_config, EXTRAP_LOW_BOUNDS, base_values["SHOCK.pressureLow"], args.seed)
    config1_pressure_high = _sample(args.n_per_config, TRAIN_BOUNDS, base_values["SHOCK.pressureHigh"], args.seed + 1)
    config2_pressure_low = _sample(args.n_per_config, TRAIN_BOUNDS, base_values["SHOCK.pressureLow"], args.seed + 2)
    config2_pressure_high = _sample(args.n_per_config, EXTRAP_HIGH_BOUNDS, base_values["SHOCK.pressureHigh"], args.seed + 3)

    print(f"Config 1 pressureLow range (extrapolated below training): {EXTRAP_LOW_BOUNDS[0] * base_values['SHOCK.pressureLow']}-{EXTRAP_LOW_BOUNDS[1] * base_values['SHOCK.pressureLow']}")
    print(f"Config 2 pressureHigh range (extrapolated above training): {EXTRAP_HIGH_BOUNDS[0] * base_values['SHOCK.pressureHigh']}-{EXTRAP_HIGH_BOUNDS[1] * base_values['SHOCK.pressureHigh']}")

    configs = {
        "config1": list(zip(config1_pressure_low, config1_pressure_high)),
        "config2": list(zip(config2_pressure_low, config2_pressure_high)),
    }

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    case_idx = 0
    for config_name, ics in configs.items():
        for width in args.filter_widths:
            for ic_index, (p_low, p_high) in enumerate(ics):
                params = {
                    "SHOCK.pressureLow": p_low,
                    "SHOCK.pressureHigh": p_high,
                    "SHOCK.densityLow": density_pressure_ratio["SHOCK.densityLow"] * p_low,
                    "SHOCK.densityHigh": density_pressure_ratio["SHOCK.densityHigh"] * p_high,
                    FILTER_WIDTH_PARAM: width,
                }
                case_id = f"case_{case_idx:04d}"
                case_dir = out_root / case_id
                case_dir.mkdir(parents=True, exist_ok=True)
                (case_dir / "ShockTube.input").write_text(render_case_input(base_text, params))

                manifest_rows.append(
                    {
                        "case_id": case_id,
                        "config": config_name,
                        "ic_index": ic_index,
                        "spectral_filter_width": width,
                        **{p: params[p] for p in SWEPT_PARAMS},
                    }
                )
                case_idx += 1

    manifest_path = out_root / "manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["case_id", "config", "ic_index", "spectral_filter_width", *SWEPT_PARAMS]
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(
        f"Wrote {len(manifest_rows)} cases "
        f"({args.n_per_config} ICs x {len(args.filter_widths)} widths x 2 configs) under {out_root}/"
    )
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
