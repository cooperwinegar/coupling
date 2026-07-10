"""Generate a Latin-Hypercube sweep of ShockTube.input cases.

Independently varies SHOCK.pressureLow and SHOCK.pressureHigh over
[0.5x, 2x] of their base-input values; SHOCK.densityLow/densityHigh are then
*derived* from pressure via the base input's own density/pressure ratios --
densityLow = (base densityLow / base pressureLow) * pressureLow, and
similarly for the high side -- so densityLow/pressureLow and
densityHigh/pressureHigh are exactly constant across every case (matching
the base input's ratios), rather than varying independently. Since the ratio
is fixed, density still spans [0.5x, 2x] of its own base value automatically.
Everything else (grid, dt, number_time_steps, quadrant, solver flags) is
copied verbatim.

Note: sampling pressureLow/pressureHigh in these ranges can never invert the
shock (low > high), because the two ranges never overlap -- e.g. base
pressure is [10000, 100000], so pressureLow samples from [5000, 20000] and
pressureHigh from [50000, 200000]; density, sharing the same ratio, follows
suit.

Writes one directory per case:
    runs/case_0000/ShockTube.input
    runs/case_0001/ShockTube.input
    ...
and a manifest (runs/manifest.csv) recording the sampled/derived parameters
per case, for provenance and for tagging training data by case later.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

from scipy.stats.qmc import LatinHypercube

SWEPT_PARAMS = ("SHOCK.densityLow", "SHOCK.densityHigh", "SHOCK.pressureLow", "SHOCK.pressureHigh")
SAMPLED_PARAMS = ("SHOCK.pressureLow", "SHOCK.pressureHigh")
_RATIO_OF = {
    "SHOCK.densityLow": "SHOCK.pressureLow",
    "SHOCK.densityHigh": "SHOCK.pressureHigh",
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
    ap.add_argument("--n-cases", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    base_path = Path(args.base_input)
    base_values = parse_base_input(base_path)
    base_text = base_path.read_text()
    print("Base values:", base_values)

    density_pressure_ratio = {
        density_param: base_values[density_param] / base_values[pressure_param]
        for density_param, pressure_param in _RATIO_OF.items()
    }
    print("Fixed density/pressure ratios (held constant across all cases):", density_pressure_ratio)

    bounds = {p: (0.5 * base_values[p], 2.0 * base_values[p]) for p in SAMPLED_PARAMS}
    print("Sample ranges:", bounds)

    sampler = LatinHypercube(d=len(SAMPLED_PARAMS), seed=args.seed)
    unit_samples = sampler.random(n=args.n_cases)  # (n_cases, 2) in [0,1)

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    for case_idx, unit_row in enumerate(unit_samples):
        case_id = f"case_{case_idx:04d}"
        params = {}
        for param, u in zip(SAMPLED_PARAMS, unit_row):
            lo, hi = bounds[param]
            params[param] = lo + u * (hi - lo)
        for density_param, pressure_param in _RATIO_OF.items():
            params[density_param] = density_pressure_ratio[density_param] * params[pressure_param]

        case_dir = out_root / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        (case_dir / "ShockTube.input").write_text(render_case_input(base_text, params))

        manifest_rows.append({"case_id": case_id, **params})

    manifest_path = out_root / "manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["case_id", *SWEPT_PARAMS])
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"Wrote {args.n_cases} cases under {out_root}/ and manifest to {manifest_path}")


if __name__ == "__main__":
    main()
