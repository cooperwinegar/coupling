"""Generate a Latin-Hypercube sweep of ShockTube.input cases.

Varies exactly 4 parameters -- SHOCK.densityLow, SHOCK.densityHigh,
SHOCK.pressureLow, SHOCK.pressureHigh -- each independently over
[0.5x, 2x] of its value in the base input file. Everything else
(grid, dt, number_time_steps, quadrant, solver flags) is copied verbatim.

Note: sampling each parameter independently in these ranges can never
invert the shock (low > high), because the two ranges never overlap --
e.g. base density is [0.125, 1.0], so densityLow samples from
[0.0625, 0.25] and densityHigh from [0.5, 2.0].

Writes one directory per case:
    runs/case_0000/ShockTube.input
    runs/case_0001/ShockTube.input
    ...
and a manifest (runs/manifest.csv) recording the sampled parameters per case,
for provenance and for tagging training data by case later.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

from scipy.stats.qmc import LatinHypercube

SWEPT_PARAMS = ("SHOCK.densityLow", "SHOCK.densityHigh", "SHOCK.pressureLow", "SHOCK.pressureHigh")


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

    bounds = {p: (0.5 * v, 2.0 * v) for p, v in base_values.items()}
    print("Sample ranges:", bounds)

    sampler = LatinHypercube(d=len(SWEPT_PARAMS), seed=args.seed)
    unit_samples = sampler.random(n=args.n_cases)  # (n_cases, 4) in [0,1)

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    for case_idx, unit_row in enumerate(unit_samples):
        case_id = f"case_{case_idx:04d}"
        params = {}
        for param, u in zip(SWEPT_PARAMS, unit_row):
            lo, hi = bounds[param]
            params[param] = lo + u * (hi - lo)

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
