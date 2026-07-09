"""Run main_gpu for every case produced by generate_cases.py, and collect
each case's CGNS output under plot/<case_id>/.

main_gpu always writes to a hardcoded "plot/" relative to its cwd, so each
case is run with cwd set to its own runs/<case_id>/ directory (keeping
outputs from colliding), then that case's plot/ is moved up into the
top-level plot/<case_id>/ so ml/dataset.py can find it.

Must be run on the machine with the Linux/CUDA build of main_gpu -- this
script only orchestrates subprocess calls, it doesn't run the solver itself.

Usage:
    python3 -m ml.run_sweep --runs-dir runs --main-gpu ./main_gpu --plot-dir plot
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def run_case(case_dir: Path, main_gpu: Path, plot_root: Path) -> bool:
    case_id = case_dir.name
    dest = plot_root / case_id
    if dest.exists() and any(dest.iterdir()):
        print(f"[{case_id}] already has output in {dest}, skipping")
        return True

    log_path = case_dir / "run.log"
    print(f"[{case_id}] running {main_gpu} ShockTube.input (cwd={case_dir}) ...")
    with open(log_path, "w") as log:
        result = subprocess.run(
            [str(main_gpu), "ShockTube.input"],
            cwd=case_dir,
            stdout=log,
            stderr=subprocess.STDOUT,
        )

    if result.returncode != 0:
        print(f"[{case_id}] FAILED (exit {result.returncode}) -- see {log_path}")
        return False

    case_plot_dir = case_dir / "plot"
    if not case_plot_dir.exists():
        print(f"[{case_id}] main_gpu exited 0 but produced no plot/ dir -- see {log_path}")
        return False

    dest.mkdir(parents=True, exist_ok=True)
    for f in case_plot_dir.iterdir():
        shutil.move(str(f), str(dest / f.name))
    case_plot_dir.rmdir()

    print(f"[{case_id}] OK -> {dest}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--main-gpu", default="./main_gpu")
    ap.add_argument("--plot-dir", default="plot")
    args = ap.parse_args()

    main_gpu = Path(args.main_gpu).resolve()
    if not main_gpu.exists():
        raise FileNotFoundError(f"main_gpu binary not found at {main_gpu}")

    runs_root = Path(args.runs_dir)
    case_dirs = sorted(d for d in runs_root.iterdir() if d.is_dir() and d.name.startswith("case_"))
    if not case_dirs:
        raise FileNotFoundError(f"No case_* directories under {runs_root} -- run generate_cases.py first")

    plot_root = Path(args.plot_dir)
    plot_root.mkdir(parents=True, exist_ok=True)

    failures = []
    for case_dir in case_dirs:
        ok = run_case(case_dir, main_gpu, plot_root)
        if not ok:
            failures.append(case_dir.name)

    print(f"\nDone: {len(case_dirs) - len(failures)}/{len(case_dirs)} cases succeeded")
    if failures:
        print("Failed cases:", ", ".join(failures))


if __name__ == "__main__":
    main()
