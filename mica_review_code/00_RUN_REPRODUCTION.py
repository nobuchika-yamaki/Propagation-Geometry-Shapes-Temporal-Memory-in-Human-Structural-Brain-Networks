#!/usr/bin/env python3
"""Run the frozen MICA-MICs analysis pipeline used for the manuscript.

This convenience driver does not implement any scientific analysis itself. It calls the
frozen analysis scripts in the required order with explicit input/output paths.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

SCRIPTS = [
    "05_MICA_MICS_GEOMETRY_SPECIFIC_SCALE_VALIDATION.py",
    "06_MICA_MICS_MEMORY_DISTRIBUTION_AND_FALSIFICATION.py",
    "07_MICA_MICS_MEMORY_MAXLAG1024_EXTENSION_v1_3.py",
    "08_MICA_MICS_EMPIRICAL_SC_WEIGHT_ROBUSTNESS_v1_2.py",
    "09_MICA_MICS_CONDUCTION_COMPENSATION_ROBUSTNESS_v1_1.py",
]


def run(cmd: list[str]) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Reproduce the MICA-MICs manuscript analyses")
    p.add_argument("--data-root", required=True,
                   help="Path to MICA-MICs MICs_release/derivatives/micapipe")
    p.add_argument("--out-root", default="./reproduction_outputs",
                   help="Parent directory for all generated outputs")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--self-tests-only", action="store_true")
    p.add_argument("--skip-self-tests", action="store_true")
    a = p.parse_args()

    if a.workers < 1:
        raise ValueError("--workers must be >= 1")

    py = sys.executable
    core = HERE / SCRIPTS[0]
    out_root = Path(a.out_root).expanduser().resolve()
    data_root = Path(a.data_root).expanduser().resolve()

    if not data_root.exists() and not a.self_tests_only:
        raise FileNotFoundError(f"MICA-MICs data root not found: {data_root}")

    if not a.skip_self_tests:
        for name in SCRIPTS:
            script = HERE / name
            cmd = [py, str(script), "--self-test"]
            if name.startswith(("07_", "08_", "09_")):
                cmd += ["--core-script", str(core)]
            run(cmd)

    if a.self_tests_only:
        return

    out_root.mkdir(parents=True, exist_ok=True)
    out05 = out_root / "05_primary_equal"
    out06 = out_root / "06_iid_and_descriptive"
    out07 = out_root / "07_extended_memory"
    out08 = out_root / "08_empirical_sc_weighting"
    out09 = out_root / "09_conduction_compensation"

    # Primary equal-weight analysis: memory, persistent-environment task, and A/B/C tests.
    run([
        py, str(core),
        "--mode", "full", "--phase", "all", "--weighting", "equal",
        "--data-root", str(data_root), "--outdir", str(out05),
        "--workers", str(a.workers),
    ])

    # Descriptive memory/performance summaries and IID temporal-dependence falsification.
    run([
        py, str(HERE / SCRIPTS[1]),
        "--phase", "all", "--analysis-root", str(out05),
        "--data-root", str(data_root), "--core-script", str(core),
        "--outdir", str(out06), "--workers", str(a.workers),
    ])

    # Extended memory spectrum to lag 1024. Clean reproduction does not reuse legacy checkpoints.
    run([
        py, str(HERE / SCRIPTS[2]),
        "--core-script", str(core), "--data-root", str(data_root),
        "--outdir", str(out07),
        "--old128-profile", str(out06 / "06_01_lag_profile_subject.csv"),
        "--no-reuse-v1-checkpoints", "--workers", str(a.workers),
    ])

    # Empirical structural-connectivity weighting robustness.
    run([
        py, str(HERE / SCRIPTS[3]),
        "--phase", "all", "--core-script", str(core),
        "--data-root", str(data_root), "--outdir", str(out08),
        "--equal-root", str(out05), "--equal-extended-root", str(out07),
        "--workers", str(a.workers),
    ])

    # Distance-dependent conduction-compensation sensitivity analysis.
    run([
        py, str(HERE / SCRIPTS[4]),
        "--phase", "all", "--core-script", str(core),
        "--data-root", str(data_root), "--outdir", str(out09),
        "--equal-root", str(out05), "--equal-extended-root", str(out07),
        "--workers", str(a.workers),
    ])

    print(f"\nReproduction pipeline completed. Outputs: {out_root}", flush=True)


if __name__ == "__main__":
    main()
