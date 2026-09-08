# MICA-MICs analysis code for peer review

This directory contains the frozen analysis scripts underlying the manuscript analyses. The original scientific script filenames and version identifiers are retained so that inter-script version checks and imports remain auditable. `00_RUN_REPRODUCTION.py` is a convenience driver only; it contains no scientific analysis logic.

## Data required

The analyses use the publicly released MICA-MICs structural-connectivity and tract-length derivatives. The required input is the MICA-MICs `MICs_release/derivatives/micapipe` directory. Resting-state functional MRI is not used.

The analysis expects the full released cohort used in the manuscript (50 participants) and the Schaefer-200 structural-connectivity and tract-length derivatives discoverable by the frozen core script.

## Files and manuscript role

- `05_MICA_MICS_GEOMETRY_SPECIFIC_SCALE_VALIDATION.py` — frozen primary analysis (`SCRIPT_VERSION = 05-mica-geometry-specific-v1.0`). Builds the participant-specific Schaefer-200 recurrent networks, implements GEOMETRIC, PAIR_SHUFFLED, PAIR_NARROW, and ZERO delay conditions, runs the lag-128 memory assay and persistent-environment task, performs validation-only propagation-scale selection, and computes the prespecified A/B/C participant-level tests.
- `06_MICA_MICS_MEMORY_DISTRIBUTION_AND_FALSIFICATION.py` — descriptive lag/profile and performance-surface analyses plus the IID temporal-dependence falsification (`06-mica-memory-distribution-falsification-v1.0`). The paired observed-minus-IID contrasts reported in the manuscript are generated here.
- `07_MICA_MICS_MEMORY_MAXLAG1024_EXTENSION_v1_3.py` — frozen extended-memory analysis (`07-mica-memory-maxlag1024-v1.3`). Extends held-out memory measurement to lag 1024 using explicit independently seeded prehistory while preserving the original analyzed input sequence.
- `08_MICA_MICS_EMPIRICAL_SC_WEIGHT_ROBUSTNESS_v1_2.py` — frozen empirical structural-connectivity weighting analysis (`08-mica-empirical-sc-weight-robustness-v1.2`). Uses log(1 + SC) incoming-weight normalization and otherwise retains the primary design.
- `09_MICA_MICS_CONDUCTION_COMPENSATION_ROBUSTNESS_v1_1.py` — frozen conduction-compensation sensitivity analysis (`09-mica-conduction-compensation-robustness-v1.1`). Evaluates beta = 0, 0.25, 0.50, 0.75, and 1.00, where beta progressively removes relative tract-length dependence of propagation delay.
- `00_RUN_REPRODUCTION.py` — convenience wrapper that executes the five frozen scripts in dependency order.
- `requirements.txt` — exact package versions used for the included code audit.
- `SELF_TEST_RESULTS.txt` — self-test output obtained from all five frozen analysis scripts before packaging.
- `MANIFEST_SHA256.txt` — SHA-256 checksums for package files.

## Manuscript output map

- Fig. 1: primary temporal-depth output from script 05 plus lag-1024 capacity/spectrum output from script 07.
- Fig. 2: persistent-environment validation/test outputs from script 05, with performance-surface summaries from script 06.
- Fig. 3: GEOMETRIC / PAIR_SHUFFLED / PAIR_NARROW comparisons from scripts 05 and 07.
- Fig. 4: IID temporal-dependence control and paired observed-minus-IID contrasts from script 06.
- Fig. 5A-B: empirical structural-connectivity weighting from script 08.
- Fig. 5C-D: conduction-compensation sensitivity analysis from script 09.
- Table 2: prespecified A/B/C inferential families from script 05.

## Validated environment

The package was audited with Python 3.13.5 and the package versions in `requirements.txt`. NumPy, pandas, and SciPy are required. Numba is used by the core when available and materially improves runtime.

## First check: self-tests

From this directory:

```bash
python3 00_RUN_REPRODUCTION.py \
  --data-root /path/to/MICs_release/derivatives/micapipe \
  --self-tests-only
```

The data path is not accessed when `--self-tests-only` is used, but the argument is retained for a single consistent command interface. All five included frozen scripts passed their internal self-tests in the packaged state; the exact output is recorded in `SELF_TEST_RESULTS.txt`.

## Full reproduction

```bash
python3 00_RUN_REPRODUCTION.py \
  --data-root /path/to/MICs_release/derivatives/micapipe \
  --out-root ./reproduction_outputs \
  --workers 4
```

The driver runs the analyses in this order:

1. primary equal-weight analysis;
2. descriptive analyses and IID temporal-dependence falsification;
3. extended memory spectrum through lag 1024;
4. empirical structural-connectivity weighting robustness;
5. conduction-compensation sensitivity analysis.

The scientific parameter grids are frozen inside the individual analysis scripts. The wrapper changes only filesystem paths and worker count.

## Direct execution

Each script can also be executed independently. Run `python3 <script> --help` for the complete command-line interface. For the primary manuscript analysis, use `--mode full`; smoke mode is a computational check and is not used for scientific inference.

## Reproducibility notes

- Participant is the independent statistical unit (N = 50).
- Primary propagation scales: S = 1, 2, 4, 8, 16, 32, 64.
- Recurrent coupling values: r = 0.5, 0.7, 0.9.
- Environmental persistence values: T_env = 4, 8, 16, 32, 64.
- Internal realizations: 8 per participant.
- Primary temporal depth uses lags 1–128; the extended capacity analysis uses lags 1–1024.
- Validation data select S*; held-out test data never select S*.
- The propagation scale S and conduction-compensation beta are dimensionless model parameters and are not estimates of anatomical brain size or absolute conduction time.
- The primary analysis uses equal incoming absolute-weight normalization. Empirical structural-connectivity weighting is a separate robustness analysis.
- The IID control preserves the binary marginal distribution, observation noise, scale grid, network conditions, validation-based selection rule, and test procedure while removing temporal dependence.

## What is not included

No participant-identifying data or empirical functional time series are included. The public MICA-MICs structural derivatives must be obtained separately. The package contains analysis code only and does not alter the released dataset.
