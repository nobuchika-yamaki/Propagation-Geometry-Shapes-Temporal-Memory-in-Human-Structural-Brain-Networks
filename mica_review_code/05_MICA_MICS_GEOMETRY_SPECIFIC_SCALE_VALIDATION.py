#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
05_MICA_MICS_GEOMETRY_SPECIFIC_SCALE_VALIDATION.py

Corrected MICA-MICs validation of the Physical-Scale hypothesis.

Scientific policy
-----------------
This script is intentionally MICA-MICs-specific. It is NOT a universal data loader.
It tests the source-model effects on MICA-MICs after correcting known weaknesses in
old 04, without tuning scientific parameters to MICA outcomes.

The analysis has three pre-specified inferential families:

A) SOURCE-EFFECT REPLICATION (continuity with the source hypothesis)
   A1. GEOMETRIC TD slope versus log2(S) > 0.
   A2. GEOMETRIC rho(log2(T_env), log2(S*)) > 0.
   A3. Slope of held-out (GEOMETRIC - ZERO) accuracy versus log2(T_env) > 0.

B) ANATOMICAL-ASSIGNMENT MECHANISM
   B1. TD-scaling slope: GEOMETRIC > PAIR_SHUFFLED.
   B2. Demand matching: rho_geo > rho_pair_shuffled.
   B3. Functional value: slope[(accuracy_geo - accuracy_pair_shuffled) vs log2(T_env)] > 0.

C) DELAY-HETEROGENEITY MECHANISM
   C1. TD-scaling slope: GEOMETRIC > PAIR_NARROW.
   C2. Demand matching: rho_geo > rho_pair_narrow.
   C3. Functional value: slope[(accuracy_geo - accuracy_pair_narrow) vs log2(T_env)] > 0.

Why these controls are non-circular
-----------------------------------
ZERO removes additional propagation delay only.
PAIR_SHUFFLED preserves the exact undirected tract-delay multiset at every S but
permutes one fixed tract assignment per subject x replicate across ALL scales.
Thus scale is not confounded with a new shuffle at each S.
PAIR_NARROW preserves the exact total undirected integer delay at every S while
minimizing variance. q+1 assignments are placed on tracts with the largest original
geometric delays, preserving anatomical rank as far as mathematically possible.
Thus it isolates heterogeneity without deliberately randomizing anatomy.

Known corrections relative to old 04
------------------------------------
1. Delay controls operate on undirected anatomical tract pairs, so both directions
   of one tract always receive the same delay.
2. PAIR_SHUFFLED uses one fixed continuous-geometry permutation across S, preventing
   the old scale-by-randomization confound.
3. Delay normalization uses Lmax among RETAINED model edges only. Non-model edges
   cannot set the delay scale of the simulated network.
4. Robust triangular-matrix symmetrization does not halve one-sided release values.
5. Scientific parameters are constants, not command-line tuning knobs.
6. Primary equal-weight transplant and pre-specified empirical-SC-weight robustness
   use identical tasks, scales, couplings, seeds and controls.

The empirical fMRI data are not used.
The model is discrete-time and must not be described as a DDE.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except Exception:
    NUMBA_AVAILABLE = False
    njit = None


# =============================================================================
# FROZEN SCIENTIFIC CONSTANTS -- NOT CLI-TUNABLE
# =============================================================================
SCRIPT_VERSION = "05-mica-geometry-specific-v1.0"
SEED_NAMESPACE = "MICA_GEOMETRY_SPECIFIC_FROZEN_V1"
ATLAS_SCALE = 200
SOURCE_CONNECTION_FRACTION = 16.0 / 127.0
TARGET_MEAN_DEGREE = int(round(SOURCE_CONNECTION_FRACTION * (ATLAS_SCALE - 1)))  # 25
SCALES = np.asarray([1, 2, 4, 8, 16, 32, 64], dtype=np.int64)
COUPLINGS = np.asarray([0.5, 0.7, 0.9], dtype=np.float64)
T_ENVS = np.asarray([4, 8, 16, 32, 64], dtype=np.int64)
FULL_INTERNAL_REPS = 8
FULL_MEMORY_WASHOUT = 1000
FULL_MEMORY_TRAIN = 10000
FULL_MEMORY_TEST = 10000
FULL_MAX_LAG = 128
FULL_ENV_WASHOUT = 1000
FULL_ENV_TRAIN = 10000
FULL_ENV_VAL = 5000
FULL_ENV_TEST = 10000
ENV_NOISE_SD = 2.0
FULL_BOOTSTRAP_SAMPLES = 10000
FULL_PERMUTATION_SAMPLES = 10000

DEFAULT_DATA_ROOT = "~/Downloads/mica-mics/MICs_release/derivatives/micapipe"
DEFAULT_OUT_ROOT = "~/Desktop/05_MICA_MICS_GEOMETRY_SPECIFIC_SCALE_VALIDATION"

CONDITIONS = ("geometric", "pair_shuffled", "pair_narrow", "zero")


@dataclass(frozen=True)
class Config:
    mode: str
    phase: str
    weighting: str
    data_root: str
    outdir: str
    workers: int
    internal_reps: int
    memory_washout: int
    memory_train: int
    memory_test: int
    max_lag: int
    env_washout: int
    env_train: int
    env_val: int
    env_test: int
    bootstrap_samples: int
    permutation_samples: int
    max_subjects: int
    checkpoint: bool


@dataclass(frozen=True)
class SubjectFiles:
    subject: str
    session: str
    sc_file: str
    length_file: str


@dataclass
class NetworkTemplate:
    n: int
    k_target: int
    src: np.ndarray
    dst: np.ndarray
    delta: np.ndarray                 # directed, paired consecutive
    selected_sc: np.ndarray           # directed, paired consecutive
    undirected_delta: np.ndarray      # one value per anatomical tract
    undirected_sc: np.ndarray
    undirected_i: np.ndarray
    undirected_j: np.ndarray
    retained_lengths_mm: np.ndarray
    lmax_retained_mm: float
    in_degree: np.ndarray


# =============================================================================
# Utilities
# =============================================================================
def stable_seed(*parts: object) -> int:
    s = "|".join(str(x) for x in (SEED_NAMESPACE,) + parts).encode("utf-8")
    return int(hashlib.sha256(s).hexdigest()[:16], 16) % (2**32 - 1)


def config_signature(cfg: Config) -> str:
    scientific = {
        "script_version": SCRIPT_VERSION,
        "seed_namespace": SEED_NAMESPACE,
        "atlas_scale": ATLAS_SCALE,
        "connection_fraction": SOURCE_CONNECTION_FRACTION,
        "target_mean_degree": TARGET_MEAN_DEGREE,
        "scales": SCALES.tolist(),
        "couplings": COUPLINGS.tolist(),
        "T_envs": T_ENVS.tolist(),
        "weighting": cfg.weighting,
        "internal_reps": cfg.internal_reps,
        "memory": [cfg.memory_washout, cfg.memory_train, cfg.memory_test, cfg.max_lag],
        "environment": [cfg.env_washout, cfg.env_train, cfg.env_val, cfg.env_test, ENV_NOISE_SD],
    }
    return hashlib.sha256(json.dumps(scientific, sort_keys=True).encode()).hexdigest()[:20]


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Corrected MICA-MICs geometry-specific physical-scale validation")
    p.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    p.add_argument("--phase", choices=("memory", "environment", "all"), default="memory")
    p.add_argument("--weighting", choices=("equal", "empirical"), default="equal",
                   help="equal = primary exact-transplant weighting; empirical = pre-specified robustness")
    p.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    p.add_argument("--outdir", default="")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--no-checkpoint", action="store_true")
    p.add_argument("--self-test", action="store_true")
    a = p.parse_args()

    if a.self_test:
        result = run_self_test()
        print(json.dumps(result, indent=2))
        raise SystemExit(0 if result.get("all_pass") else 2)

    if a.workers < 1:
        raise ValueError("--workers must be >= 1")

    if a.mode == "smoke":
        reps = 1
        mw, mtr, mte, ml = 300, 1200, 1200, 32
        ew, etr, ev, ete = 300, 1200, 600, 1200
        boot, perm, max_sub = 500, 1000, 3
    else:
        reps = FULL_INTERNAL_REPS
        mw, mtr, mte, ml = FULL_MEMORY_WASHOUT, FULL_MEMORY_TRAIN, FULL_MEMORY_TEST, FULL_MAX_LAG
        ew, etr, ev, ete = FULL_ENV_WASHOUT, FULL_ENV_TRAIN, FULL_ENV_VAL, FULL_ENV_TEST
        boot, perm, max_sub = FULL_BOOTSTRAP_SAMPLES, FULL_PERMUTATION_SAMPLES, 0

    outdir = a.outdir.strip()
    if not outdir:
        outdir = str(Path(DEFAULT_OUT_ROOT).expanduser() / a.weighting)

    return Config(
        mode=a.mode,
        phase=a.phase,
        weighting=a.weighting,
        data_root=str(Path(a.data_root).expanduser()),
        outdir=str(Path(outdir).expanduser()),
        workers=int(a.workers),
        internal_reps=reps,
        memory_washout=mw,
        memory_train=mtr,
        memory_test=mte,
        max_lag=ml,
        env_washout=ew,
        env_train=etr,
        env_val=ev,
        env_test=ete,
        bootstrap_samples=boot,
        permutation_samples=perm,
        max_subjects=max_sub,
        checkpoint=not a.no_checkpoint,
    )


# =============================================================================
# MICA-MICs input handling
# =============================================================================
def load_numeric_txt(path: Path) -> np.ndarray:
    last = None
    for delim in (",", None, "\t", ";"):
        try:
            arr = np.genfromtxt(path, delimiter=delim, dtype=float)
            if arr.size and np.isfinite(arr).any():
                arr = np.asarray(arr, dtype=np.float64)
                if arr.ndim == 1:
                    arr = arr.reshape(1, -1)
                return arr
        except Exception as e:
            last = e
    raise ValueError(f"no finite numeric values in {path}; last={last}")


def cortical_indices(full_n: int, atlas_n: int = ATLAS_SCALE) -> np.ndarray:
    if full_n == atlas_n:
        return np.arange(atlas_n, dtype=int)
    if full_n == 216 and atlas_n == 200:
        return np.r_[np.arange(15, 115), np.arange(116, 216)].astype(int)
    if full_n == 250 and atlas_n == 200:
        base = 48
        return np.r_[np.arange(base + 1, base + 101), np.arange(base + 102, base + 202)].astype(int)
    raise ValueError(f"unsupported matrix size {full_n} for atlas {atlas_n}")


def coerce_square_cortex(a: np.ndarray, atlas_n: int = ATLAS_SCALE) -> np.ndarray:
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"matrix must be square, got {a.shape}")
    idx = cortical_indices(a.shape[0], atlas_n)
    return np.asarray(a[np.ix_(idx, idx)], dtype=np.float64)


def robust_positive_symmetrize(a: np.ndarray) -> np.ndarray:
    """Average when both directions are positive; preserve the positive side when one-sided."""
    a = np.asarray(a, dtype=np.float64)
    a = np.where(np.isfinite(a), a, 0.0)
    pos = a > 0
    both = pos & pos.T
    one = pos ^ pos.T
    out = np.zeros_like(a, dtype=np.float64)
    out[both] = 0.5 * (a[both] + a.T[both])
    out[one] = np.maximum(a, a.T)[one]
    np.fill_diagonal(out, 0.0)
    return out


def discover_subjects(cfg: Config) -> List[SubjectFiles]:
    root = Path(cfg.data_root)
    if not root.exists():
        raise FileNotFoundError(f"data root does not exist: {root}")
    scs = sorted(root.rglob("*atlas-schaefer200_desc-sc.txt"))
    out: List[SubjectFiles] = []
    for sc in scs:
        subject = next((x for x in sc.parts if x.startswith("sub-")), None)
        session = next((x for x in sc.parts if x.startswith("ses-")), "ses-01")
        if not subject:
            continue
        lf = sc.with_name(sc.name.replace("_desc-sc.txt", "_desc-edgeLength.txt"))
        if lf.exists():
            out.append(SubjectFiles(subject, session, str(sc), str(lf)))
    uniq = {(x.subject, x.session): x for x in out}
    vals = [uniq[k] for k in sorted(uniq)]
    if cfg.max_subjects > 0:
        vals = vals[:cfg.max_subjects]
    if not vals:
        raise RuntimeError("No complete MICA-MICs SC + edgeLength subject found")
    if cfg.mode == "full" and len(vals) != 50:
        raise RuntimeError(f"full MICA analysis requires all 50 subjects; found {len(vals)}")
    return vals


def build_empirical_template(sf: SubjectFiles) -> NetworkTemplate:
    sc = coerce_square_cortex(load_numeric_txt(Path(sf.sc_file)))
    ln = coerce_square_cortex(load_numeric_txt(Path(sf.length_file)))
    if sc.shape != ln.shape:
        raise ValueError(f"SC/length shape mismatch {sc.shape} vs {ln.shape}")

    sc_sym = robust_positive_symmetrize(sc)
    ln_sym = robust_positive_symmetrize(ln)
    n = ATLAS_SCALE

    iu, ju = np.triu_indices(n, k=1)
    valid = (sc_sym[iu, ju] > 0) & (ln_sym[iu, ju] > 0)
    vi, vj = iu[valid], ju[valid]
    scores = sc_sym[vi, vj]

    target_undirected = int(round(n * TARGET_MEAN_DEGREE / 2.0))
    if scores.size < target_undirected:
        raise ValueError(f"only {scores.size} valid undirected edges; need {target_undirected}")

    # Deterministic strongest-edge selection. Stable sort plus canonical edge order.
    order = np.argsort(scores, kind="mergesort")[-target_undirected:]
    sel_i, sel_j = vi[order], vj[order]
    canon = np.lexsort((sel_j, sel_i))
    sel_i, sel_j = sel_i[canon], sel_j[canon]

    retained_lengths = ln_sym[sel_i, sel_j].astype(np.float64)
    retained_sc = sc_sym[sel_i, sel_j].astype(np.float64)
    if not np.all(np.isfinite(retained_lengths)) or np.any(retained_lengths <= 0):
        raise ValueError("retained edges contain invalid tract lengths")
    lmax = float(retained_lengths.max())
    if not np.isfinite(lmax) or lmax <= 0:
        raise ValueError("invalid retained-edge Lmax")
    delta_u = np.clip(retained_lengths / lmax, 0.0, 1.0)

    # Each undirected tract is represented as two consecutive directed recurrence edges.
    src = np.empty(2 * target_undirected, dtype=np.int32)
    dst = np.empty(2 * target_undirected, dtype=np.int32)
    src[0::2], dst[0::2] = sel_j, sel_i
    src[1::2], dst[1::2] = sel_i, sel_j
    delta = np.repeat(delta_u, 2).astype(np.float64)
    selected_sc = np.repeat(retained_sc, 2).astype(np.float64)

    indeg = np.bincount(dst, minlength=n).astype(np.int32)
    if np.any(indeg == 0):
        raise ValueError(f"selected graph contains {int(np.sum(indeg==0))} isolated cortical nodes")
    if not np.allclose(delta[0::2], delta[1::2]):
        raise AssertionError("tract-pair delta symmetry failed")

    return NetworkTemplate(
        n=n,
        k_target=TARGET_MEAN_DEGREE,
        src=src,
        dst=dst,
        delta=delta,
        selected_sc=selected_sc,
        undirected_delta=delta_u,
        undirected_sc=retained_sc,
        undirected_i=sel_i.astype(np.int32),
        undirected_j=sel_j.astype(np.int32),
        retained_lengths_mm=retained_lengths,
        lmax_retained_mm=lmax,
        in_degree=indeg,
    )


# =============================================================================
# Delay architecture: pair-preserving and scale-consistent
# =============================================================================
def geometric_delays(delta: np.ndarray, scale: int) -> np.ndarray:
    return np.floor(np.asarray(delta, dtype=np.float64) * int(scale) + 0.5).astype(np.int32)


def expand_undirected_delays(d_u: np.ndarray) -> np.ndarray:
    return np.repeat(np.asarray(d_u, dtype=np.int32), 2)


def pair_shuffled_delta(delta_u: np.ndarray, seed: int) -> np.ndarray:
    """One fixed permutation of continuous tract geometry, reused at every scale."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(delta_u))
    out = np.asarray(delta_u, dtype=np.float64)[perm]
    if not np.allclose(np.sort(out), np.sort(delta_u), rtol=0, atol=0):
        raise AssertionError("pair-shuffled continuous geometry multiset changed")
    return out


def pair_narrow_undirected(geo_u: np.ndarray, delta_u: np.ndarray) -> np.ndarray:
    """
    Preserve exact total integer delay and minimize variance.
    q+1 values are assigned to tracts with the largest original geometric delays;
    ties are resolved by continuous geometric delta then canonical edge index.
    This avoids introducing a second random anatomical scrambling.
    """
    geo_u = np.asarray(geo_u, dtype=np.int32)
    delta_u = np.asarray(delta_u, dtype=np.float64)
    E = geo_u.size
    total = int(np.sum(geo_u, dtype=np.int64))
    q, m = divmod(total, E)
    out = np.full(E, q, dtype=np.int32)
    if m:
        # ascending lexsort, then take the largest m; edge index gives deterministic final tie break.
        idx = np.arange(E)
        order = np.lexsort((idx, delta_u, geo_u))
        high = order[-m:]
        out[high] = q + 1
    if int(out.sum()) != total:
        raise AssertionError("PAIR_NARROW failed exact total-delay preservation")
    if out.max() - out.min() > 1:
        raise AssertionError("PAIR_NARROW did not achieve minimum integer variance")
    return out


def delay_bundle(template: NetworkTemplate, shuffled_delta_u: np.ndarray, scale: int) -> Dict[str, np.ndarray]:
    geo_u = geometric_delays(template.undirected_delta, scale)
    shf_u = geometric_delays(shuffled_delta_u, scale)
    nar_u = pair_narrow_undirected(geo_u, template.undirected_delta)

    # Exact multiset match at each S despite using one fixed continuous permutation across S.
    if not np.array_equal(np.sort(geo_u), np.sort(shf_u)):
        raise AssertionError("PAIR_SHUFFLED integer-delay multiset mismatch")
    if int(geo_u.sum()) != int(nar_u.sum()):
        raise AssertionError("PAIR_NARROW total mismatch")

    return {
        "geometric": expand_undirected_delays(geo_u),
        "pair_shuffled": expand_undirected_delays(shf_u),
        "pair_narrow": expand_undirected_delays(nar_u),
        "zero": np.zeros(template.src.size, dtype=np.int32),
    }


# =============================================================================
# Dynamics and readouts
# =============================================================================
def make_signs_and_input(template: NetworkTemplate, subject: str, rep: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(stable_seed(subject, rep, "network_randomness"))
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=template.src.size).astype(np.float64)
    b = rng.choice(np.asarray([-1.0, 1.0]), size=template.n).astype(np.float64)
    return signs, b


def base_weights(template: NetworkTemplate, signs: np.ndarray, weighting: str) -> np.ndarray:
    if weighting == "equal":
        denom = template.in_degree[template.dst].astype(np.float64)
        base = signs / denom
    elif weighting == "empirical":
        mag = np.log1p(np.maximum(template.selected_sc, 0.0))
        denom_by_node = np.bincount(template.dst, weights=mag, minlength=template.n).astype(np.float64)
        if np.any(denom_by_node <= 0):
            raise ValueError("empirical weighting has zero incoming magnitude at a node")
        base = signs * mag / denom_by_node[template.dst]
    else:
        raise ValueError(f"unknown weighting {weighting}")

    sums = np.bincount(template.dst, weights=np.abs(base), minlength=template.n)
    if not np.allclose(sums, 1.0, rtol=1e-12, atol=1e-12):
        raise AssertionError("incoming absolute recurrent-weight normalization failed")
    return base.astype(np.float64)


if NUMBA_AVAILABLE:
    @njit(cache=True)
    def _simulate_linear_numba(src, dst, base_weight, delays, b, u, r):
        n = b.shape[0]
        T = u.shape[0]
        out = np.zeros((T, n), dtype=np.float64)
        for t in range(T):
            cur = np.empty(n, dtype=np.float64)
            for i in range(n):
                cur[i] = b[i] * u[t]
            for e in range(src.shape[0]):
                tt = t - 1 - delays[e]
                if tt >= 0:
                    cur[dst[e]] += r * base_weight[e] * out[tt, src[e]]
            for i in range(n):
                out[t, i] = cur[i]
        return out
else:
    _simulate_linear_numba = None


def simulate_linear(template: NetworkTemplate, base: np.ndarray, delays: np.ndarray,
                    b: np.ndarray, u: np.ndarray, r: float) -> np.ndarray:
    if NUMBA_AVAILABLE:
        return _simulate_linear_numba(template.src, template.dst, base, delays, b, u, float(r))
    out = np.zeros((len(u), template.n), dtype=np.float64)
    for t in range(len(u)):
        cur = b * u[t]
        for e in range(template.src.size):
            tt = t - 1 - int(delays[e])
            if tt >= 0:
                cur[template.dst[e]] += float(r) * base[e] * out[tt, template.src[e]]
        out[t] = cur
    return out


def ols_fit_centered(X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    xm = X.mean(axis=0)
    ym = Y.mean(axis=0)
    Xc = X - xm
    Yc = Y - ym
    B = np.linalg.pinv(Xc.T @ Xc, hermitian=True) @ (Xc.T @ Yc)
    a = ym - xm @ B
    return a, B


def memory_targets(u: np.ndarray, start: int, length: int, max_lag: int) -> np.ndarray:
    idx = np.arange(start, start + length)
    return np.column_stack([u[idx - k] for k in range(1, max_lag + 1)])


def memory_metrics(template: NetworkTemplate, base: np.ndarray, b: np.ndarray, delays: np.ndarray,
                   r: float, train_u: np.ndarray, test_u: np.ndarray, cfg: Config) -> Tuple[np.ndarray, float, float]:
    xtr = simulate_linear(template, base, delays, b, train_u, r)
    xte = simulate_linear(template, base, delays, b, test_u, r)
    s = cfg.memory_washout
    Xtr = xtr[s:s + cfg.memory_train]
    Xte = xte[s:s + cfg.memory_test]
    Ytr = memory_targets(train_u, s, cfg.memory_train, cfg.max_lag)
    Yte = memory_targets(test_u, s, cfg.memory_test, cfg.max_lag)
    a, B = ols_fit_centered(Xtr, Ytr)
    pred = a + Xte @ B
    M = np.zeros(cfg.max_lag, dtype=np.float64)
    for q in range(cfg.max_lag):
        y, p = Yte[:, q], pred[:, q]
        if y.std() > 0 and p.std() > 0:
            c = np.corrcoef(y, p)[0, 1]
            M[q] = float(c * c) if np.isfinite(c) else 0.0
    MC = float(M.sum())
    lags = np.arange(1, cfg.max_lag + 1, dtype=np.float64)
    TD = float((lags @ M) / MC) if MC > 0 else float("nan")
    return M, MC, TD


def fit_scalar_readout(X: np.ndarray, y: np.ndarray) -> Tuple[float, np.ndarray]:
    xm, ym = X.mean(axis=0), float(y.mean())
    Xc, yc = X - xm, y - ym
    beta = np.linalg.pinv(Xc.T @ Xc, hermitian=True) @ (Xc.T @ yc)
    a = ym - float(xm @ beta)
    return a, beta


def accuracy_and_mse(X: np.ndarray, z: np.ndarray, a: float, beta: np.ndarray) -> Tuple[float, float]:
    pred = a + X @ beta
    cls = np.where(pred >= 0.0, 1.0, -1.0)
    return float(np.mean(cls == z)), float(np.mean((pred - z) ** 2))


def make_environment_sequence(total: int, Tenv: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    z = np.empty(total, dtype=np.float64)
    z[0] = 1.0 if rng.random() < 0.5 else -1.0
    h = 1.0 / float(Tenv)
    flips = rng.random(total - 1) < h
    for t in range(1, total):
        z[t] = -z[t - 1] if flips[t - 1] else z[t - 1]
    u = z + rng.normal(0.0, ENV_NOISE_SD, size=total)
    return z, u


def env_metrics(template: NetworkTemplate, base: np.ndarray, b: np.ndarray, delays: np.ndarray,
                r: float, seqs: Dict[str, Tuple[np.ndarray, np.ndarray]], cfg: Config) -> Tuple[float, float, float, float]:
    Xs, zs = {}, {}
    lengths = {"train": cfg.env_train, "val": cfg.env_val, "test": cfg.env_test}
    for part in ("train", "val", "test"):
        z, u = seqs[part]
        x = simulate_linear(template, base, delays, b, u, r)
        s = cfg.env_washout
        Xs[part] = x[s:s + lengths[part]]
        zs[part] = z[s:s + lengths[part]]
    a, beta = fit_scalar_readout(Xs["train"], zs["train"])
    va, vm = accuracy_and_mse(Xs["val"], zs["val"], a, beta)
    ta, tm = accuracy_and_mse(Xs["test"], zs["test"], a, beta)
    return va, vm, ta, tm


# =============================================================================
# Checkpoints and workers
# =============================================================================
def checkpoint_path(cfg: Config, sf: SubjectFiles, rep: int, phase: str) -> Path:
    d = Path(cfg.outdir) / "checkpoints"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{sf.subject}_{sf.session}_rep{rep:02d}_{phase}_{config_signature(cfg)}.json"


def load_checkpoint_if_valid(path: Path, cfg: Config) -> Optional[Dict[str, object]]:
    if not cfg.checkpoint or not path.exists():
        return None
    try:
        z = json.loads(path.read_text())
        if z.get("config_signature") == config_signature(cfg):
            return z
    except Exception:
        return None
    return None


def memory_worker(payload) -> Dict[str, object]:
    sf, cfg_dict, rep = payload
    cfg = Config(**cfg_dict)
    cp = checkpoint_path(cfg, sf, rep, "memory")
    old = load_checkpoint_if_valid(cp, cfg)
    if old is not None:
        return old

    template = build_empirical_template(sf)
    signs, b = make_signs_and_input(template, sf.subject, rep)
    base = base_weights(template, signs, cfg.weighting)
    shuf_delta = pair_shuffled_delta(template.undirected_delta, stable_seed(sf.subject, rep, "pair_shuffle_geometry"))

    rng_tr = np.random.default_rng(stable_seed(sf.subject, rep, "memory_train"))
    rng_te = np.random.default_rng(stable_seed(sf.subject, rep, "memory_test"))
    train_u = rng_tr.normal(0.0, 1.0, cfg.memory_washout + cfg.memory_train)
    test_u = rng_te.normal(0.0, 1.0, cfg.memory_washout + cfg.memory_test)

    rows, profiles = [], []
    zero_cache: Dict[float, Tuple[np.ndarray, float, float]] = {}
    zd = np.zeros(template.src.size, dtype=np.int32)
    for r in COUPLINGS:
        zero_cache[float(r)] = memory_metrics(template, base, b, zd, float(r), train_u, test_u, cfg)

    for S in SCALES:
        bundle = delay_bundle(template, shuf_delta, int(S))
        for cond in ("geometric", "pair_shuffled", "pair_narrow"):
            d = bundle[cond]
            for r in COUPLINGS:
                M, MC, TD = memory_metrics(template, base, b, d, float(r), train_u, test_u, cfg)
                rows.append({"condition": cond, "scale": int(S), "r": float(r), "MC": MC, "TD": TD})
                for k, mk in enumerate(M, start=1):
                    profiles.append({"condition": cond, "scale": int(S), "r": float(r), "lag": k, "M": float(mk)})
        for r in COUPLINGS:
            M, MC, TD = zero_cache[float(r)]
            rows.append({"condition": "zero", "scale": int(S), "r": float(r), "MC": MC, "TD": TD})
            for k, mk in enumerate(M, start=1):
                profiles.append({"condition": "zero", "scale": int(S), "r": float(r), "lag": k, "M": float(mk)})

    result = {
        "config_signature": config_signature(cfg),
        "subject": sf.subject,
        "session": sf.session,
        "rep": rep,
        "weighting": cfg.weighting,
        "lmax_retained_mm": template.lmax_retained_mm,
        "rows": rows,
        "profiles": profiles,
    }
    if cfg.checkpoint:
        cp.write_text(json.dumps(result))
    return result


def environment_worker(payload) -> Dict[str, object]:
    sf, cfg_dict, rep = payload
    cfg = Config(**cfg_dict)
    cp = checkpoint_path(cfg, sf, rep, "environment")
    old = load_checkpoint_if_valid(cp, cfg)
    if old is not None:
        return old

    template = build_empirical_template(sf)
    signs, b = make_signs_and_input(template, sf.subject, rep)
    base = base_weights(template, signs, cfg.weighting)
    shuf_delta = pair_shuffled_delta(template.undirected_delta, stable_seed(sf.subject, rep, "pair_shuffle_geometry"))

    # Delay architectures depend on S but NOT on T_env.
    bundles = {int(S): delay_bundle(template, shuf_delta, int(S)) for S in SCALES}

    rows = []
    for Tenv in T_ENVS:
        seqs = {}
        for part, L in (("train", cfg.env_train), ("val", cfg.env_val), ("test", cfg.env_test)):
            seed = stable_seed(sf.subject, rep, "env", int(Tenv), part)
            seqs[part] = make_environment_sequence(cfg.env_washout + L, int(Tenv), seed)

        for S in SCALES:
            bundle = bundles[int(S)]
            for cond in ("geometric", "pair_shuffled", "pair_narrow"):
                d = bundle[cond]
                for r in COUPLINGS:
                    va, vm, ta, tm = env_metrics(template, base, b, d, float(r), seqs, cfg)
                    rows.append({
                        "condition": cond, "T_env": int(Tenv), "scale": int(S), "r": float(r),
                        "val_accuracy": va, "val_mse": vm, "test_accuracy": ta, "test_mse": tm,
                    })

        # ZERO is S-invariant; simulate once per r and duplicate across nominal S
        zd = np.zeros(template.src.size, dtype=np.int32)
        for r in COUPLINGS:
            va, vm, ta, tm = env_metrics(template, base, b, zd, float(r), seqs, cfg)
            for S in SCALES:
                rows.append({
                    "condition": "zero", "T_env": int(Tenv), "scale": int(S), "r": float(r),
                    "val_accuracy": va, "val_mse": vm, "test_accuracy": ta, "test_mse": tm,
                })

    result = {
        "config_signature": config_signature(cfg),
        "subject": sf.subject,
        "session": sf.session,
        "rep": rep,
        "weighting": cfg.weighting,
        "rows": rows,
    }
    if cfg.checkpoint:
        cp.write_text(json.dumps(result))
    return result


# =============================================================================
# Aggregation
# =============================================================================
def slope_y_on_log2s(scales: Sequence[float], y: Sequence[float]) -> float:
    x = np.log2(np.asarray(scales, dtype=float))
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 2:
        return float("nan")
    return float(np.polyfit(x[ok], y[ok], 1)[0])


def auc_log2_scale(scales: Sequence[float], y: Sequence[float]) -> float:
    x = np.log2(np.asarray(scales, dtype=float))
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 2:
        return float("nan")
    order = np.argsort(x[ok])
    return float(np.trapezoid(y[ok][order], x[ok][order]))


def rho_log_env_scale(Tenv: np.ndarray, Sstar: np.ndarray) -> float:
    if len(np.unique(Sstar)) == 1:
        return 0.0
    z = spearmanr(np.log2(Tenv.astype(float)), np.log2(Sstar.astype(float)))
    return float(z.statistic) if np.isfinite(z.statistic) else float("nan")


def aggregate_memory(results: List[Dict[str, object]], out: Path) -> pd.DataFrame:
    rows, profiles = [], []
    for z in results:
        for r in z["rows"]:
            rows.append({"subject": z["subject"], "session": z["session"], "rep": z["rep"], "weighting": z["weighting"], **r})
        for r in z["profiles"]:
            profiles.append({"subject": z["subject"], "session": z["session"], "rep": z["rep"], "weighting": z["weighting"], **r})
    df = pd.DataFrame(rows)
    df.to_csv(out / "05_01_memory_condition_level.csv", index=False)
    pd.DataFrame(profiles).to_csv(out / "05_02_memory_profiles.csv", index=False)

    subject_rows = []
    for sub, ds in df.groupby("subject"):
        acc: Dict[str, List[float]] = {
            "A1_geo_TD_slope": [],
            "B1_geo_minus_pairshuffled_TD_slope": [],
            "C1_geo_minus_pairnarrow_TD_slope": [],
            "geo_minus_pairshuffled_MC_AUC": [],
            "geo_minus_pairshuffled_TD_AUC": [],
            "geo_minus_pairnarrow_MC_AUC": [],
            "geo_minus_pairnarrow_TD_AUC": [],
        }
        for (rep, r), drr in ds.groupby(["rep", "r"]):
            byc = {c: drr[drr.condition == c].sort_values("scale") for c in CONDITIONS}
            slopes_td = {c: slope_y_on_log2s(byc[c].scale, byc[c].TD) for c in ("geometric", "pair_shuffled", "pair_narrow")}
            acc["A1_geo_TD_slope"].append(slopes_td["geometric"])
            acc["B1_geo_minus_pairshuffled_TD_slope"].append(slopes_td["geometric"] - slopes_td["pair_shuffled"])
            acc["C1_geo_minus_pairnarrow_TD_slope"].append(slopes_td["geometric"] - slopes_td["pair_narrow"])
            for q in ("MC", "TD"):
                auc_geo = auc_log2_scale(byc["geometric"].scale, byc["geometric"][q])
                auc_shf = auc_log2_scale(byc["pair_shuffled"].scale, byc["pair_shuffled"][q])
                auc_nar = auc_log2_scale(byc["pair_narrow"].scale, byc["pair_narrow"][q])
                acc[f"geo_minus_pairshuffled_{q}_AUC"].append(auc_geo - auc_shf)
                acc[f"geo_minus_pairnarrow_{q}_AUC"].append(auc_geo - auc_nar)
        subject_rows.append({"subject": sub, **{k: float(np.nanmean(v)) for k, v in acc.items()}})

    sdf = pd.DataFrame(subject_rows).sort_values("subject")
    sdf.to_csv(out / "05_03_memory_subject_effects.csv", index=False)
    return sdf


def choose_scale_from_validation(d: pd.DataFrame) -> Tuple[int, float, float, float]:
    d = d.sort_values("scale")
    vmax = float(d.val_accuracy.max())
    # Frozen source rule: exact tie -> smallest S. Test outcome never breaks ties.
    bestS = int(d.loc[d.val_accuracy == vmax, "scale"].min())
    z = d[d.scale == bestS].iloc[0]
    return bestS, float(z.val_accuracy), float(z.test_accuracy), float(z.test_mse)


def aggregate_environment(results: List[Dict[str, object]], out: Path) -> pd.DataFrame:
    rows = []
    for z in results:
        for r in z["rows"]:
            rows.append({"subject": z["subject"], "session": z["session"], "rep": z["rep"], "weighting": z["weighting"], **r})
    df = pd.DataFrame(rows)
    df.to_csv(out / "05_04_environment_condition_level.csv", index=False)

    selection_rows = []
    for (sub, rep, r, Tenv, cond), d in df.groupby(["subject", "rep", "r", "T_env", "condition"]):
        bestS, va, ta, tm = choose_scale_from_validation(d)
        selection_rows.append({
            "subject": sub, "rep": rep, "r": r, "T_env": Tenv, "condition": cond,
            "S_star": bestS, "val_accuracy_star": va, "test_accuracy_star": ta, "test_mse_star": tm,
        })
    sel = pd.DataFrame(selection_rows)
    sel.to_csv(out / "05_05_environment_selected_scales.csv", index=False)

    subject_rows = []
    for sub, ds in sel.groupby("subject"):
        acc: Dict[str, List[float]] = {
            "A2_geo_rho_Tenv_Sstar": [],
            "A3_geo_minus_zero_accuracy_slope": [],
            "B2_geo_minus_pairshuffled_rho_Tenv_Sstar": [],
            "B3_geo_minus_pairshuffled_accuracy_slope": [],
            "C2_geo_minus_pairnarrow_rho_Tenv_Sstar": [],
            "C3_geo_minus_pairnarrow_accuracy_slope": [],
        }
        for (rep, r), drr in ds.groupby(["rep", "r"]):
            byc = {c: drr[drr.condition == c].sort_values("T_env") for c in CONDITIONS}
            rho_geo = rho_log_env_scale(byc["geometric"].T_env.to_numpy(), byc["geometric"].S_star.to_numpy())
            rho_shf = rho_log_env_scale(byc["pair_shuffled"].T_env.to_numpy(), byc["pair_shuffled"].S_star.to_numpy())
            rho_nar = rho_log_env_scale(byc["pair_narrow"].T_env.to_numpy(), byc["pair_narrow"].S_star.to_numpy())
            acc["A2_geo_rho_Tenv_Sstar"].append(rho_geo)
            acc["B2_geo_minus_pairshuffled_rho_Tenv_Sstar"].append(rho_geo - rho_shf)
            acc["C2_geo_minus_pairnarrow_rho_Tenv_Sstar"].append(rho_geo - rho_nar)

            g = byc["geometric"][["T_env", "test_accuracy_star"]].rename(columns={"test_accuracy_star": "geo"})
            for control, key in (
                ("zero", "A3_geo_minus_zero_accuracy_slope"),
                ("pair_shuffled", "B3_geo_minus_pairshuffled_accuracy_slope"),
                ("pair_narrow", "C3_geo_minus_pairnarrow_accuracy_slope"),
            ):
                c = byc[control][["T_env", "test_accuracy_star"]].rename(columns={"test_accuracy_star": "ctl"})
                m = g.merge(c, on="T_env")
                acc[key].append(slope_y_on_log2s(m.T_env.to_numpy(), (m.geo - m.ctl).to_numpy()))

        subject_rows.append({"subject": sub, **{k: float(np.nanmean(v)) for k, v in acc.items()}})

    sdf = pd.DataFrame(subject_rows).sort_values("subject")
    sdf.to_csv(out / "05_06_environment_subject_effects.csv", index=False)
    return sdf


# =============================================================================
# Inference
# =============================================================================
def signflip_p(values: Sequence[float], n_perm: int, alternative: str, seed: int) -> float:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan")
    obs = float(v.mean())
    rng = np.random.default_rng(seed)
    if v.size <= 15 and (2 ** v.size) <= n_perm:
        stats = np.empty(2 ** v.size, dtype=float)
        for mask in range(2 ** v.size):
            signs = np.ones(v.size)
            for i in range(v.size):
                if (mask >> i) & 1:
                    signs[i] = -1.0
            stats[mask] = float(np.mean(signs * v))
    else:
        stats = np.empty(n_perm, dtype=float)
        for q in range(n_perm):
            signs = rng.choice(np.asarray([-1.0, 1.0]), size=v.size)
            stats[q] = float(np.mean(signs * v))
    if alternative == "greater":
        return float((1 + np.sum(stats >= obs)) / (len(stats) + 1))
    return float((1 + np.sum(np.abs(stats) >= abs(obs))) / (len(stats) + 1))


def bootstrap_ci(values: Sequence[float], n_boot: int, seed: int) -> Tuple[float, float]:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=float)
    for q in range(n_boot):
        means[q] = rng.choice(v, size=v.size, replace=True).mean()
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def holm_adjust(pvals: Sequence[float]) -> List[float]:
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan)
    good = np.where(np.isfinite(p))[0]
    order = good[np.argsort(p[good])]
    running = 0.0
    m = len(order)
    for rank, idx in enumerate(order):
        val = min(1.0, (m - rank) * p[idx])
        running = max(running, val)
        out[idx] = running
    return out.tolist()


def family_statistics(df: pd.DataFrame, columns: Sequence[Tuple[str, str]], family: str,
                      cfg: Config, outpath: Path) -> pd.DataFrame:
    rows = []
    for col, hypothesis in columns:
        if col not in df.columns:
            continue
        v = pd.to_numeric(df[col], errors="coerce").to_numpy(float)
        v = v[np.isfinite(v)]
        lo, hi = bootstrap_ci(v, cfg.bootstrap_samples, stable_seed("bootstrap", family, hypothesis, cfg.weighting))
        p = signflip_p(v, cfg.permutation_samples, "greater", stable_seed("permutation", family, hypothesis, cfg.weighting))
        rows.append({
            "family": family,
            "hypothesis": hypothesis,
            "metric": col,
            "n_subjects": int(v.size),
            "mean": float(np.mean(v)) if v.size else np.nan,
            "median": float(np.median(v)) if v.size else np.nan,
            "bootstrap_95ci_low": lo,
            "bootstrap_95ci_high": hi,
            "positive_fraction": float(np.mean(v > 0)) if v.size else np.nan,
            "raw_one_sided_signflip_p": p,
        })
    if rows:
        adj = holm_adjust([r["raw_one_sided_signflip_p"] for r in rows])
        for r, a in zip(rows, adj):
            r["holm_p"] = a
            # Conventional directional inference only; CI is reported, not an extra gate.
            r["direction_supported"] = bool(cfg.mode == "full" and r["mean"] > 0 and a < 0.05)
    out = pd.DataFrame(rows)
    out.to_csv(outpath, index=False)
    return out


def write_inference(memory_sdf: Optional[pd.DataFrame], env_sdf: Optional[pd.DataFrame], cfg: Config, out: Path):
    if memory_sdf is None:
        memory_sdf = pd.DataFrame()
    if env_sdf is None:
        env_sdf = pd.DataFrame()
    merged = memory_sdf.merge(env_sdf, on="subject", how="outer") if not memory_sdf.empty and not env_sdf.empty else (memory_sdf if not memory_sdf.empty else env_sdf)

    source_cols = [
        ("A1_geo_TD_slope", "A1_GEOMETRIC_TD_scales_with_S"),
        ("A2_geo_rho_Tenv_Sstar", "A2_GEOMETRIC_optimal_scale_matches_temporal_demand"),
        ("A3_geo_minus_zero_accuracy_slope", "A3_GEOMETRIC_advantage_over_ZERO_grows_with_temporal_demand"),
    ]
    assign_cols = [
        ("B1_geo_minus_pairshuffled_TD_slope", "B1_anatomical_assignment_strengthens_TD_scaling"),
        ("B2_geo_minus_pairshuffled_rho_Tenv_Sstar", "B2_anatomical_assignment_strengthens_demand_matching"),
        ("B3_geo_minus_pairshuffled_accuracy_slope", "B3_anatomical_assignment_value_grows_with_temporal_demand"),
    ]
    hetero_cols = [
        ("C1_geo_minus_pairnarrow_TD_slope", "C1_delay_heterogeneity_strengthens_TD_scaling"),
        ("C2_geo_minus_pairnarrow_rho_Tenv_Sstar", "C2_delay_heterogeneity_strengthens_demand_matching"),
        ("C3_geo_minus_pairnarrow_accuracy_slope", "C3_delay_heterogeneity_value_grows_with_temporal_demand"),
    ]

    family_statistics(merged, source_cols, "source_effect_replication", cfg, out / "05_07_source_effect_statistics.csv")
    family_statistics(merged, assign_cols, "anatomical_assignment", cfg, out / "05_08_assignment_statistics.csv")
    family_statistics(merged, hetero_cols, "delay_heterogeneity", cfg, out / "05_09_heterogeneity_statistics.csv")

    if not memory_sdf.empty:
        secondary = [
            ("geo_minus_pairshuffled_MC_AUC", "MC_AUC_GEOMETRIC_gt_PAIR_SHUFFLED"),
            ("geo_minus_pairshuffled_TD_AUC", "TD_AUC_GEOMETRIC_gt_PAIR_SHUFFLED"),
            ("geo_minus_pairnarrow_MC_AUC", "MC_AUC_GEOMETRIC_gt_PAIR_NARROW"),
            ("geo_minus_pairnarrow_TD_AUC", "TD_AUC_GEOMETRIC_gt_PAIR_NARROW"),
        ]
        family_statistics(memory_sdf, secondary, "memory_mechanism_secondary", cfg, out / "05_10_memory_mechanism_statistics.csv")


# =============================================================================
# Execution, preflight, documentation
# =============================================================================
def run_jobs(subjects: List[SubjectFiles], cfg: Config, phase: str) -> List[Dict[str, object]]:
    payloads = [(sf, asdict(cfg), rep) for sf in subjects for rep in range(1, cfg.internal_reps + 1)]
    worker = memory_worker if phase == "memory" else environment_worker
    results, errors = [], []
    print(f"05 {phase}: weighting={cfg.weighting}; {len(subjects)} subjects x {cfg.internal_reps} reps; workers={cfg.workers}; numba={NUMBA_AVAILABLE}", flush=True)
    if cfg.workers == 1:
        for i, p in enumerate(payloads, 1):
            sf, _, rep = p
            print(f"[{i}/{len(payloads)}] {sf.subject} rep={rep}", flush=True)
            try:
                results.append(worker(p))
            except Exception:
                tr = traceback.format_exc()
                errors.append((sf.subject, rep, tr))
                print(tr, flush=True)
    else:
        with ProcessPoolExecutor(max_workers=cfg.workers) as ex:
            futs = {ex.submit(worker, p): p for p in payloads}
            done = 0
            for fut in as_completed(futs):
                done += 1
                sf, _, rep = futs[fut]
                try:
                    results.append(fut.result())
                    print(f"[{done}/{len(payloads)}] OK {sf.subject} rep={rep}", flush=True)
                except Exception:
                    tr = traceback.format_exc()
                    errors.append((sf.subject, rep, tr))
                    print(f"[{done}/{len(payloads)}] ERROR {sf.subject} rep={rep}\n{tr}", flush=True)
    if errors:
        raise RuntimeError(f"{len(errors)} subject-replicate jobs failed")
    return results


def preflight(subjects: List[SubjectFiles], cfg: Config, out: Path):
    rows = []
    for sf in subjects:
        try:
            sc_raw = load_numeric_txt(Path(sf.sc_file))
            ln_raw = load_numeric_txt(Path(sf.length_file))
            tpl = build_empirical_template(sf)
            signs, _ = make_signs_and_input(tpl, sf.subject, 1)
            base = base_weights(tpl, signs, cfg.weighting)
            shuf = pair_shuffled_delta(tpl.undirected_delta, stable_seed(sf.subject, 1, "pair_shuffle_geometry"))
            b64 = delay_bundle(tpl, shuf, 64)
            rows.append({
                "subject": sf.subject,
                "session": sf.session,
                "status": "ok",
                "sc_shape": f"{sc_raw.shape[0]}x{sc_raw.shape[1]}",
                "length_shape": f"{ln_raw.shape[0]}x{ln_raw.shape[1]}",
                "n_regions": tpl.n,
                "target_mean_degree": tpl.k_target,
                "actual_mean_degree": float(tpl.in_degree.mean()),
                "min_degree": int(tpl.in_degree.min()),
                "max_degree": int(tpl.in_degree.max()),
                "n_isolated_regions": int(np.sum(tpl.in_degree == 0)),
                "n_undirected_edges": int(tpl.undirected_delta.size),
                "n_directed_edges": int(tpl.src.size),
                "lmax_retained_mm": tpl.lmax_retained_mm,
                "delta_min": float(tpl.undirected_delta.min()),
                "delta_max": float(tpl.undirected_delta.max()),
                "pair_symmetry_geometric_S64": bool(np.array_equal(b64["geometric"][0::2], b64["geometric"][1::2])),
                "pair_symmetry_shuffled_S64": bool(np.array_equal(b64["pair_shuffled"][0::2], b64["pair_shuffled"][1::2])),
                "pair_symmetry_narrow_S64": bool(np.array_equal(b64["pair_narrow"][0::2], b64["pair_narrow"][1::2])),
                "shuffle_multiset_exact_S64": bool(np.array_equal(np.sort(b64["geometric"][0::2]), np.sort(b64["pair_shuffled"][0::2]))),
                "narrow_total_exact_S64": bool(int(b64["geometric"][0::2].sum()) == int(b64["pair_narrow"][0::2].sum())),
                "weight_sum_abs_min": float(np.bincount(tpl.dst, weights=np.abs(base), minlength=tpl.n).min()),
                "weight_sum_abs_max": float(np.bincount(tpl.dst, weights=np.abs(base), minlength=tpl.n).max()),
                "error": "",
            })
        except Exception as e:
            rows.append({"subject": sf.subject, "session": sf.session, "status": "error", "error": repr(e)})
            print(f"[PREFLIGHT ERROR {sf.subject}] {e!r}", flush=True)
    pdf = pd.DataFrame(rows)
    pdf.to_csv(out / "05_00B_preflight.csv", index=False)
    if (pdf.status != "ok").any():
        raise RuntimeError(f"preflight failed for {int((pdf.status!='ok').sum())}/{len(pdf)} subjects")


def write_inventory(subjects: List[SubjectFiles], out: Path):
    pd.DataFrame([asdict(x) for x in subjects]).to_csv(out / "05_00_inventory.csv", index=False)


def write_readme(cfg: Config, subjects: List[SubjectFiles], out: Path):
    txt = f"""# 05 MICA-MICs Geometry-Specific Scale Validation

Script version: `{SCRIPT_VERSION}`  
Seed namespace: `{SEED_NAMESPACE}`  
Mode: `{cfg.mode}`  
Phase: `{cfg.phase}`  
Weighting: `{cfg.weighting}`  
Subjects: `{len(subjects)}`  
Config signature: `{config_signature(cfg)}`

## Anti-circularity rules

- No empirical fMRI is used.
- S, r, T_env, task lengths, noise level, connection fraction, replicate count and inferential families are constants, not CLI tuning knobs.
- Full mode requires all 50 MICA-MICs subjects.
- Training, validation and test sequences are independent.
- S* is chosen from validation only; test outcomes never choose S* or break ties.
- All delay conditions share topology, weights, signs, inputs and stochastic sequences within each subject x replicate.
- PAIR_SHUFFLED uses one fixed undirected tract permutation across all S and all T_env within a subject x replicate.
- PAIR_NARROW exactly preserves total undirected integer delay at each S and minimizes variance; it does not randomize anatomy.
- Multiplicity correction is performed within the three pre-specified scientific families.
- Smoke mode is implementation-only and is not a scientific gate.

## Frozen source-model settings

- Schaefer cortical N = {ATLAS_SCALE}
- source connection fraction = 16/127 = {SOURCE_CONNECTION_FRACTION:.9f}
- target mean degree = {TARGET_MEAN_DEGREE}
- S = {SCALES.tolist()}
- r = {COUPLINGS.tolist()}
- T_env = {T_ENVS.tolist()}
- full internal reps = {FULL_INTERNAL_REPS}
- memory full: washout {FULL_MEMORY_WASHOUT}, train {FULL_MEMORY_TRAIN}, independent test {FULL_MEMORY_TEST}, max lag {FULL_MAX_LAG}
- environment full: washout {FULL_ENV_WASHOUT}, train {FULL_ENV_TRAIN}, validation {FULL_ENV_VAL}, independent test {FULL_ENV_TEST}, noise SD {ENV_NOISE_SD}

## Delay normalization correction

After strongest-edge selection, retained tract lengths are normalized by the maximum length among RETAINED model edges only. A tract excluded from the model cannot set the model's delay scale.

## Inference families

A — source-effect replication:
A1 GEOMETRIC TD slope > 0.
A2 GEOMETRIC rho(T_env, S*) > 0.
A3 GEOMETRIC-minus-ZERO held-out accuracy advantage grows with T_env.

B — anatomical assignment:
B1 GEOMETRIC TD scaling > PAIR_SHUFFLED.
B2 GEOMETRIC demand matching > PAIR_SHUFFLED.
B3 GEOMETRIC-minus-PAIR_SHUFFLED held-out accuracy advantage grows with T_env.

C — delay heterogeneity:
C1 GEOMETRIC TD scaling > PAIR_NARROW.
C2 GEOMETRIC demand matching > PAIR_NARROW.
C3 GEOMETRIC-minus-PAIR_NARROW held-out accuracy advantage grows with T_env.

Each family uses subject-level one-sided sign-flip tests with Holm correction across its three endpoints. Bootstrap 95% CIs and positive-subject fractions are also reported. No extra ad-hoc success gate is imposed.

## Weighting

`equal` is the primary exact-transplant analysis. `empirical` is a pre-specified robustness analysis using log1p(SC) magnitudes normalized so each target receives total absolute recurrent magnitude r. The primary is never replaced by whichever weighting gives a stronger result.
"""
    (out / "05_11_README_RUN.md").write_text(txt, encoding="utf-8")


# =============================================================================
# Self-test
# =============================================================================
def run_self_test() -> Dict[str, object]:
    tests: Dict[str, bool] = {}

    idx = cortical_indices(216, 200)
    tests["legacy_216_to_200"] = bool(len(idx) == 200 and idx[0] == 15 and idx[-1] == 215 and 14 not in idx and 115 not in idx)

    # Triangular symmetrization must not halve a one-sided edge.
    A = np.zeros((4, 4), dtype=float)
    A[0, 1] = 8.0
    A[2, 3] = 5.0
    B = robust_positive_symmetrize(A)
    tests["triangular_symmetrization_preserves_values"] = bool(B[0,1] == 8.0 and B[1,0] == 8.0 and B[2,3] == 5.0)

    # Pair controls.
    du = np.asarray([0.10, 0.20, 0.35, 0.55, 0.75, 1.00], dtype=float)
    shdu = pair_shuffled_delta(du, 123)
    tests["fixed_shuffle_continuous_multiset"] = bool(np.array_equal(np.sort(shdu), np.sort(du)))
    for S in (1, 8, 64):
        gu = geometric_delays(du, S)
        su = geometric_delays(shdu, S)
        nu = pair_narrow_undirected(gu, du)
        tests[f"shuffle_integer_multiset_S{S}"] = bool(np.array_equal(np.sort(gu), np.sort(su)))
        tests[f"narrow_total_S{S}"] = bool(int(gu.sum()) == int(nu.sum()))
        tests[f"narrow_min_variance_S{S}"] = bool(int(nu.max() - nu.min()) <= 1)
        tests[f"directed_pair_symmetry_S{S}"] = bool(np.array_equal(expand_undirected_delays(gu)[0::2], expand_undirected_delays(gu)[1::2]))

    # Synthetic paired template for weight normalization and simulation.
    n = 6
    ui = np.asarray([0,0,1,1,2,2,3,4,4], dtype=np.int32)
    uj = np.asarray([1,2,2,3,3,4,5,5,0], dtype=np.int32)
    src = np.empty(2*len(ui), dtype=np.int32); dst = np.empty_like(src)
    src[0::2], dst[0::2] = uj, ui
    src[1::2], dst[1::2] = ui, uj
    indeg = np.bincount(dst, minlength=n).astype(np.int32)
    delta_u = np.linspace(.1, 1.0, len(ui))
    sc_u = np.linspace(1, 9, len(ui))
    tpl = NetworkTemplate(n, 3, src, dst, np.repeat(delta_u,2), np.repeat(sc_u,2), delta_u, sc_u, ui, uj,
                          100*delta_u, 100.0, indeg)
    signs = np.ones(src.size)
    for mode in ("equal", "empirical"):
        bw = base_weights(tpl, signs, mode)
        sums = np.bincount(dst, weights=np.abs(bw), minlength=n)
        tests[f"{mode}_weight_normalization"] = bool(np.allclose(sums, 1.0))

    b = np.where(np.arange(n)%2==0, 1.0, -1.0)
    u = np.random.default_rng(2).normal(size=300)
    base = base_weights(tpl, signs, "equal")
    d = expand_undirected_delays(geometric_delays(delta_u, 4))
    x1 = simulate_linear(tpl, base, d, b, u, 0.5)
    x2 = simulate_linear(tpl, base, d, b, u, 0.5)
    tests["simulation_deterministic"] = bool(np.allclose(x1, x2))
    tests["simulation_finite"] = bool(np.isfinite(x1).all())

    # OLS exactness.
    rng = np.random.default_rng(5)
    X = rng.normal(size=(500, 8)); beta = rng.normal(size=(8,2)); Y = 1.2 + X @ beta
    a, BB = ols_fit_centered(X, Y)
    tests["ols_exact"] = bool(np.max(np.abs((a + X @ BB) - Y)) < 1e-8)

    # Validation tie rule cannot use test data.
    td = pd.DataFrame({"scale":[1,2,4], "val_accuracy":[.8,.8,.7], "test_accuracy":[.1,.99,.5], "test_mse":[1,1,1]})
    bestS, *_ = choose_scale_from_validation(td)
    tests["validation_tie_smallest_scale_not_test"] = bool(bestS == 1)

    # Holm monotonicity sanity.
    hp = holm_adjust([0.01, 0.03, 0.20])
    tests["holm_valid"] = bool(all(0 <= x <= 1 for x in hp) and hp[0] <= hp[1] <= hp[2])
    tests["source_fraction_maps_to_k25"] = bool(TARGET_MEAN_DEGREE == 25)
    tests["numba_available"] = bool(NUMBA_AVAILABLE)
    tests["all_pass"] = bool(all(v for k,v in tests.items() if k != "numba_available"))
    return tests


def main():
    cfg = parse_args()
    t0 = time.time()
    out = Path(cfg.outdir)
    out.mkdir(parents=True, exist_ok=True)

    subjects = discover_subjects(cfg)
    write_inventory(subjects, out)
    preflight(subjects, cfg, out)
    write_readme(cfg, subjects, out)

    memory_sdf: Optional[pd.DataFrame] = None
    env_sdf: Optional[pd.DataFrame] = None

    if cfg.phase in ("memory", "all"):
        memory_sdf = aggregate_memory(run_jobs(subjects, cfg, "memory"), out)
    else:
        p = out / "05_03_memory_subject_effects.csv"
        if p.exists():
            memory_sdf = pd.read_csv(p)

    if cfg.phase in ("environment", "all"):
        env_sdf = aggregate_environment(run_jobs(subjects, cfg, "environment"), out)
    else:
        p = out / "05_06_environment_subject_effects.csv"
        if p.exists():
            env_sdf = pd.read_csv(p)

    write_inference(memory_sdf, env_sdf, cfg, out)

    audit = {
        "script_version": SCRIPT_VERSION,
        "seed_namespace": SEED_NAMESPACE,
        "config_signature": config_signature(cfg),
        "mode": cfg.mode,
        "phase": cfg.phase,
        "weighting": cfg.weighting,
        "audit_pass": True,
        "n_subjects": len(subjects),
        "scientific_inference_eligible": bool(cfg.mode == "full" and len(subjects) == 50),
        "subject_is_independent_unit": True,
        "core_scientific_parameters_cli_tunable": False,
        "predeclared_weighting_mode_cli_selectable": True,
        "fixed_shuffle_across_scales": True,
        "pair_preserving_controls": True,
        "retained_edge_lmax": True,
        "validation_only_scale_selection": True,
        "test_never_selects_scale": True,
        "smoke_is_not_scientific_gate": True,
        "numba_available": NUMBA_AVAILABLE,
        "runtime_seconds": time.time() - t0,
    }
    (out / "05_12_audit.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
