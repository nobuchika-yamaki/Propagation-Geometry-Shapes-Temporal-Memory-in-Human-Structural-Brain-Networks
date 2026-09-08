#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
08_MICA_MICS_EMPIRICAL_SC_WEIGHT_ROBUSTNESS.py

Final robustness analysis 1: empirical structural-connectivity weighting.

Scientific question
-------------------
Does the principal propagation-scale result survive when recurrent edge magnitudes are
weighted by empirical MICA-MICs structural connectivity rather than being equal within
each target node?

The robustness manipulation is exactly the empirical weighting already defined in core 05:
    g_e = log(1 + SC_e)
    |w_e| = g_e / sum_{e' -> i} g_e'
with the same recurrent signs and unit incoming absolute-weight sum per target node.

Everything else is held fixed relative to the corrected equal-weight analysis:
- 50 MICA-MICs subjects, Schaefer-200 cortex
- identical retained topology and tract lengths
- identical rep IDs 1..8
- identical signs, input vectors, train/validation/test stochastic sequences
- S = 1,2,4,8,16,32,64
- r = 0.5,0.7,0.9
- T_env = 4,8,16,32,64
- GEOMETRIC / PAIR_SHUFFLED / PAIR_NARROW / ZERO controls
- validation-only S* selection

Memory is measured to lag 1024, using the corrected explicit prehistory procedure from 07.
The original lag-128 TD is also recomputed from the first 128 lags so A1/B1/C1 remain
exactly comparable to the primary 05 inferential definitions. The 1024-lag extension is
used for capacity/horizon robustness, not to redefine TD-based primary effects.

This script is MICA-MICs-specific and imports only the frozen corrected core 05 script.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

# Prevent BLAS/LAPACK oversubscription under subject-level multiprocessing.
# These must be set before NumPy/SciPy (and the imported core) initialize thread pools.
for _var in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS",
):
    os.environ[_var] = "1"


import numpy as np
import pandas as pd


SCRIPT_VERSION = "08-mica-empirical-sc-weight-robustness-v1.2"
CHECKPOINT_SIGNATURE_VERSION = "08-mica-empirical-sc-weight-robustness-v1.1"
REQUIRED_CORE_VERSION = "05-mica-geometry-specific-v1.0"
CORE_FILENAME = "05_MICA_MICS_GEOMETRY_SPECIFIC_SCALE_VALIDATION.py"
SEED_NAMESPACE = "MICA_EMPIRICAL_SC_WEIGHT_ROBUSTNESS_V1"

MAX_LAG = 1024
LAG_BATCH = 128
WASHOUT = 1000
TRAIN_LEN = 10000
TEST_LEN = 10000
INTERNAL_REPS = 8
NULL_R2_REFERENCE = 1.0 / float(TEST_LEN - 1)
CONDITIONS = ("geometric", "pair_shuffled", "pair_narrow", "zero")

DEFAULT_CORE = f"~/Downloads/{CORE_FILENAME}"
DEFAULT_DATA_ROOT = "~/Downloads/mica-mics/MICs_release/derivatives/micapipe"
DEFAULT_OUTDIR = "~/Desktop/08_MICA_MICS_EMPIRICAL_SC_WEIGHT_ROBUSTNESS_V1_1"
DEFAULT_EQUAL_ROOT = "~/Desktop/05_MICA_MICS_GEOMETRY_SPECIFIC_SCALE_VALIDATION/equal"
DEFAULT_EQUAL_EXTENDED_ROOT = "~/Desktop/07_MICA_MICS_MEMORY_MAXLAG1024_EXTENSION_V1_3"


def stable_seed(*parts: object) -> int:
    s = "|".join(str(x) for x in (SEED_NAMESPACE,) + parts).encode("utf-8")
    return int(hashlib.sha256(s).hexdigest()[:16], 16) % (2**32 - 1)


def parse_args():
    p = argparse.ArgumentParser(description="MICA-MICs empirical-SC-weight robustness")
    p.add_argument("--phase", choices=("memory", "environment", "all"), default="all")
    p.add_argument("--core-script", default=DEFAULT_CORE)
    p.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    p.add_argument("--outdir", default=DEFAULT_OUTDIR)
    p.add_argument("--equal-root", default=DEFAULT_EQUAL_ROOT)
    p.add_argument("--equal-extended-root", default=DEFAULT_EQUAL_EXTENDED_ROOT)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--no-checkpoint", action="store_true")
    p.add_argument("--self-test", action="store_true")
    a = p.parse_args()
    if a.workers < 1:
        raise ValueError("--workers must be >=1")
    return a


_CORE_CACHE = {}


def load_core(path: str):
    path = str(Path(path).expanduser())
    cached = _CORE_CACHE.get(path)
    if cached is not None:
        return cached
    if not Path(path).exists():
        raise FileNotFoundError(f"Core 05 not found: {path}")
    name = "mica_geometry_core05_for_07"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    if getattr(mod, "SCRIPT_VERSION", None) != REQUIRED_CORE_VERSION:
        raise RuntimeError(
            f"Unexpected core version {getattr(mod, 'SCRIPT_VERSION', None)!r}; "
            f"required {REQUIRED_CORE_VERSION!r}"
        )
    _CORE_CACHE[path] = mod
    return mod


def core_config(core, data_root: str, outdir: str, phase: str):
    return core.Config(
        mode="full",
        phase=phase,
        weighting="empirical",
        data_root=str(Path(data_root).expanduser()),
        outdir=str(Path(outdir).expanduser()),
        workers=1,
        internal_reps=INTERNAL_REPS,
        memory_washout=core.FULL_MEMORY_WASHOUT,
        memory_train=core.FULL_MEMORY_TRAIN,
        memory_test=core.FULL_MEMORY_TEST,
        max_lag=core.FULL_MAX_LAG,
        env_washout=core.FULL_ENV_WASHOUT,
        env_train=core.FULL_ENV_TRAIN,
        env_val=core.FULL_ENV_VAL,
        env_test=core.FULL_ENV_TEST,
        bootstrap_samples=core.FULL_BOOTSTRAP_SAMPLES,
        permutation_samples=core.FULL_PERMUTATION_SAMPLES,
        max_subjects=0,
        checkpoint=True,
    )


def signature(core, phase: str) -> str:
    # Keep the v1.1 scientific checkpoint signature so completed/in-progress v1.1
    # rep-level checkpoints are reusable. v1.2 changes orchestration/progress only.
    x = {
        "script_version": CHECKPOINT_SIGNATURE_VERSION,
        "core_version": REQUIRED_CORE_VERSION,
        "phase": phase,
        "weighting": "empirical_log1p_incoming_normalized",
        "max_lag": MAX_LAG,
        "washout": WASHOUT,
        "train_len": TRAIN_LEN,
        "test_len": TEST_LEN,
        "reps": list(range(1, INTERNAL_REPS + 1)),
        "scales": [int(x) for x in core.SCALES],
        "couplings": [float(x) for x in core.COUPLINGS],
        "T_env": [int(x) for x in core.T_ENVS],
        "conditions": list(CONDITIONS),
    }
    return hashlib.sha256(json.dumps(x, sort_keys=True).encode()).hexdigest()[:20]


# -----------------------------------------------------------------------------
# Extended memory helpers
# -----------------------------------------------------------------------------

def required_prefix(core) -> int:
    return max(MAX_LAG - WASHOUT, int(np.max(core.SCALES)) + 1)


def build_extended_input(core, subject: str, rep: int, part: str, analysis_len: int):
    if part not in ("train", "test"):
        raise ValueError(part)
    old_seed = core.stable_seed(subject, rep, f"memory_{part}")
    old_rng = np.random.default_rng(old_seed)
    old_u = old_rng.normal(0.0, 1.0, WASHOUT + analysis_len)
    pre = required_prefix(core)
    pre_seed = core.stable_seed(subject, rep, f"memory_{part}_prefix_maxlag1024")
    pre_rng = np.random.default_rng(pre_seed)
    prefix = pre_rng.normal(0.0, 1.0, pre)
    u = np.concatenate([prefix, old_u])
    start = pre + WASHOUT
    if start - MAX_LAG < 0:
        raise RuntimeError("invalid prehistory")
    if not np.array_equal(u[pre:], old_u):
        raise AssertionError("old 05 sequence not preserved")
    return u, start


def target_block(u: np.ndarray, start: int, length: int, lo: int, hi: int) -> np.ndarray:
    idx = np.arange(start, start + length, dtype=np.int64)
    lag = np.arange(lo, hi + 1, dtype=np.int64)
    pos = idx[:, None] - lag[None, :]
    if pos.min() < 0:
        raise RuntimeError("negative target index")
    return u[pos]


def memory_scores_from_states(Xtr: np.ndarray, Xte: np.ndarray,
                              train_u: np.ndarray, test_u: np.ndarray,
                              start_tr: int, start_te: int,
                              max_lag: int = MAX_LAG,
                              lag_batch: int = LAG_BATCH) -> np.ndarray:
    Xtr = np.asarray(Xtr, dtype=np.float64)
    Xte = np.asarray(Xte, dtype=np.float64)
    xm = Xtr.mean(axis=0)
    Xtrc = Xtr - xm
    Xtec = Xte - xm
    proj = np.linalg.pinv(Xtrc.T @ Xtrc, hermitian=True) @ Xtrc.T
    M = np.zeros(max_lag, dtype=np.float64)
    for lo in range(1, max_lag + 1, lag_batch):
        hi = min(max_lag, lo + lag_batch - 1)
        Ytr = target_block(train_u, start_tr, len(Xtr), lo, hi)
        Yte = target_block(test_u, start_te, len(Xte), lo, hi)
        Ytrc = Ytr - Ytr.mean(axis=0)
        B = proj @ Ytrc
        P = Xtec @ B
        Yc = Yte - Yte.mean(axis=0)
        Pc = P - P.mean(axis=0)
        num = np.sum(Yc * Pc, axis=0)
        den = np.sqrt(np.sum(Yc * Yc, axis=0) * np.sum(Pc * Pc, axis=0))
        c = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
        vals = c * c
        vals[~np.isfinite(vals)] = 0.0
        M[lo - 1:hi] = vals
    return M


def memory_profile(core, template, base, b, delays, r,
                   train_u, test_u, start_tr, start_te) -> np.ndarray:
    xtr = core.simulate_linear(template, base, delays, b, train_u, float(r))
    xte = core.simulate_linear(template, base, delays, b, test_u, float(r))
    Xtr = xtr[start_tr:start_tr + TRAIN_LEN]
    Xte = xte[start_te:start_te + TEST_LEN]
    return memory_scores_from_states(Xtr, Xte, train_u, test_u, start_tr, start_te)


def td_at_horizon(M: np.ndarray, K: int = 128) -> float:
    q = np.asarray(M[:K], dtype=np.float64)
    mc = float(q.sum())
    if mc <= 0:
        return float("nan")
    lag = np.arange(1, K + 1, dtype=np.float64)
    return float(np.sum(lag * q) / mc)


def quantile_lag(M: np.ndarray, q: float) -> float:
    total = float(np.sum(M))
    if total <= 0:
        return float("nan")
    c = np.cumsum(M) / total
    return float(np.searchsorted(c, q, side="left") + 1)


def profile_metrics(M: np.ndarray, state_dim: int = 200) -> Dict[str, float]:
    M = np.asarray(M, dtype=np.float64)
    out: Dict[str, float] = {}
    for K in (128, 256, 512, 1024):
        out[f"MC_1_{K}"] = float(M[:K].sum())
    out["MC_1_1024_over_state_dim"] = out["MC_1_1024"] / float(state_dim)
    out["MC_1_256_over_state_dim"] = out["MC_1_256"] / float(state_dim)
    out["TD_1_128"] = td_at_horizon(M, 128)
    out["lag50_1_1024"] = quantile_lag(M, 0.50)
    out["lag90_1_1024"] = quantile_lag(M, 0.90)
    out["lag95_1_1024"] = quantile_lag(M, 0.95)
    out["mean_M_last64"] = float(M[-64:].mean())
    out["mean_M_last64_minus_null"] = out["mean_M_last64"] - NULL_R2_REFERENCE
    out["MC_257_1024"] = float(M[256:].sum())
    out["expected_null_MC_257_1024"] = float((1024 - 256) * NULL_R2_REFERENCE)
    out["MC_257_1024_excess_vs_null"] = out["MC_257_1024"] - out["expected_null_MC_257_1024"]
    return out



def group_identical_delay_patterns(entries):
    """
    Group labels that have byte-identical int32 delay vectors. Reusing one simulation
    for an identical delay vector is mathematically exact and changes no scientific setting.
    entries: iterable of (label, delays).
    """
    groups = {}
    for label, delays in entries:
        d = np.asarray(delays, dtype=np.int32)
        key = d.tobytes()
        if key not in groups:
            groups[key] = {"delays": d.copy(), "labels": [label]}
        else:
            if not np.array_equal(groups[key]["delays"], d):
                raise AssertionError("delay-pattern byte grouping inconsistency")
            groups[key]["labels"].append(label)
    return list(groups.values())


# -----------------------------------------------------------------------------
# Checkpoints
# -----------------------------------------------------------------------------

def memory_cp_path(out: Path, subject: str, rep: int, sig: str) -> Path:
    d = out / "checkpoints_memory"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{subject}_rep{rep:02d}_{sig}.npz"


def save_memory_cp(path: Path, sig: str, profile_mean_r: np.ndarray, td128_by_r: np.ndarray):
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        tmp,
        signature=np.asarray(sig),
        profile_mean_r=np.asarray(profile_mean_r, dtype=np.float32),
        td128_by_r=np.asarray(td128_by_r, dtype=np.float64),
    )
    os.replace(tmp, path)


def load_memory_cp(path: Path, sig: str):
    if not path.exists():
        return None
    try:
        z = np.load(path, allow_pickle=False)
        if str(z["signature"].item()) != sig:
            return None
        return {
            "profile_mean_r": z["profile_mean_r"].astype(np.float64),
            "td128_by_r": z["td128_by_r"].astype(np.float64),
        }
    except Exception:
        return None


def env_cp_path(out: Path, subject: str, rep: int, sig: str) -> Path:
    d = out / "checkpoints_environment"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{subject}_rep{rep:02d}_{sig}.json"


def save_env_cp(path: Path, sig: str, rows: List[Dict[str, object]]):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"signature": sig, "rows": rows}))
    os.replace(tmp, path)


def load_env_cp(path: Path, sig: str):
    if not path.exists():
        return None
    try:
        z = json.loads(path.read_text())
        if z.get("signature") != sig:
            return None
        return z.get("rows", [])
    except Exception:
        return None


# -----------------------------------------------------------------------------
# Memory worker
# -----------------------------------------------------------------------------

def sc_memory_rep(core, template, subject: str, rep: int, out: Path, sig: str, checkpoint: bool):
    cp = memory_cp_path(out, subject, rep, sig)
    if checkpoint:
        old = load_memory_cp(cp, sig)
        if old is not None:
            return old

    signs, b = core.make_signs_and_input(template, subject, rep)
    base = core.base_weights(template, signs, "empirical")
    shuf_delta = core.pair_shuffled_delta(
        template.undirected_delta,
        core.stable_seed(subject, rep, "pair_shuffle_geometry"),
    )
    train_u, start_tr = build_extended_input(core, subject, rep, "train", TRAIN_LEN)
    test_u, start_te = build_extended_input(core, subject, rep, "test", TEST_LEN)

    nc, ns, nr = len(CONDITIONS), len(core.SCALES), len(core.COUPLINGS)
    prof_mean_r = np.zeros((nc, ns, MAX_LAG), dtype=np.float64)
    td128_by_r = np.zeros((nc, ns, nr), dtype=np.float64)

    # Build all requested condition/scale delay vectors once, then deduplicate exact
    # integer patterns. ZERO across all S is automatically simulated only once; any
    # rounding-induced identity between GEOMETRIC/NARROW/etc. is also reused exactly.
    entries = []
    for si, S in enumerate(core.SCALES):
        bundle = core.delay_bundle(template, shuf_delta, int(S))
        for ci, cond in enumerate(("geometric", "pair_shuffled", "pair_narrow")):
            entries.append(((ci, si), bundle[cond]))
    zci = CONDITIONS.index("zero")
    zd = np.zeros(template.src.size, dtype=np.int32)
    for si, _S in enumerate(core.SCALES):
        entries.append(((zci, si), zd))
    pattern_groups = group_identical_delay_patterns(entries)

    for ri, r in enumerate(core.COUPLINGS):
        for g in pattern_groups:
            M = memory_profile(
                core, template, base, b, g["delays"], float(r),
                train_u, test_u, start_tr, start_te,
            )
            tdv = td_at_horizon(M, 128)
            for ci, si in g["labels"]:
                prof_mean_r[ci, si] += M / float(nr)
                td128_by_r[ci, si, ri] = tdv

    result = {"profile_mean_r": prof_mean_r, "td128_by_r": td128_by_r}
    if checkpoint:
        save_memory_cp(cp, sig, prof_mean_r, td128_by_r)
    return result


def sc_memory_subject_worker(payload):
    sf_dict, core_path, out_s, sig, checkpoint = payload
    core = load_core(core_path)
    sf = core.SubjectFiles(**sf_dict)
    template = core.build_empirical_template(sf)
    out = Path(out_s)

    profiles = []
    effect_acc = {"A1_geo_TD_slope": [], "B1_geo_minus_pairshuffled_TD_slope": [], "C1_geo_minus_pairnarrow_TD_slope": []}
    for rep in range(1, INTERNAL_REPS + 1):
        z = sc_memory_rep(core, template, sf.subject, rep, out, sig, checkpoint)
        profiles.append(z["profile_mean_r"])
        td = z["td128_by_r"]
        for ri, _r in enumerate(core.COUPLINGS):
            slopes = {}
            for ci, cond in enumerate(("geometric", "pair_shuffled", "pair_narrow")):
                slopes[cond] = core.slope_y_on_log2s(core.SCALES, td[ci, :, ri])
            effect_acc["A1_geo_TD_slope"].append(slopes["geometric"])
            effect_acc["B1_geo_minus_pairshuffled_TD_slope"].append(slopes["geometric"] - slopes["pair_shuffled"])
            effect_acc["C1_geo_minus_pairnarrow_TD_slope"].append(slopes["geometric"] - slopes["pair_narrow"])

    profile = np.mean(np.stack(profiles, axis=0), axis=0)
    effects = {k: float(np.nanmean(v)) for k, v in effect_acc.items()}
    return {
        "subject": sf.subject,
        "session": sf.session,
        "profile": profile.astype(np.float32),
        "effects": effects,
    }


# -----------------------------------------------------------------------------
# Environment worker
# -----------------------------------------------------------------------------

def sc_environment_rep(core, template, subject: str, rep: int, cfg, out: Path, sig: str, checkpoint: bool):
    cp = env_cp_path(out, subject, rep, sig)
    if checkpoint:
        old = load_env_cp(cp, sig)
        if old is not None:
            return old

    signs, b = core.make_signs_and_input(template, subject, rep)
    base = core.base_weights(template, signs, "empirical")
    shuf_delta = core.pair_shuffled_delta(
        template.undirected_delta,
        core.stable_seed(subject, rep, "pair_shuffle_geometry"),
    )

    entries = []
    for S in core.SCALES:
        bundle = core.delay_bundle(template, shuf_delta, int(S))
        for cond in ("geometric", "pair_shuffled", "pair_narrow"):
            entries.append(((cond, int(S)), bundle[cond]))
    zd = np.zeros(template.src.size, dtype=np.int32)
    for S in core.SCALES:
        entries.append((("zero", int(S)), zd))
    pattern_groups = group_identical_delay_patterns(entries)

    rows: List[Dict[str, object]] = []
    for Tenv in core.T_ENVS:
        seqs = {}
        for part, L in (("train", cfg.env_train), ("val", cfg.env_val), ("test", cfg.env_test)):
            seed = core.stable_seed(subject, rep, "env", int(Tenv), part)
            seqs[part] = core.make_environment_sequence(cfg.env_washout + L, int(Tenv), seed)

        for r in core.COUPLINGS:
            for g in pattern_groups:
                va, vm, ta, tm = core.env_metrics(
                    template, base, b, g["delays"], float(r), seqs, cfg
                )
                for cond, S in g["labels"]:
                    rows.append({
                        "condition": cond, "T_env": int(Tenv), "scale": int(S), "r": float(r),
                        "val_accuracy": va, "val_mse": vm, "test_accuracy": ta, "test_mse": tm,
                    })

    if checkpoint:
        save_env_cp(cp, sig, rows)
    return rows


def sc_environment_subject_worker(payload):
    sf_dict, core_path, data_root, out_s, sig, checkpoint = payload
    core = load_core(core_path)
    sf = core.SubjectFiles(**sf_dict)
    template = core.build_empirical_template(sf)
    cfg = core_config(core, data_root, out_s, "environment")
    out = Path(out_s)
    rows = []
    for rep in range(1, INTERNAL_REPS + 1):
        rr = sc_environment_rep(core, template, sf.subject, rep, cfg, out, sig, checkpoint)
        for x in rr:
            rows.append({"subject": sf.subject, "session": sf.session, "rep": rep, **x})
    return rows


def sc_environment_rep_worker(payload):
    """One subject x one replicate per future for faster progress and better load balancing."""
    sf_dict, rep, core_path, data_root, out_s, sig, checkpoint = payload
    core = load_core(core_path)
    sf = core.SubjectFiles(**sf_dict)
    template = core.build_empirical_template(sf)
    cfg = core_config(core, data_root, out_s, "environment")
    out = Path(out_s)

    cp = env_cp_path(out, sf.subject, int(rep), sig)
    cached_rows = load_env_cp(cp, sig) if checkpoint else None
    if cached_rows is not None:
        rr = cached_rows
        cached = True
    else:
        rr = sc_environment_rep(core, template, sf.subject, int(rep), cfg, out, sig, checkpoint)
        cached = False

    rows = [{"subject": sf.subject, "session": sf.session, "rep": int(rep), **x} for x in rr]
    return {"subject": sf.subject, "rep": int(rep), "cached": cached, "rows": rows}


# -----------------------------------------------------------------------------
# Statistics and output helpers
# -----------------------------------------------------------------------------

def bootstrap_ci(v: Sequence[float], n_boot: int, seed: int) -> Tuple[float, float]:
    x = np.asarray(v, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        means[i] = rng.choice(x, size=x.size, replace=True).mean()
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def signflip_p(v: Sequence[float], n_perm: int, seed: int, alternative: str = "greater") -> float:
    x = np.asarray(v, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    obs = float(x.mean())
    rng = np.random.default_rng(seed)
    vals = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        vals[i] = float(np.mean(x * rng.choice(np.asarray([-1.0, 1.0]), size=x.size)))
    if alternative == "greater":
        return float((1 + np.sum(vals >= obs)) / (n_perm + 1))
    return float((1 + np.sum(np.abs(vals) >= abs(obs))) / (n_perm + 1))


def holm(pvals: Sequence[float]) -> List[float]:
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


def effect_statistics(df: pd.DataFrame, specs: Sequence[Tuple[str, str, str]], outpath: Path):
    rows = []
    for family, col, label in specs:
        v = pd.to_numeric(df[col], errors="coerce").to_numpy(float)
        v = v[np.isfinite(v)]
        lo, hi = bootstrap_ci(v, 10000, stable_seed("boot", family, col))
        p = signflip_p(v, 10000, stable_seed("perm", family, col), "greater")
        rows.append({
            "family": family, "effect": label, "metric": col, "n_subjects": int(v.size),
            "mean": float(np.mean(v)), "median": float(np.median(v)),
            "bootstrap_95ci_low": lo, "bootstrap_95ci_high": hi,
            "positive_fraction": float(np.mean(v > 0)), "raw_one_sided_signflip_p": p,
        })
    out = pd.DataFrame(rows)
    for family, idx in out.groupby("family").groups.items():
        ii = list(idx)
        out.loc[ii, "holm_p_within_family"] = holm(out.loc[ii, "raw_one_sided_signflip_p"].tolist())
    out.to_csv(outpath, index=False)
    return out


def paired_robustness_summary(emp: pd.DataFrame, eq_path: Path, columns: Sequence[str], out_subject: Path, out_summary: Path):
    if not eq_path.exists():
        return None
    eq = pd.read_csv(eq_path)
    keep = ["subject"] + [c for c in columns if c in eq.columns]
    m = emp.merge(eq[keep], on="subject", suffixes=("_empirical", "_equal"), validate="one_to_one")
    subject_rows = {"subject": m.subject}
    summary = []
    for c in columns:
        ec, qc = f"{c}_empirical", f"{c}_equal"
        if ec not in m.columns or qc not in m.columns:
            continue
        diff = m[ec].to_numpy(float) - m[qc].to_numpy(float)
        subject_rows[ec] = m[ec]
        subject_rows[qc] = m[qc]
        subject_rows[f"{c}_empirical_minus_equal"] = diff
        lo, hi = bootstrap_ci(diff, 10000, stable_seed("paired", c))
        summary.append({
            "metric": c,
            "mean_equal": float(np.mean(m[qc])),
            "mean_empirical": float(np.mean(m[ec])),
            "mean_empirical_minus_equal": float(np.mean(diff)),
            "bootstrap_95ci_low_diff": lo,
            "bootstrap_95ci_high_diff": hi,
            "same_direction_fraction": float(np.mean(np.sign(m[ec]) == np.sign(m[qc]))),
        })
    pd.DataFrame(subject_rows).to_csv(out_subject, index=False)
    pd.DataFrame(summary).to_csv(out_summary, index=False)
    return pd.DataFrame(summary)


# -----------------------------------------------------------------------------
# Write memory outputs
# -----------------------------------------------------------------------------

def write_memory_outputs(core, results, out: Path, equal_root: Path, equal_ext_root: Path):
    results = sorted(results, key=lambda z: z["subject"])
    effect_df = pd.DataFrame([{"subject": z["subject"], **z["effects"]} for z in results])
    effect_df.to_csv(out / "08_SC_05_memory_subject_effects.csv", index=False)

    effect_statistics(effect_df, [
        ("source_effect", "A1_geo_TD_slope", "A1_GEOMETRIC_TD_scales_with_S"),
        ("assignment", "B1_geo_minus_pairshuffled_TD_slope", "B1_GEOMETRIC_gt_PAIR_SHUFFLED_TD_scaling"),
        ("heterogeneity", "C1_geo_minus_pairnarrow_TD_slope", "C1_GEOMETRIC_gt_PAIR_NARROW_TD_scaling"),
    ], out / "08_SC_06_memory_effect_statistics.csv")

    paired_robustness_summary(
        effect_df,
        equal_root / "05_03_memory_subject_effects.csv",
        ["A1_geo_TD_slope", "B1_geo_minus_pairshuffled_TD_slope", "C1_geo_minus_pairnarrow_TD_slope"],
        out / "08_SC_07_memory_equal_vs_empirical_subject.csv",
        out / "08_SC_08_memory_equal_vs_empirical_summary.csv",
    )

    group_acc = np.zeros((len(CONDITIONS), len(core.SCALES), MAX_LAG), dtype=np.float64)
    first = True
    metric_rows = []
    profile_path = out / "08_SC_01_subject_profile_long.csv"
    for z in results:
        P = np.asarray(z["profile"], dtype=np.float64)
        group_acc += P
        chunks = []
        for ci, cond in enumerate(CONDITIONS):
            for si, S in enumerate(core.SCALES):
                M = P[ci, si]
                chunks.append(pd.DataFrame({
                    "subject": z["subject"], "condition": cond, "scale": int(S),
                    "lag": np.arange(1, MAX_LAG + 1, dtype=np.int32), "M": M,
                }))
                row = {"subject": z["subject"], "condition": cond, "scale": int(S)}
                row.update(profile_metrics(M, state_dim=int(core.ATLAS_SCALE)))
                metric_rows.append(row)
        pd.concat(chunks, ignore_index=True).to_csv(profile_path, index=False, mode="w" if first else "a", header=first)
        first = False

    subject_metrics = pd.DataFrame(metric_rows)
    subject_metrics.to_csv(out / "08_SC_03_subject_metrics.csv", index=False)

    group = group_acc / float(len(results))
    grows, gmets = [], []
    for ci, cond in enumerate(CONDITIONS):
        for si, S in enumerate(core.SCALES):
            M = group[ci, si]
            for k, mk in enumerate(M, 1):
                grows.append({"condition": cond, "scale": int(S), "lag": k, "M_mean": float(mk)})
            row = {"condition": cond, "scale": int(S)}
            row.update(profile_metrics(M, state_dim=int(core.ATLAS_SCALE)))
            gmets.append(row)
    pd.DataFrame(grows).to_csv(out / "08_SC_02_group_profile.csv", index=False)
    gm = pd.DataFrame(gmets)
    gm.to_csv(out / "08_SC_04_group_metrics.csv", index=False)

    eq_gm_path = equal_ext_root / "07_04_group_metrics.csv"
    if eq_gm_path.exists():
        eq = pd.read_csv(eq_gm_path)
        cols = ["condition", "scale", "MC_1_128", "MC_1_256", "MC_1_1024", "MC_1_1024_over_state_dim"]
        cols = [c for c in cols if c in eq.columns]
        a = eq[cols].copy()
        b = gm[cols].copy()
        m = b.merge(a, on=["condition", "scale"], suffixes=("_empirical", "_equal"))
        for q in ("MC_1_128", "MC_1_256", "MC_1_1024", "MC_1_1024_over_state_dim"):
            if f"{q}_empirical" in m.columns:
                m[f"{q}_empirical_minus_equal"] = m[f"{q}_empirical"] - m[f"{q}_equal"]
        m.to_csv(out / "08_SC_09_extended_memory_equal_vs_empirical_group.csv", index=False)


# -----------------------------------------------------------------------------
# Write environment outputs
# -----------------------------------------------------------------------------

def write_environment_outputs(core, all_rows, out: Path, equal_root: Path):
    df = all_rows.copy() if isinstance(all_rows, pd.DataFrame) else pd.DataFrame(all_rows)
    df.to_csv(out / "08_SC_10_environment_condition_level.csv", index=False)

    sel_rows = []
    for (sub, rep, r, Tenv, cond), d in df.groupby(["subject", "rep", "r", "T_env", "condition"]):
        bestS, va, ta, tm = core.choose_scale_from_validation(d)
        sel_rows.append({
            "subject": sub, "rep": int(rep), "r": float(r), "T_env": int(Tenv), "condition": cond,
            "S_star": int(bestS), "val_accuracy_star": va, "test_accuracy_star": ta, "test_mse_star": tm,
        })
    sel = pd.DataFrame(sel_rows)
    sel.to_csv(out / "08_SC_11_environment_selected_scales.csv", index=False)

    subject_rows = []
    for sub, ds in sel.groupby("subject"):
        acc = {k: [] for k in (
            "A2_geo_rho_Tenv_Sstar", "A3_geo_minus_zero_accuracy_slope",
            "B2_geo_minus_pairshuffled_rho_Tenv_Sstar", "B3_geo_minus_pairshuffled_accuracy_slope",
            "C2_geo_minus_pairnarrow_rho_Tenv_Sstar", "C3_geo_minus_pairnarrow_accuracy_slope",
        )}
        for (rep, r), drr in ds.groupby(["rep", "r"]):
            byc = {c: drr[drr.condition == c].sort_values("T_env") for c in CONDITIONS}
            rho_g = core.rho_log_env_scale(byc["geometric"].T_env.to_numpy(), byc["geometric"].S_star.to_numpy())
            rho_s = core.rho_log_env_scale(byc["pair_shuffled"].T_env.to_numpy(), byc["pair_shuffled"].S_star.to_numpy())
            rho_n = core.rho_log_env_scale(byc["pair_narrow"].T_env.to_numpy(), byc["pair_narrow"].S_star.to_numpy())
            acc["A2_geo_rho_Tenv_Sstar"].append(rho_g)
            acc["B2_geo_minus_pairshuffled_rho_Tenv_Sstar"].append(rho_g - rho_s)
            acc["C2_geo_minus_pairnarrow_rho_Tenv_Sstar"].append(rho_g - rho_n)
            g = byc["geometric"][["T_env", "test_accuracy_star"]].rename(columns={"test_accuracy_star": "geo"})
            for ctl, key in (
                ("zero", "A3_geo_minus_zero_accuracy_slope"),
                ("pair_shuffled", "B3_geo_minus_pairshuffled_accuracy_slope"),
                ("pair_narrow", "C3_geo_minus_pairnarrow_accuracy_slope"),
            ):
                c = byc[ctl][["T_env", "test_accuracy_star"]].rename(columns={"test_accuracy_star": "ctl"})
                m = g.merge(c, on="T_env")
                acc[key].append(core.slope_y_on_log2s(m.T_env.to_numpy(), (m.geo - m.ctl).to_numpy()))
        subject_rows.append({"subject": sub, **{k: float(np.nanmean(v)) for k, v in acc.items()}})
    sdf = pd.DataFrame(subject_rows).sort_values("subject")
    sdf.to_csv(out / "08_SC_12_environment_subject_effects.csv", index=False)

    effect_statistics(sdf, [
        ("source_effect", "A2_geo_rho_Tenv_Sstar", "A2_optimal_scale_matches_temporal_demand"),
        ("source_effect", "A3_geo_minus_zero_accuracy_slope", "A3_GEOMETRIC_advantage_over_ZERO_grows_with_demand"),
        ("assignment", "B2_geo_minus_pairshuffled_rho_Tenv_Sstar", "B2_assignment_strengthens_matching"),
        ("assignment", "B3_geo_minus_pairshuffled_accuracy_slope", "B3_assignment_value_grows_with_demand"),
        ("heterogeneity", "C2_geo_minus_pairnarrow_rho_Tenv_Sstar", "C2_heterogeneity_strengthens_matching"),
        ("heterogeneity", "C3_geo_minus_pairnarrow_accuracy_slope", "C3_heterogeneity_value_grows_with_demand"),
    ], out / "08_SC_13_environment_effect_statistics.csv")

    paired_robustness_summary(
        sdf,
        equal_root / "05_06_environment_subject_effects.csv",
        [
            "A2_geo_rho_Tenv_Sstar", "A3_geo_minus_zero_accuracy_slope",
            "B2_geo_minus_pairshuffled_rho_Tenv_Sstar", "B3_geo_minus_pairshuffled_accuracy_slope",
            "C2_geo_minus_pairnarrow_rho_Tenv_Sstar", "C3_geo_minus_pairnarrow_accuracy_slope",
        ],
        out / "08_SC_14_environment_equal_vs_empirical_subject.csv",
        out / "08_SC_15_environment_equal_vs_empirical_summary.csv",
    )

    surf = df.groupby(["condition", "T_env", "scale"], as_index=False).agg(
        val_accuracy=("val_accuracy", "mean"), test_accuracy=("test_accuracy", "mean"),
        test_mse=("test_mse", "mean"),
    )
    surf.to_csv(out / "08_SC_16_environment_surface_group.csv", index=False)
    ssg = sel.groupby(["condition", "T_env"], as_index=False).agg(
        mean_S_star=("S_star", "mean"), median_S_star=("S_star", "median"),
        mean_test_accuracy_star=("test_accuracy_star", "mean"),
    )
    ssg.to_csv(out / "08_SC_17_selected_scale_group.csv", index=False)


# -----------------------------------------------------------------------------
# Preflight, self-test, README
# -----------------------------------------------------------------------------

def preflight(core, subjects, out: Path):
    rows = []
    print(f"[08 SC] preflight: {len(subjects)} subjects", flush=True)
    for i, sf in enumerate(subjects, start=1):
        try:
            t = core.build_empirical_template(sf)
            signs, _b = core.make_signs_and_input(t, sf.subject, 1)
            w = core.base_weights(t, signs, "empirical")
            sums = np.bincount(t.dst, weights=np.abs(w), minlength=t.n)
            mag = np.log1p(np.maximum(t.selected_sc, 0.0))
            rows.append({
                "subject": sf.subject, "status": "ok", "n_regions": int(t.n),
                "n_undirected_edges": int(len(t.undirected_delta)),
                "incoming_abs_weight_sum_min": float(sums.min()),
                "incoming_abs_weight_sum_max": float(sums.max()),
                "log1p_sc_min": float(mag.min()), "log1p_sc_median": float(np.median(mag)),
                "log1p_sc_max": float(mag.max()), "error": "",
            })
        except Exception as e:
            rows.append({"subject": sf.subject, "status": "error", "error": repr(e)})
        if i == 1 or i % 10 == 0 or i == len(subjects):
            print(f"[08 SC preflight {i}/{len(subjects)}]", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "08_SC_00_preflight.csv", index=False)
    if len(df) != 50 or (df.status != "ok").any():
        raise RuntimeError("SC-weight preflight failed or full 50-subject set missing")


def run_self_test(core_path: str):
    core = load_core(core_path)
    tests: Dict[str, object] = {}
    tests["core_version"] = getattr(core, "SCRIPT_VERSION", None) == REQUIRED_CORE_VERSION
    tests["core_self_test"] = bool(core.run_self_test().get("all_pass", False))
    tests["replicate_ids"] = list(range(1, INTERNAL_REPS + 1)) == list(range(1, 9))
    tests["checkpoint_signature_compatible_v1_1"] = CHECKPOINT_SIGNATURE_VERSION == "08-mica-empirical-sc-weight-robustness-v1.1"
    tests["prefix_valid"] = required_prefix(core) + WASHOUT - MAX_LAG >= 0
    tests["null_reference"] = abs(NULL_R2_REFERENCE - 1.0 / 9999.0) < 1e-15
    tests["blas_thread_caps"] = all(os.environ.get(v) == "1" for v in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"))
    _g = group_identical_delay_patterns([
        (("a", 1), np.asarray([0,1,1], dtype=np.int32)),
        (("b", 2), np.asarray([0,1,1], dtype=np.int32)),
        (("c", 3), np.asarray([1,1,1], dtype=np.int32)),
    ])
    tests["exact_delay_dedup"] = (len(_g) == 2 and sorted(len(x["labels"]) for x in _g) == [1,2])

    # Synthetic empirical weighting normalization.
    n = 6
    ui = np.asarray([0,0,1,1,2,2,3,4,4], dtype=np.int32)
    uj = np.asarray([1,2,2,3,3,4,5,5,0], dtype=np.int32)
    src = np.empty(2*len(ui), dtype=np.int32); dst = np.empty_like(src)
    src[0::2], dst[0::2] = uj, ui; src[1::2], dst[1::2] = ui, uj
    indeg = np.bincount(dst, minlength=n).astype(np.int32)
    du = np.linspace(.1, 1.0, len(ui)); scu = np.geomspace(1, 100, len(ui))
    tpl = core.NetworkTemplate(n, 3, src, dst, np.repeat(du,2), np.repeat(scu,2), du, scu, ui, uj,
                               100*du, 100.0, indeg)
    signs = np.where(np.arange(src.size)%2==0, 1.0, -1.0)
    w = core.base_weights(tpl, signs, "empirical")
    sums = np.bincount(dst, weights=np.abs(w), minlength=n)
    tests["empirical_weight_normalization"] = bool(np.allclose(sums, 1.0, atol=1e-12, rtol=1e-12))

    # Batched OLS equivalence.
    rng = np.random.default_rng(123)
    ntr, nte, p, K = 300, 260, 12, 16
    st = K + 20
    tr_u = rng.normal(size=st + ntr); te_u = rng.normal(size=st + nte)
    Xtr = rng.normal(size=(ntr,p)); Xte = rng.normal(size=(nte,p))
    fast = memory_scores_from_states(Xtr, Xte, tr_u, te_u, st, st, max_lag=K, lag_batch=5)
    Ytr = target_block(tr_u, st, ntr, 1, K); Yte = target_block(te_u, st, nte, 1, K)
    xm = Xtr.mean(0); ym = Ytr.mean(0); Xc = Xtr-xm; Yc=Ytr-ym
    B = np.linalg.pinv(Xc.T@Xc, hermitian=True)@(Xc.T@Yc); pred=(Xte-xm)@B
    slow=[]
    for q in range(K):
        c=np.corrcoef(Yte[:,q], pred[:,q])[0,1]; slow.append(c*c)
    slow=np.asarray(slow)
    tests["batched_matches_naive"] = bool(np.allclose(fast, slow, rtol=1e-10, atol=1e-12))
    tests["batched_max_abs_error"] = float(np.max(np.abs(fast-slow)))

    # Checkpoint roundtrip.
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        cp = Path(td)/"x.npz"
        P=np.zeros((4,7,1024)); T=np.zeros((4,7,3))
        save_memory_cp(cp, "sig", P, T)
        z=load_memory_cp(cp,"sig")
        tests["checkpoint_roundtrip"] = bool(z is not None and z["profile_mean_r"].shape==(4,7,1024))

    tests["all_pass"] = bool(all(v for k,v in tests.items() if k != "batched_max_abs_error"))
    return tests


def write_readme(out: Path, phase: str):
    txt = f"""# 08 MICA-MICs Empirical SC-Weight Robustness\n\nScript version: `{SCRIPT_VERSION}`  \nCore required: `{REQUIRED_CORE_VERSION}`  \nPhase: {phase}  \nSubjects: 50  \nReplicates: 1..8  \nWeighting: log1p(empirical SC), normalized to unit incoming absolute weight sum  \nMax memory lag: {MAX_LAG}  \nFinite-sample E[r^2] reference: {NULL_R2_REFERENCE:.8f}\n\n## Scientific rule\n\nThis is a one-factor robustness analysis. Only recurrent edge magnitudes change relative to the\nequal-weight primary analysis. Topology, tract lengths, delay controls, signs, input vectors,\nstochastic sequences, scales, couplings, environmental persistence values, and validation/test\nrules are unchanged. The same primary A/B/C effect definitions are recomputed. No result-dependent\nretuning or new success threshold is introduced.\n\nThe lag-1024 memory analysis is used to verify capacity/horizon robustness. TD-based A1/B1/C1\nremain defined on lags 1..128 for exact comparability with core 05.\n"""
    (out / "08_SC_99_README_RUN.md").write_text(txt)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    a = parse_args()
    core_path = str(Path(a.core_script).expanduser())
    if a.self_test:
        print(json.dumps(run_self_test(core_path), indent=2))
        return

    t0 = time.time()
    core = load_core(core_path)
    if not getattr(core, "NUMBA_AVAILABLE", False):
        raise RuntimeError("Numba is required for the full robustness run; refusing slow Python fallback.")
    out = Path(a.outdir).expanduser(); out.mkdir(parents=True, exist_ok=True)
    equal_root = Path(a.equal_root).expanduser()
    equal_ext_root = Path(a.equal_extended_root).expanduser()
    cfg = core_config(core, a.data_root, str(out), a.phase)
    print(f"[08 SC] start phase={a.phase} workers={a.workers}", flush=True)
    subjects = core.discover_subjects(cfg)
    preflight(core, subjects, out)
    write_readme(out, a.phase)

    audit = {
        "script_version": SCRIPT_VERSION,
        "core_version": REQUIRED_CORE_VERSION,
        "phase": a.phase,
        "subjects": len(subjects),
        "replicate_ids": list(range(1, INTERNAL_REPS + 1)),
        "weighting": "empirical_log1p_incoming_normalized",
        "max_lag": MAX_LAG,
    }

    if a.phase in ("memory", "all"):
        print("[08 SC] memory robustness", flush=True)
        sig = signature(core, "memory")
        payloads = [(asdict(sf), core_path, str(out), sig, not a.no_checkpoint) for sf in subjects]
        results=[]; errors=[]
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            futs={ex.submit(sc_memory_subject_worker,p):p for p in payloads}
            done=0
            for fut in as_completed(futs):
                try:
                    results.append(fut.result())
                except Exception:
                    errors.append(traceback.format_exc()); print(errors[-1], flush=True)
                done+=1; print(f"[08 SC memory {done}/{len(payloads)}]", flush=True)
        if errors:
            raise RuntimeError(f"SC memory failed for {len(errors)} subjects")
        write_memory_outputs(core, results, out, equal_root, equal_ext_root)
        audit["memory_subjects"] = len(results)

    if a.phase in ("environment", "all"):
        print("[08 SC] environment robustness", flush=True)
        sig = signature(core, "environment")
        payloads=[
            (asdict(sf), rep, core_path, str(Path(a.data_root).expanduser()), str(out), sig, not a.no_checkpoint)
            for sf in subjects for rep in range(1, INTERNAL_REPS + 1)
        ]
        errors=[]; stream_path = out / "08_SC_10_environment_condition_level.csv"
        if stream_path.exists():
            stream_path.unlink()
        first = True; n_rows = 0; cached_n = 0
        print(f"[08 SC] environment jobs={len(payloads)} (= {len(subjects)} subjects x {INTERNAL_REPS} reps)", flush=True)
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            futs={ex.submit(sc_environment_rep_worker,p):p for p in payloads}
            done=0
            for fut in as_completed(futs):
                p=futs[fut]
                try:
                    z = fut.result()
                    pdf = pd.DataFrame(z["rows"])
                    pdf.to_csv(stream_path, index=False, mode="w" if first else "a", header=first)
                    first = False; n_rows += len(pdf); cached_n += int(bool(z["cached"]))
                    status = "cached" if z["cached"] else "computed"
                    print(f"[08 SC environment {done+1}/{len(payloads)}] {z['subject']} rep={z['rep']} {status}", flush=True)
                except Exception:
                    errors.append(traceback.format_exc()); print(errors[-1], flush=True)
                    print(f"[08 SC environment {done+1}/{len(payloads)}] ERROR", flush=True)
                done+=1
        if errors:
            raise RuntimeError(f"SC environment failed for {len(errors)} subject-replicate jobs")
        env_df = pd.read_csv(stream_path)
        write_environment_outputs(core, env_df, out, equal_root)
        audit["environment_rows"] = int(n_rows)
        audit["environment_rep_jobs"] = int(len(payloads))
        audit["environment_cached_rep_jobs"] = int(cached_n)

    audit["runtime_seconds"] = float(time.time()-t0)
    (out / "08_SC_98_audit.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2), flush=True)
    print(f"Outputs: {out}", flush=True)


if __name__ == "__main__":
    main()
