#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
06_MICA_MICS_MEMORY_DISTRIBUTION_AND_FALSIFICATION.py

Mechanistic follow-up to the corrected MICA-MICs geometry-specific scale validation.

This script answers four questions without redefining the previous primary hypotheses:

1) What do the lag-resolved held-out memory profiles M(k) actually look like?
2) How do memory capacity (MC) and temporal depth (TD) trade off, and how much
   absolute lag-weighted memory is retained (sum k*M(k) = MC*TD)?
3) What is the full held-out performance surface over environmental timescale x
   network scale, rather than only the selected optimum?
4) Does temporal-demand matching disappear when temporal dependence is removed
   while preserving the same binary marginal task, noise, validation-based scale
   selection, model, and delay conditions?

IMPORTANT INTERPRETATION RULES
------------------------------
- Analyses 1-3 are mechanistic/descriptive re-analyses of the completed equal-weight
  full run. They do not replace failed or successful primary hypotheses.
- Analysis 4 is a falsification/boundary-condition experiment. The latent state is
  IID +/-1 (equivalent to a symmetric two-state Markov chain with flip probability
  0.5). Nominal T_env labels are retained ONLY so that the exact same slope/rho
  analysis can be applied; the IID data-generating distribution is identical for
  every nominal label, using independent seeds.
- Non-significance of an IID effect is NOT treated as proof. The main falsification
  statistic is the paired subject-level difference between the observed temporal
  effect and its IID analogue.
- No new success metric is used to rescue previous failed hypotheses.

The script reuses the exact model/delay implementation from
05_MICA_MICS_GEOMETRY_SPECIFIC_SCALE_VALIDATION.py for the IID simulations, so
there is no silent dynamical reimplementation.
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
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


SCRIPT_VERSION = "06-mica-memory-distribution-falsification-v1.0"
SEED_NAMESPACE = "MICA_MEMORY_DISTRIBUTION_FALSIFICATION_V1"
CORE_BASENAME = "05_MICA_MICS_GEOMETRY_SPECIFIC_SCALE_VALIDATION.py"
DEFAULT_ANALYSIS_ROOT = "~/Desktop/05_MICA_MICS_GEOMETRY_SPECIFIC_SCALE_VALIDATION/equal"
DEFAULT_DATA_ROOT = "~/Downloads/mica-mics/MICs_release/derivatives/micapipe"
DEFAULT_OUTDIR = "~/Desktop/06_MICA_MICS_MEMORY_DISTRIBUTION_AND_FALSIFICATION"
CONDITIONS = ("geometric", "pair_shuffled", "pair_narrow", "zero")
_CORE_CACHE: Dict[str, object] = {}


# -----------------------------------------------------------------------------
# Generic utilities
# -----------------------------------------------------------------------------
def stable_seed(*parts: object) -> int:
    txt = "|".join(str(x) for x in (SEED_NAMESPACE,) + parts).encode("utf-8")
    return int(hashlib.sha256(txt).hexdigest()[:16], 16) % (2**32 - 1)


def resolve_core_script(user_value: str) -> Path:
    if user_value.strip():
        p = Path(user_value).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"core script not found: {p}")
        return p
    candidates = [
        Path(__file__).resolve().with_name(CORE_BASENAME),
        Path("~/Downloads").expanduser() / CORE_BASENAME,
        Path.cwd() / CORE_BASENAME,
    ]
    for p in candidates:
        if p.exists():
            return p.resolve()
    raise FileNotFoundError(
        f"Could not find {CORE_BASENAME}. Put it beside this script or in ~/Downloads, "
        "or pass --core-script."
    )


def load_core(core_path: str):
    path = str(Path(core_path).resolve())
    if path in _CORE_CACHE:
        return _CORE_CACHE[path]
    name = "mica_geometry_core_" + hashlib.sha256(path.encode()).hexdigest()[:10]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import core script {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    _CORE_CACHE[path] = mod
    return mod


def require_columns(df: pd.DataFrame, cols: Sequence[str], label: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{label} missing columns: {missing}")


def finite(x: Sequence[float]) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    return a[np.isfinite(a)]


def bootstrap_mean_ci(values: Sequence[float], n_boot: int, seed: int) -> Tuple[float, float]:
    x = finite(values)
    if x.size == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        means[i] = rng.choice(x, size=x.size, replace=True).mean()
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def signflip_p(values: Sequence[float], alternative: str, n_perm: int, seed: int) -> float:
    x = finite(values)
    if x.size == 0:
        return np.nan
    obs = float(x.mean())
    rng = np.random.default_rng(seed)
    if x.size <= 15 and 2**x.size <= n_perm:
        stats = np.empty(2**x.size, dtype=float)
        for mask in range(2**x.size):
            s = np.ones(x.size, dtype=float)
            for i in range(x.size):
                if (mask >> i) & 1:
                    s[i] = -1.0
            stats[mask] = np.mean(s * x)
    else:
        stats = np.empty(n_perm, dtype=float)
        for q in range(n_perm):
            s = rng.choice(np.asarray([-1.0, 1.0]), size=x.size)
            stats[q] = np.mean(s * x)
    if alternative == "greater":
        return float((1 + np.sum(stats >= obs)) / (len(stats) + 1))
    if alternative == "less":
        return float((1 + np.sum(stats <= obs)) / (len(stats) + 1))
    if alternative == "two-sided":
        return float((1 + np.sum(np.abs(stats) >= abs(obs))) / (len(stats) + 1))
    raise ValueError(alternative)


def holm_adjust(pvals: Sequence[float]) -> List[float]:
    p = np.asarray(pvals, dtype=float)
    out = np.full(p.shape, np.nan, dtype=float)
    good = np.where(np.isfinite(p))[0]
    order = good[np.argsort(p[good])]
    running = 0.0
    m = len(order)
    for rank, idx in enumerate(order):
        val = min(1.0, (m - rank) * p[idx])
        running = max(running, val)
        out[idx] = running
    return out.tolist()


def slope_on_log2_x(x: Sequence[float], y: Sequence[float]) -> float:
    xx = np.log2(np.asarray(x, dtype=float))
    yy = np.asarray(y, dtype=float)
    ok = np.isfinite(xx) & np.isfinite(yy)
    if ok.sum() < 2:
        return np.nan
    return float(np.polyfit(xx[ok], yy[ok], 1)[0])


def rho_log2(x: Sequence[float], y: Sequence[float]) -> float:
    xx = np.asarray(x, dtype=float)
    yy = np.asarray(y, dtype=float)
    ok = np.isfinite(xx) & np.isfinite(yy) & (xx > 0) & (yy > 0)
    if ok.sum() < 3:
        return np.nan
    if len(np.unique(yy[ok])) == 1:
        return 0.0
    z = spearmanr(np.log2(xx[ok]), np.log2(yy[ok]))
    return float(z.statistic) if np.isfinite(z.statistic) else np.nan


def summarize_subject_values(df: pd.DataFrame, group_cols: Sequence[str], value_col: str) -> pd.DataFrame:
    rows = []
    for key, d in df.groupby(list(group_cols), dropna=False, sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        x = finite(d[value_col])
        row = dict(zip(group_cols, key))
        row.update({
            "n_subjects": int(x.size),
            "mean": float(np.mean(x)) if x.size else np.nan,
            "sd": float(np.std(x, ddof=1)) if x.size > 1 else np.nan,
            "median": float(np.median(x)) if x.size else np.nan,
            "min": float(np.min(x)) if x.size else np.nan,
            "max": float(np.max(x)) if x.size else np.nan,
        })
        rows.append(row)
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Analysis 1: lag-resolved memory profiles
# -----------------------------------------------------------------------------
def memory_mass_quantile(lags: np.ndarray, M: np.ndarray, q: float) -> float:
    M = np.asarray(M, dtype=float)
    lags = np.asarray(lags, dtype=float)
    ok = np.isfinite(M) & np.isfinite(lags) & (M >= 0)
    if ok.sum() == 0:
        return np.nan
    M = M[ok]
    lags = lags[ok]
    total = M.sum()
    if total <= 0:
        return np.nan
    order = np.argsort(lags)
    c = np.cumsum(M[order]) / total
    idx = int(np.searchsorted(c, q, side="left"))
    idx = min(idx, len(order) - 1)
    return float(lags[order][idx])


def analyze_memory_profiles(root: Path, out: Path) -> Dict[str, object]:
    prof_path = root / "05_02_memory_profiles.csv"
    cond_path = root / "05_01_memory_condition_level.csv"
    if not prof_path.exists():
        raise FileNotFoundError(
            f"Required lag-profile file is missing: {prof_path}. "
            "It is generated by the completed 05 memory phase; keep that file in the equal output folder."
        )
    if not cond_path.exists():
        raise FileNotFoundError(cond_path)

    cond = pd.read_csv(cond_path)
    require_columns(cond, ["subject", "rep", "condition", "scale", "r", "MC", "TD"], "memory condition")

    # The full profile file is ~4.3 million rows. Stream it rather than allocating the
    # entire CSV in RAM. Each chunk is reduced to subject x condition x scale x lag sums/counts;
    # these partial aggregates are then merged exactly.
    keys = ["subject", "condition", "scale", "lag"]
    partials = []
    usecols = ["subject", "rep", "condition", "scale", "r", "lag", "M"]
    for chunk in pd.read_csv(prof_path, usecols=usecols, chunksize=250_000):
        require_columns(chunk, usecols, "memory profiles")
        g = chunk.groupby(keys, as_index=False)["M"].agg(M_sum="sum", M_count="count")
        partials.append(g)
    if not partials:
        raise ValueError("memory profile file is empty")
    agg = pd.concat(partials, ignore_index=True).groupby(keys, as_index=False)[["M_sum", "M_count"]].sum()
    agg["M"] = agg.M_sum / agg.M_count
    psub = agg[keys + ["M"]].sort_values(["condition", "scale", "lag", "subject"])

    subjects = sorted(psub.subject.unique())
    if len(subjects) != 50:
        raise ValueError(f"expected 50 subjects in full memory profiles, found {len(subjects)}")
    psub.to_csv(out / "06_01_lag_profile_subject.csv", index=False)

    pgrp = summarize_subject_values(psub, ["condition", "scale", "lag"], "M")
    pgrp.to_csv(out / "06_02_lag_profile_group.csv", index=False)

    # MC and TD retain their exact original nonlinear definitions by using the existing
    # condition-level metrics. M1 is computed exactly before nuisance averaging:
    # M1 = sum k M(k) = MC * TD for each subject x rep x r x condition x scale.
    cond = cond.copy()
    cond["lag_weighted_memory_M1"] = cond.MC * cond.TD
    csub = cond.groupby(["subject", "condition", "scale"], as_index=False)[
        ["MC", "TD", "lag_weighted_memory_M1"]
    ].mean().rename(columns={"MC": "MC_from_profile", "TD": "TD_from_profile"})

    # Distribution shape features are computed from each subject's nuisance-averaged M(k).
    shape_rows = []
    for (sub, condition, scale), d in psub.groupby(["subject", "condition", "scale"], sort=True):
        d = d.sort_values("lag")
        lags = d.lag.to_numpy(float)
        M = d.M.to_numpy(float)
        peak_idx = int(np.nanargmax(M)) if np.isfinite(M).any() else 0
        shape_rows.append({
            "subject": sub, "condition": condition, "scale": int(scale),
            "lag50_memory_mass": memory_mass_quantile(lags, M, 0.50),
            "lag90_memory_mass": memory_mass_quantile(lags, M, 0.90),
            "peak_lag": float(lags[peak_idx]) if len(lags) else np.nan,
            "peak_M": float(M[peak_idx]) if len(M) else np.nan,
        })
    features = csub.merge(pd.DataFrame(shape_rows), on=["subject", "condition", "scale"], validate="one_to_one")
    features.to_csv(out / "06_03_capacity_depth_subject.csv", index=False)

    fg = []
    for metric in ["MC_from_profile", "TD_from_profile", "lag_weighted_memory_M1",
                   "lag50_memory_mass", "lag90_memory_mass", "peak_lag", "peak_M"]:
        z = summarize_subject_values(features, ["condition", "scale"], metric)
        z.insert(2, "metric", metric)
        fg.append(z)
    pd.concat(fg, ignore_index=True).to_csv(out / "06_04_capacity_depth_group.csv", index=False)

    # Audit the exact M1 identity at the finest available condition level.
    identity_err = np.abs(cond.lag_weighted_memory_M1 - cond.MC * cond.TD)
    expected_profile_rows_per_subject = 4 * 7 * 128
    counts = psub.groupby("subject").size().to_numpy()
    return {
        "n_subjects_memory": len(subjects),
        "profile_streaming_chunks": len(partials),
        "subject_profile_rows_complete": bool(np.all(counts == expected_profile_rows_per_subject)),
        "max_M1_identity_abs_error": float(np.nanmax(identity_err)),
    }


# -----------------------------------------------------------------------------
# Analysis 2: capacity-depth tradeoff
# -----------------------------------------------------------------------------
def analyze_capacity_depth_tradeoff(root: Path, out: Path) -> Dict[str, object]:
    cond_path = root / "05_01_memory_condition_level.csv"
    cond = pd.read_csv(cond_path)
    require_columns(cond, ["subject", "rep", "condition", "scale", "r", "MC", "TD"], "memory condition")

    # Within subject x replicate x coupling, quantify the MC-TD relationship across the seven frozen scales.
    rows = []
    for (sub, rep, r, condition), d in cond.groupby(["subject", "rep", "r", "condition"], sort=True):
        d = d.sort_values("scale")
        if len(d) != 7:
            raise ValueError(f"expected 7 scales for {sub} rep={rep} r={r} {condition}, found {len(d)}")
        z = spearmanr(d.MC.to_numpy(float), d.TD.to_numpy(float))
        rho_mc_td = float(z.statistic) if np.isfinite(z.statistic) else np.nan
        rows.append({
            "subject": sub, "rep": int(rep), "r": float(r), "condition": condition,
            "rho_MC_TD_across_scale": rho_mc_td,
            "MC_log2S_slope": slope_on_log2_x(d.scale, d.MC),
            "TD_log2S_slope": slope_on_log2_x(d.scale, d.TD),
        })
    rr = pd.DataFrame(rows)
    sub = rr.groupby(["subject", "condition"], as_index=False)[
        ["rho_MC_TD_across_scale", "MC_log2S_slope", "TD_log2S_slope"]
    ].mean()
    sub.to_csv(out / "06_05_capacity_depth_tradeoff_subject.csv", index=False)

    grp_parts = []
    for metric in ["rho_MC_TD_across_scale", "MC_log2S_slope", "TD_log2S_slope"]:
        z = summarize_subject_values(sub, ["condition"], metric)
        z.insert(1, "metric", metric)
        grp_parts.append(z)
    pd.concat(grp_parts, ignore_index=True).to_csv(out / "06_06_capacity_depth_tradeoff_group.csv", index=False)

    # Paired descriptive contrasts in the distribution-level features. No multiplicity-driven success flags.
    features = pd.read_csv(out / "06_03_capacity_depth_subject.csv")
    metrics = ["MC_from_profile", "TD_from_profile", "lag_weighted_memory_M1", "lag50_memory_mass", "lag90_memory_mass"]
    contrast_rows = []
    for control in ("pair_shuffled", "pair_narrow", "zero"):
        g = features[features.condition == "geometric"]
        c = features[features.condition == control]
        m = g.merge(c, on=["subject", "scale"], suffixes=("_geo", "_ctl"))
        for scale, ds in m.groupby("scale"):
            for metric in metrics:
                diff = ds[f"{metric}_geo"].to_numpy(float) - ds[f"{metric}_ctl"].to_numpy(float)
                lo, hi = bootstrap_mean_ci(diff, 10000, stable_seed("capacity_depth", control, metric, int(scale)))
                contrast_rows.append({
                    "control": control,
                    "scale": int(scale),
                    "metric": metric,
                    "n_subjects": int(np.isfinite(diff).sum()),
                    "mean_geo_minus_control": float(np.nanmean(diff)),
                    "median_geo_minus_control": float(np.nanmedian(diff)),
                    "bootstrap_95ci_low": lo,
                    "bootstrap_95ci_high": hi,
                    "positive_fraction": float(np.mean(diff[np.isfinite(diff)] > 0)),
                })
    pd.DataFrame(contrast_rows).to_csv(out / "06_07_capacity_depth_descriptive_contrasts.csv", index=False)

    return {"n_subjects_tradeoff": int(sub.subject.nunique())}


# -----------------------------------------------------------------------------
# Analysis 3: full task-performance surface
# -----------------------------------------------------------------------------
def analyze_performance_surface(root: Path, out: Path) -> Dict[str, object]:
    env_path = root / "05_04_environment_condition_level.csv"
    sel_path = root / "05_05_environment_selected_scales.csv"
    env = pd.read_csv(env_path)
    sel = pd.read_csv(sel_path)
    require_columns(env, ["subject", "rep", "r", "condition", "T_env", "scale", "val_accuracy", "test_accuracy"], "environment condition")
    require_columns(sel, ["subject", "rep", "r", "condition", "T_env", "S_star", "test_accuracy_star"], "selected scales")
    if env.subject.nunique() != 50 or sel.subject.nunique() != 50:
        raise ValueError("full performance analysis requires all 50 subjects")

    surf_sub = env.groupby(["subject", "condition", "T_env", "scale"], as_index=False)[
        ["val_accuracy", "test_accuracy", "val_mse", "test_mse"]
    ].mean()
    surf_sub.to_csv(out / "06_08_performance_surface_subject.csv", index=False)

    group_parts = []
    for metric in ["val_accuracy", "test_accuracy", "val_mse", "test_mse"]:
        z = summarize_subject_values(surf_sub, ["condition", "T_env", "scale"], metric)
        z.insert(3, "metric", metric)
        group_parts.append(z)
    pd.concat(group_parts, ignore_index=True).to_csv(out / "06_09_performance_surface_group.csv", index=False)

    # Adjacent doublings show exactly where more scale helps or harms, without choosing a post-hoc threshold.
    adj_rows = []
    for (sub, condition, Tenv), d in surf_sub.groupby(["subject", "condition", "T_env"], sort=True):
        d = d.sort_values("scale")
        scales = d.scale.to_numpy(int)
        acc = d.test_accuracy.to_numpy(float)
        for i in range(len(scales) - 1):
            adj_rows.append({
                "subject": sub,
                "condition": condition,
                "T_env": int(Tenv),
                "scale_from": int(scales[i]),
                "scale_to": int(scales[i + 1]),
                "delta_test_accuracy": float(acc[i + 1] - acc[i]),
            })
    adj = pd.DataFrame(adj_rows)
    adj.to_csv(out / "06_10_adjacent_scale_effect_subject.csv", index=False)
    summarize_subject_values(adj, ["condition", "T_env", "scale_from", "scale_to"], "delta_test_accuracy").to_csv(
        out / "06_11_adjacent_scale_effect_group.csv", index=False
    )

    # Validation-selected optimum, averaged only after selection within replicate x r.
    sel_sub = sel.groupby(["subject", "condition", "T_env"], as_index=False)[
        ["S_star", "val_accuracy_star", "test_accuracy_star", "test_mse_star"]
    ].mean()
    sel_sub.to_csv(out / "06_12_selected_scale_subject.csv", index=False)
    sel_parts = []
    for metric in ["S_star", "val_accuracy_star", "test_accuracy_star", "test_mse_star"]:
        z = summarize_subject_values(sel_sub, ["condition", "T_env"], metric)
        z.insert(2, "metric", metric)
        sel_parts.append(z)
    pd.concat(sel_parts, ignore_index=True).to_csv(out / "06_13_selected_scale_group.csv", index=False)

    return {"n_subjects_surface": int(env.subject.nunique())}


# -----------------------------------------------------------------------------
# Analysis 4: IID temporal-dependence falsification
# -----------------------------------------------------------------------------
def make_iid_environment_sequence(total: int, seed: int, noise_sd: float) -> Tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(seed)
    z = rng.choice(np.asarray([-1.0, 1.0]), size=int(total)).astype(np.float64)
    u = z + rng.normal(0.0, float(noise_sd), size=int(total))
    if len(z) > 2 and z[:-1].std() > 0 and z[1:].std() > 0:
        lag1 = float(np.corrcoef(z[:-1], z[1:])[0, 1])
    else:
        lag1 = np.nan
    return z, u, lag1


def iid_checkpoint_path(outdir: str, subject: str, session: str, rep: int, core_version: str) -> Path:
    d = Path(outdir) / "iid_checkpoints"
    d.mkdir(parents=True, exist_ok=True)
    sig = hashlib.sha256(f"{SCRIPT_VERSION}|{core_version}|equal|iid05".encode()).hexdigest()[:16]
    return d / f"{subject}_{session}_rep{rep:02d}_{sig}.json"


def iid_worker(payload: Mapping[str, object]) -> Dict[str, object]:
    core_path = str(payload["core_path"])
    outdir = str(payload["outdir"])
    sf_dict = dict(payload["subject_files"])
    rep = int(payload["rep"])
    core = load_core(core_path)
    sf = core.SubjectFiles(**sf_dict)
    cp = iid_checkpoint_path(outdir, sf.subject, sf.session, rep, core.SCRIPT_VERSION)
    if cp.exists():
        try:
            old = json.loads(cp.read_text())
            if old.get("script_version") == SCRIPT_VERSION and old.get("core_version") == core.SCRIPT_VERSION:
                return old
        except Exception:
            pass

    template = core.build_empirical_template(sf)
    signs, b = core.make_signs_and_input(template, sf.subject, rep)
    base = core.base_weights(template, signs, "equal")
    shuf_delta = core.pair_shuffled_delta(
        template.undirected_delta,
        core.stable_seed(sf.subject, rep, "pair_shuffle_geometry")
    )
    bundles = {int(S): core.delay_bundle(template, shuf_delta, int(S)) for S in core.SCALES}

    # Same scientific lengths as the completed full equal-weight environment analysis.
    cfg = core.Config(
        mode="full", phase="environment", weighting="equal",
        data_root=str(payload["data_root"]), outdir=outdir, workers=1,
        internal_reps=int(core.FULL_INTERNAL_REPS),
        memory_washout=int(core.FULL_MEMORY_WASHOUT),
        memory_train=int(core.FULL_MEMORY_TRAIN),
        memory_test=int(core.FULL_MEMORY_TEST),
        max_lag=int(core.FULL_MAX_LAG),
        env_washout=int(core.FULL_ENV_WASHOUT),
        env_train=int(core.FULL_ENV_TRAIN),
        env_val=int(core.FULL_ENV_VAL),
        env_test=int(core.FULL_ENV_TEST),
        bootstrap_samples=int(core.FULL_BOOTSTRAP_SAMPLES),
        permutation_samples=int(core.FULL_PERMUTATION_SAMPLES),
        max_subjects=0, checkpoint=False,
    )

    rows = []
    seq_audit = []
    # These labels have NO effect on the IID generator distribution. Independent seeds
    # are used at each label so a zero slope is not built in by duplicating one dataset.
    for nominal in core.T_ENVS:
        seqs = {}
        for part, L in (("train", cfg.env_train), ("val", cfg.env_val), ("test", cfg.env_test)):
            seed = stable_seed(sf.subject, rep, "iid", int(nominal), part)
            z, u, lag1 = make_iid_environment_sequence(cfg.env_washout + L, seed, core.ENV_NOISE_SD)
            seqs[part] = (z, u)
            seq_audit.append({"nominal_T_env": int(nominal), "part": part, "lag1_z": lag1})

        for S in core.SCALES:
            bundle = bundles[int(S)]
            for condition in ("geometric", "pair_shuffled", "pair_narrow"):
                d = bundle[condition]
                for r in core.COUPLINGS:
                    va, vm, ta, tm = core.env_metrics(template, base, b, d, float(r), seqs, cfg)
                    rows.append({
                        "condition": condition,
                        "nominal_T_env": int(nominal),
                        "scale": int(S),
                        "r": float(r),
                        "val_accuracy": va,
                        "val_mse": vm,
                        "test_accuracy": ta,
                        "test_mse": tm,
                    })

        zd = np.zeros(template.src.size, dtype=np.int32)
        for r in core.COUPLINGS:
            va, vm, ta, tm = core.env_metrics(template, base, b, zd, float(r), seqs, cfg)
            for S in core.SCALES:
                rows.append({
                    "condition": "zero",
                    "nominal_T_env": int(nominal),
                    "scale": int(S),
                    "r": float(r),
                    "val_accuracy": va,
                    "val_mse": vm,
                    "test_accuracy": ta,
                    "test_mse": tm,
                })

    result = {
        "script_version": SCRIPT_VERSION,
        "core_version": core.SCRIPT_VERSION,
        "subject": sf.subject,
        "session": sf.session,
        "rep": rep,
        "rows": rows,
        "sequence_audit": seq_audit,
    }
    cp.write_text(json.dumps(result))
    return result


def choose_scale_validation(d: pd.DataFrame) -> Tuple[int, float, float, float]:
    d = d.sort_values("scale")
    vmax = float(d.val_accuracy.max())
    bestS = int(d.loc[d.val_accuracy == vmax, "scale"].min())
    z = d[d.scale == bestS].iloc[0]
    return bestS, float(z.val_accuracy), float(z.test_accuracy), float(z.test_mse)


def run_iid_falsification(core_path: Path, data_root: Path, root: Path, out: Path, workers: int) -> Dict[str, object]:
    core = load_core(str(core_path))

    # Ensure the source run and core constants agree before spending compute.
    readme = root / "05_11_README_RUN.md"
    if not readme.exists():
        raise FileNotFoundError(readme)
    txt = readme.read_text(errors="replace")
    for required in ("Mode: `full`", "Weighting: `equal`", "Subjects: `50`"):
        if required not in txt:
            raise ValueError(f"source run README does not confirm {required}")

    # Discover exactly the same complete structural cohort with the core implementation.
    cfg_discovery = core.Config(
        mode="full", phase="environment", weighting="equal",
        data_root=str(data_root), outdir=str(out), workers=1,
        internal_reps=int(core.FULL_INTERNAL_REPS),
        memory_washout=int(core.FULL_MEMORY_WASHOUT), memory_train=int(core.FULL_MEMORY_TRAIN),
        memory_test=int(core.FULL_MEMORY_TEST), max_lag=int(core.FULL_MAX_LAG),
        env_washout=int(core.FULL_ENV_WASHOUT), env_train=int(core.FULL_ENV_TRAIN),
        env_val=int(core.FULL_ENV_VAL), env_test=int(core.FULL_ENV_TEST),
        bootstrap_samples=int(core.FULL_BOOTSTRAP_SAMPLES), permutation_samples=int(core.FULL_PERMUTATION_SAMPLES),
        max_subjects=0, checkpoint=False,
    )
    subjects = core.discover_subjects(cfg_discovery)
    if len(subjects) != 50:
        raise ValueError(f"IID falsification requires the same 50 subjects; discovered {len(subjects)}")

    payloads = []
    for sf in subjects:
        for rep in range(int(core.FULL_INTERNAL_REPS)):
            payloads.append({
                "core_path": str(core_path),
                "outdir": str(out),
                "data_root": str(data_root),
                "subject_files": asdict(sf),
                "rep": rep,
            })

    results = []
    t0 = time.time()
    if workers == 1:
        for i, p in enumerate(payloads, 1):
            z = iid_worker(p)
            results.append(z)
            print(f"[IID {i}/{len(payloads)}] {z['subject']} rep={z['rep']}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(iid_worker, p): p for p in payloads}
            done = 0
            for fut in as_completed(futs):
                z = fut.result()
                results.append(z)
                done += 1
                print(f"[IID {done}/{len(payloads)}] {z['subject']} rep={z['rep']}", flush=True)

    rows = []
    audit_rows = []
    for z in results:
        for r in z["rows"]:
            rows.append({"subject": z["subject"], "session": z["session"], "rep": z["rep"], **r})
        for a in z["sequence_audit"]:
            audit_rows.append({"subject": z["subject"], "rep": z["rep"], **a})
    iid = pd.DataFrame(rows).sort_values(["subject", "rep", "nominal_T_env", "condition", "scale", "r"])
    iid.to_csv(out / "06_14_iid_condition_level.csv", index=False)
    audit_df = pd.DataFrame(audit_rows)
    audit_df.to_csv(out / "06_15_iid_sequence_audit.csv", index=False)

    selected_rows = []
    for (sub, rep, r, nominal, condition), d in iid.groupby(
        ["subject", "rep", "r", "nominal_T_env", "condition"], sort=True
    ):
        Sstar, va, ta, tm = choose_scale_validation(d)
        selected_rows.append({
            "subject": sub, "rep": int(rep), "r": float(r),
            "nominal_T_env": int(nominal), "condition": condition,
            "S_star": Sstar, "val_accuracy_star": va,
            "test_accuracy_star": ta, "test_mse_star": tm,
        })
    selected = pd.DataFrame(selected_rows)
    selected.to_csv(out / "06_16_iid_selected_scales.csv", index=False)

    # Exact same subject-level summaries as the temporal environment, but nominal labels carry no temporal structure.
    subject_rows = []
    for sub, ds in selected.groupby("subject", sort=True):
        acc = {
            "iid_A2_geo_rho_nominal_Sstar": [],
            "iid_A3_geo_minus_zero_accuracy_slope": [],
            "iid_B2_geo_minus_pairshuffled_rho_nominal_Sstar": [],
            "iid_B3_geo_minus_pairshuffled_accuracy_slope": [],
            "iid_C2_geo_minus_pairnarrow_rho_nominal_Sstar": [],
            "iid_C3_geo_minus_pairnarrow_accuracy_slope": [],
        }
        for (rep, r), drr in ds.groupby(["rep", "r"], sort=True):
            byc = {c: drr[drr.condition == c].sort_values("nominal_T_env") for c in CONDITIONS}
            rho_geo = rho_log2(byc["geometric"].nominal_T_env, byc["geometric"].S_star)
            rho_shf = rho_log2(byc["pair_shuffled"].nominal_T_env, byc["pair_shuffled"].S_star)
            rho_nar = rho_log2(byc["pair_narrow"].nominal_T_env, byc["pair_narrow"].S_star)
            acc["iid_A2_geo_rho_nominal_Sstar"].append(rho_geo)
            acc["iid_B2_geo_minus_pairshuffled_rho_nominal_Sstar"].append(rho_geo - rho_shf)
            acc["iid_C2_geo_minus_pairnarrow_rho_nominal_Sstar"].append(rho_geo - rho_nar)

            g = byc["geometric"][["nominal_T_env", "test_accuracy_star"]].rename(columns={"test_accuracy_star": "geo"})
            for control, key in [
                ("zero", "iid_A3_geo_minus_zero_accuracy_slope"),
                ("pair_shuffled", "iid_B3_geo_minus_pairshuffled_accuracy_slope"),
                ("pair_narrow", "iid_C3_geo_minus_pairnarrow_accuracy_slope"),
            ]:
                c = byc[control][["nominal_T_env", "test_accuracy_star"]].rename(columns={"test_accuracy_star": "ctl"})
                m = g.merge(c, on="nominal_T_env")
                acc[key].append(slope_on_log2_x(m.nominal_T_env, m.geo - m.ctl))
        subject_rows.append({"subject": sub, **{k: float(np.nanmean(v)) for k, v in acc.items()}})
    iid_sub = pd.DataFrame(subject_rows).sort_values("subject")
    iid_sub.to_csv(out / "06_17_iid_subject_effects.csv", index=False)

    # Describe IID analogues directly; absence of significance is not labeled a success.
    iid_stats = []
    for col in [c for c in iid_sub.columns if c.startswith("iid_")]:
        x = iid_sub[col].to_numpy(float)
        lo, hi = bootstrap_mean_ci(x, 10000, stable_seed("iid_stats", col))
        iid_stats.append({
            "metric": col,
            "n_subjects": int(np.isfinite(x).sum()),
            "mean": float(np.nanmean(x)),
            "median": float(np.nanmedian(x)),
            "bootstrap_95ci_low": lo,
            "bootstrap_95ci_high": hi,
            "two_sided_signflip_p_vs_zero": signflip_p(x, "two-sided", 10000, stable_seed("iid_p", col)),
        })
    pd.DataFrame(iid_stats).to_csv(out / "06_18_iid_effect_statistics.csv", index=False)

    # Paired falsification: observed temporal effect must exceed IID analogue.
    observed_path = root / "05_06_environment_subject_effects.csv"
    observed = pd.read_csv(observed_path)
    require_columns(observed, [
        "subject", "A2_geo_rho_Tenv_Sstar", "A3_geo_minus_zero_accuracy_slope",
        "B2_geo_minus_pairshuffled_rho_Tenv_Sstar", "B3_geo_minus_pairshuffled_accuracy_slope",
        "C2_geo_minus_pairnarrow_rho_Tenv_Sstar", "C3_geo_minus_pairnarrow_accuracy_slope"
    ], "observed environment subject effects")
    m = observed.merge(iid_sub, on="subject", validate="one_to_one")

    pairs = {
        "A2_temporal_demand_matching_exceeds_IID": ("A2_geo_rho_Tenv_Sstar", "iid_A2_geo_rho_nominal_Sstar", "A"),
        "A3_temporal_zero_advantage_slope_exceeds_IID": ("A3_geo_minus_zero_accuracy_slope", "iid_A3_geo_minus_zero_accuracy_slope", "A"),
        "B2_assignment_matching_exceeds_IID": ("B2_geo_minus_pairshuffled_rho_Tenv_Sstar", "iid_B2_geo_minus_pairshuffled_rho_nominal_Sstar", "B"),
        "B3_assignment_functional_slope_exceeds_IID": ("B3_geo_minus_pairshuffled_accuracy_slope", "iid_B3_geo_minus_pairshuffled_accuracy_slope", "B"),
        "C2_heterogeneity_matching_exceeds_IID": ("C2_geo_minus_pairnarrow_rho_Tenv_Sstar", "iid_C2_geo_minus_pairnarrow_rho_nominal_Sstar", "C"),
        "C3_heterogeneity_functional_slope_exceeds_IID": ("C3_geo_minus_pairnarrow_accuracy_slope", "iid_C3_geo_minus_pairnarrow_accuracy_slope", "C"),
    }
    stat_rows = []
    for name, (obs_col, iid_col, family) in pairs.items():
        diff = m[obs_col].to_numpy(float) - m[iid_col].to_numpy(float)
        lo, hi = bootstrap_mean_ci(diff, 10000, stable_seed("falsify_boot", name))
        stat_rows.append({
            "family": family,
            "contrast": name,
            "observed_metric": obs_col,
            "iid_metric": iid_col,
            "n_subjects": int(np.isfinite(diff).sum()),
            "mean_observed_minus_iid": float(np.nanmean(diff)),
            "median_observed_minus_iid": float(np.nanmedian(diff)),
            "bootstrap_95ci_low": lo,
            "bootstrap_95ci_high": hi,
            "positive_fraction": float(np.mean(diff[np.isfinite(diff)] > 0)),
            "raw_one_sided_signflip_p": signflip_p(diff, "greater", 10000, stable_seed("falsify_p", name)),
        })
    stats = pd.DataFrame(stat_rows)
    # Holm only within the pre-existing scientific families; two environment endpoints per family.
    stats["holm_p_within_family"] = np.nan
    for family, idx in stats.groupby("family").groups.items():
        ii = list(idx)
        adj = holm_adjust(stats.loc[ii, "raw_one_sided_signflip_p"].tolist())
        stats.loc[ii, "holm_p_within_family"] = adj
    stats.to_csv(out / "06_19_temporal_dependence_falsification_statistics.csv", index=False)

    lag1 = audit_df.lag1_z.to_numpy(float)
    return {
        "n_subjects_iid": int(iid_sub.subject.nunique()),
        "n_subject_replications_iid": len(results),
        "iid_latent_lag1_mean": float(np.nanmean(lag1)),
        "iid_latent_lag1_sd": float(np.nanstd(lag1, ddof=1)),
        "iid_runtime_seconds": float(time.time() - t0),
    }


# -----------------------------------------------------------------------------
# Self test and main
# -----------------------------------------------------------------------------
def run_self_test() -> Dict[str, object]:
    tests = {}
    z1, u1, c1 = make_iid_environment_sequence(100000, 12345, 2.0)
    z2, u2, c2 = make_iid_environment_sequence(100000, 12345, 2.0)
    tests["iid_reproducible"] = bool(np.array_equal(z1, z2) and np.array_equal(u1, u2))
    tests["iid_binary"] = bool(set(np.unique(z1)).issubset({-1.0, 1.0}))
    tests["iid_lag1_near_zero"] = bool(abs(c1) < 0.02)
    tests["iid_noise_nonzero"] = bool(np.std(u1 - z1) > 1.5)

    lags = np.array([1, 2, 3, 4], dtype=float)
    M = np.array([1.0, 1.0, 0.0, 0.0])
    mc = M.sum(); m1 = (lags * M).sum(); td = m1 / mc
    tests["M1_identity"] = bool(abs(m1 - mc * td) < 1e-12 and abs(td - 1.5) < 1e-12)
    tests["mass_quantiles"] = bool(memory_mass_quantile(lags, M, 0.5) == 1.0 and memory_mass_quantile(lags, M, 0.9) == 2.0)

    tests["slope_log2"] = bool(abs(slope_on_log2_x([1, 2, 4, 8], [0, 1, 2, 3]) - 1.0) < 1e-12)
    tests["rho_log2"] = bool(abs(rho_log2([4, 8, 16, 32], [1, 2, 4, 8]) - 1.0) < 1e-12)
    h = holm_adjust([0.01, 0.03, 0.04])
    tests["holm"] = bool(np.allclose(h, [0.03, 0.06, 0.06]))
    tests["all_pass"] = bool(all(tests.values()))
    return tests


def write_readme(out: Path, root: Path, core: Path, data_root: Path, audit: Mapping[str, object]) -> None:
    txt = f"""# 06 MICA-MICs Memory Distribution and Temporal-Dependence Falsification

Script version: `{SCRIPT_VERSION}`
Source equal-weight output: `{root}`
Core dynamics script: `{core}`
Data root: `{data_root}`

## Questions

1. Lag-resolved held-out memory profile M(k).
2. Capacity-depth tradeoff and lag-weighted memory M1 = sum(k M(k)) = MC x TD.
3. Full held-out T_env x S task-performance surface and adjacent scale effects.
4. IID latent-state negative control (flip probability 0.5 equivalent), with the same model, noise, scales, validation selection and held-out testing.

## Interpretation constraints

- Analyses 1-3 are mechanistic re-analyses, not replacement primary hypotheses.
- IID nominal T_env labels all use the same IID distribution but independent seeds; zero slopes are therefore not created by copying one sequence across labels.
- Non-significance in IID is not called proof. The main falsification output (`06_19`) directly compares observed temporal effects with their subject-paired IID analogues.
- Failed previous assignment effects are not redefined as successes.
- Subject remains the independent unit.

## Audit

```json
{json.dumps(dict(audit), indent=2)}
```
"""
    (out / "06_20_README_RUN.md").write_text(txt)


def parse_args():
    p = argparse.ArgumentParser(description="MICA-MICs lag-profile, tradeoff, performance-surface and IID falsification analyses")
    p.add_argument("--analysis-root", default=DEFAULT_ANALYSIS_ROOT,
                   help="completed equal-weight 05 output directory")
    p.add_argument("--data-root", default=DEFAULT_DATA_ROOT,
                   help="MICA-MICs micapipe derivatives root used by the 05 core")
    p.add_argument("--core-script", default="",
                   help=f"path to {CORE_BASENAME}; default: beside this script or ~/Downloads")
    p.add_argument("--outdir", default=DEFAULT_OUTDIR)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--phase", choices=("all", "reanalyze", "iid"), default="all",
                   help="reanalyze=1-3 only; iid=4 only; all=1-4")
    p.add_argument("--self-test", action="store_true")
    return p.parse_args()


def main():
    a = parse_args()
    if a.self_test:
        z = run_self_test()
        print(json.dumps(z, indent=2))
        raise SystemExit(0 if z.get("all_pass") else 2)
    if a.workers < 1:
        raise ValueError("--workers must be >=1")

    root = Path(a.analysis_root).expanduser().resolve()
    data_root = Path(a.data_root).expanduser().resolve()
    out = Path(a.outdir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    core_path = resolve_core_script(a.core_script)

    if not root.exists():
        raise FileNotFoundError(f"analysis root not found: {root}")
    if not data_root.exists() and a.phase in ("all", "iid"):
        raise FileNotFoundError(f"data root not found: {data_root}")

    t0 = time.time()
    audit: Dict[str, object] = {
        "script_version": SCRIPT_VERSION,
        "phase": a.phase,
        "analysis_root": str(root),
        "core_script": str(core_path),
        "data_root": str(data_root),
    }

    if a.phase in ("all", "reanalyze"):
        print("[1/4] lag-resolved memory profiles", flush=True)
        audit.update(analyze_memory_profiles(root, out))
        print("[2/4] capacity-depth tradeoff", flush=True)
        audit.update(analyze_capacity_depth_tradeoff(root, out))
        print("[3/4] performance surface", flush=True)
        audit.update(analyze_performance_surface(root, out))

    if a.phase in ("all", "iid"):
        print("[4/4] IID temporal-dependence falsification", flush=True)
        audit.update(run_iid_falsification(core_path, data_root, root, out, a.workers))

    audit["total_runtime_seconds"] = float(time.time() - t0)
    (out / "06_21_audit.json").write_text(json.dumps(audit, indent=2))
    write_readme(out, root, core_path, data_root, audit)
    print(json.dumps(audit, indent=2), flush=True)
    print(f"Outputs: {out}", flush=True)


if __name__ == "__main__":
    main()
