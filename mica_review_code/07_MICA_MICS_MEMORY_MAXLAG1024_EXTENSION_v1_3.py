#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
07_MICA_MICS_MEMORY_MAXLAG1024_EXTENSION.py

Memory-only extension of the corrected MICA-MICs geometry-specific scale analysis.

Purpose
-------
Resolve the max-lag=128 truncation ambiguity without changing the structural model,
delay conditions, scales, couplings, subjects, internal replicates, train/test lengths,
network seeds, or the old analyzed input sequences.

The script measures held-out memory M(k) for k=1..1024 for:
  GEOMETRIC, PAIR_SHUFFLED, PAIR_NARROW, ZERO
across:
  50 subjects x 8 internal replicates x 7 scales x 3 couplings.

Key anti-artifact rule
----------------------
The old 05 memory analysis used washout=1000, so naively changing max_lag from 128
to 1024 would request negative target indices for the first 24 lags beyond washout.
This script prepends a separately seeded pre-history while preserving the exact old
05 input sequence after that prefix. The analyzed train/test samples therefore use
exactly the same old stochastic sequences; only valid earlier target history is added.

This is a memory-only mechanistic extension. It does not rerun the environment task
and does not redefine previous primary hypotheses.
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
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


SCRIPT_VERSION = "07-mica-memory-maxlag1024-v1.3"
REQUIRED_CORE_VERSION = "05-mica-geometry-specific-v1.0"
CORE_FILENAME = "05_MICA_MICS_GEOMETRY_SPECIFIC_SCALE_VALIDATION.py"

MAX_LAG = 1024
WASHOUT = 1000
TRAIN_LEN = 10000
TEST_LEN = 10000
INTERNAL_REPS = 8
LAG_BATCH = 128
CONDITIONS = ("geometric", "pair_shuffled", "pair_narrow", "zero")

DEFAULT_CORE = f"~/Downloads/{CORE_FILENAME}"
DEFAULT_DATA_ROOT = "~/Downloads/mica-mics/MICs_release/derivatives/micapipe"
DEFAULT_OUTDIR = "~/Desktop/07_MICA_MICS_MEMORY_MAXLAG1024_EXTENSION_V1_3"
DEFAULT_LEGACY_V1_OUTDIR = "~/Desktop/07_MICA_MICS_MEMORY_MAXLAG1024_EXTENSION"
LEGACY_V1_VERSION = "07-mica-memory-maxlag1024-v1.0"
NULL_R2_REFERENCE = 1.0 / float(TEST_LEN - 1)
DEFAULT_OLD128 = "~/Desktop/06_MICA_MICS_MEMORY_DISTRIBUTION_AND_FALSIFICATION/06_01_lag_profile_subject.csv"


def parse_args():
    p = argparse.ArgumentParser(description="MICA-MICs memory extension to max lag 1024")
    p.add_argument("--core-script", default=DEFAULT_CORE)
    p.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    p.add_argument("--outdir", default=DEFAULT_OUTDIR)
    p.add_argument("--legacy-v1-outdir", default=DEFAULT_LEGACY_V1_OUTDIR,
                   help="Optional completed v1.0 output; valid rep 1..7 checkpoints can be reused")
    p.add_argument("--no-reuse-v1-checkpoints", action="store_true",
                   help="Do not reuse scientifically identical rep 1..7 checkpoints from completed v1.0")
    p.add_argument("--old128-profile", default=DEFAULT_OLD128,
                   help="Optional old subject-level lag profile for k<=128 consistency audit")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--no-checkpoint", action="store_true")
    p.add_argument("--self-test", action="store_true")
    a = p.parse_args()
    if a.workers < 1:
        raise ValueError("--workers must be >=1")
    return a


def load_core(path: str):
    path = str(Path(path).expanduser())
    if not Path(path).exists():
        raise FileNotFoundError(f"Core 05 script not found: {path}")
    name = "mica_geometry_core05_for_07"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load core script: {path}")
    mod = importlib.util.module_from_spec(spec)
    # Required for Python 3.13 dataclass/module lookup.
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    if getattr(mod, "SCRIPT_VERSION", None) != REQUIRED_CORE_VERSION:
        raise RuntimeError(
            f"Unexpected core version: {getattr(mod, 'SCRIPT_VERSION', None)!r}; "
            f"required {REQUIRED_CORE_VERSION!r}"
        )
    return mod


def make_core_config(core, data_root: str, outdir: str):
    """Construct only the fields needed by core.discover_subjects()."""
    return core.Config(
        mode="full",
        phase="memory",
        weighting="equal",
        data_root=str(Path(data_root).expanduser()),
        outdir=str(Path(outdir).expanduser()),
        workers=1,
        internal_reps=INTERNAL_REPS,
        memory_washout=WASHOUT,
        memory_train=TRAIN_LEN,
        memory_test=TEST_LEN,
        max_lag=MAX_LAG,
        env_washout=core.FULL_ENV_WASHOUT,
        env_train=core.FULL_ENV_TRAIN,
        env_val=core.FULL_ENV_VAL,
        env_test=core.FULL_ENV_TEST,
        bootstrap_samples=core.FULL_BOOTSTRAP_SAMPLES,
        permutation_samples=core.FULL_PERMUTATION_SAMPLES,
        max_subjects=0,
        checkpoint=True,
    )


def scientific_signature(core_path: str, data_root: str, script_version: str = SCRIPT_VERSION) -> str:
    x = {
        "script_version": str(script_version),
        "required_core": REQUIRED_CORE_VERSION,
        "max_lag": MAX_LAG,
        "washout": WASHOUT,
        "train_len": TRAIN_LEN,
        "test_len": TEST_LEN,
        "internal_reps": INTERNAL_REPS,
        "lag_batch": LAG_BATCH,
        "conditions": CONDITIONS,
        "scales": [1, 2, 4, 8, 16, 32, 64],
        "couplings": [0.5, 0.7, 0.9],
        "weighting": "equal",
        "core_path_name": Path(core_path).name,
        "data_root_name": Path(data_root).name,
    }
    return hashlib.sha256(json.dumps(x, sort_keys=True).encode()).hexdigest()[:20]


def required_prefix(core) -> int:
    # Need enough earlier input for k=1024 when analysis begins after washout=1000.
    target_need = max(0, MAX_LAG - WASHOUT)
    # Also populate at least one full maximum recurrence lag (1 + tau_max, tau_max<=64).
    recurrence_need = int(np.max(core.SCALES)) + 1
    return max(target_need, recurrence_need)


def build_extended_input(core, subject: str, rep: int, part: str, analysis_len: int) -> Tuple[np.ndarray, int, np.ndarray]:
    """
    Preserve the exact old 05 sequence after a new, independently seeded prefix.
    Returns (extended_u, analysis_start, old_u).
    """
    if part not in ("train", "test"):
        raise ValueError(part)
    old_seed = core.stable_seed(subject, rep, f"memory_{part}")
    old_rng = np.random.default_rng(old_seed)
    old_u = old_rng.normal(0.0, 1.0, WASHOUT + analysis_len)

    prefix_len = required_prefix(core)
    prefix_seed = core.stable_seed(subject, rep, f"memory_{part}_prefix_maxlag1024")
    prefix_rng = np.random.default_rng(prefix_seed)
    prefix = prefix_rng.normal(0.0, 1.0, prefix_len)

    u = np.concatenate([prefix, old_u])
    start = prefix_len + WASHOUT
    assert start - MAX_LAG >= 0
    assert np.array_equal(u[prefix_len:], old_u)
    return u, start, old_u


def target_block(u: np.ndarray, start: int, length: int, lag_lo: int, lag_hi: int) -> np.ndarray:
    """Columns correspond to lags lag_lo..lag_hi inclusive."""
    idx = np.arange(start, start + length, dtype=np.int64)
    lags = np.arange(lag_lo, lag_hi + 1, dtype=np.int64)
    pos = idx[:, None] - lags[None, :]
    if pos.min() < 0:
        raise RuntimeError("negative target index in extended-memory analysis")
    return u[pos]


def memory_scores_from_states(Xtr: np.ndarray, Xte: np.ndarray,
                              train_u: np.ndarray, test_u: np.ndarray,
                              start_tr: int, start_te: int,
                              max_lag: int = MAX_LAG,
                              lag_batch: int = LAG_BATCH) -> np.ndarray:
    """
    Batched multivariate OLS readout, mathematically equivalent to core 05 memory scoring,
    but avoids allocating a 10000 x 1024 prediction matrix at once.
    """
    Xtr = np.asarray(Xtr, dtype=np.float64)
    Xte = np.asarray(Xte, dtype=np.float64)
    xm = Xtr.mean(axis=0)
    Xtrc = Xtr - xm
    Xtec = Xte - xm
    gram_inv = np.linalg.pinv(Xtrc.T @ Xtrc, hermitian=True)
    proj = gram_inv @ Xtrc.T  # n_features x n_train

    M = np.zeros(max_lag, dtype=np.float64)
    for lo in range(1, max_lag + 1, lag_batch):
        hi = min(max_lag, lo + lag_batch - 1)
        Ytr = target_block(train_u, start_tr, len(Xtr), lo, hi)
        Yte = target_block(test_u, start_te, len(Xte), lo, hi)

        # Fit centered OLS. Intercept is irrelevant for correlation on test data.
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


def memory_metrics_extended(core, template, base, b, delays, r: float,
                            train_u: np.ndarray, test_u: np.ndarray,
                            start_tr: int, start_te: int) -> np.ndarray:
    xtr = core.simulate_linear(template, base, delays, b, train_u, float(r))
    xte = core.simulate_linear(template, base, delays, b, test_u, float(r))
    Xtr = xtr[start_tr:start_tr + TRAIN_LEN]
    Xte = xte[start_te:start_te + TEST_LEN]
    if Xtr.shape != (TRAIN_LEN, template.n) or Xte.shape != (TEST_LEN, template.n):
        raise RuntimeError(f"bad analysis state shape: {Xtr.shape}, {Xte.shape}")
    return memory_scores_from_states(Xtr, Xte, train_u, test_u, start_tr, start_te)


def rep_checkpoint_path(outdir: Path, subject: str, rep: int, signature: str) -> Path:
    d = outdir / "checkpoints"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{subject}_rep{rep:02d}_{signature}.npz"


def load_rep_checkpoint(path: Path, signature: str):
    if not path.exists():
        return None
    try:
        z = np.load(path, allow_pickle=False)
        sig = str(z["signature"].item())
        if sig != signature:
            return None
        return {
            "profile_mean_r": z["profile_mean_r"].astype(np.float64),
            "mc_mean_r": z["mc_mean_r"].astype(np.float64),
            "td_mean_r": z["td_mean_r"].astype(np.float64),
        }
    except Exception:
        return None


def save_rep_checkpoint(path: Path, signature: str, profile_mean_r: np.ndarray,
                        mc_mean_r: np.ndarray, td_mean_r: np.ndarray):
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        tmp,
        signature=np.asarray(signature),
        profile_mean_r=profile_mean_r.astype(np.float32),
        mc_mean_r=mc_mean_r.astype(np.float64),
        td_mean_r=td_mean_r.astype(np.float64),
    )
    os.replace(tmp, path)


def one_rep(core, template, subject: str, rep: int, outdir: Path, signature: str, checkpoint: bool,
            legacy_outdir: Path | None = None, legacy_signature: str | None = None,
            allow_legacy_reuse: bool = True):
    cp = rep_checkpoint_path(outdir, subject, rep, signature)
    if checkpoint:
        old = load_rep_checkpoint(cp, signature)
        if old is not None:
            old["_legacy_reused"] = False
            return old

    # v1.0 used reps 0..7. Reps 1..7 are scientifically identical to the corrected
    # 05-aligned design and can be reused after exact legacy-signature validation.
    if allow_legacy_reuse and 1 <= int(rep) <= 7 and legacy_outdir is not None and legacy_signature:
        legacy_cp = rep_checkpoint_path(Path(legacy_outdir), subject, rep, legacy_signature)
        legacy = load_rep_checkpoint(legacy_cp, legacy_signature)
        if legacy is not None:
            if checkpoint:
                save_rep_checkpoint(
                    cp, signature,
                    legacy["profile_mean_r"], legacy["mc_mean_r"], legacy["td_mean_r"]
                )
            legacy["_legacy_reused"] = True
            return legacy

    signs, b = core.make_signs_and_input(template, subject, rep)
    base = core.base_weights(template, signs, "equal")
    shuf_delta = core.pair_shuffled_delta(
        template.undirected_delta,
        core.stable_seed(subject, rep, "pair_shuffle_geometry")
    )

    train_u, start_tr, _ = build_extended_input(core, subject, rep, "train", TRAIN_LEN)
    test_u, start_te, _ = build_extended_input(core, subject, rep, "test", TEST_LEN)

    ncond = len(CONDITIONS)
    nscale = len(core.SCALES)
    nr = len(core.COUPLINGS)
    prof = np.zeros((ncond, nscale, nr, MAX_LAG), dtype=np.float64)
    mc = np.zeros((ncond, nscale, nr), dtype=np.float64)
    td = np.zeros((ncond, nscale, nr), dtype=np.float64)

    zero_by_r: Dict[float, np.ndarray] = {}
    zd = np.zeros(template.src.size, dtype=np.int32)
    for ri, r in enumerate(core.COUPLINGS):
        M = memory_metrics_extended(core, template, base, b, zd, float(r), train_u, test_u, start_tr, start_te)
        zero_by_r[float(r)] = M

    for si, S in enumerate(core.SCALES):
        bundle = core.delay_bundle(template, shuf_delta, int(S))
        for ci, cond in enumerate(("geometric", "pair_shuffled", "pair_narrow")):
            d = bundle[cond]
            for ri, r in enumerate(core.COUPLINGS):
                M = memory_metrics_extended(core, template, base, b, d, float(r), train_u, test_u, start_tr, start_te)
                prof[ci, si, ri] = M
        ci = CONDITIONS.index("zero")
        for ri, r in enumerate(core.COUPLINGS):
            prof[ci, si, ri] = zero_by_r[float(r)]

    lags = np.arange(1, MAX_LAG + 1, dtype=np.float64)
    mc[:] = prof.sum(axis=-1)
    denom = mc.copy()
    td[:] = np.divide(
        np.sum(prof * lags[None, None, None, :], axis=-1),
        denom,
        out=np.full_like(denom, np.nan),
        where=denom > 0,
    )

    # Average nuisance coupling within this replicate. Replicate averaging happens later.
    result = {
        "profile_mean_r": prof.mean(axis=2),
        "mc_mean_r": mc.mean(axis=2),
        "td_mean_r": td.mean(axis=2),
        "_legacy_reused": False,
    }
    if checkpoint:
        # Checkpoint stores only numeric scientific arrays; runtime metadata such as
        # _legacy_reused must never be forwarded into the checkpoint writer.
        save_rep_checkpoint(
            cp, signature,
            result["profile_mean_r"], result["mc_mean_r"], result["td_mean_r"]
        )
    return result


def subject_worker(payload):
    (sf_dict, core_path, data_root, outdir_s, signature, checkpoint,
     legacy_outdir_s, legacy_signature, allow_legacy_reuse) = payload
    core = load_core(core_path)
    sf = core.SubjectFiles(**sf_dict)
    template = core.build_empirical_template(sf)
    outdir = Path(outdir_s)
    legacy_outdir = Path(legacy_outdir_s) if legacy_outdir_s else None

    profiles = []
    mcs = []
    tds = []
    legacy_reused = 0
    for rep in range(1, INTERNAL_REPS + 1):
        z = one_rep(
            core, template, sf.subject, rep, outdir, signature, checkpoint,
            legacy_outdir=legacy_outdir, legacy_signature=legacy_signature,
            allow_legacy_reuse=allow_legacy_reuse,
        )
        legacy_reused += int(bool(z.get("_legacy_reused", False)))
        profiles.append(z["profile_mean_r"])
        mcs.append(z["mc_mean_r"])
        tds.append(z["td_mean_r"])

    profile = np.mean(np.stack(profiles, axis=0), axis=0)
    mc_mean_rep_r = np.mean(np.stack(mcs, axis=0), axis=0)
    td_mean_rep_r = np.mean(np.stack(tds, axis=0), axis=0)

    return {
        "subject": sf.subject,
        "session": sf.session,
        "lmax_retained_mm": float(template.lmax_retained_mm),
        "profile": profile.astype(np.float32),
        "mc_mean_rep_r": mc_mean_rep_r,
        "td_mean_rep_r": td_mean_rep_r,
        "legacy_reused_reps": int(legacy_reused),
    }


def quantile_lag(M: np.ndarray, q: float) -> float:
    total = float(np.sum(M))
    if total <= 0:
        return float("nan")
    c = np.cumsum(M) / total
    return float(np.searchsorted(c, q, side="left") + 1)


def profile_metrics(M: np.ndarray, state_dim: int = 200) -> Dict[str, float]:
    """
    Raw M(k) metrics plus horizon and finite-sample diagnostics.

    Important: raw TD/quantiles across 1024 lags include the positive r^2 finite-sample
    floor. They are retained for transparency, but the truncation decision should rely on
    horizon-specific MC and boundary-vs-null diagnostics rather than raw 1024 TD.
    """
    M = np.asarray(M, dtype=np.float64)
    lags = np.arange(1, len(M) + 1, dtype=np.float64)
    MC = float(M.sum())
    horizons = (128, 256, 512, 1024)
    mc_h = {k: float(M[:min(k, len(M))].sum()) for k in horizons}
    null_r2 = float(NULL_R2_REFERENCE)
    last64_mean = float(M[-64:].mean())
    tail257 = float(M[256:].sum()) if len(M) > 256 else 0.0
    expected_tail257 = float(max(0, len(M) - 256) * null_r2)

    if MC <= 0:
        out = {k: float("nan") for k in (
            "MC_profile", "TD_profile", "M1", "effective_lags", "lag_sd",
            "lag50", "lag90", "lag95", "lag99", "tail_fraction_gt128",
            "tail_fraction_gt256", "tail_fraction_gt512", "last64_fraction",
            "mean_M_last64", "M_at_1024")}
    else:
        p = M / MC
        td = float(np.sum(lags * p))
        var = float(np.sum(((lags - td) ** 2) * p))
        pr = float(1.0 / np.sum(p * p))
        out = {
            "MC_profile": MC,
            "TD_profile": td,
            "M1": float(np.sum(lags * M)),
            "effective_lags": pr,
            "lag_sd": math.sqrt(max(0.0, var)),
            "lag50": quantile_lag(M, 0.50),
            "lag90": quantile_lag(M, 0.90),
            "lag95": quantile_lag(M, 0.95),
            "lag99": quantile_lag(M, 0.99),
            "tail_fraction_gt128": float(M[128:].sum() / MC),
            "tail_fraction_gt256": float(M[256:].sum() / MC),
            "tail_fraction_gt512": float(M[512:].sum() / MC),
            "last64_fraction": float(M[-64:].sum() / MC),
            "mean_M_last64": last64_mean,
            "M_at_1024": float(M[-1]),
        }

    out.update({
        "MC_1_128": mc_h[128],
        "MC_1_256": mc_h[256],
        "MC_1_512": mc_h[512],
        "MC_1_1024": mc_h[1024],
        "MC_1_1024_over_state_dim": float(mc_h[1024] / float(state_dim)),
        "MC_1_256_over_state_dim": float(mc_h[256] / float(state_dim)),
        "independence_r2_reference": null_r2,
        "expected_null_MC_1024": float(len(M) * null_r2),
        "expected_null_MC_last64": float(64 * null_r2),
        "mean_M_last64_minus_null_reference": float(last64_mean - null_r2),
        "mean_M_last64_over_null_reference": float(last64_mean / null_r2) if null_r2 > 0 else float("nan"),
        "MC_257_1024": tail257,
        "expected_null_MC_257_1024": expected_tail257,
        "MC_257_1024_excess_vs_null_reference": float(tail257 - expected_tail257),
    })
    return out


def write_outputs(core, results: List[Dict[str, object]], outdir: Path, old128_path: str, signature: str):
    results = sorted(results, key=lambda z: z["subject"])
    subjects = [z["subject"] for z in results]
    nsub = len(subjects)

    # Compact subject-level long profile: 50 x 4 x 7 x 1024 = 1,433,600 rows.
    profile_csv = outdir / "07_01_subject_profile_long.csv"
    first = True
    metric_rows = []
    group_acc = np.zeros((len(CONDITIONS), len(core.SCALES), MAX_LAG), dtype=np.float64)

    for z in results:
        subject = z["subject"]
        P = np.asarray(z["profile"], dtype=np.float64)
        group_acc += P
        chunks = []
        for ci, cond in enumerate(CONDITIONS):
            for si, S in enumerate(core.SCALES):
                M = P[ci, si]
                chunks.append(pd.DataFrame({
                    "subject": subject,
                    "condition": cond,
                    "scale": int(S),
                    "lag": np.arange(1, MAX_LAG + 1, dtype=np.int32),
                    "M": M,
                }))
                row = {
                    "subject": subject,
                    "condition": cond,
                    "scale": int(S),
                    "MC_mean_rep_r": float(z["mc_mean_rep_r"][ci, si]),
                    "TD_mean_rep_r": float(z["td_mean_rep_r"][ci, si]),
                }
                row.update(profile_metrics(M, state_dim=int(core.ATLAS_SCALE)))
                metric_rows.append(row)
        dd = pd.concat(chunks, ignore_index=True)
        dd.to_csv(profile_csv, index=False, mode="w" if first else "a", header=first)
        first = False

    subject_metrics = pd.DataFrame(metric_rows)
    subject_metrics.to_csv(outdir / "07_03_subject_metrics.csv", index=False)

    group = group_acc / float(nsub)
    group_rows = []
    group_metric_rows = []
    for ci, cond in enumerate(CONDITIONS):
        for si, S in enumerate(core.SCALES):
            M = group[ci, si]
            for k, mk in enumerate(M, start=1):
                group_rows.append({"condition": cond, "scale": int(S), "lag": k, "M_mean": float(mk)})
            row = {"condition": cond, "scale": int(S)}
            row.update(profile_metrics(M, state_dim=int(core.ATLAS_SCALE)))
            # Also aggregate subject-level nuisance-averaged MC/TD, preserving old convention.
            d = subject_metrics[(subject_metrics.condition == cond) & (subject_metrics.scale == int(S))]
            row["MC_mean_rep_r_subject_mean"] = float(d.MC_mean_rep_r.mean())
            row["TD_mean_rep_r_subject_mean"] = float(d.TD_mean_rep_r.mean())
            group_metric_rows.append(row)

    pd.DataFrame(group_rows).to_csv(outdir / "07_02_group_profile.csv", index=False)
    group_metrics = pd.DataFrame(group_metric_rows)
    group_metrics.to_csv(outdir / "07_04_group_metrics.csv", index=False)

    # Tail mass broken into non-overlapping windows; descriptive, no arbitrary pass/fail threshold.
    tail_rows = []
    windows = [(1,128), (129,256), (257,512), (513,768), (769,1024)]
    for _, r in subject_metrics.iterrows():
        subject = r["subject"]; cond = r["condition"]; S = int(r["scale"])
        z = next(x for x in results if x["subject"] == subject)
        ci = CONDITIONS.index(cond); si = list(core.SCALES).index(S)
        M = np.asarray(z["profile"], dtype=np.float64)[ci, si]
        total = M.sum()
        row = {"subject": subject, "condition": cond, "scale": S, "MC_1_1024": float(total)}
        for a,b in windows:
            mass = float(M[a-1:b].sum())
            width = int(b - a + 1)
            null_mass = float(width * NULL_R2_REFERENCE)
            row[f"MC_{a}_{b}"] = mass
            row[f"fraction_{a}_{b}"] = float(mass / total) if total > 0 else float("nan")
            row[f"expected_null_MC_{a}_{b}"] = null_mass
            row[f"excess_vs_null_MC_{a}_{b}"] = float(mass - null_mass)
        tail_rows.append(row)
    tail_df = pd.DataFrame(tail_rows)
    tail_df.to_csv(outdir / "07_05_tail_windows_subject.csv", index=False)
    gtail = tail_df.groupby(["condition","scale"], as_index=False).mean(numeric_only=True)
    gtail.to_csv(outdir / "07_06_tail_windows_group.csv", index=False)

    # Optional k<=128 consistency audit against 06 subject profiles.
    old_path = Path(old128_path).expanduser() if old128_path else None
    consistency = []
    if old_path and old_path.exists():
        old = pd.read_csv(old_path)
        old = old[old["lag"] <= 128].copy()
        for z in results:
            subject = z["subject"]
            P = np.asarray(z["profile"], dtype=np.float64)
            for ci, cond in enumerate(CONDITIONS):
                for si, S in enumerate(core.SCALES):
                    a = old[(old.subject == subject) & (old.condition == cond) & (old.scale == int(S))].sort_values("lag")
                    if len(a) != 128:
                        continue
                    new = P[ci, si, :128]
                    diff = new - a.M.to_numpy(float)
                    consistency.append({
                        "subject": subject, "condition": cond, "scale": int(S),
                        "mean_abs_diff_M_1_128": float(np.mean(np.abs(diff))),
                        "max_abs_diff_M_1_128": float(np.max(np.abs(diff))),
                        "rmse_M_1_128": float(np.sqrt(np.mean(diff * diff))),
                    })
    pd.DataFrame(consistency).to_csv(outdir / "07_07_old128_consistency.csv", index=False)

    # Summary focused on the truncation question.
    summary = {
        "script_version": SCRIPT_VERSION,
        "scientific_signature": signature,
        "n_subjects": nsub,
        "max_lag": MAX_LAG,
        "washout": WASHOUT,
        "prehistory_length": required_prefix(core),
        "train_len": TRAIN_LEN,
        "test_len": TEST_LEN,
        "internal_reps": INTERNAL_REPS,
        "conditions": list(CONDITIONS),
        "scales": [int(x) for x in core.SCALES],
        "couplings": [float(x) for x in core.COUPLINGS],
        "weighting": "equal",
        "replicate_ids": list(range(1, INTERNAL_REPS + 1)),
        "independence_r2_reference": float(NULL_R2_REFERENCE),
        "expected_null_MC_1024": float(MAX_LAG * NULL_R2_REFERENCE),
        "legacy_reused_reps_total": int(sum(int(z.get("legacy_reused_reps", 0)) for z in results)),
        "old128_consistency_rows": len(consistency),
        "old128_mean_abs_diff": float(np.mean([x["mean_abs_diff_M_1_128"] for x in consistency])) if consistency else None,
        "old128_max_abs_diff": float(np.max([x["max_abs_diff_M_1_128"] for x in consistency])) if consistency else None,
    }
    (outdir / "07_08_audit.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    readme = f"""# 07 MICA-MICs Memory Max-Lag 1024 Extension

Script version: `{SCRIPT_VERSION}`  
Core version required: `{REQUIRED_CORE_VERSION}`  
Subjects: {nsub}  
Max lag: {MAX_LAG}  
Washout: {WASHOUT}  
Pre-history added: {required_prefix(core)} samples  
Train/test: {TRAIN_LEN}/{TEST_LEN} independent samples  
Internal replicates: {INTERNAL_REPS} (IDs 1..8, matched to core 05)  
Finite-sample independence reference E[r^2]: {NULL_R2_REFERENCE:.8f}  
Weighting: equal  

## Scientific purpose

This analysis addresses only the max-lag=128 truncation ambiguity in the memory analysis.
It does not rerun the environment task and does not replace prior hypotheses.

The structural graph, GEOMETRIC/PAIR_SHUFFLED/PAIR_NARROW/ZERO definitions, scales,
couplings, signs, input vectors, and old 05 train/test random sequences are retained.
A separately seeded pre-history is prepended so lags 1..1024 are valid even though the
original washout was 1000.

## Main outputs

- `07_01_subject_profile_long.csv`: subject-level M(k), averaged over 8 reps x 3 couplings.
- `07_02_group_profile.csv`: group mean M(k).
- `07_03_subject_metrics.csv`: MC, TD, M1, effective-lag count, lag spread and tail diagnostics.
- `07_04_group_metrics.csv`: group summaries, including MC/N and finite-sample null references.
- `07_05_tail_windows_subject.csv`: non-overlapping memory mass windows.
- `07_06_tail_windows_group.csv`: group tail-window summaries.
- `07_07_old128_consistency.csv`: comparison of new k<=128 profile with the old 06 profile when available.
- `07_08_audit.json`: run configuration.

## Interpretation rule

No threshold is used to declare that the 1024 window is 'long enough'. Inspect horizon-specific
MC and the last-window / last-64-lag values against the analytic independence reference
E[r^2] = 1/(n_test-1). Raw TD, lag quantiles, effective-lag count and lag SD across all 1024
lags are retained for transparency but are not primary truncation diagnostics because the
positive finite-sample r^2 floor is amplified by long-lag weighting.
"""
    (outdir / "07_09_README_RUN.md").write_text(readme, encoding="utf-8")


def run_preflight(core, subjects, outdir: Path):
    rows = []
    for sf in subjects:
        try:
            t = core.build_empirical_template(sf)
            rows.append({
                "subject": sf.subject,
                "session": sf.session,
                "status": "ok",
                "n_regions": int(t.n),
                "n_undirected_edges": int(len(t.undirected_delta)),
                "lmax_retained_mm": float(t.lmax_retained_mm),
                "max_geo_delay_S64": int(core.geometric_delays(t.delta, 64).max()),
                "error": "",
            })
        except Exception as e:
            rows.append({"subject": sf.subject, "session": sf.session, "status": "error", "error": repr(e)})
    df = pd.DataFrame(rows)
    df.to_csv(outdir / "07_00_preflight.csv", index=False)
    bad = df[df.status != "ok"]
    if len(bad):
        raise RuntimeError(f"preflight failed for {len(bad)} subjects")
    if len(df) != 50:
        raise RuntimeError(f"full 1024 analysis requires all 50 MICA-MICs subjects; found {len(df)}")


def run_self_test(core_path: str) -> Dict[str, object]:
    core = load_core(core_path)
    tests = {}
    tests["core_version"] = getattr(core, "SCRIPT_VERSION", None) == REQUIRED_CORE_VERSION
    core_tests = core.run_self_test()
    tests["core_self_test"] = bool(core_tests.get("all_pass", False))
    tests["prefix_nonnegative"] = required_prefix(core) + WASHOUT - MAX_LAG >= 0
    tests["prefix_covers_recurrence"] = required_prefix(core) >= int(np.max(core.SCALES)) + 1
    tests["replicate_ids_match_core05"] = list(range(1, INTERNAL_REPS + 1)) == list(range(1, 9))
    tests["null_r2_reference"] = abs(NULL_R2_REFERENCE - (1.0 / 9999.0)) < 1e-15

    # Old-sequence preservation.
    u, start, old = build_extended_input(core, "sub-TEST", 0, "train", 200)
    pre = required_prefix(core)
    tests["old_sequence_preserved_exact"] = bool(np.array_equal(u[pre:], old))
    tests["target_index_valid"] = bool(start - MAX_LAG >= 0)

    # Batched OLS scores must equal a naive all-lags implementation on random states.
    rng = np.random.default_rng(123)
    ntr, nte, p, lag = 300, 260, 12, 16
    wash = 20
    prefix = lag
    tr_u = rng.normal(size=prefix + wash + ntr)
    te_u = rng.normal(size=prefix + wash + nte)
    st = prefix + wash
    Xtr = rng.normal(size=(ntr, p))
    Xte = rng.normal(size=(nte, p))
    fast = memory_scores_from_states(Xtr, Xte, tr_u, te_u, st, st, max_lag=lag, lag_batch=5)

    Ytr = target_block(tr_u, st, ntr, 1, lag)
    Yte = target_block(te_u, st, nte, 1, lag)
    xm = Xtr.mean(axis=0); ym = Ytr.mean(axis=0)
    Xc = Xtr - xm; Yc = Ytr - ym
    B = np.linalg.pinv(Xc.T @ Xc, hermitian=True) @ (Xc.T @ Yc)
    a = ym - xm @ B
    pred = a + Xte @ B
    slow = []
    for q in range(lag):
        c = np.corrcoef(Yte[:, q], pred[:, q])[0,1]
        slow.append(c*c)
    slow = np.asarray(slow)
    tests["batched_matches_naive"] = bool(np.allclose(fast, slow, rtol=1e-10, atol=1e-12))
    tests["batched_max_abs_error"] = float(np.max(np.abs(fast - slow)))

    # Checkpoint writer/loader roundtrip, including the exact call shape used by one_rep.
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        cp = Path(td) / "roundtrip.npz"
        p0 = rng.normal(size=(4, 7, 1024))
        m0 = rng.normal(size=(4, 7))
        t0 = rng.normal(size=(4, 7))
        save_rep_checkpoint(cp, "selftest-signature", p0, m0, t0)
        zz = load_rep_checkpoint(cp, "selftest-signature")
        tests["checkpoint_roundtrip"] = bool(
            zz is not None
            and np.allclose(zz["profile_mean_r"], p0, rtol=1e-6, atol=1e-6)
            and np.allclose(zz["mc_mean_r"], m0)
            and np.allclose(zz["td_mean_r"], t0)
        )

    tests["all_pass"] = all(v for k,v in tests.items() if k not in ("batched_max_abs_error",))
    return tests


def main():
    a = parse_args()
    core_path = str(Path(a.core_script).expanduser())
    if a.self_test:
        print(json.dumps(run_self_test(core_path), indent=2))
        return

    t0 = time.time()
    core = load_core(core_path)
    outdir = Path(a.outdir).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)
    cfg = make_core_config(core, a.data_root, str(outdir))
    subjects = core.discover_subjects(cfg)
    if len(subjects) != 50:
        raise RuntimeError(f"Expected 50 MICA-MICs subjects, found {len(subjects)}")
    run_preflight(core, subjects, outdir)

    signature = scientific_signature(core_path, a.data_root, SCRIPT_VERSION)
    legacy_signature = scientific_signature(core_path, a.data_root, LEGACY_V1_VERSION)
    legacy_outdir = Path(a.legacy_v1_outdir).expanduser() if a.legacy_v1_outdir else None
    allow_legacy_reuse = not a.no_reuse_v1_checkpoints
    payloads = [
        (
            vars(sf), core_path, str(Path(a.data_root).expanduser()), str(outdir),
            signature, not a.no_checkpoint,
            str(legacy_outdir) if legacy_outdir is not None else "",
            legacy_signature, allow_legacy_reuse,
        )
        for sf in subjects
    ]

    results = []
    if a.workers == 1:
        for i,payload in enumerate(payloads, start=1):
            z = subject_worker(payload)
            results.append(z)
            print(f"[07] subject {i}/50 complete: {z['subject']}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            futs = {ex.submit(subject_worker, p): p[0]["subject"] for p in payloads}
            done = 0
            for fut in as_completed(futs):
                subject = futs[fut]
                try:
                    z = fut.result()
                except Exception:
                    print(f"[07 ERROR] {subject}\n{traceback.format_exc()}", flush=True)
                    raise
                results.append(z)
                done += 1
                print(f"[07] subject {done}/50 complete: {subject}", flush=True)

    write_outputs(core, results, outdir, a.old128_profile, signature)
    elapsed = time.time() - t0
    audit_path = outdir / "07_08_audit.json"
    audit = json.loads(audit_path.read_text())
    audit["runtime_seconds"] = elapsed
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": "complete",
        "n_subjects": len(results),
        "max_lag": MAX_LAG,
        "outdir": str(outdir),
        "legacy_reused_reps_total": int(audit.get("legacy_reused_reps_total", 0)),
        "runtime_seconds": elapsed,
    }, indent=2))


if __name__ == "__main__":
    main()
