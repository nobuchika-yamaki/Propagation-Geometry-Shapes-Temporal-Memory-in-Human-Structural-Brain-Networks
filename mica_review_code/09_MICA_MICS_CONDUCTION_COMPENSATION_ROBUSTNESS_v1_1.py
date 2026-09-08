#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
09_MICA_MICS_CONDUCTION_COMPENSATION_ROBUSTNESS.py

Final robustness analysis 2: distance-dependent conduction compensation.

Scientific question
-------------------
How much of the propagation-scale effect survives when longer tracts are allowed to conduct
faster, compressing the dependence of delay on anatomical path length?

We use a dimensionless sensitivity family
    effective_delta_e(beta) = delta_e ** (1 - beta)
    tau_e(S, beta) = floor(S * effective_delta_e(beta) + 0.5)
where delta_e = L_e / L_max.

Interpretation:
- beta = 0.00: original constant-effective-velocity mapping (exact core-05 geometry)
- beta = 0.25, 0.50, 0.75: progressively stronger length-dependent velocity compensation
- beta = 1.00: complete relative distance compensation; all retained tracts have the same
  relative propagation delay at a given S.

A multiplicative global velocity factor is not separately parameterized because it is
mathematically absorbed by the existing propagation-scale parameter S. Beta therefore changes
only the relative dependence of delay on tract length, which is the biologically relevant
compensation question for this sensitivity analysis.

This is a one-factor robustness analysis. Equal recurrent weighting is retained. No pair-shuffle
or pair-narrow controls are reintroduced here; those mechanisms were already tested in the
primary analysis. The analysis asks whether the principal A-family effects persist as relative
length dependence is progressively compressed.

Beta=0 results are loaded from the completed equal-weight 05/07 analyses rather than recomputed.
Only beta>0 conditions are newly simulated, using the same subjects, rep IDs 1..8, signs, input
vectors, stochastic sequences, S grid, r grid, and T_env grid.
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
from typing import Dict, List, Sequence, Tuple

# Prevent BLAS/LAPACK oversubscription under subject-level multiprocessing.
for _var in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS",
):
    os.environ[_var] = "1"


import numpy as np
import pandas as pd


SCRIPT_VERSION = "09-mica-conduction-compensation-robustness-v1.1"
REQUIRED_CORE_VERSION = "05-mica-geometry-specific-v1.0"
CORE_FILENAME = "05_MICA_MICS_GEOMETRY_SPECIFIC_SCALE_VALIDATION.py"
SEED_NAMESPACE = "MICA_CONDUCTION_COMPENSATION_ROBUSTNESS_V1"

BETAS = np.asarray([0.00, 0.25, 0.50, 0.75, 1.00], dtype=np.float64)
NEW_BETAS = np.asarray([0.25, 0.50, 0.75, 1.00], dtype=np.float64)
MAX_LAG = 1024
LAG_BATCH = 128
WASHOUT = 1000
TRAIN_LEN = 10000
TEST_LEN = 10000
INTERNAL_REPS = 8
NULL_R2_REFERENCE = 1.0 / float(TEST_LEN - 1)

DEFAULT_CORE = f"~/Downloads/{CORE_FILENAME}"
DEFAULT_DATA_ROOT = "~/Downloads/mica-mics/MICs_release/derivatives/micapipe"
DEFAULT_OUTDIR = "~/Desktop/09_MICA_MICS_CONDUCTION_COMPENSATION_ROBUSTNESS_V1_1"
DEFAULT_EQUAL_ROOT = "~/Desktop/05_MICA_MICS_GEOMETRY_SPECIFIC_SCALE_VALIDATION/equal"
DEFAULT_EQUAL_EXTENDED_ROOT = "~/Desktop/07_MICA_MICS_MEMORY_MAXLAG1024_EXTENSION_V1_3"


def stable_seed(*parts: object) -> int:
    s = "|".join(str(x) for x in (SEED_NAMESPACE,) + parts).encode("utf-8")
    return int(hashlib.sha256(s).hexdigest()[:16], 16) % (2**32 - 1)


def parse_args():
    p = argparse.ArgumentParser(description="MICA-MICs conduction-compensation robustness")
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
    # Reuse the same module name used by 07 so Numba cache environments remain valid.
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
        weighting="equal",
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
    x = {
        "script_version": SCRIPT_VERSION,
        "core_version": REQUIRED_CORE_VERSION,
        "phase": phase,
        "betas_new": [float(x) for x in NEW_BETAS],
        "mapping": "delta**(1-beta)",
        "weighting": "equal",
        "max_lag": MAX_LAG,
        "reps": list(range(1, INTERNAL_REPS + 1)),
        "scales": [int(x) for x in core.SCALES],
        "couplings": [float(x) for x in core.COUPLINGS],
        "T_env": [int(x) for x in core.T_ENVS],
    }
    return hashlib.sha256(json.dumps(x, sort_keys=True).encode()).hexdigest()[:20]


# -----------------------------------------------------------------------------
# Compensation mapping
# -----------------------------------------------------------------------------

def compensated_delta(delta_u: np.ndarray, beta: float) -> np.ndarray:
    d = np.asarray(delta_u, dtype=np.float64)
    if np.any(d <= 0) or np.any(d > 1.0 + 1e-12):
        raise ValueError("delta must lie in (0,1]")
    b = float(beta)
    if b < 0 or b > 1:
        raise ValueError("beta must be in [0,1]")
    return np.power(d, 1.0 - b)


def compensated_delays(core, template, beta: float, scale: int) -> np.ndarray:
    du = compensated_delta(template.undirected_delta, beta)
    d_u = core.geometric_delays(du, int(scale))
    return core.expand_undirected_delays(d_u)


def delay_diagnostics(core, template, subject: str) -> List[Dict[str, object]]:
    rows = []
    for beta in BETAS:
        deff = compensated_delta(template.undirected_delta, float(beta))
        for S in core.SCALES:
            d = core.geometric_delays(deff, int(S)).astype(np.float64)
            mu = float(d.mean())
            sd = float(d.std(ddof=0))
            rows.append({
                "subject": subject,
                "beta": float(beta),
                "scale": int(S),
                "mean_delay": mu,
                "sd_delay": sd,
                "cv_delay": float(sd / mu) if mu > 0 else float("nan"),
                "min_delay": int(d.min()),
                "max_delay": int(d.max()),
                "n_unique_delays": int(np.unique(d).size),
                "mean_effective_delta": float(deff.mean()),
                "sd_effective_delta": float(deff.std(ddof=0)),
            })
    return rows



def compensated_pattern_groups(core, template):
    """Group byte-identical compensated integer-delay patterns across (beta, S)."""
    groups = {}
    for bi, beta in enumerate(NEW_BETAS):
        for si, S in enumerate(core.SCALES):
            d = np.asarray(compensated_delays(core, template, float(beta), int(S)), dtype=np.int32)
            key = d.tobytes()
            label = (bi, si, float(beta), int(S))
            if key not in groups:
                groups[key] = {"delays": d.copy(), "labels": [label]}
            else:
                if not np.array_equal(groups[key]["delays"], d):
                    raise AssertionError("compensated delay-pattern grouping inconsistency")
                groups[key]["labels"].append(label)
    return list(groups.values())


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
    prefix = np.random.default_rng(pre_seed).normal(0.0, 1.0, pre)
    u = np.concatenate([prefix, old_u])
    start = pre + WASHOUT
    if start - MAX_LAG < 0 or not np.array_equal(u[pre:], old_u):
        raise RuntimeError("extended-input construction failed")
    return u, start


def target_block(u: np.ndarray, start: int, length: int, lo: int, hi: int):
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
    Xtr = np.asarray(Xtr, dtype=np.float64); Xte = np.asarray(Xte, dtype=np.float64)
    xm = Xtr.mean(0); Xtrc = Xtr - xm; Xtec = Xte - xm
    proj = np.linalg.pinv(Xtrc.T @ Xtrc, hermitian=True) @ Xtrc.T
    M = np.zeros(max_lag, dtype=np.float64)
    for lo in range(1, max_lag + 1, lag_batch):
        hi = min(max_lag, lo + lag_batch - 1)
        Ytr = target_block(train_u, start_tr, len(Xtr), lo, hi)
        Yte = target_block(test_u, start_te, len(Xte), lo, hi)
        B = proj @ (Ytr - Ytr.mean(0))
        P = Xtec @ B
        Yc = Yte - Yte.mean(0); Pc = P - P.mean(0)
        num = np.sum(Yc * Pc, axis=0)
        den = np.sqrt(np.sum(Yc * Yc, axis=0) * np.sum(Pc * Pc, axis=0))
        c = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
        vals = c*c; vals[~np.isfinite(vals)] = 0.0
        M[lo-1:hi] = vals
    return M


def memory_profile(core, template, base, b, delays, r, train_u, test_u, start_tr, start_te):
    xtr = core.simulate_linear(template, base, delays, b, train_u, float(r))
    xte = core.simulate_linear(template, base, delays, b, test_u, float(r))
    Xtr = xtr[start_tr:start_tr + TRAIN_LEN]
    Xte = xte[start_te:start_te + TEST_LEN]
    return memory_scores_from_states(Xtr, Xte, train_u, test_u, start_tr, start_te)


def td_at_horizon(M: np.ndarray, K: int = 128) -> float:
    q = np.asarray(M[:K], dtype=np.float64); mc = float(q.sum())
    if mc <= 0:
        return float("nan")
    lag = np.arange(1, K+1, dtype=np.float64)
    return float(np.sum(lag*q)/mc)


def quantile_lag(M: np.ndarray, q: float) -> float:
    total = float(np.sum(M))
    if total <= 0:
        return float("nan")
    return float(np.searchsorted(np.cumsum(M)/total, q, side="left") + 1)


def profile_metrics(M: np.ndarray, state_dim: int = 200) -> Dict[str, float]:
    M = np.asarray(M, dtype=np.float64)
    out = {f"MC_1_{K}": float(M[:K].sum()) for K in (128,256,512,1024)}
    out["MC_1_1024_over_state_dim"] = out["MC_1_1024"] / float(state_dim)
    out["MC_1_256_over_state_dim"] = out["MC_1_256"] / float(state_dim)
    out["TD_1_128"] = td_at_horizon(M, 128)
    out["lag50_1_1024"] = quantile_lag(M, 0.50)
    out["lag90_1_1024"] = quantile_lag(M, 0.90)
    out["lag95_1_1024"] = quantile_lag(M, 0.95)
    out["mean_M_last64"] = float(M[-64:].mean())
    out["mean_M_last64_minus_null"] = out["mean_M_last64"] - NULL_R2_REFERENCE
    out["MC_257_1024"] = float(M[256:].sum())
    out["expected_null_MC_257_1024"] = float((1024-256)*NULL_R2_REFERENCE)
    out["MC_257_1024_excess_vs_null"] = out["MC_257_1024"] - out["expected_null_MC_257_1024"]
    return out


# -----------------------------------------------------------------------------
# Checkpoints
# -----------------------------------------------------------------------------

def mem_cp_path(out: Path, subject: str, rep: int, sig: str) -> Path:
    d = out / "checkpoints_memory"; d.mkdir(parents=True, exist_ok=True)
    return d / f"{subject}_rep{rep:02d}_{sig}.npz"


def save_mem_cp(path: Path, sig: str, profiles_mean_r: np.ndarray, td128_by_r: np.ndarray):
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, signature=np.asarray(sig),
                        profiles_mean_r=np.asarray(profiles_mean_r,dtype=np.float32),
                        td128_by_r=np.asarray(td128_by_r,dtype=np.float64))
    os.replace(tmp, path)


def load_mem_cp(path: Path, sig: str):
    if not path.exists():
        return None
    try:
        z=np.load(path,allow_pickle=False)
        if str(z["signature"].item()) != sig:
            return None
        return {"profiles_mean_r":z["profiles_mean_r"].astype(np.float64),
                "td128_by_r":z["td128_by_r"].astype(np.float64)}
    except Exception:
        return None


def env_cp_path(out: Path, subject: str, rep: int, sig: str) -> Path:
    d = out / "checkpoints_environment"; d.mkdir(parents=True, exist_ok=True)
    return d / f"{subject}_rep{rep:02d}_{sig}.json"


def save_env_cp(path: Path, sig: str, rows: List[Dict[str,object]]):
    tmp=path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"signature":sig,"rows":rows}))
    os.replace(tmp,path)


def load_env_cp(path: Path, sig: str):
    if not path.exists():
        return None
    try:
        z=json.loads(path.read_text())
        return z.get("rows",[]) if z.get("signature")==sig else None
    except Exception:
        return None


# -----------------------------------------------------------------------------
# Memory simulation beta>0
# -----------------------------------------------------------------------------

def cc_memory_rep(core, template, pattern_groups, subject: str, rep: int, out: Path, sig: str, checkpoint: bool):
    cp=mem_cp_path(out,subject,rep,sig)
    if checkpoint:
        old=load_mem_cp(cp,sig)
        if old is not None:
            return old

    signs,b=core.make_signs_and_input(template,subject,rep)
    base=core.base_weights(template,signs,"equal")
    train_u,start_tr=build_extended_input(core,subject,rep,"train",TRAIN_LEN)
    test_u,start_te=build_extended_input(core,subject,rep,"test",TEST_LEN)

    nb,ns,nr=len(NEW_BETAS),len(core.SCALES),len(core.COUPLINGS)
    prof=np.zeros((nb,ns,MAX_LAG),dtype=np.float64)
    td=np.zeros((nb,ns,nr),dtype=np.float64)

    # Exact integer-delay duplicates across beta/S are simulated only once per r.
    for ri,r in enumerate(core.COUPLINGS):
        for g in pattern_groups:
            M=memory_profile(core,template,base,b,g["delays"],float(r),train_u,test_u,start_tr,start_te)
            tdv=td_at_horizon(M,128)
            for bi,si,_beta,_S in g["labels"]:
                prof[bi,si]+=M/float(nr)
                td[bi,si,ri]=tdv

    if checkpoint:
        save_mem_cp(cp,sig,prof,td)
    return {"profiles_mean_r":prof,"td128_by_r":td}


def cc_memory_subject_worker(payload):
    sf_dict,core_path,out_s,sig,checkpoint=payload
    core=load_core(core_path); sf=core.SubjectFiles(**sf_dict)
    template=core.build_empirical_template(sf); out=Path(out_s)
    pattern_groups=compensated_pattern_groups(core,template)
    profiles=[]; acc={float(b):[] for b in NEW_BETAS}
    for rep in range(1,INTERNAL_REPS+1):
        z=cc_memory_rep(core,template,pattern_groups,sf.subject,rep,out,sig,checkpoint)
        profiles.append(z["profiles_mean_r"]); td=z["td128_by_r"]
        for bi,beta in enumerate(NEW_BETAS):
            for ri,_r in enumerate(core.COUPLINGS):
                acc[float(beta)].append(core.slope_y_on_log2s(core.SCALES,td[bi,:,ri]))
    P=np.mean(np.stack(profiles,axis=0),axis=0)
    effects={float(b):float(np.nanmean(v)) for b,v in acc.items()}
    return {"subject":sf.subject,"session":sf.session,"profile":P.astype(np.float32),
            "effects":effects,"delay_diag":delay_diagnostics(core,template,sf.subject),
            "unique_delay_patterns":int(len(pattern_groups))}


# -----------------------------------------------------------------------------
# Environment simulation beta>0
# -----------------------------------------------------------------------------

def cc_environment_rep(core,template,pattern_groups,subject:str,rep:int,cfg,out:Path,sig:str,checkpoint:bool):
    cp=env_cp_path(out,subject,rep,sig)
    if checkpoint:
        old=load_env_cp(cp,sig)
        if old is not None:
            return old

    signs,b=core.make_signs_and_input(template,subject,rep)
    base=core.base_weights(template,signs,"equal")
    rows=[]
    for Tenv in core.T_ENVS:
        seqs={}
        for part,L in (("train",cfg.env_train),("val",cfg.env_val),("test",cfg.env_test)):
            seed=core.stable_seed(subject,rep,"env",int(Tenv),part)
            seqs[part]=core.make_environment_sequence(cfg.env_washout+L,int(Tenv),seed)
        for r in core.COUPLINGS:
            for g in pattern_groups:
                va,vm,ta,tm=core.env_metrics(template,base,b,g["delays"],float(r),seqs,cfg)
                for _bi,_si,beta,S in g["labels"]:
                    rows.append({"beta":float(beta),"T_env":int(Tenv),"scale":int(S),"r":float(r),
                                 "val_accuracy":va,"val_mse":vm,"test_accuracy":ta,"test_mse":tm})
    if checkpoint:
        save_env_cp(cp,sig,rows)
    return rows


def cc_environment_subject_worker(payload):
    sf_dict,core_path,data_root,out_s,sig,checkpoint=payload
    core=load_core(core_path); sf=core.SubjectFiles(**sf_dict)
    template=core.build_empirical_template(sf)
    pattern_groups=compensated_pattern_groups(core,template)
    cfg=core_config(core,data_root,out_s,"environment"); out=Path(out_s)
    rows=[]
    for rep in range(1,INTERNAL_REPS+1):
        rr=cc_environment_rep(core,template,pattern_groups,sf.subject,rep,cfg,out,sig,checkpoint)
        rows.extend([{"subject":sf.subject,"session":sf.session,"rep":rep,**x} for x in rr])
    return rows


# -----------------------------------------------------------------------------
# Summary statistics
# -----------------------------------------------------------------------------

def bootstrap_ci(v:Sequence[float],n_boot:int,seed:int)->Tuple[float,float]:
    x=np.asarray(v,dtype=float); x=x[np.isfinite(x)]
    if x.size==0: return float("nan"),float("nan")
    rng=np.random.default_rng(seed); means=np.empty(n_boot)
    for i in range(n_boot): means[i]=rng.choice(x,size=x.size,replace=True).mean()
    return float(np.quantile(means,.025)),float(np.quantile(means,.975))


def summarize_by_beta(df:pd.DataFrame,value_col:str,outpath:Path,baseline_mean:float|None=None):
    rows=[]
    for beta,d in df.groupby("beta"):
        v=pd.to_numeric(d[value_col],errors="coerce").to_numpy(float); v=v[np.isfinite(v)]
        lo,hi=bootstrap_ci(v,10000,stable_seed("beta_summary",value_col,float(beta)))
        mean=float(np.mean(v))
        rows.append({"beta":float(beta),"metric":value_col,"n_subjects":int(v.size),
                     "mean":mean,"median":float(np.median(v)),"bootstrap_95ci_low":lo,
                     "bootstrap_95ci_high":hi,"positive_fraction":float(np.mean(v>0)),
                     "retention_vs_beta0":float(mean/baseline_mean) if baseline_mean not in (None,0) else np.nan})
    out=pd.DataFrame(rows).sort_values("beta"); out.to_csv(outpath,index=False); return out


# -----------------------------------------------------------------------------
# Baseline readers
# -----------------------------------------------------------------------------

def require_baselines(equal_root:Path,equal_ext_root:Path,phase:str):
    need=[]
    if phase in ("memory","all"):
        need += [equal_root/"05_03_memory_subject_effects.csv",
                 equal_ext_root/"07_03_subject_metrics.csv",
                 equal_ext_root/"07_02_group_profile.csv"]
    if phase in ("environment","all"):
        need += [equal_root/"05_04_environment_condition_level.csv",
                 equal_root/"05_05_environment_selected_scales.csv",
                 equal_root/"05_06_environment_subject_effects.csv"]
    missing=[str(p) for p in need if not p.exists()]
    if missing:
        raise FileNotFoundError("Required beta=0 baseline outputs missing:\n"+"\n".join(missing))


# -----------------------------------------------------------------------------
# Write memory outputs
# -----------------------------------------------------------------------------

def write_memory_outputs(core,results,out:Path,equal_root:Path,equal_ext_root:Path):
    results=sorted(results,key=lambda z:z["subject"])

    # New beta subject profiles/metrics.
    profile_path=out/"09_CC_01_subject_profile_long_beta_gt0.csv"; first=True
    metric_rows=[]; group_acc=np.zeros((len(NEW_BETAS),len(core.SCALES),MAX_LAG),dtype=np.float64)
    effect_rows=[]; delay_rows=[]
    for z in results:
        P=np.asarray(z["profile"],dtype=np.float64); group_acc+=P
        chunks=[]
        for bi,beta in enumerate(NEW_BETAS):
            effect_rows.append({"subject":z["subject"],"beta":float(beta),"A1_geo_TD_slope":float(z["effects"][float(beta)])})
            for si,S in enumerate(core.SCALES):
                M=P[bi,si]
                chunks.append(pd.DataFrame({"subject":z["subject"],"beta":float(beta),"scale":int(S),
                                            "lag":np.arange(1,MAX_LAG+1,dtype=np.int32),"M":M}))
                row={"subject":z["subject"],"beta":float(beta),"scale":int(S)}
                row.update(profile_metrics(M,int(core.ATLAS_SCALE))); metric_rows.append(row)
        delay_rows.extend(z["delay_diag"])
        pd.concat(chunks,ignore_index=True).to_csv(profile_path,index=False,mode="w" if first else "a",header=first)
        first=False

    new_metrics=pd.DataFrame(metric_rows)
    new_effects=pd.DataFrame(effect_rows)

    # beta=0 subject metrics from completed 07 (geometric only).
    b0m=pd.read_csv(equal_ext_root/"07_03_subject_metrics.csv")
    b0m=b0m[b0m.condition=="geometric"].copy()
    rename={"MC_1_128":"MC_1_128","MC_1_256":"MC_1_256","MC_1_512":"MC_1_512","MC_1_1024":"MC_1_1024",
            "MC_1_1024_over_state_dim":"MC_1_1024_over_state_dim","MC_1_256_over_state_dim":"MC_1_256_over_state_dim"}
    keep=["subject","scale"]+[c for c in rename if c in b0m.columns]
    b0m=b0m[keep].copy(); b0m["beta"]=0.0
    # Add TD_1_128 from old primary subject profile if absent is not needed for group mechanism table.
    combined_metrics=pd.concat([b0m,new_metrics],ignore_index=True,sort=False).sort_values(["beta","subject","scale"])
    combined_metrics.to_csv(out/"09_CC_03_subject_metrics_all_beta.csv",index=False)

    # beta=0 A1 from core 05.
    b0e=pd.read_csv(equal_root/"05_03_memory_subject_effects.csv")[["subject","A1_geo_TD_slope"]].copy()
    b0e["beta"]=0.0
    effects=pd.concat([b0e[["subject","beta","A1_geo_TD_slope"]],new_effects],ignore_index=True).sort_values(["beta","subject"])
    effects.to_csv(out/"09_CC_05_memory_subject_effects.csv",index=False)
    beta0_mean=float(b0e.A1_geo_TD_slope.mean())
    summarize_by_beta(effects,"A1_geo_TD_slope",out/"09_CC_06_memory_A1_summary.csv",beta0_mean)

    # Group profiles: beta0 from completed 07, beta>0 newly computed.
    b0gp=pd.read_csv(equal_ext_root/"07_02_group_profile.csv")
    b0gp=b0gp[b0gp.condition=="geometric"][["scale","lag","M_mean"]].copy(); b0gp["beta"]=0.0
    grows=[b0gp[["beta","scale","lag","M_mean"]]]
    new_group=group_acc/float(len(results)); gm=[]
    for bi,beta in enumerate(NEW_BETAS):
        for si,S in enumerate(core.SCALES):
            M=new_group[bi,si]
            grows.append(pd.DataFrame({"beta":float(beta),"scale":int(S),"lag":np.arange(1,MAX_LAG+1),"M_mean":M}))
            row={"beta":float(beta),"scale":int(S)}; row.update(profile_metrics(M,int(core.ATLAS_SCALE))); gm.append(row)
    group_profile=pd.concat(grows,ignore_index=True).sort_values(["beta","scale","lag"])
    group_profile.to_csv(out/"09_CC_02_group_profile_all_beta.csv",index=False)

    # Build group metric table from subject metrics to include beta0 and beta>0 uniformly.
    metric_cols=[c for c in combined_metrics.columns if c not in ("subject",)]
    group_metrics=combined_metrics.groupby(["beta","scale"],as_index=False).mean(numeric_only=True)
    group_metrics.to_csv(out/"09_CC_04_group_metrics_all_beta.csv",index=False)

    pd.DataFrame(delay_rows).sort_values(["subject","beta","scale"]).to_csv(out/"09_CC_07_delay_distribution_subject.csv",index=False)
    pd.DataFrame(delay_rows).groupby(["beta","scale"],as_index=False).mean(numeric_only=True).to_csv(out/"09_CC_08_delay_distribution_group.csv",index=False)


# -----------------------------------------------------------------------------
# Write environment outputs
# -----------------------------------------------------------------------------

def write_environment_outputs(core,rows,out:Path,equal_root:Path):
    new=rows.copy() if isinstance(rows,pd.DataFrame) else pd.DataFrame(rows)
    new.to_csv(out/"09_CC_10_environment_condition_level_beta_gt0.csv",index=False)

    # Validation-only S* for beta>0.
    sel_rows=[]
    for (sub,rep,r,Tenv,beta),d in new.groupby(["subject","rep","r","T_env","beta"]):
        bestS,va,ta,tm=core.choose_scale_from_validation(d)
        sel_rows.append({"subject":sub,"rep":int(rep),"r":float(r),"T_env":int(Tenv),"beta":float(beta),
                         "S_star":int(bestS),"val_accuracy_star":va,"test_accuracy_star":ta,"test_mse_star":tm})
    sel_new=pd.DataFrame(sel_rows)

    # beta=0 geometric + ZERO selected-scale baseline from completed core 05.
    base_sel=pd.read_csv(equal_root/"05_05_environment_selected_scales.csv")
    geo0=base_sel[base_sel.condition=="geometric"].copy(); geo0["beta"]=0.0
    geo0=geo0[["subject","rep","r","T_env","beta","S_star","val_accuracy_star","test_accuracy_star","test_mse_star"]]
    zero=base_sel[base_sel.condition=="zero"][["subject","rep","r","T_env","test_accuracy_star"]].copy()
    zero=zero.rename(columns={"test_accuracy_star":"zero_test_accuracy_star"})
    sel=pd.concat([geo0,sel_new],ignore_index=True).sort_values(["beta","subject","rep","r","T_env"])
    sel.to_csv(out/"09_CC_11_environment_selected_scales_all_beta.csv",index=False)

    # Subject A2/A3 for beta>0 using exact same ZERO baseline; beta0 from core 05.
    subject_rows=[]
    for beta in NEW_BETAS:
        d_beta=sel_new[sel_new.beta==float(beta)]
        for sub,ds in d_beta.groupby("subject"):
            a2=[]; a3=[]
            for (rep,r),drr in ds.groupby(["rep","r"]):
                drr=drr.sort_values("T_env")
                a2.append(core.rho_log_env_scale(drr.T_env.to_numpy(),drr.S_star.to_numpy()))
                z=zero[(zero.subject==sub)&(zero.rep==rep)&(zero.r==r)]
                m=drr[["T_env","test_accuracy_star"]].merge(z[["T_env","zero_test_accuracy_star"]],on="T_env")
                a3.append(core.slope_y_on_log2s(m.T_env.to_numpy(),(m.test_accuracy_star-m.zero_test_accuracy_star).to_numpy()))
            subject_rows.append({"subject":sub,"beta":float(beta),"A2_geo_rho_Tenv_Sstar":float(np.nanmean(a2)),
                                 "A3_geo_minus_zero_accuracy_slope":float(np.nanmean(a3))})
    new_eff=pd.DataFrame(subject_rows)
    b0=pd.read_csv(equal_root/"05_06_environment_subject_effects.csv")[["subject","A2_geo_rho_Tenv_Sstar","A3_geo_minus_zero_accuracy_slope"]].copy(); b0["beta"]=0.0
    effects=pd.concat([b0[["subject","beta","A2_geo_rho_Tenv_Sstar","A3_geo_minus_zero_accuracy_slope"]],new_eff],ignore_index=True)
    effects=effects.sort_values(["beta","subject"]); effects.to_csv(out/"09_CC_12_environment_subject_effects.csv",index=False)

    for col,fn in (("A2_geo_rho_Tenv_Sstar","09_CC_13_environment_A2_summary.csv"),
                   ("A3_geo_minus_zero_accuracy_slope","09_CC_14_environment_A3_summary.csv")):
        bmean=float(b0[col].mean()); summarize_by_beta(effects[["subject","beta",col]],col,out/fn,bmean)

    # Group performance surface including beta=0 geometric.
    base_cond=pd.read_csv(equal_root/"05_04_environment_condition_level.csv")
    b0c=base_cond[base_cond.condition=="geometric"].copy(); b0c["beta"]=0.0
    b0c=b0c[["subject","rep","r","T_env","scale","beta","val_accuracy","val_mse","test_accuracy","test_mse"]]
    allc=pd.concat([b0c,new],ignore_index=True,sort=False)
    surf=allc.groupby(["beta","T_env","scale"],as_index=False).agg(
        val_accuracy=("val_accuracy","mean"),test_accuracy=("test_accuracy","mean"),test_mse=("test_mse","mean"))
    surf.to_csv(out/"09_CC_15_environment_surface_group.csv",index=False)
    sg=sel.groupby(["beta","T_env"],as_index=False).agg(mean_S_star=("S_star","mean"),median_S_star=("S_star","median"),
                                                         mean_test_accuracy_star=("test_accuracy_star","mean"))
    sg.to_csv(out/"09_CC_16_selected_scale_group.csv",index=False)


# -----------------------------------------------------------------------------
# Preflight / self-test / README
# -----------------------------------------------------------------------------

def preflight(core,subjects,out:Path):
    rows=[]
    for sf in subjects:
        try:
            t=core.build_empirical_template(sf)
            d0=compensated_delays(core,t,0.0,64); d1=compensated_delays(core,t,1.0,64)
            rows.append({"subject":sf.subject,"status":"ok","n_regions":int(t.n),"n_undirected_edges":int(len(t.undirected_delta)),
                         "beta0_matches_core_S64":bool(np.array_equal(d0,core.expand_undirected_delays(core.geometric_delays(t.undirected_delta,64)))),
                         "beta1_all_equal_S64":bool(np.all(d1==64)),"error":""})
        except Exception as e:
            rows.append({"subject":sf.subject,"status":"error","error":repr(e)})
    df=pd.DataFrame(rows); df.to_csv(out/"09_CC_00_preflight.csv",index=False)
    if len(df)!=50 or (df.status!="ok").any() or not df.beta0_matches_core_S64.all() or not df.beta1_all_equal_S64.all():
        raise RuntimeError("conduction-compensation preflight failed")


def run_self_test(core_path:str):
    core=load_core(core_path); tests={}
    tests["core_version"]=getattr(core,"SCRIPT_VERSION",None)==REQUIRED_CORE_VERSION
    tests["core_self_test"]=bool(core.run_self_test().get("all_pass",False))
    d=np.asarray([.1,.25,.5,1.0])
    tests["beta0_identity_continuous"]=bool(np.allclose(compensated_delta(d,0.0),d))
    tests["beta1_equal_continuous"]=bool(np.allclose(compensated_delta(d,1.0),1.0))
    tests["compensation_compresses_continuous_sd"]=bool(np.std(compensated_delta(d,.75)) < np.std(compensated_delta(d,.25)) < np.std(d))
    tests["replicate_ids"]=list(range(1,INTERNAL_REPS+1))==list(range(1,9))
    tests["prefix_valid"]=required_prefix(core)+WASHOUT-MAX_LAG>=0
    tests["blas_thread_caps"] = all(os.environ.get(v) == "1" for v in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"))

    # Synthetic template delay endpoints.
    n=4; ui=np.asarray([0,0,1,2],dtype=np.int32); uj=np.asarray([1,2,2,3],dtype=np.int32)
    src=np.empty(8,dtype=np.int32); dst=np.empty(8,dtype=np.int32)
    src[0::2],dst[0::2]=uj,ui; src[1::2],dst[1::2]=ui,uj
    indeg=np.bincount(dst,minlength=n).astype(np.int32); du=np.asarray([.1,.3,.6,1.0]); scu=np.ones(4)
    tpl=core.NetworkTemplate(n,2,src,dst,np.repeat(du,2),np.repeat(scu,2),du,scu,ui,uj,100*du,100.0,indeg)
    tests["beta0_delay_identity"]=bool(np.array_equal(compensated_delays(core,tpl,0.0,16),core.expand_undirected_delays(core.geometric_delays(du,16))))
    tests["beta1_delay_equal"]=bool(np.all(compensated_delays(core,tpl,1.0,16)==16))
    _pg = compensated_pattern_groups(core, tpl)
    _labels = sum((g["labels"] for g in _pg), [])
    tests["delay_dedup_preserves_all_beta_scale_labels"] = len(_labels) == len(NEW_BETAS) * len(core.SCALES)
    tests["delay_dedup_not_expansive"] = len(_pg) <= len(_labels)

    # Batched OLS equivalence.
    rng=np.random.default_rng(1); ntr,nte,p,K=300,250,8,12; st=40
    tr=rng.normal(size=st+ntr); te=rng.normal(size=st+nte); Xtr=rng.normal(size=(ntr,p)); Xte=rng.normal(size=(nte,p))
    fast=memory_scores_from_states(Xtr,Xte,tr,te,st,st,max_lag=K,lag_batch=5)
    Ytr=target_block(tr,st,ntr,1,K); Yte=target_block(te,st,nte,1,K); xm=Xtr.mean(0); Xc=Xtr-xm
    B=np.linalg.pinv(Xc.T@Xc,hermitian=True)@(Xc.T@(Ytr-Ytr.mean(0))); P=(Xte-xm)@B
    slow=np.asarray([np.corrcoef(Yte[:,q],P[:,q])[0,1]**2 for q in range(K)])
    tests["batched_matches_naive"]=bool(np.allclose(fast,slow,rtol=1e-10,atol=1e-12))
    tests["batched_max_abs_error"]=float(np.max(np.abs(fast-slow)))

    import tempfile
    with tempfile.TemporaryDirectory() as td:
        cp=Path(td)/"x.npz"; P=np.zeros((4,7,1024)); T=np.zeros((4,7,3)); save_mem_cp(cp,"s",P,T); z=load_mem_cp(cp,"s")
        tests["checkpoint_roundtrip"]=bool(z is not None and z["profiles_mean_r"].shape==(4,7,1024))
    tests["all_pass"]=bool(all(v for k,v in tests.items() if k!="batched_max_abs_error"))
    return tests


def write_readme(out:Path,phase:str):
    txt=f"""# 09 MICA-MICs Conduction-Compensation Robustness\n\nScript version: `{SCRIPT_VERSION}`  \nCore required: `{REQUIRED_CORE_VERSION}`  \nPhase: {phase}  \nSubjects: 50  \nReplicates: 1..8  \nWeighting: equal  \nBetas: 0, 0.25, 0.50, 0.75, 1.00  \nNew simulations: beta > 0 only  \nMax memory lag: {MAX_LAG}\n\n## Mapping\n\n`effective_delta = delta ** (1 - beta)` and `tau = round(S * effective_delta)`.\nBeta=0 is the original mapping. Beta=1 removes relative tract-length differences at fixed S.\nA common multiplicative velocity factor is absorbed by S, so beta changes only length-dependent\nrelative compensation.\n\n## Scientific rule\n\nThis is a sensitivity trajectory, not a new optimization. Beta values are fixed before running and\nno beta is selected from outcomes. Equal weights, topology, subjects, rep IDs, signs, inputs,\nstochastic sequences, S grid, r grid, T_env grid, validation-only S* selection, and held-out tests\nare unchanged. Beta=0 is loaded from the completed equal-weight 05/07 analyses.\n\nPrimary robustness quantities are A1 (TD-vs-scale slope on lags 1..128), A2 (T_env-vs-S* matching),\nand A3 (GEOMETRIC-minus-ZERO held-out advantage vs temporal demand), summarized continuously\nacross beta with subject bootstrap confidence intervals. No new significance gate is introduced.\n"""
    (out/"09_CC_99_README_RUN.md").write_text(txt)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    a=parse_args(); core_path=str(Path(a.core_script).expanduser())
    if a.self_test:
        print(json.dumps(run_self_test(core_path),indent=2)); return
    t0=time.time(); core=load_core(core_path)
    if not getattr(core,"NUMBA_AVAILABLE",False):
        raise RuntimeError("Numba is required for the full robustness run; refusing slow Python fallback.")
    out=Path(a.outdir).expanduser(); out.mkdir(parents=True,exist_ok=True)
    equal_root=Path(a.equal_root).expanduser(); equal_ext=Path(a.equal_extended_root).expanduser()
    require_baselines(equal_root,equal_ext,a.phase)
    cfg=core_config(core,a.data_root,str(out),a.phase); subjects=core.discover_subjects(cfg)
    preflight(core,subjects,out); write_readme(out,a.phase)
    audit={"script_version":SCRIPT_VERSION,"core_version":REQUIRED_CORE_VERSION,"phase":a.phase,"subjects":len(subjects),
           "replicate_ids":list(range(1,INTERNAL_REPS+1)),"betas":[float(x) for x in BETAS],"new_betas":[float(x) for x in NEW_BETAS],
           "mapping":"delta**(1-beta)","weighting":"equal","max_lag":MAX_LAG}

    if a.phase in ("memory","all"):
        print("[09 CC] memory robustness",flush=True); sig=signature(core,"memory")
        payloads=[(asdict(sf),core_path,str(out),sig,not a.no_checkpoint) for sf in subjects]
        results=[]; errors=[]
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            futs={ex.submit(cc_memory_subject_worker,p):p for p in payloads}; done=0
            for fut in as_completed(futs):
                try: results.append(fut.result())
                except Exception: errors.append(traceback.format_exc()); print(errors[-1],flush=True)
                done+=1; print(f"[09 CC memory {done}/{len(payloads)}]",flush=True)
        if errors: raise RuntimeError(f"CC memory failed for {len(errors)} subjects")
        write_memory_outputs(core,results,out,equal_root,equal_ext); audit["memory_subjects"]=len(results)

    if a.phase in ("environment","all"):
        print("[09 CC] environment robustness",flush=True); sig=signature(core,"environment")
        payloads=[(asdict(sf),core_path,str(Path(a.data_root).expanduser()),str(out),sig,not a.no_checkpoint) for sf in subjects]
        errors=[]; stream_path=out/"09_CC_10_environment_condition_level_beta_gt0.csv"
        if stream_path.exists(): stream_path.unlink()
        first=True; n_rows=0
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            futs={ex.submit(cc_environment_subject_worker,p):p for p in payloads}; done=0
            for fut in as_completed(futs):
                try:
                    part=fut.result(); pdf=pd.DataFrame(part)
                    pdf.to_csv(stream_path,index=False,mode="w" if first else "a",header=first)
                    first=False; n_rows+=len(pdf)
                except Exception: errors.append(traceback.format_exc()); print(errors[-1],flush=True)
                done+=1; print(f"[09 CC environment {done}/{len(payloads)}]",flush=True)
        if errors: raise RuntimeError(f"CC environment failed for {len(errors)} subjects")
        env_df=pd.read_csv(stream_path)
        write_environment_outputs(core,env_df,out,equal_root); audit["environment_rows_beta_gt0"]=int(n_rows)

    audit["runtime_seconds"]=float(time.time()-t0); (out/"09_CC_98_audit.json").write_text(json.dumps(audit,indent=2))
    print(json.dumps(audit,indent=2),flush=True); print(f"Outputs: {out}",flush=True)


if __name__=="__main__":
    main()
