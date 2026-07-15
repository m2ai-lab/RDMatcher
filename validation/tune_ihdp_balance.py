"""Balance-only grid search for the seven IHDP validation methods.

No outcome, potential outcome, ATT, or bias value is read by this script.
Threshold policies are selected by minimum mean absolute covariate SMD among
configurations with treated retention at least 90 percent.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
sys.path.insert(0, ".")

from datasets import load_ihdp_single
from rdmatcher import RDMatcher
from comparison_methods import run_matchit_ps, run_matchit_mahalanobis, run_matchit_maha_hybrid

GOWER_THRESHOLDS = (0.10, 0.125, 0.15, 0.175, 0.20, 0.225, 0.25, 0.275, 0.30)
MAHA_THRESHOLDS = (4.00, 4.25, 4.50, 4.75, 5.00, 5.25, 5.50, 5.75, 6.00)
PS_CALIPERS = (0.20,)
K_CANDIDATES = 250
OUT_PATH = "validation/plots/ihdp_balance_threshold_tuning.csv"


def absolute_smds(matched, numeric, categorical):
    t = matched[matched.exposure_status == 1]
    c = matched[matched.exposure_status == 0]
    out = {}
    for col in numeric:
        den = np.sqrt((t[col].std(ddof=1) ** 2 + c[col].std(ddof=1) ** 2) / 2)
        out[col] = 0.0 if den == 0 else abs((t[col].mean() - c[col].mean()) / den)
    for col in categorical:
        pt, pc = t[col].astype(str).eq("1").mean(), c[col].astype(str).eq("1").mean()
        den = np.sqrt((pt * (1 - pt) + pc * (1 - pc)) / 2)
        out[col] = 0.0 if den == 0 else abs((pt - pc) / den)
    return out


def run_rdm(df, numeric, categorical, threshold, caliper=None, maha=False):
    if caliper is not None and pd.isna(caliper):
        caliper = None
    matcher = RDMatcher(pop_df=df, patient_id_col="patient_id", exposure_status="exposure_status",
                         features_numeric=numeric, features_categorical=categorical,
                         process_features=False, onehot=maha, debug=False, log_to_console=False)
    if caliper is not None:
        matcher.fit_propensity_model(formula=" + ".join(numeric + categorical), random_state=404)
    return matcher.rare_matching(
        threshold=threshold, n_neighbors=1, k_candidates=K_CANDIDATES, method="multi",
        distance_metric="mahalanobis" if maha else "gower", global_optimal=True,
        competitive_match=True, ps_hybrid=caliper is not None, ps_caliper=caliper,
        ps_caliper_strict=True, diagnostics=False, return_matched_data=True,
        gower_sd_weights=not maha, gower_sd_weights_mult=1.96, gower_sd_reference="controls")


def run_matchit(df, numeric, categorical, method, caliper):
    cov = numeric + categorical
    formula = "exposure_status ~ " + " + ".join(cov)
    if method == "PSM (MatchIt)":
        return run_matchit_ps(df, "exposure_status", "outcome", cov, formula=formula, caliper=caliper, ratio=1).matched_df
    if method == "Maha (MatchIt)":
        return run_matchit_mahalanobis(df, "exposure_status", "outcome", cov, formula=formula, ratio=1).matched_df
    return run_matchit_maha_hybrid(df, "exposure_status", "outcome", cov, formula=formula, caliper=caliper, ratio=1).matched_df


def grid():
    return ([ ("RDM", x, np.nan, False) for x in GOWER_THRESHOLDS ] +
            [ ("PSM+RDM", x, p, False) for p in PS_CALIPERS for x in GOWER_THRESHOLDS ] +
            [ ("Maha (RDM)", x, np.nan, True) for x in MAHA_THRESHOLDS ] +
            [ ("PSM+Maha (RDM)", x, p, True) for p in PS_CALIPERS for x in MAHA_THRESHOLDS ] +
            [ ("PSM (MatchIt)", np.nan, p, None) for p in PS_CALIPERS ] +
            [ ("Maha (MatchIt)", np.nan, np.nan, None) ] +
            [ ("PSM+Maha (MatchIt)", np.nan, p, None) for p in PS_CALIPERS ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-rep", type=int, default=0)
    ap.add_argument("--n-reps", type=int, default=1)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--stop", type=int)
    ap.add_argument("--aggregate", action="store_true")
    args = ap.parse_args()
    out = Path(OUT_PATH)
    if args.aggregate:
        paths = sorted(out.parent.glob("ihdp_balance_threshold_tuning_*.csv"))
        if not paths: raise FileNotFoundError("No IHDP balance chunks found")
        df = pd.concat((pd.read_csv(p) for p in paths), ignore_index=True)
        df.to_csv(out, index=False)
        eligible = df[df.retention >= .90].sort_values(["mean_abs_smd", "max_abs_smd"])
        print(eligible.groupby("method", sort=False).head(3).to_string(index=False))
        return
    configs = grid(); stop = len(configs) if args.stop is None else args.stop
    rows = []
    for rep in range(args.start_rep, min(args.start_rep + args.n_reps, 100)):
        df = load_ihdp_single(rep); meta = df.attrs["meta"]
        numeric, categorical = meta["features_numeric"], meta["features_categorical"]
        n_treated = int((df.exposure_status == 1).sum())
        for i, (method, threshold, caliper, maha) in enumerate(configs[args.start:stop], args.start + 1):
            started = perf_counter()
            if maha is None:
                matched = run_matchit(df, numeric, categorical, method, caliper)
            else:
                matched = run_rdm(df, numeric, categorical, threshold, caliper, maha)
            if matched is None: matched = pd.DataFrame()
            matched = matched[matched.match_group.notna()].copy() if not matched.empty else matched
            smds = absolute_smds(matched, numeric, categorical) if not matched.empty else {x: np.nan for x in numeric + categorical}
            n_match = int((matched.exposure_status == 1).sum()) if not matched.empty else 0
            rows.append({"replication": rep, "method": method, "threshold": threshold,
                         "ps_caliper": caliper, "k_candidates": K_CANDIDATES,
                         "mean_abs_smd": float(np.nanmean(list(smds.values()))),
                         "rms_smd": float(np.sqrt(np.nanmean(np.square(list(smds.values()))))),
                         "max_abs_smd": float(np.nanmax(list(smds.values()))),
                         "matched_treated": n_match, "retention": n_match / n_treated,
                         "match_time_sec": perf_counter() - started,
                         **{f"smd_{k}": v for k, v in smds.items()}})
            print(f"rep={rep} [{i}/{stop}] {method} threshold={threshold} caliper={caliper}", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    chunk = out.with_name(f"ihdp_balance_threshold_tuning_{args.start_rep}_{args.start_rep + args.n_reps - 1}_{args.start}_{stop}.csv")
    pd.DataFrame(rows).to_csv(chunk, index=False); print(f"Saved {chunk}")


if __name__ == "__main__": main()
