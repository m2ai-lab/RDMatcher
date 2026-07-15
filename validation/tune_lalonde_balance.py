"""Balance-only threshold tuning for RDMatcher methods on LaLonde CPS.

The outcome is not used in any score, ranking, or saved result.
"""

from __future__ import annotations

import argparse
import os
import sys
from itertools import product
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
sys.path.insert(0, ".")

from datasets import load_lalonde
from rdmatcher import RDMatcher
from comparison_methods import run_matchit_ps, run_matchit_mahalanobis, run_matchit_maha_hybrid


GOWER_THRESHOLDS = (0.10, 0.125, 0.15, 0.175, 0.20, 0.225, 0.25, 0.275, 0.30)
MAHA_THRESHOLDS = (1.00, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00)
PS_CALIPERS = (0.10, 0.15, 0.20)
K_CANDIDATES = 250
OUT_PATH = "validation/plots/lalonde_balance_threshold_tuning.csv"


def _absolute_smds(matched: pd.DataFrame, numeric: list[str], categorical: list[str]) -> dict[str, float]:
    treated = matched[matched["exposure_status"] == 1]
    controls = matched[matched["exposure_status"] == 0]
    values: dict[str, float] = {}
    for column in numeric:
        denominator = np.sqrt((treated[column].std(ddof=1) ** 2 + controls[column].std(ddof=1) ** 2) / 2)
        values[column] = 0.0 if denominator == 0 else abs((treated[column].mean() - controls[column].mean()) / denominator)
    for column in categorical:
        p_treated, p_control = treated[column].mean(), controls[column].mean()
        denominator = np.sqrt((p_treated * (1 - p_treated) + p_control * (1 - p_control)) / 2)
        values[column] = 0.0 if denominator == 0 else abs((p_treated - p_control) / denominator)
    return values


def _parse_args():
    parser = argparse.ArgumentParser(description="Balance-only threshold tuning on LaLonde CPS.")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int)
    parser.add_argument("--aggregate", action="store_true")
    return parser.parse_args()


def _aggregate_chunks():
    out_path = Path(OUT_PATH)
    paths = sorted(out_path.parent.glob("lalonde_balance_threshold_tuning_*.csv"))
    if not paths:
        raise FileNotFoundError("No balance-tuning chunks were found to aggregate.")
    results = pd.concat((pd.read_csv(path) for path in paths), ignore_index=True)
    if results.duplicated(["method", "threshold", "ps_caliper"]).any() or len(results) != 79:
        raise ValueError("Balance-tuning chunks are incomplete or contain duplicate configurations.")
    results = results.sort_values(["method", "threshold"]).reset_index(drop=True)
    results.to_csv(out_path, index=False)
    eligible = results[results["retention"] >= 0.90].sort_values(["mean_abs_smd", "max_abs_smd", "match_time_sec"])
    print("Balance-optimal configurations with at least 90% retention:")
    print(eligible.groupby("method", sort=False).head(3).to_string(index=False))
    print(f"\nSaved: {out_path}")


def main():
    args = _parse_args()
    if args.aggregate:
        _aggregate_chunks()
        return
    df = load_lalonde(use_re74=True, external_control="cps")
    meta = df.attrs["meta"]
    numeric, categorical = meta["features_numeric"], meta["features_categorical"]
    n_treated = int((df["exposure_status"] == 1).sum())
    grid = []
    grid.extend(("RDM", threshold, np.nan) for threshold in GOWER_THRESHOLDS)
    grid.extend(("PSM+RDM", threshold, caliper) for caliper in PS_CALIPERS for threshold in GOWER_THRESHOLDS)
    grid.extend(("Maha (RDM)", threshold, np.nan) for threshold in MAHA_THRESHOLDS)
    grid.extend(("PSM+Maha (RDM)", threshold, caliper) for caliper in PS_CALIPERS for threshold in MAHA_THRESHOLDS)
    grid.extend(("PSM (MatchIt)", np.nan, caliper) for caliper in PS_CALIPERS)
    grid.append(("Maha (MatchIt)", np.nan, np.nan))
    grid.extend(("PSM+Maha (MatchIt)", np.nan, caliper) for caliper in PS_CALIPERS)
    stop = len(grid) if args.stop is None else args.stop
    if not 0 <= args.start < stop <= len(grid):
        raise ValueError("Requested grid range is invalid.")
    selected_grid = grid[args.start:stop]

    rows = []
    for index, (method, threshold, caliper) in enumerate(selected_grid, start=args.start + 1):
        started = perf_counter()
        if method == "PSM (MatchIt)":
            covariates = numeric + categorical
            result = run_matchit_ps(df, "exposure_status", "outcome", covariates,
                                    formula="exposure_status ~ " + " + ".join(covariates),
                                    caliper=caliper, ratio=1)
            matched = result.matched_df
        elif method == "Maha (MatchIt)":
            covariates = numeric + categorical
            result = run_matchit_mahalanobis(df, "exposure_status", "outcome", covariates,
                                             formula="exposure_status ~ " + " + ".join(covariates),
                                             ratio=1)
            matched = result.matched_df
        elif method == "PSM+Maha (MatchIt)":
            covariates = numeric + categorical
            result = run_matchit_maha_hybrid(df, "exposure_status", "outcome", covariates,
                                             formula="exposure_status ~ " + " + ".join(covariates),
                                             caliper=caliper, ratio=1)
            matched = result.matched_df
        else:
            matcher = RDMatcher(pop_df=df, patient_id_col="patient_id", exposure_status="exposure_status",
                             features_numeric=numeric, features_categorical=categorical,
                             process_features=False, onehot=("Maha" in method),
                             debug=False, log_to_console=False)
            is_maha = "Maha" in method
            if method.startswith("PSM+"):
                matcher.fit_propensity_model(formula=" + ".join(numeric + categorical), random_state=404)
            matched = matcher.rare_matching(
                threshold=threshold, n_neighbors=1, k_candidates=K_CANDIDATES,
                method="multi", distance_metric="mahalanobis" if is_maha else "gower",
                global_optimal=True, competitive_match=True,
                ps_hybrid=method.startswith("PSM+"), ps_caliper=caliper if method.startswith("PSM+") else None,
                ps_caliper_strict=True, diagnostics=False, return_matched_data=True,
                gower_sd_weights=not is_maha, gower_sd_weights_mult=1.96,
                gower_sd_reference="controls",
            )
        matched = matched[matched["match_group"].notna()].copy()
        smds = _absolute_smds(matched, numeric, categorical)
        row = {
            "method": method,
            "threshold": threshold,
            "ps_caliper": caliper if method in {"PSM+RDM", "PSM+Maha (RDM)", "PSM (MatchIt)", "PSM+Maha (MatchIt)"} else np.nan,
            "k_candidates": K_CANDIDATES,
            "mean_abs_smd": float(np.mean(list(smds.values()))),
            "rms_smd": float(np.sqrt(np.mean(np.square(list(smds.values()))))),
            "max_abs_smd": float(max(smds.values())),
            "matched_treated": int((matched["exposure_status"] == 1).sum()),
            "retention": float((matched["exposure_status"] == 1).sum() / n_treated),
            "match_time_sec": perf_counter() - started,
            **{f"smd_{feature}": value for feature, value in smds.items()},
        }
        rows.append(row)
        print(
            f"[{index:>2}/{len(grid)}] {method:<15} threshold={threshold:<5g} "
            f"max={row['max_abs_smd']:.3f} mean={row['mean_abs_smd']:.3f} "
            f"retention={row['retention']:.1%}", flush=True,
        )

    results = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    out_path = Path(OUT_PATH).with_name(f"lalonde_balance_threshold_tuning_{args.start}_{stop}.csv")
    results.to_csv(out_path, index=False)
    eligible = results[results["retention"] >= 0.90].sort_values(
        ["mean_abs_smd", "max_abs_smd", "match_time_sec"]
    )
    print("\nBalance-optimal configurations with at least 90% retention:")
    print(eligible.groupby("method", sort=False).head(3).to_string(index=False))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
