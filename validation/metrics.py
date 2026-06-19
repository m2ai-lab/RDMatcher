"""Shared metrics computation for validation suite.

Ported from EXAMPLE_SIMULATION.ipynb — exact same implementations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def original_sd_calculator(df, features_numeric, exposure_col="exposure_status"):
    raw_controls = df[df[exposure_col] == 0][features_numeric]
    raw_cases = df[df[exposure_col] == 1][features_numeric]
    var_ctrl = raw_controls.var()
    var_case = raw_cases.var()
    baseline_sd = np.sqrt((var_ctrl + var_case) / 2)
    baseline_sd[baseline_sd == 0] = 1.0
    return baseline_sd.to_dict()


def calculate_pairwise_differences(df_match, df_original, features_numeric, features_categorical, original_sds=None):
    matched_df = df_match.dropna(subset=["match_group"])
    treated = matched_df[matched_df["exposure_status"] == 1]
    controls = matched_df[matched_df["exposure_status"] == 0]

    merged = pd.merge(
        treated[["match_group"] + features_numeric + features_categorical],
        controls[["match_group"] + features_numeric + features_categorical],
        on="match_group",
        suffixes=("_t", "_c"),
    )

    diff_df = pd.DataFrame({"match_group": merged["match_group"]})

    if original_sds is None:
        original_sds = original_sd_calculator(df_original, features_numeric)

    for col in features_numeric:
        raw_diff = merged[f"{col}_t"] - merged[f"{col}_c"]
        if col in original_sds:
            diff_df[col] = raw_diff / original_sds[col]
        else:
            diff_df[col] = raw_diff

    for col in features_categorical:
        diff_df[col] = (merged[f"{col}_t"] != merged[f"{col}_c"]).astype(int)

    return diff_df


def summarize_iter_quality(df_match, df_original, features_numeric, features_categorical):
    diff_df = calculate_pairwise_differences(df_match, df_original, features_numeric, features_categorical)

    summary_numeric = {}
    for col in features_numeric:
        abs_diff = diff_df[col].abs()
        summary_numeric[f"{col}_meanpd"] = abs_diff.mean()
        summary_numeric[f"{col}_medianpd"] = abs_diff.median()
        summary_numeric[f"{col}_maxpd"] = abs_diff.max()

    summary_categorical = {}
    for col in features_categorical:
        summary_categorical[f"{col}_pmr"] = diff_df[col].mean()

    return summary_numeric, summary_categorical


def _calculate_smd_numeric(prop_exposed, prop_control, var_list):
    smds = {}
    for var in var_list:
        mean1 = prop_exposed[var].mean()
        mean2 = prop_control[var].mean()
        std1 = prop_exposed[var].std()
        std2 = prop_control[var].std()
        pooled_std = np.sqrt((std1**2 + std2**2) / 2)
        smds[var] = abs(mean1 - mean2) / pooled_std if pooled_std != 0 else 0.0
    return smds


def _calculate_smd_categorical(prop_exposed, prop_control, var_list):
    smds = {}
    for feature in var_list:
        p1 = prop_exposed.get(feature, 0)
        p2 = prop_control.get(feature, 0)
        if p1 == 0 and p2 == 0:
            smds[feature] = 0.0
            continue
        variance = (p1 * (1 - p1) + p2 * (1 - p2)) / 2
        smds[feature] = abs(p1 - p2) / np.sqrt(variance) if variance > 0 else 0.0
    return smds


def summary_stats_table(data, features_numeric, features_categorical, exposure_col="exposure_status", smd_threshold=0.1):
    rows = []
    exposed_data = data[data[exposure_col] == 1]
    control_data = data[data[exposure_col] == 0]
    if exposed_data.empty or control_data.empty:
        return None

    if features_numeric:
        smds_num = _calculate_smd_numeric(exposed_data, control_data, features_numeric)
        for feature in features_numeric:
            rows.append({
                "Feature": feature,
                "Mean_Exposed": exposed_data[feature].mean(),
                "Std_Exposed": exposed_data[feature].std(),
                "Mean_Control": control_data[feature].mean(),
                "Std_Control": control_data[feature].std(),
                "SMD": smds_num.get(feature, np.nan),
            })

    if features_categorical:
        for feature in features_categorical:
            prop_exposed = exposed_data[feature].value_counts(normalize=True)
            prop_control = control_data[feature].value_counts(normalize=True)
            smds_cat = _calculate_smd_categorical(prop_exposed, prop_control, feature)
            for category, smd_val in smds_cat.items():
                rows.append({
                    "Feature": f"{feature}_{str(category)[:15]}",
                    "Mean_Exposed": prop_exposed.get(category, 0) * 100,
                    "Std_Exposed": np.nan,
                    "Mean_Control": prop_control.get(category, 0) * 100,
                    "Std_Control": np.nan,
                    "SMD": smd_val,
                })

    summary_df = pd.DataFrame(rows)
    if not summary_df.empty:
        summary_df["SMD Result"] = summary_df["SMD"].apply(lambda x: "OK" if x < smd_threshold else "BAD")
        return np.round(summary_df, 3)
    return pd.DataFrame()


def set_r_style():
    plt.rcParams.update({
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
    })
    sns.set_style("darkgrid", {
        "axes.facecolor": "#F3F3F3",
        "grid.color": "white",
        "axes.edgecolor": "none",
        "xtick.bottom": True,
        "ytick.left": True,
    })


def calculate_mas(smd_table):
    if smd_table is None or not isinstance(smd_table, pd.DataFrame) or smd_table.empty:
        return np.nan
    return smd_table["SMD"].abs().mean()
