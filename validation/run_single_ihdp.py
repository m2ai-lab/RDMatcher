"""IHDP — 100 replications, 9 standardized methods.

Methods:
  a. RDM                        — standalone Gower, threshold=0.1, equal weights
  b. PSM (RDM)                  — RDM propensity only, euclidean, caliper=0.2, 1:1
  c. PSM+RDM                    — PS caliper + Gower matching (single pass)
  d. MatchIt+RDM                — MatchIt PS (K=3, caliper=0.2) → RDM Gower (0.3, equal weights)
  e. PSM (MatchIt)              — MatchIt PS ratio=1, caliper=0.2
  f. Mahalanobis (MatchIt)      — MatchIt distance='mahalanobis', ratio=1
  g. PSM+Mahalanobis (MatchIt)  — MatchIt PSM+Mahalanobis restricted, caliper=0.2
  h. Mahalanobis (RDM)          — RDMatcher Mahalanobis, threshold=2.0
  i. PSM+Maha (RDM)             — RDMatcher PS caliper + Mahalanobis, threshold=2.0
"""

from __future__ import annotations

import sys
import time
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

sys.path.insert(0, "src")
sys.path.insert(0, ".")

from datasets import load_ihdp_single
from comparison_methods import (
    run_matchit_ps,
    run_matchit_mahalanobis,
    run_matchit_maha_hybrid,
    build_gower_weights_dict,
)
from metrics import (
    summary_stats_table,
    summarize_iter_quality,
    original_sd_calculator,
    calculate_mas,
    set_r_style,
)
from rdmatcher import RDMatcher

N_REPLICATIONS = 100
PUBLISHED_ATT = 4.0
CALIPER = 0.2
RDM_THRESHOLD = 0.1
GOWER_THRESHOLD = 0.3
MAHA_THRESHOLD = 2.0
OUT_DIR = "validation/plots"

# Caliper scale: "logit" matches MatchIt std.caliper=TRUE on linear predictor.
# Set to "score" to use propensity probability scale instead.
CALIPER_SCALE = "logit"


# ---------------------------------------------------------------------------
# Method runners
# ---------------------------------------------------------------------------

def run_rdm(df, numeric, categorical):
    """a. Standalone RDM Gower."""
    gw = build_gower_weights_dict(numeric, categorical,
                                  numeric_weight=1, categorical_weight=1)
    t0 = time.time()
    matcher = RDMatcher(
        pop_df=df, patient_id_col="patient_id", exposure_status="exposure_status",
        features_numeric=numeric, features_categorical=categorical,
        process_features=False, onehot=False, debug=False, log_to_console=False,
    )
    matched = matcher.rare_matching(
        threshold=RDM_THRESHOLD, n_neighbors=1, k_candidates=500,
        method="multi", distance_metric="gower",
        global_optimal=True, competitive_match=True,
        diagnostics=False, return_matched_data=True, gower_weights=gw,
    )
    elapsed = time.time() - t0
    matched = matched[matched["match_group"].notna()].copy()
    return matched, elapsed


def _get_rdm_caliper(matcher):
    """Compute caliper threshold on the appropriate propensity scale."""
    treated = matcher.pop_processed.loc[matcher.pop_processed["exposure_status"] == 1]
    if CALIPER_SCALE == "score":
        sd = treated["propensity_score"].std()
    else:
        sd = treated["propensity_logit"].std()
    return CALIPER * sd


def run_psm_rdm(df, numeric, categorical):
    """b. PSM (RDM) — propensity only, euclidean, caliper=0.2, 1:1."""
    formula = " + ".join(numeric + categorical)
    t0 = time.time()
    matcher = RDMatcher(
        pop_df=df, patient_id_col="patient_id", exposure_status="exposure_status",
        features_numeric=numeric, features_categorical=categorical,
        process_features=False, onehot=False, debug=False, log_to_console=False,
    )
    matcher.fit_propensity_model(formula=formula, random_state=404)
    caliper_threshold = _get_rdm_caliper(matcher)
    matched = matcher.rare_matching(
        threshold=caliper_threshold, n_neighbors=1, k_candidates=1000,
        method="propensity", distance_metric="euclidean",
        propensity_col="propensity_logit",
        global_optimal=True, competitive_match=True,
        diagnostics=False, return_matched_data=True,
    )
    elapsed = time.time() - t0
    matched = matched[matched["match_group"].notna()].copy()
    return matched, elapsed


def run_psm_rdm_hybrid(df, numeric, categorical):
    """c. PSM+RDM — PS caliper + Gower matching (single pass, our method)."""
    formula = " + ".join(numeric + categorical)
    gw = build_gower_weights_dict(numeric, categorical,
                                  numeric_weight=1, categorical_weight=1)
    t0 = time.time()
    matcher = RDMatcher(
        pop_df=df, patient_id_col="patient_id", exposure_status="exposure_status",
        features_numeric=numeric, features_categorical=categorical,
        process_features=False, onehot=False, debug=False, log_to_console=False,
    )
    matcher.fit_propensity_model(formula=formula, random_state=404)
    matched = matcher.rare_matching(
        threshold=GOWER_THRESHOLD, n_neighbors=1, k_candidates=500,
        method="multi", distance_metric="gower",
        global_optimal=True, competitive_match=True,
        ps_hybrid=True, ps_caliper=CALIPER, ps_caliper_strict=True,
        diagnostics=False, return_matched_data=True, gower_weights=gw,
    )
    elapsed = time.time() - t0
    matched = matched[matched["match_group"].notna()].copy()
    return matched, elapsed


def run_matchit_rdm(df, numeric, categorical):
    """d. MatchIt+RDM — MatchIt PS (K=3, caliper=0.2) → RDM Gower (0.3)."""
    covariates = numeric + categorical
    formula_r = "exposure_status ~ " + " + ".join(covariates)
    gw = build_gower_weights_dict(numeric, categorical,
                                  numeric_weight=1, categorical_weight=1)

    t0 = time.time()
    # Stage 1: MatchIt PS K=3
    result = run_matchit_ps(df, "exposure_status", "outcome", covariates,
                            formula=formula_r, caliper=CALIPER, ratio=3)
    s1 = result.matched_df
    if s1 is None or s1.empty:
        return None, time.time() - t0

    # Stage 2: RDM Gower 1:1
    matcher2 = RDMatcher(
        pop_df=s1, patient_id_col="patient_id", exposure_status="exposure_status",
        features_numeric=numeric, features_categorical=categorical,
        process_features=False, onehot=False, debug=False, log_to_console=False,
    )
    matched = matcher2.rare_matching(
        threshold=GOWER_THRESHOLD, n_neighbors=1, k_candidates=500,
        method="multi", distance_metric="gower",
        global_optimal=True, competitive_match=True,
        diagnostics=False, return_matched_data=True, gower_weights=gw,
    )
    elapsed = time.time() - t0
    matched = matched[matched["match_group"].notna()].copy()
    return matched, elapsed


def run_psm_matchit(df, numeric, categorical):
    """e. PSM (MatchIt) — ratio=1, caliper=0.2."""
    covariates = numeric + categorical
    formula_r = "exposure_status ~ " + " + ".join(covariates)
    t0 = time.time()
    result = run_matchit_ps(df, "exposure_status", "outcome", covariates,
                            formula=formula_r, caliper=CALIPER, ratio=1)
    elapsed = time.time() - t0
    if result.matched_df is None or result.matched_df.empty:
        return None, elapsed
    return result.matched_df, elapsed


def run_maha_matchit(df, numeric, categorical):
    """f. Mahalanobis (MatchIt) — distance='mahalanobis', ratio=1."""
    covariates = numeric + categorical
    formula_r = "exposure_status ~ " + " + ".join(covariates)
    t0 = time.time()
    result = run_matchit_mahalanobis(df, "exposure_status", "outcome", covariates,
                                     formula=formula_r, ratio=1)
    elapsed = time.time() - t0
    if result.matched_df is None or result.matched_df.empty:
        return None, elapsed
    return result.matched_df, elapsed


def run_psm_maha_matchit(df, numeric, categorical):
    """g. PSM+Mahalanobis (MatchIt) — mahvars restricted, caliper=0.2."""
    covariates = numeric + categorical
    formula_r = "exposure_status ~ " + " + ".join(covariates)
    t0 = time.time()
    result = run_matchit_maha_hybrid(df, "exposure_status", "outcome", covariates,
                                     formula=formula_r, caliper=CALIPER, ratio=1)
    elapsed = time.time() - t0
    if result.matched_df is None or result.matched_df.empty:
        return None, elapsed
    return result.matched_df, elapsed


def run_rdm_maha(df, numeric, categorical):
    """h. Mahalanobis (RDM) — RDMatcher Mahalanobis."""
    t0 = time.time()
    matcher = RDMatcher(
        pop_df=df, patient_id_col="patient_id", exposure_status="exposure_status",
        features_numeric=numeric, features_categorical=categorical,
        process_features=False, onehot=True, debug=False, log_to_console=False,
    )
    matched = matcher.rare_matching(
        threshold=MAHA_THRESHOLD, n_neighbors=1, k_candidates=500,
        method="multi", distance_metric="mahalanobis",
        global_optimal=True, competitive_match=True,
        diagnostics=False, return_matched_data=True,
    )
    elapsed = time.time() - t0
    matched = matched[matched["match_group"].notna()].copy()
    return matched, elapsed


def run_psm_rdm_maha_hybrid(df, numeric, categorical):
    """i. PSM+Maha (RDM) — PS caliper + Mahalanobis matching."""
    formula = " + ".join(numeric + categorical)
    t0 = time.time()
    matcher = RDMatcher(
        pop_df=df, patient_id_col="patient_id", exposure_status="exposure_status",
        features_numeric=numeric, features_categorical=categorical,
        process_features=False, onehot=True, debug=False, log_to_console=False,
    )
    matcher.fit_propensity_model(formula=formula, random_state=404)
    matched = matcher.rare_matching(
        threshold=MAHA_THRESHOLD, n_neighbors=1, k_candidates=500,
        method="multi", distance_metric="mahalanobis",
        global_optimal=True, competitive_match=True,
        ps_hybrid=True, ps_caliper=CALIPER, ps_caliper_strict=True,
        diagnostics=False, return_matched_data=True,
    )
    elapsed = time.time() - t0
    matched = matched[matched["match_group"].notna()].copy()
    return matched, elapsed


# ---------------------------------------------------------------------------
# Method config
# ---------------------------------------------------------------------------

METHODS = {
    "RDM": run_rdm,
    "PSM (RDM)": run_psm_rdm,
    "PSM+RDM": run_psm_rdm_hybrid,
    "MatchIt+RDM": run_matchit_rdm,
    "PSM (MatchIt)": run_psm_matchit,
    "Mahalanobis (MatchIt)": run_maha_matchit,
    "PSM+Mahalanobis (MatchIt)": run_psm_maha_matchit,
    "Mahalanobis (RDM)": run_rdm_maha,
    "PSM+Maha (RDM)": run_psm_rdm_maha_hybrid,
}

COLORS = {
    "RDM": "#052049",
    "PSM (RDM)": "#4B0082",
    "PSM+RDM": "#16A0AC",
    "MatchIt+RDM": "#E76F51",
    "PSM (MatchIt)": "#32A03E",
    "Mahalanobis (MatchIt)": "#A238BA",
    "PSM+Mahalanobis (MatchIt)": "#C42882",
    "Mahalanobis (RDM)": "#6C4AB6",
    "PSM+Maha (RDM)": "#C06C84",
}
CRUDE_COLOR = "#4b5563"


# ---------------------------------------------------------------------------
# Run single replication
# ---------------------------------------------------------------------------

def run_single_rep(rep_idx):
    """Run all configured methods on a single IHDP replication."""
    df = load_ihdp_single(rep_idx)
    meta = df.attrs["meta"]
    numeric = meta["features_numeric"]
    categorical = meta["features_categorical"]

    # True ATT from potential outcomes
    treated = df[df["exposure_status"] == 1]
    true_att = float((treated["mu1"] - treated["mu0"]).mean())

    n_treated_orig = int((df["exposure_status"] == 1).sum())
    crude_att = df[df["exposure_status"] == 1]["outcome"].mean() - \
                df[df["exposure_status"] == 0]["outcome"].mean()

    rep_results = []

    for name, func in METHODS.items():
        try:
            matched, elapsed = func(df, numeric, categorical)
            if matched is None or matched.empty:
                rep_results.append({
                    "seed": rep_idx, "method": name, "true_att": true_att,
                    "crude_att": crude_att, "att": np.nan, "retention": np.nan,
                    "mas": np.nan, "time": elapsed,
                })
                continue

            n_matched = int((matched["exposure_status"] == 1).sum())
            att = matched[matched["exposure_status"] == 1]["outcome"].mean() - \
                  matched[matched["exposure_status"] == 0]["outcome"].mean()
            retention = n_matched / n_treated_orig if n_treated_orig > 0 else 0
            smd = summary_stats_table(matched, numeric, categorical)
            mas = calculate_mas(smd) if isinstance(smd, pd.DataFrame) else np.nan

            rep_results.append({
                "seed": rep_idx, "method": name, "true_att": true_att,
                "crude_att": crude_att, "att": att, "retention": retention,
                "mas": mas, "time": elapsed,
            })
        except Exception as e:
            rep_results.append({
                "seed": rep_idx, "method": name, "true_att": true_att,
                "crude_att": crude_att, "att": np.nan, "retention": np.nan,
                "mas": np.nan, "time": np.nan, "error": str(e),
            })

    return rep_results


# ---------------------------------------------------------------------------
# Plot (boxplots over 100 reps)
# ---------------------------------------------------------------------------

def make_plot(results_df, true_effect, save_path=None):
    """4-row figure matching EXAMPLE_SIMULATION.ipynb format."""
    set_r_style()

    method_order = list(METHODS.keys())
    palette = {m: COLORS[m] for m in method_order}

    fig = plt.figure(figsize=(16, 26))
    gs = fig.add_gridspec(4, 2, hspace=0.5, wspace=0.2)

    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, :])
    ax3a = fig.add_subplot(gs[2, 0])
    ax3b = fig.add_subplot(gs[2, 1])
    ax4 = fig.add_subplot(gs[3, :])
    all_axes = [ax1, ax2, ax3a, ax3b, ax4]

    # --- ROW 1: ATT Distribution (KDE) ---
    att_data = results_df[results_df["method"].isin(method_order)][["att", "method"]].dropna()
    att_data["Algorithm"] = att_data["method"]
    crude_data = pd.DataFrame({
        "att": results_df.drop_duplicates("seed")["crude_att"].dropna(),
        "Algorithm": "Crude Unmatched",
    })
    combined = pd.concat([att_data[["att", "Algorithm"]], crude_data], ignore_index=True)

    kde_palette = palette.copy()
    kde_palette["Crude Unmatched"] = CRUDE_COLOR
    kde_order = ["Crude Unmatched"] + method_order

    sns.kdeplot(data=combined, x="att", hue="Algorithm", fill=True, ax=ax1,
                palette=kde_palette, alpha=0.1, linewidth=2, hue_order=kde_order)
    ax1.axvline(true_effect, color="#ef4444", linestyle="--", linewidth=2.5,
                label=f"True ATT ({true_effect:.2f})", zorder=10)
    ax1.set_title("Treatment Effect Recovery (ATT)", fontsize=15, fontweight="bold")
    ax1.set_xlabel("Estimate")
    if ax1.get_legend():
        ax1.legend()

    # --- ROW 2: Covariate Balance (MAS boxplot) ---
    plot_df = results_df[results_df["method"].isin(method_order)].copy()
    sns.boxplot(data=plot_df.dropna(subset=["mas"]), x="mas", y="method",
                hue="method", palette=palette, ax=ax2,
                boxprops=dict(alpha=1.0), dodge=False,
                order=method_order, hue_order=method_order)
    ax2.axvline(0.1, color="#b91c1c", linestyle=":", linewidth=2, label="Threshold (0.1)")
    ax2.set_title("Covariate Balance (Maximum Absolute Standardized Difference)",
                  fontsize=15, fontweight="bold")
    ax2.set_xlabel("Max Absolute SMD")

    # --- ROW 3: Bias (histogram-style) ---
    bias_data = plot_df[["method", "att"]].dropna().copy()
    bias_data["bias"] = bias_data["att"] - true_effect
    sns.boxplot(data=bias_data, x="bias", y="method", hue="method",
                palette=palette, ax=ax3a, boxprops=dict(alpha=0.8),
                dodge=False, order=method_order, hue_order=method_order)
    ax3a.axvline(0, color="black", linestyle="-", linewidth=1)
    ax3a.set_title("ATT Bias (Estimated - True)", fontsize=14, fontweight="bold")
    ax3a.set_xlabel("Bias")
    ax3a.tick_params(axis="x", rotation=45)
    if ax3a.get_legend():
        ax3a.get_legend().remove()

    # --- ROW 3 right: Retention ---
    sns.boxplot(data=plot_df.dropna(subset=["retention"]), x="retention", y="method",
                hue="method", palette=palette, ax=ax3b,
                boxprops=dict(alpha=0.8), dodge=False,
                order=method_order, hue_order=method_order)
    ax3b.set_title("Match Retention Rate", fontsize=14, fontweight="bold")
    ax3b.set_xlabel("Retention")
    ax3b.tick_params(axis="x", rotation=45)
    if ax3b.get_legend():
        ax3b.get_legend().remove()

    # --- ROW 4: Execution Time ---
    sns.boxplot(data=plot_df.dropna(subset=["time"]), x="time", y="method",
                hue="method", palette=palette, ax=ax4,
                boxprops=dict(alpha=1.0), dodge=False,
                order=method_order, hue_order=method_order)
    ax4.set_title("Execution Time (Seconds)", fontsize=15, fontweight="bold")
    ax4.set_xlabel("Seconds")

    for ax in all_axes:
        sns.despine(ax=ax, left=True, bottom=True)
        ax.grid(True, which="major", linestyle="-", linewidth=1.2)
        if ax in [ax3a, ax3b]:
            ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.5)

    plt.suptitle("IHDP Matching Diagnostics (N=100 reps, true ATT=4.0)",
                 fontsize=16, fontweight="bold", y=0.91)

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=300, facecolor="white")
        print(f"  Saved: {save_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary(results_df, true_effect):
    method_order = list(METHODS.keys())

    # Crude
    crude = results_df.drop_duplicates("seed")["crude_att"].dropna()
    crude_bias = crude.mean() - true_effect
    crude_rmse = np.sqrt(((crude - true_effect) ** 2).mean())

    header = f"{'Method':<30} {'ATT':>10} {'Bias':>10} {'RMSE':>10} {'MAS':>8} {'Time':>10} {'Ret':>8}"
    print("\n" + header)
    print("-" * len(header))
    print(f"{'Crude Unmatched':<30} {crude.mean():>10.3f} {crude_bias:>10.3f} "
          f"{crude_rmse:>10.3f} {'N/A':>8} {'N/A':>10} {'100.0%':>8}")

    for method in method_order:
        group = results_df[results_df["method"] == method]
        est = group["att"].dropna()
        if est.empty:
            print(f"{method:<30} {'NaN':>10} {'NaN':>10} {'NaN':>10} {'NaN':>8} {'NaN':>10} {'NaN':>8}")
            continue
        bias = est.mean() - true_effect
        rmse = np.sqrt(((est - true_effect) ** 2).mean())
        mas = group["mas"].dropna().mean()
        t_mean = group["time"].dropna().mean()
        ret = group["retention"].dropna().mean()
        print(f"{method:<30} {est.mean():>10.3f} {bias:>10.3f} {rmse:>10.3f} "
              f"{mas:>8.3f} {t_mean:>9.2f}s {ret:>7.1%}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 80)
    print("IHDP — 100 Replications, 9 Standardized Methods")
    print(f"  True ATT = {PUBLISHED_ATT}")
    print(f"  Caliper = {CALIPER} | RDM threshold = {RDM_THRESHOLD} | Gower threshold = {GOWER_THRESHOLD}")
    print("=" * 80)

    all_results = []
    t_total = time.time()
    for rep_idx in range(N_REPLICATIONS):
        rep_results = run_single_rep(rep_idx)
        all_results.extend(rep_results)
        if rep_idx % 10 == 0:
            n_ok = sum(1 for r in rep_results if not np.isnan(r.get("att", np.nan)))
            print(f"  Rep {rep_idx:>3}: {n_ok}/{len(METHODS)} methods OK")
    total_time = time.time() - t_total
    print(f"\nTotal runtime: {total_time:.1f}s ({total_time / N_REPLICATIONS:.2f}s/rep)")

    results_df = pd.DataFrame(all_results)
    os.makedirs(OUT_DIR, exist_ok=True)
    results_df.to_csv(f"{OUT_DIR}/ihdp_7methods_raw.csv", index=False)

    print_summary(results_df, PUBLISHED_ATT)

    make_plot(results_df, PUBLISHED_ATT, f"{OUT_DIR}/ihdp_7methods.png")

    print(f"\n  Saved: {OUT_DIR}/ihdp_7methods.csv")
    results_df.to_csv(f"{OUT_DIR}/ihdp_7methods.csv", index=False)


if __name__ == "__main__":
    main()
