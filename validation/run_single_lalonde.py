"""Single full-dataset run of LaLonde CPS with the nine research methods.

Methods (in display order):
  a. PSM (MatchIt)              — MatchIt PS ratio=1, caliper=0.2
  a2. PSM (RDM)                 — RDMatcher raw-logit propensity, caliper=0.2
  a3. Scaled Euclidean (MatchIt)— MatchIt scaled Euclidean nearest neighbor
  b. Mahalanobis (MatchIt)      — MatchIt distance='mahalanobis', ratio=1
  c. PSM+Mahalanobis (MatchIt)  — MatchIt PSM+Mahalanobis restricted, caliper=0.2
  d. RDM                        — Gower RDMatcher, unbounded distance
  e. PSM+RDM                    — PS caliper 0.20 + unbounded Gower RDMatcher
  f. Mahalanobis (RDM)          — unbounded RDMatcher Mahalanobis
  g. PSM+Maha (RDM)             — PS caliper 0.20 + unbounded RDMatcher Mahalanobis
"""

from __future__ import annotations

import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

sys.path.insert(0, "src")
sys.path.insert(0, ".")

from datasets import load_lalonde
from research_config import RESEARCH_CONFIG
from comparison_methods import (
    run_matchit_ps,
    run_matchit_scaled_euclidean,
    run_matchit_mahalanobis,
    run_matchit_maha_hybrid,
)
from metrics import (
    summary_stats_table,
    summarize_iter_quality,
    original_sd_calculator,
    calculate_pairwise_differences,
    calculate_mas,
    set_r_style,
)
from rdmatcher import RDMatcher

PUBLISHED_ATT = 1794.0
CALIPER = 0.2  # MatchIt PSM and MatchIt hybrid policy caliper
PSM_RDM_CALIPER = 0.20
PSM_RDM_MAHA_CALIPER = 0.20
# Distance thresholds and candidate caps are intentionally disabled for this
# validation pass. RDMatcher supplies k_candidates dynamically by default.
RDM_THRESHOLD = float("inf")
GOWER_THRESHOLD = float("inf")
RDM_MAHA_THRESHOLD = float("inf")
PSM_RDM_MAHA_THRESHOLD = float("inf")
K_CANDIDATES = None
GOWER_SD_WEIGHTS_MULT = 1.96

# Selected without using outcomes: propensity caliper fixed a priori at 0.20;
# among tested RDMatcher thresholds, retain configurations with treated
# retention >= 0.90 and max absolute SMD <= 0.20, then choose minimum RMS SMD.

# Caliper scale: "logit" matches MatchIt std.caliper=TRUE on linear predictor.
# Set to "score" to use propensity probability scale instead.
CALIPER_SCALE = "logit"


# ---------------------------------------------------------------------------
# Method runners (display order)
# ---------------------------------------------------------------------------

def run_psm_matchit(df, numeric, categorical):
    """PSM (MatchIt) — ratio=1, caliper=0.2."""
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
    """Mahalanobis (MatchIt) — distance='mahalanobis', ratio=1."""
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
    """PSM+Mahalanobis (MatchIt) — mahvars restricted, caliper=0.2."""
    covariates = numeric + categorical
    formula_r = "exposure_status ~ " + " + ".join(covariates)
    t0 = time.time()
    result = run_matchit_maha_hybrid(df, "exposure_status", "outcome", covariates,
                                     formula=formula_r, caliper=CALIPER, ratio=1)
    elapsed = time.time() - t0
    if result.matched_df is None or result.matched_df.empty:
        return None, elapsed
    return result.matched_df, elapsed


def run_rdm(df, numeric, categorical):
    """RDM — Gower RDMatcher using the research configuration."""
    t0 = time.time()
    matcher = RDMatcher(
        pop_df=df, patient_id_col="patient_id", exposure_status="exposure_status",
        features_numeric=numeric, features_categorical=categorical,
        process_features=False, onehot=False, debug=False, log_to_console=False,
    )
    matched = matcher.rare_matching(
        threshold=RDM_THRESHOLD, n_neighbors=1, k_candidates=K_CANDIDATES,
        method="multi", distance_metric="gower",
        global_optimal=True, competitive_match=True,
        diagnostics=False, return_matched_data=True,
        gower_sd_weights=True, gower_sd_weights_mult=GOWER_SD_WEIGHTS_MULT,
        gower_sd_reference="controls",
    )
    elapsed = time.time() - t0
    matched = matched[matched["match_group"].notna()].copy()
    return matched, elapsed


def run_psm_rdm_hybrid(df, numeric, categorical):
    """PSM+RDM — PS caliper plus Gower RDMatcher, research configuration."""
    formula = " + ".join(numeric + categorical)
    t0 = time.time()
    matcher = RDMatcher(
        pop_df=df, patient_id_col="patient_id", exposure_status="exposure_status",
        features_numeric=numeric, features_categorical=categorical,
        process_features=False, onehot=False, debug=False, log_to_console=False,
    )
    matcher.fit_propensity_model(formula=formula, random_state=404)
    matched = matcher.rare_matching(
        threshold=GOWER_THRESHOLD, n_neighbors=1, k_candidates=K_CANDIDATES,
        method="multi", distance_metric="gower",
        global_optimal=True, competitive_match=True,
        ps_hybrid=True, ps_caliper=PSM_RDM_CALIPER, ps_caliper_strict=True,
        diagnostics=False, return_matched_data=True,
        gower_sd_weights=True, gower_sd_weights_mult=GOWER_SD_WEIGHTS_MULT,
        gower_sd_reference="controls",
    )
    elapsed = time.time() - t0
    matched = matched[matched["match_group"].notna()].copy()
    return matched, elapsed


def run_psm_rdm_only(df, numeric, categorical):
    """PSM (RDM) — raw-logit propensity-only matching with fixed 0.20 caliper."""
    formula = " + ".join(numeric + categorical)
    t0 = time.time()
    matcher = RDMatcher(
        pop_df=df, patient_id_col="patient_id", exposure_status="exposure_status",
        features_numeric=numeric, features_categorical=categorical,
        process_features=False, onehot=False, debug=False, log_to_console=False,
    )
    matcher.fit_propensity_model(formula=formula, random_state=404)
    matched = matcher.rare_matching(
        threshold=float("inf"), n_neighbors=1, k_candidates=K_CANDIDATES,
        method="propensity", distance_metric="euclidean",
        propensity_caliper=CALIPER, global_optimal=True, competitive_match=True,
        diagnostics=False, return_matched_data=True,
    )
    elapsed = time.time() - t0
    return matched[matched["match_group"].notna()].copy(), elapsed


def run_scaled_euclidean_matchit(df, numeric, categorical):
    """Scaled Euclidean (MatchIt) — nearest-neighbor ratio 1:1."""
    covariates = numeric + categorical
    formula_r = "exposure_status ~ " + " + ".join(covariates)
    t0 = time.time()
    result = run_matchit_scaled_euclidean(
        df, "exposure_status", "outcome", covariates, formula=formula_r, ratio=1,
    )
    matched = result.matched_df
    if matched is None or matched.empty:
        return None, time.time() - t0
    return matched[matched["match_group"].notna()].copy(), time.time() - t0


def run_rdm_maha(df, numeric, categorical):
    """Mahalanobis (RDM) — RDMatcher Mahalanobis, research configuration."""
    t0 = time.time()
    matcher = RDMatcher(
        pop_df=df, patient_id_col="patient_id", exposure_status="exposure_status",
        features_numeric=numeric, features_categorical=categorical,
        process_features=False, onehot=True, debug=False, log_to_console=False,
    )
    matched = matcher.rare_matching(
        threshold=RDM_MAHA_THRESHOLD, n_neighbors=1, k_candidates=K_CANDIDATES,
        method="multi", distance_metric="mahalanobis",
        global_optimal=True, competitive_match=True,
        diagnostics=False, return_matched_data=True,
    )
    elapsed = time.time() - t0
    matched = matched[matched["match_group"].notna()].copy()
    return matched, elapsed


def run_psm_rdm_maha_hybrid(df, numeric, categorical):
    """PSM+Maha (RDM) — PS caliper plus Mahalanobis RDMatcher."""
    formula = " + ".join(numeric + categorical)
    t0 = time.time()
    matcher = RDMatcher(
        pop_df=df, patient_id_col="patient_id", exposure_status="exposure_status",
        features_numeric=numeric, features_categorical=categorical,
        process_features=False, onehot=True, debug=False, log_to_console=False,
    )
    matcher.fit_propensity_model(formula=formula, random_state=404)
    matched = matcher.rare_matching(
        threshold=PSM_RDM_MAHA_THRESHOLD, n_neighbors=1, k_candidates=K_CANDIDATES,
        method="multi", distance_metric="mahalanobis",
        global_optimal=True, competitive_match=True,
        ps_hybrid=True, ps_caliper=PSM_RDM_MAHA_CALIPER, ps_caliper_strict=True,
        diagnostics=False, return_matched_data=True,
    )
    elapsed = time.time() - t0
    matched = matched[matched["match_group"].notna()].copy()
    return matched, elapsed


METHOD_RUNNERS = {
    "RDM": run_rdm,
    "PSM_RDM": run_psm_rdm_hybrid,
    "PSM": run_psm_rdm_only,
    "RDM_Mahalanobis": run_rdm_maha,
    "PSM_RDM_Mahalanobis": run_psm_rdm_maha_hybrid,
    "MatchIt": run_psm_matchit,
    "MatchIt_ScaledEuclidean": run_scaled_euclidean_matchit,
    "Mahalanobis": run_maha_matchit,
    "Hybrid_Maha_MatchIt": run_psm_maha_matchit,
}
METHODS = {spec["name"]: METHOD_RUNNERS[key] for key, spec in RESEARCH_CONFIG.items()}
COLORS = {spec["name"]: spec["color"] for spec in RESEARCH_CONFIG.values()}


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_row(name, matched_df, df_orig, numeric, categorical, n_treated_orig, elapsed):
    """Compute all metrics for a single method run."""
    if matched_df is None or matched_df.empty:
        return {
            "Method": name, "ATT": np.nan, "Bias": np.nan,
            "MAS": np.nan, "MaxPD_values": [],
            "MaxSMD": np.nan,
            "Time": np.nan, "Retention": np.nan, "N": 0,
        }

    n_matched = int((matched_df["exposure_status"] == 1).sum())
    att = matched_df[matched_df["exposure_status"] == 1]["outcome"].mean() - \
          matched_df[matched_df["exposure_status"] == 0]["outcome"].mean()
    retention = n_matched / n_treated_orig if n_treated_orig > 0 else 0
    smd = summary_stats_table(matched_df, numeric, categorical)
    mas = calculate_mas(smd) if isinstance(smd, pd.DataFrame) else np.nan
    max_smd = float(smd["SMD"].abs().max()) if isinstance(smd, pd.DataFrame) and not smd.empty else np.nan

    # Per-feature SMD values for boxplot
    smd_values = []
    if isinstance(smd, pd.DataFrame) and "SMD" in smd.columns:
        smd_values = smd["SMD"].abs().tolist()

    # Per-pair max absolute standardized difference across numeric features
    maxpd_values = []
    try:
        diff_df = calculate_pairwise_differences(matched_df, df_orig, numeric, categorical)
        if not diff_df.empty:
            num_cols = [c for c in diff_df.columns if c in numeric]
            if num_cols:
                maxpd_values = diff_df[num_cols].abs().max(axis=1).tolist()
    except Exception:
        pass

    return {
        "Method": name,
        "ATT": att,
        "Bias": att - PUBLISHED_ATT,
        "MAS": mas,
        "MaxSMD": max_smd,
        "SMD_values": smd_values,
        "MaxPD_values": maxpd_values,
        "Time": elapsed,
        "Retention": retention,
        "N": n_matched,
    }


# ---------------------------------------------------------------------------
# Plot: 4 panels — Bias, SMD, MaxPD, Time
# ---------------------------------------------------------------------------

def make_plot(results_df, save_path=None):
    """4-panel figure matching EXAMPLE_SIMULATION.ipynb style."""
    set_r_style()

    plot_df = results_df[results_df["Method"].isin(METHODS.keys())].copy()
    # Reversed so PSM+RDM is at the bottom (matplotlib barh puts first item at y=0)
    method_order = list(reversed([m for m in METHODS.keys() if m in plot_df["Method"].values]))
    color_palette = {m: COLORS[m] for m in METHODS.keys()}

    # Build long-format DataFrame for MaxPD boxplot
    maxpd_rows = []
    for _, row in plot_df.iterrows():
        m = row["Method"]
        for val in row.get("MaxPD_values", []):
            maxpd_rows.append({"Method": m, "Value": val})
    maxpd_df = pd.DataFrame(maxpd_rows)

    # Figure layout
    fig = plt.figure(figsize=(16, 26))
    gs = fig.add_gridspec(4, 2, hspace=0.5, wspace=0.2)
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, :])
    ax3 = fig.add_subplot(gs[2, :])
    ax4 = fig.add_subplot(gs[3, :])
    all_axes = [ax1, ax2, ax3, ax4]

    # --- ROW 1: ATT BIAS (horizontal bar) ---
    bias_vals = plot_df.set_index("Method")["Bias"].reindex(method_order)
    bars = ax1.barh(method_order, bias_vals.values,
                    color=[color_palette[m] for m in method_order],
                    edgecolor="white", height=0.6, alpha=0.8)
    ax1.axvline(0, color="black", linestyle="-", linewidth=1.2)
    ax1.set_xlabel("Bias ($)")
    ax1.set_title("ATT Bias (Estimated − Published)", fontsize=15, fontweight="bold")
    for bar, val in zip(bars, bias_vals.values):
        if not np.isnan(val):
            ax1.text(bar.get_width() + (5 if val >= 0 else -5),
                     bar.get_y() + bar.get_height() / 2,
                     f"${val:+,.0f}", va="center", fontsize=10,
                     ha="left" if val >= 0 else "right")

    # --- ROW 2: maximum absolute SMD (horizontal bar) ---
    max_smd_vals = plot_df.set_index("Method")["MaxSMD"].reindex(method_order)
    ax2.barh(method_order, max_smd_vals.values,
             color=[color_palette[m] for m in method_order],
             edgecolor="white", height=0.6, alpha=0.8)
    ax2.axvline(0.1, color="#b91c1c", linestyle=":", linewidth=2, label="Threshold (0.1)")
    ax2.set_xlabel("Maximum Absolute SMD")
    ax2.set_title("Covariate Balance (Maximum Absolute SMD)",
                  fontsize=15, fontweight="bold")
    ax2.legend(fontsize=10)

    # --- ROW 3: MaxPD per-pair boxplot (horizontal) ---
    # sns.boxplot places the FIRST item in order at the TOP of the y-axis.
    # Since method_order is already reversed (PSM+RDM first), it goes at top.
    # We need to pass the ORIGINAL order so PSM+RDM ends up at the bottom.
    boxplot_order = list(reversed(method_order))
    if not maxpd_df.empty:
        sns.boxplot(data=maxpd_df, x='Value', y='Method',
                    palette=color_palette, ax=ax3,
                    order=boxplot_order,
                    boxprops=dict(alpha=0.8), fliersize=2)
    ax3.set_title("Pairwise Match Quality (Max Abs. Stand. Difference per Pair)",
                  fontsize=15, fontweight="bold")
    ax3.set_xlabel("Max Abs. Stand. Difference")
    ax3.set_ylabel("")

    # --- ROW 4: EXECUTION TIME (horizontal bar) ---
    time_vals = plot_df.set_index("Method")["Time"].reindex(method_order)
    ax4.barh(method_order, time_vals.values,
             color=[color_palette[m] for m in method_order],
             edgecolor="white", height=0.6, alpha=0.8)
    ax4.set_xlabel("Seconds")
    ax4.set_title("Execution Time (Seconds)", fontsize=15, fontweight="bold")

    # Style: despine + grid
    for ax in all_axes:
        sns.despine(ax=ax, left=True, bottom=True)
        ax.grid(True, which='major', linestyle='-', linewidth=1.2)
    for ax in [ax2, ax3]:
        ax.grid(True, axis='x', linestyle='--', linewidth=0.5, alpha=0.5)

    plt.suptitle("LaLonde CPS — Matching Method Comparison", fontsize=16,
                 fontweight='bold', y=0.91)

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=600, facecolor="white")
        print(f"  Saved: {save_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_table(results_list):
    header = f"{'Method':<30} {'ATT':>10} {'Bias':>10} {'MAS':>8} {'MaxPD':>8} {'N':>5} {'Ret':>8} {'Time':>8}"
    print("\n" + header)
    print("-" * len(header))
    for r in results_list:
        att_s = f"${r['ATT']:>8,.0f}" if not np.isnan(r.get("ATT", np.nan)) else "     NaN"
        bias_s = f"${r['Bias']:>8,.0f}" if not np.isnan(r.get("Bias", np.nan)) else "     NaN"
        mas_s = f"{r['MAS']:.3f}" if not np.isnan(r.get("MAS", np.nan)) else "  NaN"
        maxpd_vals = r.get("MaxPD_values", [])
        maxpd_s = f"{np.mean(maxpd_vals):.3f}" if maxpd_vals else "  NaN"
        ret_s = f"{r['Retention']:.1%}" if not np.isnan(r.get("Retention", np.nan)) else "  NaN"
        t_s = f"{r['Time']:.2f}s" if not np.isnan(r.get("Time", np.nan)) else "  NaN"
        print(f"{r['Method']:<30} {att_s:>10} {bias_s:>10} {mas_s:>8} {maxpd_s:>8} {r['N']:>5} {ret_s:>8} {t_s:>8}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 80)
    print("LaLonde CPS — Single Run, Nine Research Methods")
    print(f"  Published ATT ≈ ${PUBLISHED_ATT:,.0f}")
    print(
        f"  Calipers = PSM+RDM {PSM_RDM_CALIPER}/MatchIt {CALIPER}, PSM+Maha (RDM) {PSM_RDM_MAHA_CALIPER} | RDM threshold = {RDM_THRESHOLD} | "
        f"PSM+RDM threshold = {GOWER_THRESHOLD} | Maha thresholds = "
        f"{RDM_MAHA_THRESHOLD}/{PSM_RDM_MAHA_THRESHOLD} | "
        f"candidates = RDMatcher default ({K_CANDIDATES})"
    )
    print("=" * 80)

    df = load_lalonde(use_re74=True, external_control="cps")
    meta = df.attrs["meta"]
    numeric = meta["features_numeric"]
    categorical = meta["features_categorical"]
    n_treated = int((df["exposure_status"] == 1).sum())

    print(f"  Loaded: {len(df)} subjects ({n_treated} treated, "
          f"{len(df) - n_treated} controls)")

    crude_att = df[df["exposure_status"] == 1]["outcome"].mean() - \
                df[df["exposure_status"] == 0]["outcome"].mean()
    print(f"  Crude ATT: ${crude_att:,.0f}")
    print()

    # Add crude row
    all_results = [{
        "Method": "Crude Unmatched", "ATT": crude_att,
        "Bias": crude_att - PUBLISHED_ATT,
        "MAS": np.nan, "MaxPD": np.nan,
        "Time": np.nan, "Retention": 1.0, "N": n_treated,
    }]

    for name, func in METHODS.items():
        print(f"  Running: {name} ...", end=" ", flush=True)
        try:
            matched, elapsed = func(df, numeric, categorical)
            row = compute_row(name, matched, df, numeric, categorical, n_treated, elapsed)
            all_results.append(row)
            att_s = f"${row['ATT']:>8,.0f}" if not np.isnan(row['ATT']) else "NaN"
            maxpd_vals = row.get("MaxPD_values", [])
            maxpd_s = f"{np.mean(maxpd_vals):.3f}" if maxpd_vals else "NaN"
            print(f"ATT={att_s}  Bias=${row['Bias']:+,.0f}  MAS={row['MAS']:.3f}  "
                  f"MaxPD={maxpd_s}  "
                  f"N={row['N']:>4}  Ret={row['Retention']:.1%}  "
                  f"Time={elapsed:.2f}s")
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback; traceback.print_exc()
            all_results.append({
                "Method": name, "ATT": np.nan, "Bias": np.nan,
                "MAS": np.nan, "MaxPD_values": [],
                "Time": np.nan, "Retention": np.nan, "N": 0,
            })

    results_df = pd.DataFrame(all_results)

    print_table(all_results)
    print(f"\n  Published ATT: ${PUBLISHED_ATT:,.0f}")

    # Save plots
    out_dir = "validation/plots"
    make_plot(results_df, f"{out_dir}/lalonde_cps_research_methods.png")

    # Save CSV
    results_df.to_csv(f"{out_dir}/lalonde_cps_research_methods.csv", index=False)
    print(f"  Saved CSV: {out_dir}/lalonde_cps_research_methods.csv")


if __name__ == "__main__":
    main()
