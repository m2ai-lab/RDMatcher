"""IHDP validation with the seven research methods used in the simulations."""

from __future__ import annotations

import argparse
import sys
import time
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

sys.path.insert(0, "src")
sys.path.insert(0, ".")

from datasets import load_ihdp_single
from research_config import RESEARCH_CONFIG
from comparison_methods import (
    run_matchit_ps,
    run_matchit_mahalanobis,
    run_matchit_maha_hybrid,
)
from metrics import (
    summary_stats_table,
    summarize_iter_quality,
    original_sd_calculator,
    calculate_mas,
    calculate_pairwise_differences,
    set_r_style,
)
from rdmatcher import RDMatcher

N_REPLICATIONS = 100
PUBLISHED_ATT = 4.0
CALIPER = 0.2
RDM_THRESHOLD = 0.30
GOWER_THRESHOLD = 0.15
MAHA_THRESHOLD = 4.5
PSM_RDM_MAHA_THRESHOLD = 5.0
K_CANDIDATES = 250
GOWER_SD_WEIGHTS_MULT = 1.96
OUT_DIR = "validation/plots"

# Caliper scale: "logit" matches MatchIt std.caliper=TRUE on linear predictor.
# Set to "score" to use propensity probability scale instead.
CALIPER_SCALE = "logit"


# ---------------------------------------------------------------------------
# Method runners
# ---------------------------------------------------------------------------

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
    """PSM+RDM — propensity-caliper eligibility plus Gower allocation."""
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
        ps_hybrid=True, ps_caliper=CALIPER, ps_caliper_strict=True,
        diagnostics=False, return_matched_data=True,
        gower_sd_weights=True, gower_sd_weights_mult=GOWER_SD_WEIGHTS_MULT,
        gower_sd_reference="controls",
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
        threshold=MAHA_THRESHOLD, n_neighbors=1, k_candidates=K_CANDIDATES,
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
        threshold=PSM_RDM_MAHA_THRESHOLD, n_neighbors=1, k_candidates=K_CANDIDATES,
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

METHOD_RUNNERS = {
    "RDM": run_rdm,
    "PSM_RDM": run_psm_rdm_hybrid,
    "RDM_Mahalanobis": run_rdm_maha,
    "PSM_RDM_Mahalanobis": run_psm_rdm_maha_hybrid,
    "MatchIt": run_psm_matchit,
    "Mahalanobis": run_maha_matchit,
    "Hybrid_Maha_MatchIt": run_psm_maha_matchit,
}
METHODS = {spec["name"]: METHOD_RUNNERS[key] for key, spec in RESEARCH_CONFIG.items()}
COLORS = {spec["name"]: spec["color"] for spec in RESEARCH_CONFIG.values()}
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
            mas = float(smd["SMD"].abs().max()) if isinstance(smd, pd.DataFrame) and not smd.empty else np.nan
            pairwise = calculate_pairwise_differences(matched, df, numeric, categorical)
            numeric_mismatch = {
                f"num_{col}_maxpd": float(pairwise[col].abs().max()) for col in numeric if col in pairwise
            }
            categorical_mismatch = {
                f"cat_{col}_pmr": float(pairwise[col].mean()) for col in categorical if col in pairwise
            }

            rep_results.append({
                "seed": rep_idx, "method": name, "true_att": true_att,
                "crude_att": crude_att, "att": att, "retention": retention,
                "mas": mas, "time": elapsed,
                **numeric_mismatch, **categorical_mismatch,
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
    """Bias, pairwise differences, balance, retention, and runtime diagnostics."""
    set_r_style()

    method_order = list(METHODS.keys())
    palette = {m: COLORS[m] for m in method_order}

    plot_df = results_df[results_df["method"].isin(method_order)].copy()
    fig, axes = plt.subplots(3, 2, figsize=(18, 24), constrained_layout=True)
    ax1, ax2, ax3, ax4, ax5, ax6 = axes.flat

    bias_data = plot_df[["method", "att"]].dropna().copy()
    bias_data["bias"] = bias_data["att"] - true_effect
    crude = results_df.drop_duplicates("seed")[["crude_att"]].dropna().rename(columns={"crude_att": "att"})
    crude["method"] = "Crude Unmatched"
    crude["bias"] = crude["att"] - true_effect
    bias_data = pd.concat([bias_data, crude], ignore_index=True)
    bias_palette = palette | {"Crude Unmatched": CRUDE_COLOR}
    sns.boxplot(data=bias_data, x="bias", y="method", hue="method", palette=bias_palette,
                ax=ax1, dodge=False, order=["Crude Unmatched"] + method_order,
                hue_order=["Crude Unmatched"] + method_order)
    ax1.axvline(0, color="black", linewidth=1); ax1.set_title("ATT Bias (Estimated − True)")
    ax1.set_xlabel("Bias")
    if ax1.get_legend(): ax1.get_legend().remove()

    def _difference_long(prefix):
        cols = [c for c in plot_df.columns if c.startswith(prefix)]
        if not cols: return pd.DataFrame()
        out = plot_df[["method"] + cols].copy()
        out["value"] = out[cols].max(axis=1)
        return out[["method", "value"]].dropna()

    numeric_long = _difference_long("num_")
    if not numeric_long.empty:
        sns.boxplot(data=numeric_long, x="method", y="value", hue="method", palette=palette, ax=ax2,
                    order=method_order, hue_order=method_order, dodge=False)
    ax2.set_title("Aggregate Pairwise Difference (Maximum across Numeric Features)")
    ax2.set_xlabel(""); ax2.set_ylabel("Max absolute standardized difference per pair")
    ax2.tick_params(axis="x", rotation=45)
    if ax2.get_legend(): ax2.legend(title="Method", fontsize=8)

    categorical_long = _difference_long("cat_")
    if not categorical_long.empty:
        sns.boxplot(data=categorical_long, x="method", y="value", hue="method", palette=palette, ax=ax3,
                    order=method_order, hue_order=method_order, dodge=False)
    ax3.set_title("Aggregate Pairwise Mismatch Rate (Maximum across Categorical Features)")
    ax3.set_xlabel(""); ax3.set_ylabel("Max mismatch rate per pair")
    ax3.set_ylim(-0.05, 1.05); ax3.tick_params(axis="x", rotation=45)
    if ax3.get_legend(): ax3.legend(title="Method", fontsize=8)

    sns.boxplot(data=plot_df.dropna(subset=["mas"]), x="mas", y="method", hue="method",
                palette=palette, ax=ax4, dodge=False, order=method_order, hue_order=method_order)
    ax4.set_title("Covariate Balance (Maximum Absolute SMD)"); ax4.set_xlabel("Max absolute SMD")
    if ax4.get_legend(): ax4.get_legend().remove()
    sns.boxplot(data=plot_df.dropna(subset=["retention"]), x="retention", y="method", hue="method",
                palette=palette, ax=ax5, dodge=False, order=method_order, hue_order=method_order)
    ax5.set_title("Match Retention Rate"); ax5.set_xlabel("Retention")
    if ax5.get_legend(): ax5.get_legend().remove()
    sns.boxplot(data=plot_df.dropna(subset=["time"]), x="time", y="method", hue="method",
                palette=palette, ax=ax6, dodge=False, order=method_order, hue_order=method_order)
    ax6.set_title("Execution Time (Seconds)"); ax6.set_xlabel("Seconds")
    if ax6.get_legend(): ax6.get_legend().remove()

    all_axes = [ax1, ax2, ax3, ax4, ax5, ax6]

    for ax in all_axes:
        sns.despine(ax=ax, left=True, bottom=True)
        ax.grid(True, which="major", linestyle="-", linewidth=1.2)
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

def _parse_args():
    parser = argparse.ArgumentParser(description="Run IHDP validation with the research methods.")
    parser.add_argument("--start-rep", type=int, default=0)
    parser.add_argument("--n-reps", type=int, default=N_REPLICATIONS)
    parser.add_argument("--aggregate", action="store_true")
    return parser.parse_args()


def _aggregate_chunks():
    output_dir = Path(OUT_DIR)
    paths = sorted(output_dir.glob("ihdp_research_methods_*_*.csv"))
    if not paths:
        raise FileNotFoundError("No IHDP chunk CSVs were found to aggregate.")

    results_df = pd.concat((pd.read_csv(path) for path in paths), ignore_index=True)
    expected_seeds = set(range(N_REPLICATIONS))
    expected_methods = set(METHODS)
    observed_seeds = set(results_df["seed"].unique())
    observed_methods = set(results_df["method"].unique())
    if observed_seeds != expected_seeds or observed_methods != expected_methods:
        raise ValueError(
            "Incomplete chunk results: "
            f"seeds={len(observed_seeds)}/{N_REPLICATIONS}, "
            f"methods={sorted(observed_methods)}"
        )
    if results_df.duplicated(["seed", "method"]).any():
        raise ValueError("Chunk results contain duplicate seed-method rows.")

    results_df = results_df.sort_values(["seed", "method"]).reset_index(drop=True)
    results_df.to_csv(output_dir / "ihdp_research_methods_raw.csv", index=False)
    print_summary(results_df, PUBLISHED_ATT)
    make_plot(results_df, PUBLISHED_ATT, output_dir / "ihdp_research_methods.png")
    results_df.to_csv(output_dir / "ihdp_research_methods.csv", index=False)
    print(f"\n  Saved: {output_dir / 'ihdp_research_methods.csv'}")


def main():
    args = _parse_args()
    if args.aggregate:
        _aggregate_chunks()
        return
    if args.start_rep < 0 or args.n_reps < 1:
        raise ValueError("--start-rep must be non-negative and --n-reps must be at least 1.")
    stop_rep = min(args.start_rep + args.n_reps, N_REPLICATIONS)
    if args.start_rep >= stop_rep:
        raise ValueError("The requested replication range is empty.")

    print("=" * 80)
    print(f"IHDP — Replications {args.start_rep}-{stop_rep - 1}, Seven Research Methods")
    print(f"  True ATT = {PUBLISHED_ATT}")
    print(
        f"  Caliper = {CALIPER} | RDM threshold = {RDM_THRESHOLD} | "
        f"PSM+RDM threshold = {GOWER_THRESHOLD} | Maha threshold = {MAHA_THRESHOLD} | "
        f"candidates = {K_CANDIDATES}"
    )
    print("=" * 80)

    all_results = []
    t_total = time.time()
    for rep_idx in range(args.start_rep, stop_rep):
        rep_results = run_single_rep(rep_idx)
        all_results.extend(rep_results)
        if rep_idx % 10 == 0:
            n_ok = sum(1 for r in rep_results if not np.isnan(r.get("att", np.nan)))
            print(f"  Rep {rep_idx:>3}: {n_ok}/{len(METHODS)} methods OK")
    total_time = time.time() - t_total
    print(f"\nTotal runtime: {total_time:.1f}s ({total_time / (stop_rep - args.start_rep):.2f}s/rep)")

    results_df = pd.DataFrame(all_results)
    os.makedirs(OUT_DIR, exist_ok=True)
    if args.start_rep != 0 or stop_rep != N_REPLICATIONS:
        chunk_path = Path(OUT_DIR) / f"ihdp_research_methods_{args.start_rep:03d}_{stop_rep - 1:03d}.csv"
        results_df.to_csv(chunk_path, index=False)
        print(f"\n  Saved chunk: {chunk_path}")
        return

    results_df.to_csv(f"{OUT_DIR}/ihdp_research_methods_raw.csv", index=False)

    print_summary(results_df, PUBLISHED_ATT)

    make_plot(results_df, PUBLISHED_ATT, f"{OUT_DIR}/ihdp_research_methods.png")

    print(f"\n  Saved: {OUT_DIR}/ihdp_research_methods.csv")
    results_df.to_csv(f"{OUT_DIR}/ihdp_research_methods.csv", index=False)


if __name__ == "__main__":
    main()
