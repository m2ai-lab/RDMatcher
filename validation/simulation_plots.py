from __future__ import annotations

import ast
import os
import math
from graphlib import TopologicalSorter
from pathlib import Path
from typing import Any

import matplotlib


def _running_in_notebook() -> bool:
    try:
        from IPython import get_ipython

        shell = get_ipython()
        if shell is None:
            return False
        return shell.__class__.__name__ == "ZMQInteractiveShell"
    except Exception:
        return False


if os.environ.get("SIMULATION_PLOTS_BACKEND", "").lower() == "agg" or not _running_in_notebook():
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression


def set_r_style() -> None:
    sns.set_style(
        "darkgrid",
        {
            "axes.facecolor": "#F3F3F3",
            "grid.color": "white",
            "axes.edgecolor": "none",
            "xtick.bottom": True,
            "ytick.left": True,
        },
    )


def format_feature_name(col: str, label_map: dict[str, str] | None) -> str:
    if not label_map:
        return col
    if col in label_map:
        return label_map[col]
    for base_key, readable_name in label_map.items():
        if col.startswith(f"{base_key}_"):
            category_val = col.replace(f"{base_key}_", "")
            return f"{readable_name}: {category_val}"
    return col


def _coerce_pair_diff_records(cell: Any) -> list[dict[str, Any]]:
    if isinstance(cell, list):
        return [record for record in cell if isinstance(record, dict)]
    if isinstance(cell, str):
        try:
            parsed = ast.literal_eval(cell)
        except (ValueError, SyntaxError):
            return []
        if isinstance(parsed, list):
            return [record for record in parsed if isinstance(record, dict)]
    return []


def _build_pairwise_df_from_long_results(
    results_df: pd.DataFrame,
    method_name_map: dict[str, str],
    label_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    parsed_pairwise_data: list[dict[str, Any]] = []

    if "pair_diffs" in results_df.columns:
        for _, row in results_df.iterrows():
            algo_name = method_name_map.get(row.get("method"))
            if algo_name is None:
                continue
            for record in _coerce_pair_diff_records(row.get("pair_diffs")):
                feature = str(record.get("feature", ""))
                metric = str(record.get("metric", ""))
                value = record.get("value")
                if feature and metric and value is not None:
                    parsed_pairwise_data.append(
                        {
                            "Seed": row.get("seed", row.get("iteration")),
                            "Method": algo_name,
                            "Feature": format_feature_name(feature, label_map),
                            "Metric": metric,
                            "Value": float(value),
                        }
                    )
    else:
        for _, row in results_df.iterrows():
            algo_name = method_name_map.get(row.get("method"))
            if algo_name is None:
                continue
            if isinstance(row.get("num"), dict):
                for key, val in row["num"].items():
                    if key.endswith("_meanpd"):
                        parsed_pairwise_data.append(
                            {
                                "Seed": row.get("seed", row.get("iteration")),
                                "Method": algo_name,
                                "Feature": format_feature_name(key.replace("_meanpd", ""), label_map),
                                "Metric": "ASD",
                                "Value": float(val),
                            }
                        )
            if isinstance(row.get("cat"), dict):
                for key, val in row["cat"].items():
                    if key.endswith("_pmr"):
                        parsed_pairwise_data.append(
                            {
                                "Seed": row.get("seed", row.get("iteration")),
                                "Method": algo_name,
                                "Feature": format_feature_name(key.replace("_pmr", ""), label_map),
                                "Metric": "PMR",
                                "Value": float(val),
                            }
                        )

    return pd.DataFrame(parsed_pairwise_data)


def _summarize_pairwise_prevalence(pairwise_df: pd.DataFrame) -> pd.DataFrame:
    if pairwise_df.empty:
        return pairwise_df

    # PMR records are raw 0/1 mismatch indicators for each matched pair and feature.
    # Summarize them to mismatch prevalence within seed-method-feature so the plot
    # reflects how often assignments are mismatched rather than pooled binary draws.
    pmr = pairwise_df[pairwise_df["Metric"] == "PMR"]
    if not pmr.empty:
        pmr = (
            pmr.groupby(["Seed", "Method", "Feature", "Metric"], as_index=False)["Value"]
            .mean()
        )

    # ASD records are already absolute pairwise distances; summarize them to the
    # within-seed mean by feature to keep the scale comparable across methods.
    asd = pairwise_df[pairwise_df["Metric"] == "ASD"]
    if not asd.empty:
        asd = (
            asd.groupby(["Seed", "Method", "Feature", "Metric"], as_index=False)["Value"]
            .mean()
        )

    return pd.concat([pmr, asd], ignore_index=True)


def plot_imbalanced_dgp_diagnostics(
    df_obs: pd.DataFrame,
    features_numeric: list[str],
    features_categorical: list[str],
    treatment_node: str,
    outcome_node: str,
    label_map: dict[str, str] | None = None,
    title_prefix: str = "Medium",
    save_path: str | Path | None = None,
):
    set_r_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    plt.subplots_adjust(wspace=0.3)

    df_encoded = pd.get_dummies(df_obs, columns=features_categorical, drop_first=True, dtype=int)
    encoded_covariates = features_numeric + [
        col for col in df_encoded.columns if col.startswith(tuple([f"{c}_" for c in features_categorical]))
    ]

    treated = df_encoded[df_encoded[treatment_node] == 1]
    controls = df_encoded[df_encoded[treatment_node] == 0]
    sample_size = min(len(controls), len(treated) * 10)
    diag_controls = controls.sample(n=sample_size, random_state=42)
    df_diag = pd.concat([treated, diag_controls]).copy()

    X_diag = df_diag[encoded_covariates]
    y_diag = df_diag[treatment_node]
    ps_model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    ps_model.fit(X_diag, y_diag)
    df_diag["diagnostic_ps"] = ps_model.predict_proba(X_diag)[:, 1]

    sns.kdeplot(
        data=df_diag[df_diag[treatment_node] == 1],
        x="diagnostic_ps",
        fill=True,
        color="#052049",
        label="Treated (All)",
        ax=axes[0],
        linewidth=2,
    )
    sns.kdeplot(
        data=df_diag[df_diag[treatment_node] == 0],
        x="diagnostic_ps",
        fill=True,
        color="#878D96",
        label="Control (1:10 Sample)",
        ax=axes[0],
        linewidth=2,
    )
    axes[0].set_title("Diagnostic Positivity (Feature Overlap)", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Estimated Propensity Score (Downsampled Model)")
    axes[0].set_ylabel("Density")
    axes[0].legend(loc="upper right")

    smd_records = []
    treated_df_full = df_encoded[df_encoded[treatment_node] == 1]
    control_df_full = df_encoded[df_encoded[treatment_node] == 0]
    for col in encoded_covariates:
        mean1 = treated_df_full[col].mean()
        mean0 = control_df_full[col].mean()
        var1 = treated_df_full[col].var()
        var0 = control_df_full[col].var()
        pooled_sd = math.sqrt((var1 + var0) / 2.0)
        smd = 0 if pooled_sd == 0 else abs((mean1 - mean0) / pooled_sd)
        smd_records.append({"Feature": format_feature_name(col, label_map), "Absolute SMD": smd})

    df_smd = pd.DataFrame(smd_records).sort_values(by="Absolute SMD", ascending=False)
    sns.barplot(data=df_smd, y="Feature", x="Absolute SMD", ax=axes[1], color="#506380")
    axes[1].set_title("Empirical Confounding Strength", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Absolute Standardized Mean Difference (SMD)")
    axes[1].axvline(0.1, color="black", linestyle="--", alpha=0.5)
    axes[1].set_ylabel("")
    axes[1].set_xlim(0, max(2, float(df_smd["Absolute SMD"].max()) * 1.05 if not df_smd.empty else 2))

    for ax in axes:
        sns.despine(ax=ax, left=True, bottom=True)
        ax.grid(True, axis="x", linestyle="-", linewidth=1.2)
        ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.5)

    plt.suptitle(f"Data Generation: {title_prefix} Support", fontsize=16, fontweight="bold", y=1)
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=300)

    return fig, axes


def plot_matching_correlation_heatmap(
    df_complete: pd.DataFrame,
    features_numeric: list[str],
    features_categorical: list[str],
    label_map: dict[str, str] | None = None,
    title_prefix: str = "Medium",
    save_path: str | Path | None = None,
):
    from .dgp import correlation_heatmap_frame

    set_r_style()
    corr = correlation_heatmap_frame(
        df_complete=df_complete,
        features_numeric=features_numeric,
        features_categorical=features_categorical,
    )
    display_cols = [format_feature_name(col, label_map) for col in corr.columns]
    corr_display = corr.copy()
    corr_display.columns = display_cols
    corr_display.index = display_cols

    fig_scale = max(10, min(22, len(display_cols) * 0.75))
    fig, ax = plt.subplots(figsize=(fig_scale, fig_scale))
    sns.heatmap(
        corr_display,
        cmap="coolwarm",
        center=0.0,
        vmin=-1.0,
        vmax=1.0,
        square=True,
        linewidths=0.3,
        cbar_kws={"shrink": 0.7, "label": "Correlation"},
        ax=ax,
    )
    ax.set_title(
        f"Matching Covariate Correlation Heatmap\n{title_prefix}",
        fontsize=15,
        fontweight="bold",
    )
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=300)

    return fig, ax, corr


def plot_dag_structure(
    dag_spec: dict[str, dict[str, Any]],
    treatment_node: str = "exposure_status",
    outcome_node: str = "outcome",
    label_map: dict[str, str] | None = None,
    title_prefix: str = "Medium",
    save_path: str | Path | None = None,
):
    set_r_style()

    role_palette = {
        "confounder": "#4C78A8",
        "prognostic": "#F58518",
        "proxy": "#72B7B2",
        "instrument": "#E45756",
        "mediator": "#54A24B",
        "collider": "#B279A2",
        "treatment": "#2F2F2F",
        "outcome": "#111111",
    }
    match_edge_palette = {True: "#1A1A1A", False: "#9E9E9E"}

    dependencies = {node: set(spec.get("parents", [])) for node, spec in dag_spec.items()}
    generations = list(TopologicalSorter(dependencies).static_order())

    level_map: dict[str, int] = {}
    for node in generations:
        parents = dag_spec[node].get("parents", [])
        level_map[node] = 0 if not parents else 1 + max(level_map[parent] for parent in parents)

    nodes_by_level: dict[int, list[str]] = {}
    for node, level in level_map.items():
        nodes_by_level.setdefault(level, []).append(node)

    for level_nodes in nodes_by_level.values():
        level_nodes.sort(key=lambda name: (dag_spec[name].get("role", ""), name))

    positions: dict[str, tuple[float, float]] = {}
    max_width = max(len(nodes) for nodes in nodes_by_level.values())
    for level in sorted(nodes_by_level):
        nodes = nodes_by_level[level]
        y_coords = np.linspace(0.1, 0.9, len(nodes)) if len(nodes) > 1 else np.array([0.5])
        for idx, node in enumerate(nodes):
            x = level / max(1, (max(nodes_by_level) if nodes_by_level else 1))
            y = y_coords[idx]
            positions[node] = (x, y)

    fig_width = max(12, 3.2 * (max(level_map.values()) + 1))
    fig_height = max(7, 1.0 * max_width + 2)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    for child, spec in dag_spec.items():
        child_x, child_y = positions[child]
        for parent in spec.get("parents", []):
            parent_x, parent_y = positions[parent]
            arrow = FancyArrowPatch(
                (parent_x + 0.03, parent_y),
                (child_x - 0.03, child_y),
                arrowstyle="-|>",
                mutation_scale=10,
                linewidth=1.2,
                color="#6B6B6B",
                alpha=0.75,
                connectionstyle="arc3,rad=0.0",
            )
            ax.add_patch(arrow)

    for node, spec in dag_spec.items():
        x, y = positions[node]
        role = spec.get("role", "proxy")
        node_color = role_palette.get(role, "#72B7B2")
        is_match = bool(spec.get("match", False))
        edge_color = "#000000" if node in {treatment_node, outcome_node} else match_edge_palette[is_match]
        edge_width = 2.4 if is_match else 1.4
        marker = "s" if node == treatment_node else ("D" if node == outcome_node else "o")
        size = 2200 if node in {treatment_node, outcome_node} else 1800

        ax.scatter(
            [x],
            [y],
            s=size,
            c=node_color,
            edgecolors=edge_color,
            linewidths=edge_width,
            marker=marker,
            zorder=3,
        )
        ax.text(
            x,
            y,
            format_feature_name(node, label_map),
            ha="center",
            va="center",
            fontsize=9,
            color="white" if role in {"treatment", "outcome"} else "black",
            fontweight="bold" if node in {treatment_node, outcome_node} else "normal",
            zorder=4,
            wrap=True,
        )

    role_handles = [
        Line2D([0], [0], marker="o", color="none", label=role.title(), markerfacecolor=color, markeredgecolor="black", markersize=10)
        for role, color in role_palette.items()
        if role not in {"treatment", "outcome"}
    ]
    special_handles = [
        Line2D([0], [0], marker="s", color="none", label="Treatment", markerfacecolor=role_palette["treatment"], markeredgecolor="black", markersize=10),
        Line2D([0], [0], marker="D", color="none", label="Outcome", markerfacecolor=role_palette["outcome"], markeredgecolor="black", markersize=10),
        Line2D([0], [0], marker="o", color="none", label="Matched-on Node", markerfacecolor="white", markeredgecolor="black", markeredgewidth=2.4, markersize=10),
        Line2D([0], [0], marker="o", color="none", label="Excluded Node", markerfacecolor="white", markeredgecolor="#9E9E9E", markeredgewidth=1.4, markersize=10),
    ]
    ax.legend(
        handles=role_handles + special_handles,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=False,
        title="Node Semantics",
    )
    ax.set_title(f"DAG Structure\n{title_prefix}", fontsize=16, fontweight="bold")
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=300)

    return fig, ax


def plot_simulation(
    results_df: pd.DataFrame,
    true_effect: float,
    algo_config: dict[str, dict[str, Any]] | None = None,
    label_map: dict[str, str] | None = None,
    save_path: str | Path | None = None,
    title_prefix: str = "Medium",
):
    set_r_style()

    def calculate_mas(smd_table):
        if smd_table is None or not isinstance(smd_table, pd.DataFrame):
            return np.nan
        return smd_table["SMD"].abs().mean()

    algo_config = algo_config or {}
    runtime_metric = "match_time_sec" if any(col.endswith("_match_time_sec") for col in results_df.columns) else "time"
    att_cols, mas_cols, time_cols = [], [], []
    name_lookup, color_palette = {}, {}
    for prefix, config in algo_config.items():
        method_name = config["name"]
        att_col = f"{prefix}_att"
        smd_col = f"{prefix}_smd"
        mas_col = f"{prefix}_mas"
        time_col = f"{prefix}_{runtime_metric}"
        retention_col = f"{prefix}_retention"
        num_col = f"{prefix}_num"
        cat_col = f"{prefix}_cat"
        pair_diffs_col = f"{prefix}_pair_diffs"

        color_palette[method_name] = config["color"]

        if att_col in results_df.columns:
            att_cols.append(att_col)
            name_lookup[att_col] = method_name
            name_lookup[mas_col] = method_name

            if smd_col in results_df.columns:
                results_df[mas_col] = results_df[smd_col].apply(calculate_mas)
                mas_cols.append(mas_col)

            if time_col in results_df.columns:
                time_cols.append(time_col)
                name_lookup[time_col] = method_name

            if retention_col in results_df.columns:
                name_lookup[retention_col] = method_name

        if pair_diffs_col in results_df.columns:
            name_lookup[pair_diffs_col] = method_name

    long_cols = []
    for prefix in algo_config:
        for metric in ["att", "smd", runtime_metric, "retention", "num", "cat", "pair_diffs"]:
            col = f"{prefix}_{metric}"
            if col in results_df.columns:
                long_cols.append(col)

    if "iteration" in results_df.columns and "method" not in results_df.columns:
        long_results_df = results_df[["iteration"] + long_cols].copy()
        long_results_df = long_results_df.melt(id_vars=["iteration"], value_vars=long_cols, var_name="method_metric", value_name="value")
        long_results_df[["method", "metric"]] = long_results_df["method_metric"].str.extract(
            rf"^(.*)_(att|smd|{runtime_metric}|retention|num|cat|pair_diffs)$"
        )
        long_results_df = long_results_df.dropna(subset=["method", "metric"])
        long_results_df = long_results_df.pivot(index=["iteration", "method"], columns="metric", values="value").reset_index()
        if runtime_metric in long_results_df.columns and "time" not in long_results_df.columns:
            long_results_df["time"] = long_results_df[runtime_metric]
    else:
        long_results_df = results_df.copy()

    pairwise_df = _build_pairwise_df_from_long_results(long_results_df, {k: v["name"] for k, v in algo_config.items()}, label_map=label_map)

    fig = plt.figure(figsize=(16, 26), constrained_layout=True)
    gs = fig.add_gridspec(4, 2, hspace=0.5, wspace=0.2)
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, :])
    ax3a = fig.add_subplot(gs[2, 0])
    ax3b = fig.add_subplot(gs[2, 1])
    ax4 = fig.add_subplot(gs[3, :])
    all_axes = [ax1, ax2, ax3a, ax3b, ax4]

    if att_cols:
        id_vars = ["iteration"] if "iteration" in results_df.columns else []
        plot_att_df = results_df.melt(id_vars=id_vars, value_vars=att_cols, var_name="col", value_name="Estimate")
        plot_att_df["Algorithm"] = plot_att_df["col"].map(name_lookup)
        sns.kdeplot(data=plot_att_df, x="Estimate", hue="Algorithm", fill=True, ax=ax1, palette=color_palette, alpha=0.1, linewidth=2)
        ax1.axvline(true_effect, color="#ef4444", linestyle="--", linewidth=2.5, label=f"True Effect ({true_effect})", zorder=10)
    ax1.set_title("Treatment Effect Recovery (ATT)", fontsize=15, fontweight="bold")

    if mas_cols:
        id_vars = ["iteration"] if "iteration" in results_df.columns else []
        plot_mas_df = results_df.melt(id_vars=id_vars, value_vars=mas_cols, var_name="col", value_name="Mean Absolute SMD")
        plot_mas_df["Algorithm"] = plot_mas_df["col"].map(name_lookup)
        sns.boxplot(data=plot_mas_df, x="Mean Absolute SMD", y="Algorithm", hue="Algorithm", palette=color_palette, ax=ax2, boxprops=dict(alpha=1.0), dodge=False)
        ax2.axvline(0.1, color="#b91c1c", linestyle=":", linewidth=2, label="Threshold (0.1)")
    ax2.set_title("Covariate Balance (Standardized Mean Difference)", fontsize=15, fontweight="bold")

    if not pairwise_df.empty:
        df_cat = pairwise_df[pairwise_df["Metric"] == "PMR"]
        df_num = pairwise_df[pairwise_df["Metric"] == "ASD"]
        if not df_cat.empty:
            sns.boxplot(data=df_cat, x="Feature", y="Value", hue="Method", palette=color_palette, ax=ax3a, boxprops=dict(alpha=0.8), fliersize=2)
            ax3a.set_title("Pairwise Mismatch Rate (Categorical)", fontsize=15, fontweight="bold")
            ax3a.set_ylabel("Mismatch Rate (Proportion)")
            ax3a.set_xlabel("")
            ax3a.set_ylim(-0.05, 1.05)
            ax3a.tick_params(axis="x", rotation=45)
            if ax3a.get_legend():
                ax3a.get_legend().remove()
        if not df_num.empty:
            sns.boxplot(data=df_num, x="Feature", y="Value", hue="Method", palette=color_palette, ax=ax3b, boxprops=dict(alpha=0.8), fliersize=2)
            ax3b.set_title("Pairwise Absolute Standardized Difference (Numeric)", fontsize=15, fontweight="bold")
            ax3b.set_ylabel("Absolute Standardized Difference")
            ax3b.set_xlabel("")
            ax3b.tick_params(axis="x", rotation=45)
            if ax3b.get_legend():
                ax3b.get_legend().remove()

    time_cols = [c for c in time_cols if c in results_df.columns]
    if time_cols:
        id_vars = ["iteration"] if "iteration" in results_df.columns else []
        plot_time_df = results_df.melt(id_vars=id_vars, value_vars=time_cols, var_name="col", value_name="Seconds")
        plot_time_df["Algorithm"] = plot_time_df["col"].map(name_lookup)
        sns.boxplot(data=plot_time_df, x="Seconds", y="Algorithm", hue="Algorithm", palette=color_palette, ax=ax4, boxprops=dict(alpha=1.0), dodge=False)
    ax4.set_title("Execution Time (Seconds)", fontsize=15, fontweight="bold")

    for ax in all_axes:
        sns.despine(ax=ax, left=True, bottom=True)
        ax.tick_params(axis="both", labelsize=14)
        ax.grid(True, which="major", linestyle="-", linewidth=1.2)
        if ax in [ax3a, ax3b]:
            ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.5)

    fig.suptitle(f"Matching Diagnostics: {title_prefix} Support", fontsize=16, fontweight="bold")
    # Explicitly reserve a small top margin for the suptitle without relying on
    # tight_layout(), which warns on this multi-axes diagnostics figure.
    fig.get_layout_engine().set(rect=(0.0, 0.0, 1.0, 0.97))

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=300)

    metrics = []
    for prefix, config in algo_config.items():
        att_c, mas_c, time_c, retention_c = f"{prefix}_att", f"{prefix}_mas", f"{prefix}_{runtime_metric}", f"{prefix}_retention"
        if att_c in results_df.columns:
            est = results_df[att_c].dropna()
            M = len(est)
            bias = est.mean() - true_effect
            mse_est = ((est - true_effect) ** 2).mean()
            rmse = np.sqrt(mse_est)
            if M > 1:
                mcse_bias = est.std(ddof=1) / np.sqrt(M)
                mcse_mse = ((est - true_effect) ** 2).std(ddof=1) / np.sqrt(M)
                mcse_rmse = mcse_mse / (2 * rmse) if rmse > 0 else 0.0
            else:
                mcse_bias, mcse_rmse = np.nan, np.nan
            m_mas = results_df[mas_c].mean() if mas_c in results_df.columns else np.nan
            m_retention = results_df[retention_c].mean() if retention_c in results_df.columns else np.nan
            metrics.append(
                [
                    config["name"],
                    f"{bias:.3f} ({mcse_bias:.3f})",
                    f"{rmse:.3f} ({mcse_rmse:.3f})",
                    f"{m_mas:.3f}" if not np.isnan(m_mas) else "N/A",
                    f"{m_retention:.1%}" if not np.isnan(m_retention) else "N/A",
                ]
            )

    return fig, metrics


def plot_retention_dist(
    results_df: pd.DataFrame,
    algo_config: dict[str, dict[str, Any]],
    save_path: str | Path | None = None,
    title: str = "Sample Retention Distribution",
    title_prefix: str = "Low",
):
    set_r_style()

    method_name_map = {k: v["name"] for k, v in algo_config.items()}
    color_palette = {v["name"]: v["color"] for v in algo_config.values()}
    algo_order = [method_name_map[k] for k in algo_config if k in method_name_map]
    retention_cols = []
    name_lookup = {}
    for prefix, config in algo_config.items():
        retention_col = f"{prefix}_retention"
        if retention_col in results_df.columns:
            retention_cols.append(retention_col)
            name_lookup[retention_col] = config["name"]

    if retention_cols:
        plot_df = results_df.melt(
            id_vars=["iteration"] if "iteration" in results_df.columns else [],
            value_vars=retention_cols,
            var_name="col",
            value_name="Retention",
        )
        plot_df["Algorithm"] = plot_df["col"].map(name_lookup)
    elif {"method", "retention"}.issubset(results_df.columns):
        plot_df = results_df.loc[results_df["method"].isin(algo_config.keys()), ["method", "retention"]].copy()
        plot_df = plot_df.rename(columns={"retention": "Retention"})
        plot_df["Algorithm"] = plot_df["method"].map(method_name_map)
    else:
        raise ValueError("No retention data identified in the DataFrame.")

    plot_df = plot_df.dropna(subset=["Algorithm", "Retention"])
    if plot_df.empty:
        raise ValueError("No retention data identified in the DataFrame.")
    observed_algorithms = plot_df["Algorithm"].dropna().unique().tolist()
    algo_order = [name for name in algo_order if name in observed_algorithms]

    fig, ax = plt.subplots(figsize=(12, 4))
    sns.boxplot(
        data=plot_df,
        x="Retention",
        y="Algorithm",
        hue="Algorithm",
        palette=color_palette,
        order=algo_order,
        hue_order=algo_order,
        ax=ax,
        boxprops=dict(alpha=0.85),
        dodge=False,
        legend=False,
    )
    ax.set_title(f"{title}: {title_prefix} Support", fontsize=15, fontweight="bold")
    ax.set_xlabel("Proportion Retained", fontsize=14)
    ax.set_ylabel("")
    ax.set_xlim(-0.05, 1.05)

    sns.despine(ax=ax, left=True, bottom=True)
    ax.tick_params(axis="both", labelsize=14)
    ax.grid(True, axis="x", which="major", linestyle="-", linewidth=1.2)
    ax.grid(False, axis="y")

    plt.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
    return fig, ax


def plot_simulation_advanced(
    results_df: pd.DataFrame,
    algo_config: dict[str, dict[str, Any]],
    true_effect: float | None = None,
    save_path: str | Path | None = None,
    title_prefix: str = "Medium",
):
    """
    Paper-style summary plot for long-format simulation results.

    Expects one row per seed-method pair with columns:
    - method
    - att
    - crude_att
    - smd
    - time or match_time_sec
    - retention
    - num
    - cat
    """
    set_r_style()
    plt.rcParams.update({"axes.labelsize": 16, "xtick.labelsize": 14, "ytick.labelsize": 14})

    def calculate_max_abs_smd(smd_table: pd.DataFrame | None) -> float:
        if smd_table is None or not isinstance(smd_table, pd.DataFrame) or smd_table.empty:
            return np.nan
        return float(smd_table["SMD"].abs().max())

    iter_col = "seed" if "seed" in results_df.columns else "iteration"

    if "true_att" in results_df.columns and results_df["true_att"].notna().any():
        truth_by_iter = (
            results_df[[iter_col, "true_att"]]
            .dropna(subset=["true_att"])
            .drop_duplicates(subset=[iter_col])
            .rename(columns={"true_att": "_true_effect"})
        )
    else:
        if true_effect is None:
            raise ValueError("plot_simulation_advanced() requires either results_df['true_att'] or a scalar true_effect.")
        truth_by_iter = pd.DataFrame({iter_col: results_df[iter_col].drop_duplicates(), "_true_effect": true_effect})

    plot_df = results_df.copy()
    runtime_col = "match_time_sec" if "match_time_sec" in plot_df.columns else "time"
    plot_df["mas"] = plot_df["smd"].apply(calculate_max_abs_smd)
    plot_df = plot_df.merge(truth_by_iter, on=iter_col, how="left")
    if true_effect is not None:
        plot_df["_true_effect"] = plot_df["_true_effect"].fillna(true_effect)

    method_name_map = {k: v["name"] for k, v in algo_config.items()}
    color_palette = {v["name"]: v["color"] for v in algo_config.values()}
    algo_order = [method_name_map[k] for k in algo_config if k in method_name_map]
    kde_order = ["Crude Unmatched"] + algo_order

    plot_df["Algorithm"] = plot_df["method"].map(method_name_map)
    plot_df = plot_df.dropna(subset=["Algorithm"]).copy()

    pairwise_df = _summarize_pairwise_prevalence(
        _build_pairwise_df_from_long_results(plot_df, method_name_map)
    )

    crude_plot_df = (
        results_df[[iter_col, "crude_att"]]
        .drop_duplicates(subset=[iter_col])
        .merge(truth_by_iter, on=iter_col, how="left")
        .dropna(subset=["crude_att"])
        .copy()
    )
    if true_effect is not None:
        crude_plot_df["_true_effect"] = crude_plot_df["_true_effect"].fillna(true_effect)
    crude_series = crude_plot_df["crude_att"]

    att_plot_df = plot_df[[iter_col, "att", "Algorithm", "_true_effect"]].dropna(subset=["att"]).copy()
    crude_plot_df = crude_plot_df.rename(columns={"crude_att": "att"})
    crude_plot_df["Algorithm"] = "Crude Unmatched"
    combined_att_df = pd.concat(
        [
            att_plot_df[[iter_col, "att", "Algorithm", "_true_effect"]],
            crude_plot_df[[iter_col, "att", "Algorithm", "_true_effect"]],
        ],
        ignore_index=True,
    )
    combined_att_df["bias"] = combined_att_df["att"] - combined_att_df["_true_effect"]

    kde_palette = color_palette.copy()
    kde_palette["Crude Unmatched"] = "#4b5563"

    fig = plt.figure(figsize=(16, 26))
    gs = fig.add_gridspec(4, 2, hspace=0.5, wspace=0.2)
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, :])
    ax3a = fig.add_subplot(gs[2, 0])
    ax3b = fig.add_subplot(gs[2, 1])
    ax4 = fig.add_subplot(gs[3, :])
    all_axes = [ax1, ax2, ax3a, ax3b, ax4]

    sns.boxplot(
        data=combined_att_df,
        x="bias",
        y="Algorithm",
        hue="Algorithm",
        palette=kde_palette,
        ax=ax1,
        boxprops=dict(alpha=1.0),
        dodge=False,
        order=kde_order,
        hue_order=kde_order,
    )
    ax1.axvline(0, color="#ef4444", linestyle="--", linewidth=2.5, label="Zero Bias", zorder=10)
    ax1.set_title("Treatment Effect Bias (Estimate - True Effect)", fontsize=15, fontweight="bold")
    ax1.set_xlabel("ATT Bias")
    ax1.set_ylabel("")
    if ax1.get_legend():
        ax1.get_legend().remove()

    sns.boxplot(
        data=plot_df.dropna(subset=["mas"]),
        x="mas",
        y="Algorithm",
        hue="Algorithm",
        palette=color_palette,
        ax=ax2,
        boxprops=dict(alpha=1.0),
        dodge=False,
        order=algo_order,
        hue_order=algo_order,
    )
    ax2.axvline(0.1, color="#b91c1c", linestyle=":", linewidth=2, label="Threshold (0.1)")
    ax2.set_title("Covariate Balance (Maximum Absolute Standardized Difference)", fontsize=15, fontweight="bold")
    ax2.set_xlabel("Max Absolute SMD")
    ax2.set_ylabel("")
    if ax2.get_legend():
        ax2.get_legend().remove()

    if not pairwise_df.empty:
        df_cat = pairwise_df[pairwise_df["Metric"] == "PMR"]
        df_num = pairwise_df[pairwise_df["Metric"] == "ASD"]

        if not df_cat.empty:
            sns.boxplot(
                data=df_cat,
                x="Method",
                y="Value",
                hue="Method",
                palette=color_palette,
                ax=ax3a,
                boxprops=dict(alpha=0.8),
                fliersize=2,
                dodge=False,
                order=algo_order,
                hue_order=algo_order,
            )
            ax3a.set_title("Pairwise Categorical Mismatch Prevalence", fontsize=14, fontweight="bold")
            ax3a.set_ylabel("Mismatch prevalence across matched pairs")
            ax3a.set_xlabel("")
            ax3a.set_ylim(-0.05, 1.05)
            ax3a.tick_params(axis="x", rotation=45)
            for label in ax3a.get_xticklabels():
                label.set_ha("right")
                label.set_rotation_mode("anchor")
            if ax3a.get_legend():
                ax3a.get_legend().remove()

        if not df_num.empty:
            sns.boxplot(
                data=df_num,
                x="Method",
                y="Value",
                hue="Method",
                palette=color_palette,
                ax=ax3b,
                boxprops=dict(alpha=0.8),
                fliersize=2,
                dodge=False,
                order=algo_order,
                hue_order=algo_order,
            )
            ax3b.set_title("Mean Pairwise Absolute Standardized Difference", fontsize=14, fontweight="bold")
            ax3b.set_ylabel("Mean absolute standardized difference")
            ax3b.set_xlabel("")
            ax3b.tick_params(axis="x", rotation=45)
            for label in ax3b.get_xticklabels():
                label.set_ha("right")
                label.set_rotation_mode("anchor")
            if ax3b.get_legend():
                ax3b.get_legend().remove()

    sns.boxplot(
        data=plot_df.dropna(subset=[runtime_col]),
        x=runtime_col,
        y="Algorithm",
        hue="Algorithm",
        palette=color_palette,
        ax=ax4,
        boxprops=dict(alpha=1.0),
        dodge=False,
        order=algo_order,
        hue_order=algo_order,
    )
    ax4.set_title("Execution Time (Seconds)", fontsize=15, fontweight="bold")
    ax4.set_xlabel("Seconds")
    ax4.set_ylabel("")
    if ax4.get_legend():
        ax4.get_legend().remove()

    metrics = []
    if not crude_series.empty:
        crude_errors = crude_plot_df["att"] - crude_plot_df["_true_effect"]
        crude_bias = float(crude_errors.mean())
        crude_rmse = float(np.sqrt((crude_errors**2).mean()))
        metrics.append(["Crude Unmatched", f"{crude_series.mean():.3f}", f"{crude_bias:.3f}", f"{crude_rmse:.3f}", "N/A", "N/A", "100.0%"])

    for raw_method, config in algo_config.items():
        algo_name = config["name"]
        group = plot_df[plot_df["method"] == raw_method]
        if group.empty:
            continue

        est = group["att"].dropna()
        if not est.empty:
            errors = group.loc[group["att"].notna(), "att"] - group.loc[group["att"].notna(), "_true_effect"]
            bias = float(errors.mean())
            rmse = float(np.sqrt((errors**2).mean()))
            est_mean = f"{est.mean():.3f}"
        else:
            bias = np.nan
            rmse = np.nan
            est_mean = "N/A"

        m_mas = group["mas"].dropna().mean()
        m_time = group[runtime_col].dropna().mean()
        m_retention = group["retention"].dropna().mean()
        metrics.append(
            [
                algo_name,
                est_mean,
                f"{bias:.3f}" if not np.isnan(bias) else "N/A",
                f"{rmse:.3f}" if not np.isnan(rmse) else "N/A",
                f"{m_mas:.3f}" if not np.isnan(m_mas) else "N/A",
                f"{m_time:.3f}s" if not np.isnan(m_time) else "N/A",
                f"{m_retention:.1%}" if not np.isnan(m_retention) else "N/A",
            ]
        )

    for ax in all_axes:
        sns.despine(ax=ax, left=True, bottom=True)
        ax.grid(True, which="major", linestyle="-", linewidth=1.2)
        if ax in [ax3a, ax3b]:
            ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.5)

    plt.suptitle(f"Matching Diagnostics: {title_prefix} Support", fontsize=16, fontweight="bold", y=0.91)
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=600)
    plt.show()
    return fig, metrics
