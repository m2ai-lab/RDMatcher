"""Comparison methods for benchmarking against RDMatcher.

Methods aligned with the simulation notebook (EXAMPLE_SIMULATION.ipynb):
- MatchIt PS (nearest, GLM, caliper=0.2) via rpy2
- MatchIt Mahalanobis (nearest) via rpy2
- Custom Mahalanobis matching (greedy)
- Unadjusted comparison
- Two-stage RDMatcher pipeline (PSM → Gower)

Effect estimation:
- Binary outcomes (RHC): Conditional logistic regression (CLR)
- Continuous outcomes (LaLonde): OLS with clustered SEs by match_group
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


# ---------------------------------------------------------------------------
# Covariate preparation helper
# ---------------------------------------------------------------------------

def prepare_covariates(
    df: pd.DataFrame,
    covariates: list[str],
) -> tuple[np.ndarray, dict[str, dict]]:
    """Convert covariates to a float matrix, label-encoding string columns and imputing NaN with column means."""
    out = df[covariates].copy()
    label_maps: dict[str, dict] = {}

    for col in covariates:
        if out[col].dtype == "object" or (out[col].dtype.name == "category"):
            codes, uniques = pd.factorize(out[col], sort=True)
            out[col] = codes.astype(float)
            label_maps[col] = dict(enumerate(uniques))
        else:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    X = out.values.astype(float)
    col_mean = np.nanmean(X, axis=0)
    col_mean = np.where(np.isnan(col_mean), 0.0, col_mean)
    X = np.where(np.isnan(X), col_mean, X)

    return X, label_maps


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class MethodResult:
    name: str
    effect_size: float
    matched_indices: list[tuple[int, int]] = field(default_factory=list)
    matched_df: pd.DataFrame | None = None
    runtime_seconds: float = 0.0
    n_treated: int = 0
    n_control: int = 0


# ---------------------------------------------------------------------------
# Build matched DataFrame from indices
# ---------------------------------------------------------------------------

def build_matched_df(
    df: pd.DataFrame,
    matched_indices: list[tuple[int, int]],
) -> pd.DataFrame:
    """Convert a list of (treated_idx, control_idx) pairs into a matched DataFrame with match_group."""
    rows = []
    for pair_id, (ti, ci) in enumerate(matched_indices):
        row_t = df.loc[ti].copy()
        row_t["match_group"] = pair_id
        row_t["exposure_status"] = 1
        rows.append(row_t)
        row_c = df.loc[ci].copy()
        row_c["match_group"] = pair_id
        row_c["exposure_status"] = 0
        rows.append(row_c)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Unadjusted (naive) comparison
# ---------------------------------------------------------------------------

def unadjusted_or(df: pd.DataFrame, treatment: str, outcome: str) -> float:
    """Compute unadjusted odds ratio."""
    exposed = df[df[treatment] == 1][outcome]
    control = df[df[treatment] == 0][outcome]
    a = (exposed == 1).sum()
    b = (exposed == 0).sum()
    c = (control == 1).sum()
    d = (control == 0).sum()
    return (a * d) / max(b * c, 1e-10)


def unadjusted_att(df: pd.DataFrame, treatment: str, outcome: str) -> float:
    """Compute unadjusted average treatment effect on the treated."""
    exposed = df[df[treatment] == 1][outcome].mean()
    control = df[df[treatment] == 0][outcome].mean()
    return exposed - control


# ---------------------------------------------------------------------------
# Mahalanobis distance matching (custom, greedy, without replacement)
# ---------------------------------------------------------------------------

def mahalanobis_matching(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    covariates: list[str],
    n_neighbors: int = 1,
    caliper: float | None = None,
) -> MethodResult:
    """Greedy nearest-neighbour matching on Mahalanobis distance."""
    t0 = time.time()

    treated = df[df[treatment] == 1].copy()
    control = df[df[treatment] == 0].copy()

    X_t, _ = prepare_covariates(treated, covariates)
    X_c, _ = prepare_covariates(control, covariates)

    combined = np.vstack([X_t, X_c])
    cov = np.cov(combined.T) + np.eye(len(covariates)) * 1e-6
    cov_inv = np.linalg.inv(cov)

    dists = cdist(X_t, X_c, metric="mahalanobis", VI=cov_inv)

    matched_pairs = []
    used_controls: set[int] = set()

    for i in range(dists.shape[0]):
        sorted_c = np.argsort(dists[i])
        count = 0
        for j in sorted_c:
            if caliper is not None and dists[i, j] > caliper:
                break
            if j not in used_controls:
                matched_pairs.append((treated.index[i], control.index[j]))
                used_controls.add(j)
                count += 1
                if count >= n_neighbors:
                    break

    matched_treated_idx = [p[0] for p in matched_pairs]
    matched_control_idx = [p[1] for p in matched_pairs]

    if len(matched_pairs) == 0:
        return MethodResult(
            name="Mahalanobis",
            effect_size=float("nan"),
            matched_indices=[],
            runtime_seconds=time.time() - t0,
            n_treated=0,
            n_control=0,
        )

    t_mean = df.loc[matched_treated_idx, outcome].mean()
    c_mean = df.loc[matched_control_idx, outcome].mean()

    matched_df = build_matched_df(df, matched_pairs)

    return MethodResult(
        name="Mahalanobis",
        effect_size=t_mean - c_mean,
        matched_indices=matched_pairs,
        matched_df=matched_df,
        runtime_seconds=time.time() - t0,
        n_treated=len(matched_treated_idx),
        n_control=len(matched_control_idx),
    )


# ---------------------------------------------------------------------------
# NA imputation for MatchIt (R does not accept NAs in covariates)
# ---------------------------------------------------------------------------

def _impute_for_matchit(df: pd.DataFrame, covariates: list[str]) -> pd.DataFrame:
    """Impute missing values: median for numeric, mode for categorical."""
    out = df.copy()
    for col in covariates:
        if col not in out.columns:
            continue
        if out[col].dtype.kind in "iufb":  # numeric or bool
            out[col] = out[col].fillna(out[col].median())
        else:
            mode_val = out[col].mode()
            if len(mode_val) > 0:
                out[col] = out[col].fillna(mode_val.iloc[0])
    return out


# ---------------------------------------------------------------------------
# MatchIt via rpy2 — Propensity Score Matching
# ---------------------------------------------------------------------------

def run_matchit_ps(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    covariates: list[str],
    formula: str | None = None,
    caliper: float = 0.2,
    ratio: int = 1,
    replace: bool = False,
) -> MethodResult:
    """Propensity score matching using R's MatchIt via rpy2.

    Matches the simulation notebook: method='nearest', distance='glm',
    link='linear.logit', caliper=0.2, std.caliper=TRUE.

    Returns MethodResult with matched_df containing match_group column.
    """
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.packages import importr

    t0 = time.time()

    # Impute NAs — MatchIt does not accept missing values
    df = _impute_for_matchit(df, covariates)

    with pandas2ri.converter.context():
        r_df = ro.conversion.py2rpy(df)

    matchit = importr("MatchIt")

    if formula is None:
        rhs = " + ".join(covariates)
        formula = f"{treatment} ~ {rhs}"

    r_formula = ro.Formula(formula)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = matchit.matchit(
                r_formula,
                data=r_df,
                method="nearest",
                distance="glm",
                link="linear.logit",
                caliper=caliper,
                std_caliper=True,
                ratio=ratio,
                replace=replace,
            )
    except Exception as e:
        warnings.warn(f"MatchIt PS failed: {e}")
        return MethodResult(
            name="MatchIt PS",
            effect_size=float("nan"),
            runtime_seconds=time.time() - t0,
        )

    matched_r = matchit.match_data(m)

    with pandas2ri.converter.context():
        matched_df = ro.conversion.rpy2py(matched_r)

    if "subclass" in matched_df.columns:
        matched_df = matched_df.rename(columns={"subclass": "match_group"})

    n_treated = int((matched_df[treatment] == 1).sum())
    n_control = int((matched_df[treatment] == 0).sum())

    return MethodResult(
        name="MatchIt PS",
        effect_size=float("nan"),  # computed later by caller
        matched_df=matched_df,
        runtime_seconds=time.time() - t0,
        n_treated=n_treated,
        n_control=n_control,
    )


def run_matchit_scaled_euclidean(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    covariates: list[str],
    formula: str | None = None,
    ratio: int = 1,
    replace: bool = False,
) -> MethodResult:
    """Nearest-neighbor matching on MatchIt's scaled Euclidean distance."""
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.packages import importr

    t0 = time.time()
    df = _impute_for_matchit(df, covariates)
    with pandas2ri.converter.context():
        r_df = ro.conversion.py2rpy(df)
    matchit = importr("MatchIt")
    if formula is None:
        formula = f"{treatment} ~ " + " + ".join(covariates)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = matchit.matchit(
                ro.Formula(formula), data=r_df, method="nearest",
                distance="scaled_euclidean", ratio=ratio, replace=replace,
            )
        matched_r = matchit.match_data(m)
        with pandas2ri.converter.context():
            matched_df = ro.conversion.rpy2py(matched_r)
        if "subclass" in matched_df.columns:
            matched_df = matched_df.rename(columns={"subclass": "match_group"})
    except Exception as e:
        warnings.warn(f"MatchIt scaled Euclidean failed: {e}")
        return MethodResult(name="MatchIt scaled Euclidean", effect_size=float("nan"), runtime_seconds=time.time()-t0)
    return MethodResult(
        name="MatchIt scaled Euclidean", effect_size=float("nan"), matched_df=matched_df,
        runtime_seconds=time.time()-t0,
        n_treated=int((matched_df[treatment] == 1).sum()),
        n_control=int((matched_df[treatment] == 0).sum()),
    )


# ---------------------------------------------------------------------------
# MatchIt via rpy2 — Mahalanobis matching
# ---------------------------------------------------------------------------

def run_matchit_mahalanobis(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    covariates: list[str],
    formula: str | None = None,
    caliper: float | None = None,
    ratio: int = 1,
    replace: bool = False,
) -> MethodResult:
    """Mahalanobis matching using R's MatchIt via rpy2.

    Matches the simulation notebook: distance='mahalanobis', method='nearest'.
    """
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.packages import importr

    t0 = time.time()

    df = _impute_for_matchit(df, covariates)

    with pandas2ri.converter.context():
        r_df = ro.conversion.py2rpy(df)

    matchit = importr("MatchIt")

    if formula is None:
        rhs = " + ".join(covariates)
        formula = f"{treatment} ~ {rhs}"

    r_formula = ro.Formula(formula)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mah_caliper = None
            if caliper is not None:
                mah_caliper = ro.FloatVector([float(caliper)])
                mah_caliper.names = ro.StrVector(["distance"])
            match_args = dict(
                formula=r_formula,
                data=r_df,
                distance="mahalanobis",
                method="nearest",
                std_caliper=False,
                ratio=ratio,
                replace=replace,
            )
            if mah_caliper is not None:
                match_args["caliper"] = mah_caliper
            m = matchit.matchit(**match_args)
    except Exception as e:
        warnings.warn(f"MatchIt Mahalanobis failed: {e}")
        return MethodResult(
            name="MatchIt Mahalanobis",
            effect_size=float("nan"),
            runtime_seconds=time.time() - t0,
        )

    matched_r = matchit.match_data(m)

    with pandas2ri.converter.context():
        matched_df = ro.conversion.rpy2py(matched_r)

    if "subclass" in matched_df.columns:
        matched_df = matched_df.rename(columns={"subclass": "match_group"})

    n_treated = int((matched_df[treatment] == 1).sum())
    n_control = int((matched_df[treatment] == 0).sum())

    return MethodResult(
        name="MatchIt Mahalanobis",
        effect_size=float("nan"),
        matched_df=matched_df,
        runtime_seconds=time.time() - t0,
        n_treated=n_treated,
        n_control=n_control,
    )


# ---------------------------------------------------------------------------
# MatchIt via rpy2 — Mahalanobis hybrid (PS distance + mahvars)
# ---------------------------------------------------------------------------

def run_matchit_maha_hybrid(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    covariates: list[str],
    formula: str | None = None,
    caliper: float = 0.2,
    ratio: int = 1,
    replace: bool = False,
) -> MethodResult:
    """MatchIt Mahalanobis hybrid: PS-based matching with mahvars refinement.

    Matches the simulation notebook's run_mahamatchit():
    distance='glm', link='linear.logit', mahvars=~covariates, caliper=0.2, std.caliper=TRUE.
    """
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.packages import importr

    t0 = time.time()

    df = _impute_for_matchit(df, covariates)

    with pandas2ri.converter.context():
        r_df = ro.conversion.py2rpy(df)

    matchit = importr("MatchIt")

    if formula is None:
        rhs = " + ".join(covariates)
        formula = f"{treatment} ~ {rhs}"

    r_formula = ro.Formula(formula)
    mahvars_formula = ro.Formula("~ " + " + ".join(covariates))

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = matchit.matchit(
                r_formula,
                data=r_df,
                distance="glm",
                link="linear.logit",
                method="nearest",
                caliper=caliper,
                std_caliper=True,
                mahvars=mahvars_formula,
                ratio=ratio,
                replace=replace,
            )
    except Exception as e:
        warnings.warn(f"MatchIt Mahalanobis hybrid failed: {e}")
        return MethodResult(
            name="MatchIt + Mahalanobis",
            effect_size=float("nan"),
            matched_df=None,
            runtime_seconds=time.time() - t0,
        )

    matched_r = matchit.match_data(m)

    with pandas2ri.converter.context():
        matched_df = ro.conversion.rpy2py(matched_r)

    if "subclass" in matched_df.columns:
        matched_df = matched_df.rename(columns={"subclass": "match_group"})
    matched_df = matched_df.drop(columns=["weights"], errors="ignore")

    n_treated = int((matched_df[treatment] == 1).sum())
    n_control = int((matched_df[treatment] == 0).sum())

    return MethodResult(
        name="MatchIt + Mahalanobis",
        effect_size=float("nan"),
        matched_df=matched_df,
        runtime_seconds=time.time() - t0,
        n_treated=n_treated,
        n_control=n_control,
    )


# ---------------------------------------------------------------------------
# McNemar OR from matched pairs (binary outcomes)
# ---------------------------------------------------------------------------

def mcnemar_or(matched_df: pd.DataFrame, treatment: str = "exposure_status", outcome: str = "outcome") -> dict:
    """Compute McNemar's paired odds ratio from 1:1 matched pairs."""
    from scipy.stats import chi2

    treated = matched_df[matched_df[treatment] == 1].set_index("match_group")
    control = matched_df[matched_df[treatment] == 0].set_index("match_group")

    common = treated.index.intersection(control.index)
    t = treated.loc[common, outcome].astype(float)
    c = control.loc[common, outcome].astype(float)

    b = int(((t == 1) & (c == 0)).sum())
    c_count = int(((t == 0) & (c == 1)).sum())

    or_val = b / max(c_count, 1e-10)

    if b > 0 and c_count > 0:
        log_or = np.log(or_val)
        se = np.sqrt(1.0 / b + 1.0 / c_count)
        z = 1.96
        ci_lo = np.exp(log_or - z * se)
        ci_hi = np.exp(log_or + z * se)
        chi2_stat = (abs(b - c_count) - 1) ** 2 / max(b + c_count, 1e-10)
        p_value = 1 - chi2.cdf(chi2_stat, df=1)
    else:
        ci_lo, ci_hi, p_value = float("nan"), float("nan"), float("nan")

    return {
        "or": or_val,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "p_value": p_value,
        "discordant_b": b,
        "discordant_c": c_count,
        "n_pairs": len(common),
    }


# ---------------------------------------------------------------------------
# Conditional logistic regression from matched pairs (binary outcomes)
# ---------------------------------------------------------------------------

def conditional_logistic_or(matched_df: pd.DataFrame, treatment: str = "exposure_status",
                            outcome: str = "outcome") -> dict:
    """Estimate OR from matched data using conditional logistic regression (Cox PH trick)."""
    try:
        from statsmodels.duration.hazard_regression import PHReg
    except ImportError:
        try:
            from statsmodels.phreg import PHReg
        except ImportError:
            warnings.warn("statsmodels PHReg not available; skipping CLR")
            return {"or": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"), "n_pairs": 0}

    df = matched_df[matched_df["match_group"].notna()].copy()
    df = df.sort_values("match_group")

    groups = df["match_group"].unique()
    records = []
    for g in groups:
        grp = df[df["match_group"] == g]
        t_row = grp[grp[treatment] == 1]
        c_row = grp[grp[treatment] == 0]
        if len(t_row) != 1 or len(c_row) != 1:
            continue
        t_out = int(t_row[outcome].values[0])
        c_out = int(c_row[outcome].values[0])
        records.append({"pair": g, "treatment": 1, "event": t_out, "time": 1})
        records.append({"pair": g, "treatment": 0, "event": c_out, "time": 1})

    n_pairs = len(records) // 2
    if n_pairs < 5:
        warnings.warn("Too few matched pairs for CLR")
        return {"or": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"), "n_pairs": n_pairs}

    pair_df = pd.DataFrame(records)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = PHReg(
                endog=pair_df["time"].values,
                exog=pair_df[["treatment"]].values,
                status=pair_df["event"].values,
                ties="breslow",
            )
            result = model.fit(disp=0)
            beta = result.params[0]
            se = result.bse[0]
            z = 1.96
            or_val = np.exp(beta)
            ci_lo = np.exp(beta - z * se)
            ci_hi = np.exp(beta + z * se)
    except Exception as e:
        warnings.warn(f"CLR failed: {e}")
        return {"or": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"), "n_pairs": n_pairs}

    return {"or": or_val, "ci_lo": ci_lo, "ci_hi": ci_hi, "n_pairs": n_pairs}


# ---------------------------------------------------------------------------
# ATT estimation via OLS with clustered SEs (continuous outcomes)
# ---------------------------------------------------------------------------

def estimate_att_ols(
    matched_df: pd.DataFrame,
    outcome_col: str = "outcome",
    treatment_col: str = "exposure_status",
    subclass_col: str = "match_group",
) -> dict:
    """Estimate ATT from matched data using OLS with cluster-robust SEs.

    Mirrors the simulation notebook's estimate_att_post_matching():
    - OLS: outcome ~ treatment + covariates, clustered by match_group
    - For simplicity, we use outcome ~ treatment with clustered SEs
      (covariate adjustment already done via matching)

    Returns dict with att, se, ci_lo, ci_hi, t_stat, p_value.
    """
    import statsmodels.api as sm

    df = matched_df[matched_df[subclass_col].notna()].copy()

    y = df[outcome_col].astype(float).values
    X = df[[treatment_col]].astype(float).values
    X = sm.add_constant(X)
    clusters = df[subclass_col].values

    try:
        model = sm.OLS(y, X)
        result = model.fit(cov_type="cluster", cov_kwds={"groups": clusters})

        # treatment coefficient is index 1 (after constant)
        att = result.params[1]
        se = result.bse[1]
        ci = result.conf_int()
        if hasattr(ci, 'iloc'):
            ci_lo = ci.iloc[1, 0]
            ci_hi = ci.iloc[1, 1]
        else:
            ci_lo = ci[1, 0]
            ci_hi = ci[1, 1]
        t_stat = result.tvalues[1]
        p_value = result.pvalues[1]
    except Exception as e:
        warnings.warn(f"OLS ATT estimation failed: {e}")
        att = df[df[treatment_col] == 1][outcome_col].mean() - df[df[treatment_col] == 0][outcome_col].mean()
        return {"att": att, "se": float("nan"), "ci_lo": float("nan"),
                "ci_hi": float("nan"), "t_stat": float("nan"), "p_value": float("nan")}

    return {
        "att": att,
        "se": se,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "t_stat": t_stat,
        "p_value": p_value,
    }


# ---------------------------------------------------------------------------
# SMD computation helper
# ---------------------------------------------------------------------------

def compute_smd(
    df: pd.DataFrame,
    treatment: str,
    covariates: list[str],
    matched_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute standardised mean differences before and optionally after matching."""
    rows = []
    for col in covariates:
        series = df[col]
        is_obj = series.dtype == "object" or series.dtype.name == "category"
        if is_obj:
            series = pd.to_numeric(series, errors="coerce")
        is_binary_or_cat = is_obj or series.nunique() <= 2

        if is_binary_or_cat:
            def _smd(data, _col=col):
                s = data[_col]
                if s.dtype == "object" or s.dtype.name == "category":
                    s = pd.to_numeric(s, errors="coerce")
                mask = data[treatment] == 1
                t = s[mask]
                c = s[~mask]
                p_t = t.mean()
                p_c = c.mean()
                pooled = np.sqrt((p_t * (1 - p_t) + p_c * (1 - p_c)) / 2)
                return abs(p_t - p_c) / max(pooled, 1e-10)
        else:
            def _smd(data, _col=col):
                mask = data[treatment] == 1
                t = data.loc[mask, _col]
                c = data.loc[~mask, _col]
                pooled = np.sqrt((t.std() ** 2 + c.std() ** 2) / 2)
                return abs(t.mean() - c.mean()) / max(pooled, 1e-10)

        row = {"feature": col, "SMD_before": _smd(df)}
        if matched_df is not None:
            row["SMD_after"] = _smd(matched_df)
        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Per-match quality metrics (MAPD and PM)
# ---------------------------------------------------------------------------

def match_quality_metrics(
    matched_df: pd.DataFrame,
    numeric: list[str],
    categorical: list[str],
    full_df: pd.DataFrame | None = None,
    treatment: str = "exposure_status",
) -> dict:
    """Compute normalized per-match-pair quality metrics.

    Numeric: Max absolute standardized difference per pair (|treated - control|
    normalised by feature range from the full population), averaged across pairs.
    Categorical: Mismatch rate per pair (proportion of categorical features that
    differ), averaged across pairs.

    Both metrics are on a [0, 1] scale when features are well-behaved.

    Returns dict with:
    - mapd: Mean (across pairs) of max standardised absolute difference (numeric)
    - pm: Mean (across pairs) of categorical mismatch rate
    - mapd_per_feature: dict[feature -> mean standardised absolute difference]
    - pm_per_feature: dict[feature -> proportion mismatched]
    """
    ref = full_df if full_df is not None else matched_df

    treated = matched_df[matched_df[treatment] == 1].copy()
    control = matched_df[matched_df[treatment] == 0].copy()

    if "match_group" not in matched_df.columns:
        return {"mapd": float("nan"), "pm": float("nan"),
                "mapd_per_feature": {}, "pm_per_feature": {}}

    control_groups = control.groupby("match_group")

    feat_ranges = {}
    for feat in numeric:
        vals = pd.to_numeric(ref[feat], errors="coerce").dropna()
        r = vals.max() - vals.min()
        feat_ranges[feat] = r if r > 0 else 1.0

    mapd_per_feature = {}
    pm_per_feature = {}
    per_pair_mapd = []
    per_pair_pm = []

    for feat in numeric:
        feat_diffs = []
        for _, t_row in treated.iterrows():
            mg = t_row["match_group"]
            if mg not in control_groups.groups:
                continue
            c_rows = control_groups.get_group(mg)
            pair_diffs = []
            for _, c_row in c_rows.iterrows():
                tv = pd.to_numeric(t_row[feat], errors="coerce")
                cv = pd.to_numeric(c_row[feat], errors="coerce")
                if pd.notna(tv) and pd.notna(cv):
                    pair_diffs.append(abs(tv - cv) / feat_ranges[feat])
            if pair_diffs:
                feat_diffs.append(float(np.mean(pair_diffs)))
        mapd_per_feature[feat] = float(np.mean(feat_diffs)) if feat_diffs else float("nan")

    for feat in categorical:
        feat_mismatches = []
        for _, t_row in treated.iterrows():
            mg = t_row["match_group"]
            if mg not in control_groups.groups:
                continue
            c_rows = control_groups.get_group(mg)
            pair_mismatches = []
            for _, c_row in c_rows.iterrows():
                tv = str(t_row[feat])
                cv = str(c_row[feat])
                pair_mismatches.append(0.0 if tv == cv else 1.0)
            if pair_mismatches:
                feat_mismatches.append(float(np.mean(pair_mismatches)))
        pm_per_feature[feat] = float(np.mean(feat_mismatches)) if feat_mismatches else float("nan")

    treated_groups = treated.set_index("match_group")
    for mg in treated_groups.index:
        if mg not in control_groups.groups:
            continue
        c_rows = control_groups.get_group(mg)
        t_row = treated_groups.loc[mg]
        if isinstance(t_row, pd.DataFrame):
            t_row = t_row.iloc[0]

        for _, c_row in c_rows.iterrows():
            num_diffs = []
            for feat in numeric:
                tv = pd.to_numeric(t_row[feat], errors="coerce")
                cv = pd.to_numeric(c_row[feat], errors="coerce")
                if pd.notna(tv) and pd.notna(cv):
                    num_diffs.append(abs(tv - cv) / feat_ranges[feat])
            if num_diffs:
                per_pair_mapd.append(float(np.max(num_diffs)))

            cat_mismatches = []
            for feat in categorical:
                tv = str(t_row[feat])
                cv = str(c_row[feat])
                cat_mismatches.append(0.0 if tv == cv else 1.0)
            if cat_mismatches:
                per_pair_pm.append(float(np.mean(cat_mismatches)))

    return {
        "mapd": float(np.mean(per_pair_mapd)) if per_pair_mapd else float("nan"),
        "pm": float(np.mean(per_pair_pm)) if per_pair_pm else float("nan"),
        "mapd_per_feature": mapd_per_feature,
        "pm_per_feature": pm_per_feature,
    }


# ---------------------------------------------------------------------------
# Two-stage RDMatcher pipeline: Propensity -> Gower refinement
# ---------------------------------------------------------------------------

def build_gower_weights_dict(
    numeric: list[str],
    categorical: list[str],
    numeric_weight: float = 1.0,
    categorical_weight: float = 1.0,
) -> dict:
    """Build a gower_weights dict: numeric features get numeric_weight, categorical get categorical_weight."""
    w = {}
    for f in numeric:
        w[f] = numeric_weight
    for f in categorical:
        w[f] = categorical_weight
    return w


def run_two_stage_rdmatcher(
    df: pd.DataFrame,
    numeric: list[str],
    categorical: list[str],
    formula: str,
    ps_threshold: float,
    gower_threshold: float,
    n_neighbors_stage1: int = 3,
    k_candidates: int = 500,
    gower_weights: dict | None = None,
    ps_distance_metric: str = "gower",
) -> dict:
    """Two-stage RDM pipeline: PS matching (1:K) -> Gower refinement (1:1)."""
    from rdmatcher import RDMatcher

    covariates = numeric + categorical

    # --- Stage 1: Propensity matching ---
    matcher1 = RDMatcher(
        pop_df=df,
        patient_id_col="patient_id",
        exposure_status="exposure_status",
        features_numeric=numeric,
        features_categorical=categorical,
        process_features=False,
        onehot=False,
        debug=False,
        log_to_console=False,
    )
    matcher1.fit_propensity_model(formula=formula, random_state=404)

    t0_s1 = time.time()
    try:
        s1_matched = matcher1.rare_matching(
            threshold=ps_threshold,
            n_neighbors=n_neighbors_stage1,
            k_candidates=k_candidates,
            method="propensity",
            distance_metric=ps_distance_metric,
            global_optimal=True,
            competitive_match=True,
            diagnostics=False,
            return_matched_data=True,
        )
    except Exception as e:
        return {"error": f"Stage 1 failed: {e}", "total_runtime": time.time() - t0_s1,
                "n_stage1_matched": 0, "n_stage2_matched": 0}
    s1_runtime = time.time() - t0_s1

    s1_matched = s1_matched[s1_matched["match_group"].notna()].copy()
    n_stage1 = s1_matched[s1_matched["exposure_status"] == 1].shape[0]
    if n_stage1 == 0:
        return {"error": "Stage 1 matched 0 pairs", "total_runtime": s1_runtime,
                "n_stage1_matched": 0, "n_stage2_matched": 0}

    # --- Stage 2: Gower refinement on stage-1 output ---
    matcher2 = RDMatcher(
        pop_df=s1_matched,
        patient_id_col="patient_id",
        exposure_status="exposure_status",
        features_numeric=numeric,
        features_categorical=categorical,
        process_features=False,
        onehot=False,
        debug=False,
        log_to_console=False,
    )

    t0_s2 = time.time()
    try:
        s2_matched = matcher2.rare_matching(
            threshold=gower_threshold,
            n_neighbors=1,
            k_candidates=k_candidates,
            method="multi",
            distance_metric="gower",
            global_optimal=True,
            competitive_match=True,
            diagnostics=False,
            return_matched_data=True,
            gower_weights=gower_weights,
        )
    except Exception as e:
        return {"error": f"Stage 2 failed: {e}", "total_runtime": s1_runtime + time.time() - t0_s2,
                "n_stage1_matched": n_stage1, "n_stage2_matched": 0}
    s2_runtime = time.time() - t0_s2

    s2_matched = s2_matched[s2_matched["match_group"].notna()].copy()
    n_stage2 = s2_matched[s2_matched["exposure_status"] == 1].shape[0]

    return {
        "matched_df": s2_matched,
        "stage1_runtime": s1_runtime,
        "stage2_runtime": s2_runtime,
        "total_runtime": s1_runtime + s2_runtime,
        "n_stage1_matched": n_stage1,
        "n_stage2_matched": n_stage2,
    }


def run_two_stage_prefitted(
    matcher_prefitted,
    df: pd.DataFrame,
    numeric: list[str],
    categorical: list[str],
    ps_threshold: float,
    gower_threshold: float,
    n_neighbors_stage1: int = 3,
    k_candidates: int = 500,
    gower_weights: dict | None = None,
    ps_distance_metric: str = "gower",
) -> dict:
    """Two-stage RDM pipeline with a pre-fitted propensity matcher for stage 1."""
    from rdmatcher import RDMatcher

    t0_s1 = time.time()
    try:
        s1_matched = matcher_prefitted.rare_matching(
            threshold=ps_threshold,
            n_neighbors=n_neighbors_stage1,
            k_candidates=k_candidates,
            method="propensity",
            distance_metric=ps_distance_metric,
            global_optimal=True,
            competitive_match=True,
            diagnostics=False,
            return_matched_data=True,
        )
    except Exception as e:
        return {"error": f"Stage 1 failed: {e}", "total_runtime": time.time() - t0_s1,
                "n_stage1_matched": 0, "n_stage2_matched": 0}
    s1_runtime = time.time() - t0_s1

    s1_matched = s1_matched[s1_matched["match_group"].notna()].copy()
    n_stage1 = s1_matched[s1_matched["exposure_status"] == 1].shape[0]
    if n_stage1 == 0:
        return {"error": "Stage 1 matched 0 pairs", "total_runtime": s1_runtime,
                "n_stage1_matched": 0, "n_stage2_matched": 0}

    matcher2 = RDMatcher(
        pop_df=s1_matched,
        patient_id_col="patient_id",
        exposure_status="exposure_status",
        features_numeric=numeric,
        features_categorical=categorical,
        process_features=False,
        onehot=False,
        debug=False,
        log_to_console=False,
    )

    t0_s2 = time.time()
    try:
        s2_matched = matcher2.rare_matching(
            threshold=gower_threshold,
            n_neighbors=1,
            k_candidates=k_candidates,
            method="multi",
            distance_metric="gower",
            global_optimal=True,
            competitive_match=True,
            diagnostics=False,
            return_matched_data=True,
            gower_weights=gower_weights,
        )
    except Exception as e:
        return {"error": f"Stage 2 failed: {e}", "total_runtime": s1_runtime + time.time() - t0_s2,
                "n_stage1_matched": n_stage1, "n_stage2_matched": 0}
    s2_runtime = time.time() - t0_s2

    s2_matched = s2_matched[s2_matched["match_group"].notna()].copy()
    n_stage2 = s2_matched[s2_matched["exposure_status"] == 1].shape[0]

    return {
        "matched_df": s2_matched,
        "stage1_runtime": s1_runtime,
        "stage2_runtime": s2_runtime,
        "total_runtime": s1_runtime + s2_runtime,
        "n_stage1_matched": n_stage1,
        "n_stage2_matched": n_stage2,
    }
