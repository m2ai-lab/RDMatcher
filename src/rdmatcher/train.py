import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import GridSearchCV
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Literal, Optional, Dict, List, Tuple
import logging
import warnings
from sklearn.exceptions import ConvergenceWarning





logger = logging.getLogger('rdmatcher.train')


def _get_encoded_features(df, features, drop_first=True):
    """Helper to One-Hot Encode categorical features just for the model matrix."""
    X = df[features].copy()
    cat_cols = X.select_dtypes(include=['object', 'category']).columns

    if len(cat_cols) > 0:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=drop_first, dtype=float)

    return X


def build_psm_model_matrix(df, original_to_processed: Optional[Dict[str, List[str]]], processed_df, formula_terms: Optional[List[Tuple[str, ...]]], processed_to_original: Optional[Dict[str, str]] = None):
    """
    Build a modeling DataFrame (X) for propensity score modeling.
    - original_to_processed: dict mapping original feature -> list of processed cols
    - processed_df: DataFrame with processed columns (one-hoted etc)
    - formula_terms: list of tuples (terms parsed) from formula.parse_formula
    Returns: X_model (DataFrame), metadata: mapping info
    """
    processed_to_original = processed_to_original or {}
    original_to_processed = original_to_processed or {}

    specified_bases = {}
    for term in formula_terms:
        for name in term:
            if name in processed_to_original:
                base = processed_to_original[name]
                specified_bases.setdefault(base, set()).add(('child', name))
            elif name in original_to_processed:
                base = name
                specified_bases.setdefault(base, set()).add(('base', name))
            else:
                matches = [p for p in processed_df.columns if p.startswith(f"{name}_")]
                if matches:
                    base = name
                    specified_bases.setdefault(base, set()).add(('base', name))
                else:
                    raise ValueError(f"Unknown feature '{name}' when resolving formula. Valid originals: {sorted(list(original_to_processed.keys()))}")

    for base, specs in specified_bases.items():
        has_base = any(s[0] == 'base' for s in specs)
        has_child = any(s[0] == 'child' for s in specs)
        if has_base and has_child:
            raise ValueError(
                f"Formula mixes parent and child specificity for feature '{base}'. "
                "Use either the original feature name (e.g., 'sex') to apply the same relationship to all children, "
                "or explicitly specify a single child processed column name, but not both."
            )

    used = []
    mapping = {}
    for term in formula_terms:
        if len(term) == 1:
            name = term[0]
            if name in original_to_processed:
                procs = original_to_processed.get(name, [])
            elif name in processed_to_original:
                procs = [name]
            else:
                procs = [p for p in processed_df.columns if p.startswith(f"{name}_")]
            if not procs:
                raise ValueError(f"Unknown original feature '{name}' when resolving formula. Valid: {sorted(list(original_to_processed.keys()))}")
            mapping.setdefault(name, []).extend(procs)
            for p in procs:
                if p not in used:
                    used.append(p)

    X = pd.DataFrame(index=processed_df.index)
    for col in used:
        if col not in processed_df.columns:
            raise ValueError(f"Processed column '{col}' not found in processed dataframe")
        X[col] = processed_df[col].astype(float)

    interaction_cols = []
    for term in formula_terms:
        if len(term) <= 1:
            continue
        lists = []
        for orig in term:
            if orig in processed_df.columns:
                procs = [orig]
            else:
                procs = original_to_processed.get(orig)
            if procs is None or len(procs) == 0:
                procs = [p for p in processed_df.columns if p.startswith(f"{orig}_")]
            if not procs:
                raise ValueError(f"Unknown original feature '{orig}' when resolving interaction {term}")
            lists.append(procs)
        import itertools
        for combo in itertools.product(*lists):
            name = ':'.join(combo)
            interaction_cols.append(name)
            vals = processed_df[combo[0]].astype(float)
            for c in combo[1:]:
                vals = vals * processed_df[c].astype(float)
            X[name] = vals

    X = X.fillna(0.0)
    meta = {
        'main_processed': used,
        'interaction_cols': interaction_cols,
        'mapping': mapping
    }
    try:
        logger.info(f"PSM design matrix: main={len(used)} cols, interactions={len(interaction_cols)} cols")
        logger.debug(f"PSM mapping (original->processed): {mapping}")
        logger.debug(f"PSM interaction columns: {interaction_cols}")
    except Exception:
        pass
    return X, meta















def propensity_logits_simple(df_in, exposure_status, all_features,
                             downsample_ratio=None,
                             n_bags=None,
                             n_hard_mining=None,
                             random_state=404,
                             debug=False,
                             original_to_processed: Optional[Dict[str, List[str]]] = None,
                             processed_to_original: Optional[Dict[str, str]] = None,
                             formula_terms: Optional[List[Tuple[str, ...]]] = None,
                             estimator=None,
                             estimator_kwargs: Optional[Dict] = None,
                             **kwargs):
    """
    Calculate propensity scores/logits with support for Bagging, Downsampling,
    and Hard Negative Mining within a unified API.

    Strategies (Priority Order):
    1. n_hard_mining: Uses 2-step mining (Scout -> Filter -> Final Refined Model).
    2. n_bags: Uses Ensemble Bagging (Average of k models).
    3. downsample_ratio: Uses simple random downsampling (Train subset -> Predict All).
    4. Default: Standard Logistic Regression on full dataset.

    Args:
        df_in (pd.DataFrame): Input DataFrame.
        exposure_status (str): Binary column name.
        all_features (list): List of feature names.
        downsample_ratio (float): Ratio of Controls:Cases for simple downsampling.
                                  e.g., 5.0 means 5 controls per 1 case.
        n_bags (int): Number of bags for ensemble.
        n_hard_mining (int): Number of "hard" controls to mine for the final model.
                             e.g., 2000.
        **kwargs: Passed to LogisticRegression (e.g., class_weight).

    Returns:
        tuple: (df_in with columns added, Series of logits)
    """
    if debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    strategies = [n_hard_mining, n_bags, downsample_ratio]
    active_strategies = [s for s in strategies if s is not None and s > 0]
    if len(active_strategies) > 1:
        raise ValueError("Only one of n_hard_mining, n_bags, or downsample_ratio can be active at a time.")

    model_kwargs = dict(kwargs)
    if estimator_kwargs:
        model_kwargs.update(estimator_kwargs)

    model_kwargs.setdefault('max_iter', 10_000)
    model_kwargs.setdefault('class_weight', None)
    model_kwargs.setdefault('solver', 'newton-cholesky')
    model_kwargs.pop('penalty', None)
    model_kwargs.setdefault('C', float('inf'))
    model_kwargs.setdefault('tol', 1e-10)

    df = df_in.copy()
    if estimator is None:
        estimator = LogisticRegression

    convergence_warnings = []
    fitted_estimators = []

    def _fit_checked(model, X, y):
        """Fit a model and validate convergence and finite coefficients."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always', ConvergenceWarning)
            model.fit(X, y)
        messages = [str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)]
        convergence_warnings.extend(messages)
        if messages:
            raise RuntimeError("Propensity model failed to converge: " + "; ".join(messages))
        coefficients = getattr(model, 'coef_', None)
        if coefficients is not None and not np.isfinite(np.asarray(coefficients, dtype=float)).all():
            raise RuntimeError("Propensity model produced non-finite coefficients.")
        fitted_estimators.append(model)
        return model

    def _make_estimator(rs=None):
        if isinstance(estimator, type):
            try:
                if rs is not None:
                    return estimator(random_state=rs, **model_kwargs)
                return estimator(**model_kwargs)
            except TypeError:
                return estimator(**model_kwargs)
        else:
            return clone(estimator)

    if formula_terms:
        if original_to_processed is None:
            raise ValueError("original_to_processed mapping required when formula_terms provided")
        X_all, meta = build_psm_model_matrix(df, original_to_processed, df, formula_terms, processed_to_original=processed_to_original)
    else:
        X_all = _get_encoded_features(df, all_features)
        meta = {'main_processed': list(X_all.columns), 'interaction_cols': [], 'mapping': {}}

    y_all = df[exposure_status].values

    idx_cases = np.where(y_all == 1)[0]
    idx_controls = np.where(y_all == 0)[0]

    probs = None
    logit = None

    if n_hard_mining and n_hard_mining > 0:
        logger.info(f"Strategy: Hard Negative Mining (Targeting top {n_hard_mining} hard controls).")

        n_scout = min(len(idx_controls), len(idx_cases) * 10)
        rng = np.random.default_rng(random_state)
        idx_scout = np.concatenate([idx_cases, rng.choice(idx_controls, n_scout, replace=False)])

        scout_clf = _make_estimator(random_state)
        _fit_checked(scout_clf, _safe_slice(X_all, idx_scout), y_all[idx_scout])

        probs_controls = scout_clf.predict_proba(_safe_slice(X_all, idx_controls))[:, 1]

        top_k_idx = np.argsort(probs_controls)[-n_hard_mining:]
        idx_hard = idx_controls[top_k_idx]

        idx_final_train = np.concatenate([idx_cases, idx_hard])
        final_clf = _make_estimator(random_state)
        _fit_checked(final_clf, _safe_slice(X_all, idx_final_train), y_all[idx_final_train])

        if hasattr(final_clf, 'decision_function'):
            logit = final_clf.decision_function(X_all)
            probs = 1.0 / (1.0 + np.exp(-logit))
        else:
            probs = final_clf.predict_proba(X_all)[:, 1]

    elif n_bags and n_bags > 1:
        logger.info(f"Strategy: Ensemble Bagging ({n_bags} bags).")
        probs, logit = _train_bagged_ensemble(X_all, y_all, n_bags, random_state, estimator=estimator, estimator_kwargs=model_kwargs)

    elif downsample_ratio and downsample_ratio > 0:
        logger.info(f"Strategy: Simple Downsampling (Ratio 1:{downsample_ratio}).")

        n_controls_needed = int(len(idx_cases) * downsample_ratio)
        if n_controls_needed < len(idx_controls):
            rng = np.random.default_rng(random_state)
            idx_sampled_controls = rng.choice(idx_controls, n_controls_needed, replace=False)
            idx_train = np.concatenate([idx_cases, idx_sampled_controls])
        else:
            idx_train = np.arange(len(y_all))

        clf = _make_estimator(random_state)
        _fit_checked(clf, _safe_slice(X_all, idx_train), y_all[idx_train])

        probs = clf.predict_proba(X_all)[:, 1]

    else:
        logger.info("Strategy: Standard Logistic Regression (Full Cohort).")
        clf = _make_estimator(random_state)
        _fit_checked(clf, X_all, y_all)
        probs = clf.predict_proba(X_all)[:, 1]

    if logit is None:
        if hasattr(clf, 'decision_function'):
            logit = clf.decision_function(X_all)
            probs = 1.0 / (1.0 + np.exp(-logit))
        else:
            probs = np.clip(np.asarray(probs, dtype=float), 1e-12, 1 - 1e-12)
            logit = np.log(probs / (1.0 - probs))

    probs = np.asarray(probs, dtype=float)
    logit = np.asarray(logit, dtype=float)
    if not np.isfinite(probs).all() or not np.isfinite(logit).all():
        raise RuntimeError("Propensity model produced non-finite scores.")

    fitted = fitted_estimators[-1] if fitted_estimators else None
    meta.update({
        'design_matrix_columns': list(X_all.columns),
        'backend': f'{fitted.__class__.__module__}.{fitted.__class__.__name__}' if fitted is not None else None,
        'solver': getattr(fitted, 'solver', model_kwargs.get('solver')) if fitted is not None else model_kwargs.get('solver'),
        'tolerance': getattr(fitted, 'tol', model_kwargs.get('tol')) if fitted is not None else model_kwargs.get('tol'),
        'max_iter': getattr(fitted, 'max_iter', model_kwargs.get('max_iter')) if fitted is not None else model_kwargs.get('max_iter'),
        'n_iter': np.asarray(getattr(fitted, 'n_iter_', [])).tolist() if fitted is not None else [],
        'iteration_counts': [np.asarray(getattr(model, 'n_iter_', [])).tolist() for model in fitted_estimators],
        'convergence_warnings': convergence_warnings,
        'score_summary': {
            'n': int(logit.size), 'logit_min': float(logit.min()), 'logit_max': float(logit.max()),
            'score_min': float(probs.min()), 'score_max': float(probs.max()),
        },
    })

    df_in['propensity_score'] = probs
    df_in['propensity_logit'] = logit

    return df_in, df_in['propensity_logit'], meta


def _safe_slice(X, indices):
    """Helper to slice X whether it's a DataFrame or numpy array/sparse matrix"""
    if hasattr(X, "iloc"):
        return X.iloc[indices]
    return X[indices]


def _train_bagged_ensemble(X, y, n_bags, random_state, **kwargs):
    """Helper for bagging strategy"""
    estimator = kwargs.pop('estimator', LogisticRegression)
    estimator_kwargs = kwargs.pop('estimator_kwargs', {})
    base_model = estimator(random_state=random_state, **estimator_kwargs)

    idx_cases = np.where(y == 1)[0]
    idx_controls = np.where(y == 0)[0]

    rng = np.random.default_rng(random_state)
    shuffled_controls = rng.permutation(idx_controls)
    control_chunks = np.array_split(shuffled_controls, n_bags)

    total_probs = np.zeros(X.shape[0])
    total_logits = np.zeros(X.shape[0])

    for chunk in control_chunks:
        train_idx = np.concatenate([idx_cases, chunk])
        clf = clone(base_model)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always', ConvergenceWarning)
            clf.fit(_safe_slice(X, train_idx), y[train_idx])
        convergence = [str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)]
        if convergence:
            raise RuntimeError("Propensity bagging model failed to converge: " + "; ".join(convergence))
        if hasattr(clf, 'coef_') and not np.isfinite(np.asarray(clf.coef_, dtype=float)).all():
            raise RuntimeError("Propensity bagging model produced non-finite coefficients.")
        bag_probs = clf.predict_proba(X)[:, 1]
        total_probs += bag_probs
        if hasattr(clf, 'decision_function'):
            total_logits += clf.decision_function(X)
        else:
            bag_probs = np.clip(np.asarray(bag_probs, dtype=float), 1e-12, 1 - 1e-12)
            total_logits += np.log(bag_probs / (1.0 - bag_probs))

    return total_probs / n_bags, total_logits / n_bags






def plot_ps_losses(train_losses, val_losses, auc_score):
    sns.set_theme(style="whitegrid")
    set1_colors = sns.color_palette("Set1")

    plt.figure(figsize=(10, 6))
    sns.lineplot(x=range(len(train_losses)), y=train_losses, label='Training Loss', color=set1_colors[0])
    sns.lineplot(x=range(len(val_losses)), y=val_losses, label='Validation Loss', color=set1_colors[1])
    plt.xlabel('Iteration')
    plt.ylabel('Log loss')
    plt.title(f'Training and validation loss over time (AUC: {auc_score:.3f})')
    plt.legend()
    plt.show()
