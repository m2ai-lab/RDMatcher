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


# from .logger import rdlogger



# logger = rdlogger(__name__, level='INFO')
logger = logging.getLogger('rdmatcher.train')


# -----------------------------------------------------------------------------
# Helper: Safe Encoding (Used by both functions)
# -----------------------------------------------------------------------------
def _get_encoded_features(df, features, drop_first=True):
    """
    Helper to One-Hot Encode categorical features just for the model matrix.
    Returns a dataframe of numeric features ready for Sklearn.
    """
    X = df[features].copy()
    # Select object and category columns
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
    # Build helper lookups for formula name resolution
    processed_to_original = processed_to_original or {}
    original_to_processed = original_to_processed or {}

    # Determine base original name for each identifier used in the formula
    # and detect if the user mixed base and child specificity for the same feature
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
                # Try to match processed columns that start with name + '_'
                matches = [p for p in processed_df.columns if p.startswith(f"{name}_")]
                if matches:
                    # treat as base
                    base = name
                    specified_bases.setdefault(base, set()).add(('base', name))
                else:
                    raise ValueError(f"Unknown feature '{name}' when resolving formula. Valid originals: {sorted(list(original_to_processed.keys()))}")

    # Check for mixed specificity (both base and child used for same original)
    for base, specs in specified_bases.items():
        has_base = any(s[0] == 'base' for s in specs)
        has_child = any(s[0] == 'child' for s in specs)
        if has_base and has_child:
            raise ValueError(
                f"Formula mixes parent and child specificity for feature '{base}'. "
                "Use either the original feature name (e.g., 'sex') to apply the same relationship to all children, "
                "or explicitly specify a single child processed column name, but not both."
            )

    # Collect main-effect processed columns
    used = []
    mapping = {}
    for term in formula_terms:
        if len(term) == 1:
            name = term[0]
            # resolve to list of processed columns
            if name in original_to_processed:
                procs = original_to_processed.get(name, [])
            elif name in processed_to_original:
                procs = [name]
            else:
                # fallback: any processed cols that start with name + '_'
                procs = [p for p in processed_df.columns if p.startswith(f"{name}_")]
            if not procs:
                raise ValueError(f"Unknown original feature '{name}' when resolving formula. Valid: {sorted(list(original_to_processed.keys()))}")
            mapping.setdefault(name, []).extend(procs)
            for p in procs:
                if p not in used:
                    used.append(p)

    X = pd.DataFrame(index=processed_df.index)
    # Add main effects
    for col in used:
        if col not in processed_df.columns:
            raise ValueError(f"Processed column '{col}' not found in processed dataframe")
        X[col] = processed_df[col].astype(float)

    # Add interactions (terms with len>1)
    interaction_cols = []
    for term in formula_terms:
        if len(term) <= 1:
            continue
        # Expand list of processed columns for each component
        lists = []
        for orig in term:
            # If user provided a processed child name, use it directly
            if orig in processed_df.columns:
                procs = [orig]
            else:
                procs = original_to_processed.get(orig)
            if procs is None or len(procs) == 0:
                # fallback: any processed cols that start with orig + '_'
                procs = [p for p in processed_df.columns if p.startswith(f"{orig}_")]
            if not procs:
                raise ValueError(f"Unknown original feature '{orig}' when resolving interaction {term}")
            lists.append(procs)
        # cross product
        import itertools
        for combo in itertools.product(*lists):
            name = ':'.join(combo)
            interaction_cols.append(name)
            # compute product
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
    # Log summary of the expansion for transparency
    try:
        logger.info(f"PSM design matrix: main={len(used)} cols, interactions={len(interaction_cols)} cols")
        logger.debug(f"PSM mapping (original->processed): {mapping}")
        logger.debug(f"PSM interaction columns: {interaction_cols}")
    except Exception:
        pass
    return X, meta

        # Work on a copy of the raw data
    # df = self.pop.copy()

    # # Build the preprocessor using our custom pipeline.
    # preprocessor = build_preprocessing_pipeline(
    #     features_numeric=self.features_numeric,
    #     features_categorical=self.features_categorical,
    #     features_log=self.features_log,
    #     features_bin=self.features_bin,
    #     bin_method=self.bin_method,
    #     bin_width=self.bin_width,
    #     onehot=self.onehot
    # )

    # Apply preprocessing and get a final DataFrame.
    # self.logger.info("Applying preprocessing pipeline to the dataset.")
    # self.pop_processed = apply_preprocessing_pipeline(df, preprocessor, self.patient_id_col, self.exposure_status)



# ------------------------------
# Propensity Score Calculation Functions
# ------------------------------
# def propensity_logits_simple(df_in, exposure_status, all_features, random_state=404, debug=False, **kwargs):
#     """
#     Calculate propensity scores using logistic regression.

#     This function takes a DataFrame and calculates propensity scores for a 
#     specified binary exposure variable using a logistic regression model. 
#     The propensity scores represent the probability of the exposure variable 
#     being 1, given the other features in the DataFrame.

#     Args:
#         df_in (pd.DataFrame): Input DataFrame containing the features and the 
#                               binary exposure variable.
#         exposure_status (str): The name of the column in the DataFrame that 
#                                represents the binary exposure variable. 

#     Returns:
#         tuple: A tuple containing:
#             - pd.DataFrame: A copy of the input DataFrame with an additional 
#                             column 'propensity_logit' containing the calculated 
#                             propensity logit.
#             - pd.Series: A Series containing the propensity logits.

#     Notes:
#         - The logistic regression model is initialized with a random state of 404 
#           and uses class balancing to handle imbalanced datasets.
#     """
#     if debug:
#         logger.setLevel(logging.DEBUG)
#     else:
#         logger.setLevel(logging.INFO)
#     logger.info("Calculating propensity logits using simple logistic regression.")

#     df = df_in.copy()
#     # X = df[all_features]
#     X = _get_encoded_features(df, all_features)
#     y = df[exposure_status]
#     logistic = LogisticRegression(random_state=random_state, max_iter=2000)
#     logistic.fit(X, y)
    
#     probs = logistic.predict_proba(X)[:, 1]
#     eps = 1e-12
#     logit = np.log(probs.clip(eps, 1 - eps) / (1 - probs.clip(eps, 1 - eps)))
#     df['propensity_score'] = probs
#     df['propensity_logit'] = logit
#     logger.info("Propensity logits calculated successfully.")

#     return df, df['propensity_logit']


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

    # make sure only one strategy is active
    strategies = [n_hard_mining, n_bags, downsample_ratio]
    active_strategies = [s for s in strategies if s is not None and s > 0]
    if len(active_strategies) > 1:
        raise ValueError("Only one of n_hard_mining, n_bags, or downsample_ratio can be active at a time.")

    # allow max_iter and other logistic params to be passed via kwargs
    if 'max_iter' not in kwargs:
        kwargs['max_iter'] = 2000
    if 'class_weight' not in kwargs:
        # not balanced by default since some strategies handle imbalance differently, but you can set it if you want
        kwargs['class_weight'] = None
    if 'solver' not in kwargs:
        kwargs['solver'] = 'lbfgs'
    if 'penalty' not in kwargs:
        kwargs['penalty'] = None
    if 'tol' not in kwargs:
        kwargs['tol'] = 1e-4

    # 1. Build model matrix depending on formula_terms and mapping
    df = df_in.copy()
    # default estimator handling: accept class or instance
    estimator_kwargs = estimator_kwargs or {}
    if estimator is None:
        estimator = LogisticRegression

    def _make_estimator(rs=None):
        # If estimator is a class, instantiate with random_state if possible
        if isinstance(estimator, type):
            try:
                if rs is not None:
                    return estimator(random_state=rs, **estimator_kwargs)
                return estimator(**estimator_kwargs)
            except TypeError:
                # estimator doesn't accept random_state
                return estimator(**estimator_kwargs)
        else:
            # estimator provided as instance: clone for each use
            return clone(estimator)

    # If formula_terms provided, build X_model using mapping
    if formula_terms:
        if original_to_processed is None:
            raise ValueError("original_to_processed mapping required when formula_terms provided")
        X_all, meta = build_psm_model_matrix(df, original_to_processed, df, formula_terms, processed_to_original=processed_to_original)
    else:
        X_all = _get_encoded_features(df, all_features)
        meta = {'main_processed': list(X_all.columns), 'interaction_cols': [], 'mapping': {}}

    y_all = df[exposure_status].values
    
    # Identify indices
    idx_cases = np.where(y_all == 1)[0]
    idx_controls = np.where(y_all == 0)[0]
    
    probs = None # To be filled by one of the strategies

    # --- STRATEGY 1: HARD NEGATIVE MINING ---
    if n_hard_mining and n_hard_mining > 0:
        logger.info(f"Strategy: Hard Negative Mining (Targeting top {n_hard_mining} hard controls).")
        
        # A. Scout Step (Train on 1:10 downsample to find the boundary)
        n_scout = min(len(idx_controls), len(idx_cases) * 10)
        rng = np.random.default_rng(random_state)
        idx_scout = np.concatenate([idx_cases, rng.choice(idx_controls, n_scout, replace=False)])
        
        scout_clf = LogisticRegression(random_state=random_state, **kwargs)
        scout_clf.fit(_safe_slice(X_all, idx_scout), y_all[idx_scout])
        
        # Predict on ALL controls to find the hardest ones
        # (We use the scout to scan the massive 1M cohort)
        probs_controls = scout_clf.predict_proba(_safe_slice(X_all, idx_controls))[:, 1]
        
        # Select top hardest controls
        top_k_idx = np.argsort(probs_controls)[-n_hard_mining:]
        idx_hard = idx_controls[top_k_idx]
        
        # B. Final Step (Train on Cases + Hard Controls)
        idx_final_train = np.concatenate([idx_cases, idx_hard])
        final_clf = LogisticRegression(random_state=random_state, **kwargs)
        final_clf.fit(_safe_slice(X_all, idx_final_train), y_all[idx_final_train])
        
        # PREDICT ON EVERYONE
        probs = final_clf.predict_proba(X_all)[:, 1]

    # --- STRATEGY 2: BAGGING ---
    elif n_bags and n_bags > 1:
        logger.info(f"Strategy: Ensemble Bagging ({n_bags} bags).")
        # Uses the helper function defined previously, but passed X_all/y_all
        probs = _train_bagged_ensemble(X_all, y_all, n_bags, random_state, estimator=estimator, estimator_kwargs=estimator_kwargs, **kwargs)

    # --- STRATEGY 3: SIMPLE DOWNSAMPLING ---
    elif downsample_ratio and downsample_ratio > 0:
        logger.info(f"Strategy: Simple Downsampling (Ratio 1:{downsample_ratio}).")
        
        n_controls_needed = int(len(idx_cases) * downsample_ratio)
        if n_controls_needed < len(idx_controls):
            rng = np.random.default_rng(random_state)
            idx_sampled_controls = rng.choice(idx_controls, n_controls_needed, replace=False)
            idx_train = np.concatenate([idx_cases, idx_sampled_controls])
        else:
            idx_train = np.arange(len(y_all)) # Use all if ratio exceeds available data
            
        clf = _make_estimator(random_state)
        clf.fit(_safe_slice(X_all, idx_train), y_all[idx_train])
        
        # PREDICT ON EVERYONE
        # Crucial: We trained on a subset, but we score the whole population
        # so you can match cases to controls that weren't in the training set.
        probs = clf.predict_proba(X_all)[:, 1]

    # --- STRATEGY 4: STANDARD (FULL) ---
    else:
        logger.info("Strategy: Standard Logistic Regression (Full Cohort).")
        clf = _make_estimator(random_state)
        clf.fit(X_all, y_all)
        probs = clf.predict_proba(X_all)[:, 1]

    # 4. Convert to Logits and Assign
    # Using the original index from df_in guarantees alignment
    # Use decision_function directly for numerically precise logit values
    logit = clf.decision_function(X_all)
    probs = 1.0 / (1.0 + np.exp(-logit))
    
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
    
    total_probs = np.zeros(X.shape[0]) # Align with full X rows
    
    # Check if X is sparse/numpy to optimize prediction containers if needed
    # But for simplicity, we predict on full X each time
    
    for chunk in control_chunks:
        train_idx = np.concatenate([idx_cases, chunk])
        clf = clone(base_model)
        clf.fit(_safe_slice(X, train_idx), y[train_idx])
        total_probs += clf.predict_proba(X)[:, 1]
        
    return total_probs / n_bags



# legacy full propensity implementation removed in favor of formula-driven unified API



# ------------------------------
# Plotting Function for Propensity Score Losses
# ------------------------------
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
