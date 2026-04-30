import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import GridSearchCV
from sklearn.base import clone
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Literal
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

    # 1. ENCODING: Must be done on the FULL dataset to ensure consistent columns/alignment
    # We work with indices to avoid the "reset_index" bug.
    df = df_in.copy()
    X_all = _get_encoded_features(df, all_features)
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
        probs = _train_bagged_ensemble(X_all, y_all, n_bags, random_state, **kwargs)

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
            
        clf = LogisticRegression(random_state=random_state, **kwargs)
        clf.fit(_safe_slice(X_all, idx_train), y_all[idx_train])
        
        # PREDICT ON EVERYONE
        # Crucial: We trained on a subset, but we score the whole population
        # so you can match cases to controls that weren't in the training set.
        probs = clf.predict_proba(X_all)[:, 1]

    # --- STRATEGY 4: STANDARD (FULL) ---
    else:
        logger.info("Strategy: Standard Logistic Regression (Full Cohort).")
        clf = LogisticRegression(random_state=random_state, **kwargs)
        clf.fit(X_all, y_all)
        probs = clf.predict_proba(X_all)[:, 1]

    # 4. Convert to Logits and Assign
    # Using the original index from df_in guarantees alignment
    eps = 1e-12
    logit = np.log(probs.clip(eps, 1 - eps) / (1 - probs.clip(eps, 1 - eps)))
    
    df_in['propensity_score'] = probs
    df_in['propensity_logit'] = logit
    
    return df_in, df_in['propensity_logit']


def _safe_slice(X, indices):
    """Helper to slice X whether it's a DataFrame or numpy array/sparse matrix"""
    if hasattr(X, "iloc"):
        return X.iloc[indices]
    return X[indices]


def _train_bagged_ensemble(X, y, n_bags, random_state, **kwargs):
    """Helper for bagging strategy"""
    base_model = LogisticRegression(random_state=random_state, **kwargs)
    
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



def propensity_logits_full(df_in, exposure_status, all_features, random_state=404, max_iter=2000, penalty=None, 
                           solver: Literal['saga', 'liblinear'] = 'saga', debug=False, **kwargs):

    if debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    logger.info("Calculating propensity logits using full logistic regression with interaction terms.")

    degree = kwargs.get('degree', 1) # Default to 1 if not specified (standard interactions often implied 2)
    # If the user specifically asks for interaction terms, usually they mean degree=2
    if 'degree' not in kwargs and 'interaction_only' not in kwargs:
        # Slight safety check: if they call "full" they might expect interactions
        # but your code defaults to 1. Keeping your default 1.
        pass

    cv_folds = kwargs.get('cv_folds', None)
    
    # Create a copy of the dataframe
    df = df_in.copy()
    y = df[exposure_status]

    # --- CHANGE 1: Encode Categoricals FIRST ---
    # We must turn "Sex" -> "Sex_Male" before we can multiply it by "Age"
    X_encoded = _get_encoded_features(df, all_features)

    # --- CHANGE 2: Polynomial Features on Encoded Data ---
    # interaction_only=True is usually safer for encoded data to prevent 
    # Boolean^2 (which is just Boolean) and reduce dimensionality.
    interaction_only = kwargs.get('interaction_only', False) 
    
    poly = PolynomialFeatures(degree=degree, interaction_only=interaction_only, include_bias=True)
    
    # This might create a VERY large sparse matrix if you have many categories
    X_interactions = poly.fit_transform(X_encoded)
    
    feature_names = poly.get_feature_names_out(X_encoded.columns)
    
    if debug:
        logger.debug(f"Interaction terms created: {len(feature_names)}")

    # Convert to DF for clean sklearn handling
    X_inter_df = pd.DataFrame(
        X_interactions,
        columns=feature_names,
        index=X_encoded.index
    )
    
    # --- CHANGE 3: Handle Parameters for CV/Fit ---
    # (Same logic as your previous code, but using X_inter_df)
    
    param_grid = kwargs.get('param_grid', {'C': [0.01, 0.1, 1, 10, 100]})
    scoring = kwargs.get('scoring', 'roc_auc')
    n_jobs = kwargs.get('n_jobs', -1)

    if cv_folds is not None:
        logistic_cv = GridSearchCV(
            LogisticRegression(
                penalty=penalty,
                solver=solver,
                random_state=random_state,
                max_iter=max_iter
            ),
            param_grid,
            cv=cv_folds,
            scoring=scoring,
            n_jobs=n_jobs
        )
        logger.info("Performing cross-validation...")
        logistic_cv.fit(X_inter_df, y)
        best_C = logistic_cv.best_params_['C']
        logger.info(f"Best C: {best_C}")
    else:
        best_C = 1.0

    # Final Fit
    logistic = LogisticRegression(
        penalty=penalty,
        C=best_C,
        solver=solver,
        random_state=random_state, 
        class_weight='balanced',
        max_iter=max_iter
    )
    logistic.fit(X_inter_df, y)
    
    # Assign back to ORIGINAL DF
    probs = logistic.predict_proba(X_inter_df)[:, 1]
    eps = 1e-12
    logit = np.log(probs.clip(eps, 1 - eps) / (1 - probs.clip(eps, 1 - eps)))
    df['propensity_score'] = probs
    df['propensity_logit'] = logit
    logger.info("Propensity logits calculated successfully with interaction terms.")
    
    return df, df['propensity_logit']



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
