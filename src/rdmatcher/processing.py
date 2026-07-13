import logging
import numpy as np
import pandas as pd
from typing import Literal
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from .logger import rdlogger

# logger = rdlogger(__name__, level="INFO")
logger = logging.getLogger('rdmatcher.processing')

# ------------------------------
# Custom Transformer: BinningTransformer
# ------------------------------
class BinningTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, bin_width=10):
        self.bin_width = bin_width

    def fit(self, X, y=None):
        # If X is a DataFrame, store its column names.
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.array(X.columns)
        else:
            self.feature_names_in_ = np.array([])
        return self

    def transform(self, X):
        # Expecting X as an array or DataFrame.
        X_array = X.values if hasattr(X, "values") else X
        return np.floor_divide(X_array - 1, self.bin_width) + 1

    def get_feature_names_out(self, features_in=None):
        if features_in is None:
            if self.feature_names_in_ is not None and len(self.feature_names_in_) > 0:
                features_in = self.feature_names_in_
            else:
                raise ValueError("No feature names provided.")
        return np.array([f"binned({name})" for name in features_in])



# ------------------------------
# Custom Transformer: LogTransformer
# ------------------------------
class LogTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.array(X.columns)
        else:
            self.feature_names_in_ = np.array([])
        return self

    def transform(self, X):
        # Use np.log1p for transformation.
        X_array = X.values if hasattr(X, "values") else X
        return np.log1p(X_array)

    def get_feature_names_out(self, features_in=None):
        if features_in is None:
            if self.feature_names_in_ is not None and len(self.feature_names_in_) > 0:
                features_in = self.feature_names_in_
            else:
                raise ValueError("No feature names available for LogTransformer")
        return np.array([f"log({name})" for name in features_in])



# ------------------------------
# Custom Transformer: ScalerWithNames
# ------------------------------
class ScalerWithNames(StandardScaler):
    def fit(self, X, y=None):
        # If X is a DataFrame, store its columns.
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.array(X.columns)
        else:
            self.feature_names_in_ = np.array([])
        return super().fit(X, y)

    def get_feature_names_out(self, features_in=None):
        # Scaling does not change names; just pass them through.
        if features_in is None:
            if self.feature_names_in_ is not None and len(self.feature_names_in_) > 0:
                return np.array(self.feature_names_in_)
            else:
                raise ValueError("No feature names to return from scaler.")
        return np.array(features_in)



# ------------------------------
# Build the Preprocessing Pipeline
# ------------------------------
def build_preprocessing_pipeline(
    features_numeric,
    features_categorical,
    bin_method: Literal["scaler", 'binned', 'binned_scaler', None],
    bin_width: int,
    features_log=None,
    features_bin=None,
    onehot: bool = False,
    onehot_scalar: bool = False,
    onehot_drop: Literal['first', 'if_binary', None] = 'first'
):
    """
    Build a robust preprocessing pipeline that applies custom transformations:
      - For numeric features: allow log-transformation and/or binning (with or without scaling).
      - For categorical features: One-hot encode and then scale (to put on a comparable scale).

    Parameters
    ----------
    features_numeric : list of str
        List of names for numeric features.
    features_categorical : list of str
        List of names for categorical features.
    onehot : bool, default False
        Whether to one-hot encode categorical features.
    onehot_scalar : bool, default False
        If True, applies one-hot encoding followed by scaling to categorical features.
    onehot_drop : {'first', 'if_binary', None}, default 'first'
        Drop policy passed to ``OneHotEncoder``.
    features_log : list of str, optional
        Subset of numeric features to be log-transformed with np.log1p.
    features_bin : list of str, optional
        Subset of numeric features to be binned.
    bin_method : str, default "scaler"
        Options:
          - "scaler": Use raw numeric features (and apply scaling).
          - "binned": For features in features_bin, only apply binning (treated as ordinal).
          - "binned_scaler": For features in features_bin, apply binning then scaling.
    bin_width : int, default 10
        Bin width for binning transformation. Applies the same bin width to all features in features_bin.

    Returns
    -------
    preprocessor : ColumnTransformer
        A ColumnTransformer that applies the designated transformations.
    """
    features_log = features_log or []
    features_bin = features_bin or []

    numeric_transformers = []

    # 1. Pipeline for features requiring binning.
    if features_bin:
        if bin_method == 'binned':
            binned_pipeline = Pipeline([
                ("binning", BinningTransformer(bin_width=bin_width))
            ])
        elif bin_method == 'binned_scaler':
            binned_pipeline = Pipeline([
                ("binning", BinningTransformer(bin_width=bin_width)),
                ("scaler", ScalerWithNames())
            ])
        elif bin_method is None:
            binned_pipeline = "passthrough"
        else:
            # Fallback: simply scale these features.
            binned_pipeline = Pipeline([("scaler", ScalerWithNames())])
        numeric_transformers.append(("binned", binned_pipeline, features_bin))

    # 2. Pipeline for features to be log-transformed (if not binned).
    numeric_log = [feat for feat in features_numeric if feat in features_log and feat not in features_bin]
    if numeric_log:
        log_pipeline = Pipeline([
            ("log", LogTransformer()),
            ("scaler", ScalerWithNames())
        ])
        numeric_transformers.append(("log_numeric", log_pipeline, numeric_log))

    # 3. Pipeline for remaining numeric features.
    remaining_numeric = [feat for feat in features_numeric if feat not in features_bin and feat not in features_log]
    if remaining_numeric:
        if bin_method is None:
            plain_pipeline = "passthrough"
        else:
            plain_pipeline = Pipeline([
                ("scaler", ScalerWithNames())
            ])
        numeric_transformers.append(("plain_numeric", plain_pipeline, remaining_numeric))

    # 4. Pipeline for categorical features: one-hot encode then (optionally) scale.
    if onehot:
        categorical_pipeline = Pipeline([
            ("onehot", OneHotEncoder(drop=onehot_drop, sparse_output=False))
        ])
    elif onehot_scalar:
        categorical_pipeline = Pipeline([
            ("onehot", OneHotEncoder(drop=onehot_drop, sparse_output=False)),
            ("scaler", ScalerWithNames())
        ])
    else:
        # if onehot is false, simply pass the categorical features through without any transformation
        categorical_pipeline = "passthrough"

    # Combine numeric and categorical transformers into a ColumnTransformer.
    preprocessor = ColumnTransformer(
        transformers = numeric_transformers + [("categorical", categorical_pipeline, features_categorical)],
        verbose_feature_names_out=False
    )

    preprocessor.set_output(transform="pandas")

    return preprocessor


# ------------------------------
# Helper Function: Apply Preprocessing and Return DataFrame
# ------------------------------
def apply_preprocessing_pipeline(df_in, preprocessor, patient_id_col, exposure_status):
    """
    Fit the preprocessor on DataFrame df, transform it, and return a DataFrame
    with the transformed data and appropriate column names.

    Parameters
    ----------
    df_in : pd.DataFrame
        Original input data.
    preprocessor : ColumnTransformer
        A fitted (or unfitted) ColumnTransformer.

    Returns
    -------
    df_out : pd.DataFrame
         DataFrame with transformed features and the new feature names.
    """
    # only fit relative to the control group
    # removed this. Fitting on the entire dataset is better
    # df_ctl = df_in[df_in[exposure_status] == 0].copy()

    # Fit the preprocessor on the data.
    preprocessor_fitted = preprocessor.fit(df_in)
    # Transform the data.
    df_processed = preprocessor_fitted.transform(df_in)
    # Retrieve output feature names.
    output_feature_names = preprocessor_fitted.get_feature_names_out(input_features=df_in.columns.tolist())
    # Build a DataFrame with the transformed data and output feature names.
    df_out = pd.concat([
        df_in[[patient_id_col, exposure_status]].reset_index(drop=True), 
        pd.DataFrame(df_processed, columns=output_feature_names).reset_index(drop=True)
    ], axis=1)

    return df_out


def build_propensity_preprocessor(features_numeric,
                                  features_categorical,
                                  features_log=None,
                                  features_bin=None,
                                  bin_method: Literal["scaler", 'binned', 'binned_scaler', None] = 'scaler',
                                  bin_width: int = 10,
                                  onehot_scalar: bool = False,
                                  onehot_drop: Literal['first', 'if_binary', None] = 'first'):
    """
    Convenience wrapper that builds a preprocessing pipeline configured for
    propensity model building: always enables one-hot encoding for categoricals
    (drop-first) while exposing the same feature options as the main pipeline.
    Returns a ColumnTransformer with output set to pandas.
    """
    return build_preprocessing_pipeline(
        features_numeric=features_numeric,
        features_categorical=features_categorical,
        bin_method=bin_method,
        bin_width=bin_width,
        features_log=features_log,
        features_bin=features_bin,
        onehot=True,
        onehot_scalar=onehot_scalar,
        onehot_drop=onehot_drop
    )
