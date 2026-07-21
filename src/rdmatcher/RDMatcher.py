import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Literal, Optional
import logging
import os
import matplotlib.pyplot as plt
import seaborn as sns

from .processing import build_preprocessing_pipeline, apply_preprocessing_pipeline
from .train import propensity_logits_simple
from .formula import parse_formula
from .matcher import Matcher
from .matching import matching_diagnostics
from .utils import hide_columns




class RDMatcher:
    def __init__(
        self,
        pop_df,
        patient_id_col,
        exposure_status,
        features_numeric,
        features_categorical,
        features_datetime=None,
        features_log=None,
        features_bin=None,
        process_features=False,
        bin_method=None,
        bin_width=10,
        onehot=False,
        onehot_scalar=False,
        onehot_drop='first',
        debug=False,
        log_file: str = "rdmatcher.log",
        log_to_console: bool = False
    ):

        self.pop_df = pop_df
        self.control_pop = None
        self.exposure_pop = None
        self.patient_id_col = patient_id_col
        self.exposure_status = exposure_status
        self.features_numeric = features_numeric
        self.features_categorical = features_categorical
        self.all_features = features_numeric + features_categorical
        self.features_datetime = features_datetime if features_datetime is not None else []
        self.features_log = features_log if features_log is not None else []
        self.features_bin = features_bin if features_bin is not None else []
        self.bin_method = bin_method
        self.bin_width = bin_width
        self.onehot = onehot
        self.onehot_scalar = onehot_scalar
        self.onehot_drop = onehot_drop
        self.debug = debug
        self._process_features_enabled = process_features

        # logging
        self._configure_logger(debug, log_file=log_file, console=log_to_console)
        self.logger.info("Initializing RDMatcher with provided control and exposure populations.")
        
        # Validate inputs and concatenate dataframes
        self.logger.info("Validating inputs and combining control and exposure populations.")
        self._validate_inputs_and_combine()
        self.logger.info("Inputs validated and populations combined successfully.")

        # Convert datetime features to numeric and add them to features_numeric
        if self.features_datetime:
            self.pop = self._convert_datetime_to_numeric()
            self.features_numeric += [f"{col}_days" for col in self.features_datetime]
            self.all_features += [f"{col}_days" for col in self.features_datetime]
            self.logger.info("Datetime features converted to numeric and added to features_numeric.")

        # Preprocess features
        if not process_features:
            self.logger.info("Skipping feature preprocessing as process_features is set to False.")
            self.pop_processed = self.pop.copy()
        else:
            if onehot:
                self.logger.info("One-hot encoding is enabled and will be applied to non-Gower matching paths; Gower will keep raw categorical semantics.")
            else:
                self.logger.info("One-hot encoding disabled. Recommended for Gower distance.")
            self._process_features()
            
        
        # initialize as None
        self.matched_data = None
        self.pop_matched = None
        self.matched_exposed = None
        self.matched_control = None
        self.unmatched_exposed = None
        
        
        self.logger.info("Dataset loaded. Raw data available in self.pop.")

        # Build feature name lineage maps for weight resolution
        # These map processed columns (used for matching) back to the original feature names
        from .feature_map import build_feature_name_maps

        processed_cols = [col for col in getattr(self, 'pop_processed', pd.DataFrame()).columns if col not in [self.patient_id_col, self.exposure_status]]
        p2o, o2p = build_feature_name_maps(processed_cols, self.features_numeric, self.features_categorical, self.features_datetime)
        self._processed_to_original = p2o
        self._original_to_processed = o2p


    def _build_feature_name_maps(self):
        # wrapper kept for backward compatibility; compute maps using shared helper
        from .feature_map import build_feature_name_maps

        processed_cols = [col for col in getattr(self, 'pop_processed', pd.DataFrame()).columns if col not in [self.patient_id_col, self.exposure_status]]
        p2o, o2p = build_feature_name_maps(processed_cols, self.features_numeric, self.features_categorical, self.features_datetime)
        self._processed_to_original = p2o
        self._original_to_processed = o2p



    def _configure_logger(self, debug: bool, log_file, console: bool = False):
        self.logger = logging.getLogger('rdmatcher')
        level = logging.DEBUG if debug else logging.INFO
        self.logger.setLevel(level)

        # 1. Clear existing handlers to allow reconfiguration
        # If we don't do this, we can't "turn off" the console or switch log files
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        # Define the formatter once for both handlers
        fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # 2. Setup Console Handler (Optional)
        if console:
            console_h = logging.StreamHandler()
            console_h.setLevel(level)
            console_h.setFormatter(fmt)
            self.logger.addHandler(console_h)

        # 3. Setup File Handler (Optional)
        if log_file:
            # Ensure directory exists
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)

            file_h = logging.FileHandler(log_file)
            file_h.setLevel(level)
            file_h.setFormatter(fmt)
            self.logger.addHandler(file_h)

        # Stop at parent; don't send to root
        self.logger.propagate = False

        # Optional sanity check
        self.logger.debug(
            f"Logger configured. Level={level}, Handlers={len(self.logger.handlers)} "
            f"(Console={console}, File={log_file})"
        )
    

    def _validate_inputs_and_combine(self):

        # Ensure pop_df is a DataFrame
        if not isinstance(self.pop_df, pd.DataFrame):
            raise ValueError("pop_df should be a pandas DataFrame.")
        
        # Split pop_df into control and exposure populations
        self.control_pop = self.pop_df[self.pop_df[self.exposure_status] == 0].copy()
        self.logger.info(f"Control population size: {len(self.control_pop)}")
        self.exposure_pop = self.pop_df[self.pop_df[self.exposure_status] == 1].copy()
        self.logger.info(f"Exposure population size: {len(self.exposure_pop)}")
 
        # check that features_log are all in features_numeric
        if self.features_log:
            for feature in self.features_log:
                if feature not in self.features_numeric:
                    raise ValueError(f"Feature {feature} is in features_log but not in features_numeric.")
        # check that features_bin are all in features_numeric
        if self.features_bin:
            for feature in self.features_bin:
                if feature not in self.features_numeric:
                    raise ValueError(f"Feature {feature} is in features_bin but not in features_numeric.")

        # Check for missing features in control and exposure populations
        for feature_set, name in [
            (self.features_numeric, "features_numeric"),
            (self.features_categorical, "features_categorical"),
            (self.features_datetime, "features_datetime"),
            (self.features_log, "features_log"),
            (self.features_bin, "features_bin"),
        ]:
            if feature_set:
                missing_features = {
                    feature
                    for feature in feature_set
                    if feature not in self.control_pop.columns or feature not in self.exposure_pop.columns
                }
                if missing_features:
                    raise ValueError(f"Some {name} are missing in control_pop or exposure_pop: {missing_features}")

        # Check for required columns in both populations
        for col in [self.exposure_status, self.patient_id_col]:
            if col not in self.control_pop.columns or col not in self.exposure_pop.columns:
                raise ValueError(f"{col} is missing in control_pop or exposure_pop.")

        # Concatenate dataframes and validate combined population
        try:
            self.pop = pd.concat([self.control_pop, self.exposure_pop]).reset_index(drop=True)
        except ValueError as e:
            raise ValueError("Error concatenating control_pop and exposure_pop. Check the columns and data types.") from e

        if self.pop[self.patient_id_col].isnull().any():
            raise ValueError("Missing patient IDs found in the combined dataset.")
        if self.pop[self.patient_id_col].duplicated().any():
            raise ValueError("Duplicate patient IDs found in the combined dataset.")
        if self.pop[self.exposure_status].isnull().any():
            raise ValueError("Missing exposure status found in the combined dataset.")

    

    def _convert_datetime_to_numeric(self):
        pop_copy = self.pop.copy()
        for col in self.features_datetime:
            self.logger.debug(f"Converting datetime feature '{col}' to numeric days since the first global occurrence.")
            # Convert to datetime if not already
            if not pd.api.types.is_datetime64_any_dtype(pop_copy[col]):
                pop_copy[col] = pd.to_datetime(pop_copy[col], errors='coerce')
            min_date = pop_copy[col].min()
            pop_copy[f"{col}_days"] = (pop_copy[col] - min_date).dt.total_seconds() / (24 * 60 * 60)
        return pop_copy

    
    def _process_features(self):
        """
        Preprocess features by converting datetime to numeric values, binning (optional),
        standardizing numeric features, and one-hot encoding categorical features.
        
        Once processed, the data is stored in self.pop_processed and self.all_features
        is updated accordingly.
        
        Parameters:
        -----------
        bin_features : list or None, optional
            List of features to be binned.
        binning_denominator : int, default=10
            Denominator for binning if specified.
        """
        # Prevent re-processing if already processed
        if hasattr(self, '_is_processed') and self._is_processed:
            self.logger.info("Data has already been preprocessed. Skipping redundant processing.")
            return

        # Work on a copy of the raw data
        df = self.pop.copy()

        # Build the preprocessor using our custom pipeline.
        preprocessor = build_preprocessing_pipeline(
            features_numeric=self.features_numeric,
            features_categorical=self.features_categorical,
            features_log=self.features_log,
            features_bin=self.features_bin,
            bin_method=self.bin_method,
            bin_width=self.bin_width,
            onehot=self.onehot,
            onehot_scalar=self.onehot_scalar,
            onehot_drop=self.onehot_drop
        )

        # Apply preprocessing and get a final DataFrame.
        self.logger.info("Applying preprocessing pipeline to the dataset.")
        self.pop_processed = apply_preprocessing_pipeline(df, preprocessor, self.patient_id_col, self.exposure_status)
        self._is_processed = True
        # update all_features to reflect any changes
        self.all_features = [col for col in self.pop_processed.columns if col not in [self.patient_id_col, self.exposure_status]]
        self.logger.info("Preprocessing complete. Data available in self.pop_processed.")


    # public method to preprocess features
    def process_features(self):
        """
        Public method to preprocess features if not already done.
        """
        self._process_features()

    # ------------------------------------------------------------------
    # Feature-name mapping helpers
    # ------------------------------------------------------------------
    def _build_processed_to_original(self, local_processed, original_to_processed):
        """Map each processed column back to its original feature name.

        For one-hot expansions like ``black_1`` we strip the suffix and map
        back to the original categorical ``black``.  The critical guard is
        that the *base* name itself must **not** already be an original
        feature — otherwise a numeric column like ``drug_dose`` whose prefix
        ``drug`` happens to be a categorical would be mis-mapped.
        """
        result = {}
        for original, processed_cols in (original_to_processed or {}).items():
            for processed_col in processed_cols:
                result[processed_col] = original

        for c in local_processed.columns:
            if c in (self.patient_id_col, self.exposure_status):
                continue
            result.setdefault(c, c)
        return result

    def _build_gower_sd_weights(
        self,
        matching_data,
        match_features,
        gower_sd_reference="controls",
        gower_sd_weights_mult=1.96,
        numeric_block_weight=0.70,
        categorical_block_weight=0.30,
    ):
        """Build Gower weights that make SD-scale numeric shifts comparable.

        Standard Gower divides numeric differences by the reference range. For
        heavy-tailed numeric features, rare extremes can make meaningful
        numeric differences too small. These weights multiply each numeric
        feature by range / (multiplier * SD), then normalize numeric and
        categorical feature blocks to fixed total weights.
        """
        if gower_sd_weights_mult <= 0:
            raise ValueError("gower_sd_weights_mult must be positive.")
        if numeric_block_weight < 0 or categorical_block_weight < 0:
            raise ValueError("Gower SD block weights must be non-negative.")
        if gower_sd_reference not in {"controls", "pooled"}:
            raise ValueError(
                "gower_sd_reference must be one of {'controls', 'pooled'}."
            )

        if gower_sd_reference == "controls":
            control_mask = matching_data[self.exposure_status] == 0
            reference = matching_data.loc[control_mask, match_features]
            reference_label = "control"
        else:
            reference = matching_data.loc[:, match_features]
            reference_label = "pooled"
        if reference.empty:
            raise ValueError(
                f"Cannot build Gower SD weights without {reference_label} reference records."
            )

        numeric_features = [f for f in self.features_numeric if f in match_features]
        categorical_features = [f for f in self.features_categorical if f in match_features]
        weights = {}

        if numeric_features and numeric_block_weight > 0:
            numeric = reference[numeric_features].astype(float)
            ranges = numeric.max(skipna=True) - numeric.min(skipna=True)
            sd = numeric.std(skipna=True, ddof=1)
            raw = ranges / (float(gower_sd_weights_mult) * sd)
            raw = raw.replace([np.inf, -np.inf], np.nan).fillna(1.0)
            raw = raw.clip(lower=1e-8)
            raw_sum = float(raw.sum())
            if raw_sum <= 0 or not np.isfinite(raw_sum):
                raise ValueError("Failed to build valid numeric Gower SD weights.")
            for feature in numeric_features:
                weights[feature] = float(numeric_block_weight * raw[feature] / raw_sum)

        if categorical_features and categorical_block_weight > 0:
            categorical_weight = float(categorical_block_weight) / len(categorical_features)
            weights.update({feature: categorical_weight for feature in categorical_features})

        missing = [feature for feature in match_features if feature not in weights]
        for feature in missing:
            weights[feature] = 0.0

        self.logger.info(
            f"Built Gower SD weights using {reference_label} reference pool "
            f"({len(reference)} rows): "
            f"mult={gower_sd_weights_mult}, numeric_block_weight={numeric_block_weight}, "
            f"categorical_block_weight={categorical_block_weight}, weights={weights}"
        )
        return weights

    def _build_multi_matching_view(self, distance_metric: str):
        """Build the cohort view used by multi-covariate matching.

        The returned DataFrame is always derived from ``self.pop`` and only
        includes the user-declared matching features plus the id/exposure
        columns. This avoids leakage from propensity-specific columns that may
        be present in ``self.pop`` or ``self.pop_processed``.
        """
        match_features = list(dict.fromkeys(self.features_numeric + self.features_categorical))
        base_cols = match_features + [self.patient_id_col, self.exposure_status]

        if distance_metric == 'gower':
            matching_data = self.pop[base_cols].copy()
            processed_to_original_map = {f: f for f in match_features}
            original_to_processed_map = {f: [f] for f in match_features}
            return matching_data, match_features, processed_to_original_map, original_to_processed_map

        # Non-Gower metrics require a numeric matrix. If the cohort includes
        # categoricals, one-hot encoding must be enabled at class construction.
        if self.features_categorical and not self.onehot:
            raise ValueError(
                "Non-Gower distance with categorical features requires onehot=True so the cohort "
                "can be encoded into a numeric matrix before matching."
            )

        features_log = self.features_log if self._process_features_enabled else []
        features_bin = self.features_bin if self._process_features_enabled else []
        bin_method = self.bin_method if self._process_features_enabled else None

        preprocessor = build_preprocessing_pipeline(
            features_numeric=self.features_numeric,
            features_categorical=self.features_categorical,
            bin_method=bin_method,
            bin_width=self.bin_width,
            features_log=features_log,
            features_bin=features_bin,
            onehot=self.onehot,
            onehot_scalar=bool(self.onehot_scalar and self._process_features_enabled),
            onehot_drop=self.onehot_drop,
        )
        matching_data = apply_preprocessing_pipeline(self.pop, preprocessor, self.patient_id_col, self.exposure_status)
        processed_cols = [c for c in matching_data.columns if c not in [self.patient_id_col, self.exposure_status]]

        from .feature_map import build_feature_name_maps

        processed_to_original_map, original_to_processed_map = build_feature_name_maps(
            processed_cols,
            self.features_numeric,
            self.features_categorical,
            self.features_datetime,
        )

        # Keep the mapping available for downstream feature-weight and diagnostic logic.
        self._processed_to_original = processed_to_original_map
        self._original_to_processed = original_to_processed_map
        self._matching_processed_cols = processed_cols

        return matching_data, processed_cols, processed_to_original_map, original_to_processed_map

    # calculate propensity scores
    def fit_propensity_model(self, formula: Optional[str]=None, random_state=404, **kwargs):
        """
        Calculate propensity scores using logistic regression.

        This function takes a DataFrame and calculates propensity scores for a 
        specified binary exposure variable using a logistic regression model. 
        The propensity scores represent the probability of the exposure variable 
        being 1, given the other features in the DataFrame.

        Args:
            random_state (int): Random state for reproducibility.

        Returns:
            tuple: A tuple containing:
                - pd.DataFrame: A copy of the input DataFrame with an additional 
                                column 'propensity_logit' containing the calculated 
                                propensity scores.
                - pd.Series: A Series containing the propensity scores.
        """
        # Check that the preprocessed data is available.
        # if not hasattr(self, "pop_processed"):
        #     raise ValueError("Preprocessed features not found. Please run _process_features() first.")

        # Build a dedicated preprocessing view for propensity modeling to ensure categoricals are one-hot encoded
        self.logger.info("Building preprocessing pipeline for propensity modeling (one-hot encoding enabled).")
        from .processing import build_propensity_preprocessor

        pre = build_propensity_preprocessor(
            features_numeric=self.features_numeric,
            features_categorical=self.features_categorical,
            features_log=self.features_log,
            features_bin=self.features_bin,
            bin_method='scaler',
            bin_width=self.bin_width,
            onehot_scalar=self.onehot_scalar,
            onehot_drop=self.onehot_drop
        )
        pop_for_psm = apply_preprocessing_pipeline(self.pop.copy(), pre, self.patient_id_col, self.exposure_status)
        local_processed = pop_for_psm
        # build mapping original->processed using shared helper to ensure consistency
        from .feature_map import build_feature_name_maps
        processed_cols = [c for c in local_processed.columns if c not in [self.patient_id_col, self.exposure_status]]
        _, original_to_processed = build_feature_name_maps(processed_cols, self.features_numeric, self.features_categorical, self.features_datetime)

        # Parse formula
        if formula is None:
            self.logger.info("No formula provided for propensity modeling; using all original features as main effects.")
            # build a formula that selects all original features
            formula_terms = [(f,) for f in (self.features_numeric + self.features_categorical)]
        else:
            formula_terms = parse_formula(formula)

        # Call unified propensity function: uses processed DF and original->processed mapping
        psm_df, self.propensity_logits, psm_meta = propensity_logits_simple(
            local_processed,
            self.exposure_status,
            all_features=[c for c in local_processed.columns if c not in [self.patient_id_col, self.exposure_status]],
            original_to_processed=original_to_processed,
            processed_to_original=self._build_processed_to_original(local_processed, original_to_processed),
            formula_terms=formula_terms,
            random_state=random_state,
            **kwargs
        )

        # The fitted OneHotEncoder is the authoritative record of omitted
        # reference categories.  Expose it alongside the resolved matrix so a
        # formula can be reproduced outside RDMatcher.
        categorical_references = {}
        try:
            encoder = pre.named_transformers_['categorical'].named_steps['onehot']
            if self.onehot_drop == 'first':
                categorical_references = {
                    feature: categories[0]
                    for feature, categories in zip(self.features_categorical, encoder.categories_)
                    if len(categories)
                }
            elif self.onehot_drop == 'if_binary':
                categorical_references = {
                    feature: categories[0]
                    for feature, categories in zip(self.features_categorical, encoder.categories_)
                    if len(categories) == 2
                }
        except (AttributeError, KeyError):
            self.logger.warning("Could not determine propensity categorical reference levels.")

        # We want to keep propensity scores separate from the main matching feature space.
        # Merge propensity_logit back into self.pop (original population) by patient id.
        if 'propensity_logit' in psm_df.columns:
            prop_map = psm_df[[self.patient_id_col, 'propensity_logit']].set_index(self.patient_id_col)
            # also merge propensity_score if present
            if 'propensity_score' in psm_df.columns:
                prop_map = psm_df[[self.patient_id_col, 'propensity_score', 'propensity_logit']].set_index(self.patient_id_col)
            if psm_df[self.patient_id_col].duplicated().any() or self.pop[self.patient_id_col].duplicated().any():
                raise ValueError("Patient IDs must be unique for propensity-score alignment.")
            # Re-fitting replaces prior scalar scores.  All joins are by ID;
            # positional assignment is deliberately forbidden here.
            score_columns = ['propensity_score', 'propensity_logit']
            self.pop = self.pop.drop(columns=[c for c in score_columns if c in self.pop], errors='ignore')
            self.pop = self.pop.merge(
                prop_map.reset_index(), on=self.patient_id_col, how='left', validate='one_to_one'
            )
            if self.pop[['propensity_score', 'propensity_logit']].isna().any().any():
                raise RuntimeError("Propensity-score ID merge left unmatched patients.")
            # Also add to pop_processed if it exists, aligning indices by patient id
            if hasattr(self, 'pop_processed') and self.pop_processed is not None:
                self.pop_processed = self.pop_processed.drop(
                    columns=[c for c in score_columns if c in self.pop_processed], errors='ignore'
                ).merge(prop_map.reset_index(), on=self.patient_id_col, how='left', validate='one_to_one')
                if self.pop_processed[['propensity_score', 'propensity_logit']].isna().any().any():
                    raise RuntimeError("Propensity-score ID merge left unmatched processed patients.")

            # Add propensity_logit to all_features by default so it becomes
            # available as a scalar matching feature. We intentionally add only
            # propensity_logit (not propensity_score) per user instruction.
            if 'propensity_logit' not in self.all_features:
                self.all_features.append('propensity_logit')

            # Rebuild feature name maps so the new propensity column is recognized
            # by the gower-weights resolver and other mapping utilities.
            try:
                self._build_feature_name_maps()
            except Exception:
                # If rebuilding maps fails, log a warning but continue
                self.logger.warning("Failed to rebuild feature name maps after adding propensity_logit.")

        # store metadata about propensity feature resolution for transparency
        self._propensity_feature_map = {
            'formula': formula,
            'parsed_terms': formula_terms,
            'meta': psm_meta,
            'original_to_processed_psm': original_to_processed,
            'design_matrix_columns': psm_meta['design_matrix_columns'],
            'categorical_reference_levels': categorical_references,
        }
        # Keep the externally visible score vector ID-indexed as well.
        self.propensity_logits = self.pop.set_index(self.patient_id_col)['propensity_logit'].copy()

        self.logger.info("Propensity scores calculated and merged as scalar columns. propensity_logit has been added to all_features.")

    def get_propensity_feature_map(self):
        """Return metadata about the last fitted propensity model.

        Returns a dict with keys:
          - formula: the formula string used (or None)
          - parsed_terms: list of parsed terms
          - meta: design-matrix and optimizer metadata, including solver,
            tolerance, iterations, warnings, and score summary
          - original_to_processed_psm: mapping used for the propensity modeling view
          - design_matrix_columns and categorical_reference_levels: the
            resolved formula matrix and omitted one-hot reference categories
        """
        return getattr(self, '_propensity_feature_map', {})


    def plot_propensity_coverage(self, compare_matching=False, figsize=(10, 6), save_path=None):
        """
        Plot the distribution of propensity scores before and after matching (if compare_matching is True).

        Parameters:
        -----------
        compare_matching : bool, default=False
            If True, display a panel comparing the propensity score distributions before and after matching.
            If False, display only the distribution before matching.
        figsize : tuple, default=(10, 6)
            Figure size in inches.
        save_path : str, optional
            If provided, save the figure to this path.
        
        Returns:
        --------
        fig, axes : tuple
            Matplotlib figure and list of axes objects.
        """
        # Check that the necessary data exist
        if not hasattr(self, "propensity_logits") and not hasattr(self, "pop"):
            raise ValueError("Propensity scores not found. Please run calculate_propensity_logits() first.")
        
        if compare_matching and not hasattr(self, "pop_matched"):
            raise ValueError("Matched data not found. Please run rare_matching() first.")
        
        # Determine the number of subplots needed
        n_axes = 2 if compare_matching else 1
        fig, axes = plt.subplots(1, n_axes, figsize=figsize)
        if n_axes == 1:
            axes = [axes]
        
        # Define the order for exposure status using the specified column name (assumed stored in self.exposure_status)
        exposure_status_order = sorted(self.pop[self.exposure_status].unique(), reverse=True)
        
        # Plot before matching using a single call with hue
        sns.kdeplot(
            data=self.pop,
            x='propensity_logit',
            hue=self.exposure_status,
            hue_order=exposure_status_order,
            palette='Set1',
            fill=True,
            alpha=0.5,
            common_norm=False,
            ax=axes[0]
        )
        axes[0].set_title('Propensity Score Distribution Before Matching')
        axes[0].set_xlabel('Propensity Score')
        axes[0].set_ylabel('Density')
        axes[0].set_xticks([0, 1])
        legend = axes[0].get_legend()
        if legend:
            legend.set_title("Exposure Status")
        
        # Plot after matching if requested using a single call with hue
        if compare_matching:
            sns.kdeplot(
                data=self.pop_matched,
                x='propensity_logit',
                hue=self.exposure_status,
                hue_order=exposure_status_order,
                palette='Set1',
                fill=True,
                alpha=0.5,
                common_norm=False,
                ax=axes[1]
            )
            axes[1].set_title('Propensity Score Distribution After Matching')
            axes[1].set_xlabel('Propensity Score')
            axes[1].set_ylabel('Density')
            axes[1].set_xticks([0, 1])
            legend = axes[1].get_legend()
            if legend:
                legend.set_title("Exposure Status")
        
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

        return fig, axes



    #-------------------------------
    # Matching
    #-------------------------------
    def rare_matching(self, 
                      threshold: float, 
                      n_neighbors: int, 
                      k_candidates: Optional[int] = None,
                      method: Literal['multi', 'propensity'] = 'multi',
                      distance_metric: Literal['gower', 'euclidean', 'cosine', 'mahalanobis'] = 'gower',
                      global_optimal: bool = True, 
                      replacement: bool = False, 
                      competitive_match: bool = True,
                      ps_hybrid: bool = False,
                      ps_caliper: float = 0.2,
                      propensity_caliper: Optional[float] = None,
                      ps_caliper_strict: bool = True,
                      gower_sd_weights: bool = False,
                      gower_sd_reference: Literal['controls', 'pooled'] = 'controls',
                      gower_sd_weights_mult: float = 1.96,
                      n_jobs: Optional[int] = 1,
                      parallel_chunk_size: Optional[int] = None,
                      streaming: str = 'auto',
                      stream_block_size: Optional[int] = None,
                      stream_threshold_gb: float = 1.0,
                      memory_limit_gb: Optional[float] = None,
                      return_unmatched_cases: bool = True,
                      **kwargs):
        """
        Perform optimal matching for rare disease populations using the refactored Matcher class.
        
        Parameters:
        -----------
        threshold : float
            The maximum allowable distance for matching.
        n_neighbors : int
            The number of neighbors to consider for each exposed individual.
        k_candidates : int, optional
            The number of candidate controls to consider (pre-filter) for each
            exposed individual. If omitted, defaults in the underlying matcher
            to ``n_exposed * n_neighbors``.
        method : 'multi' or 'propensity'
            'multi': Uses multi-covariate matching (Gower/Euclidean/Mahalanobis).
            'propensity': Matches solely on propensity score.
        gower_sd_weights : bool, default=False
            If True, automatically build Gower weights so SD-scale numeric
            differences are not diluted by large observed ranges.
        gower_sd_reference : {'controls', 'pooled'}, default='controls'
            Reference pool used when building SD-based Gower weights.
            'controls' uses control records only and preserves ATT-style donor
            scaling. 'pooled' is a sensitivity-analysis option that uses both
            treated and control records from the current matching view.
        gower_sd_weights_mult : float, default=1.96
            Numeric SD multiplier used when building SD-based Gower weights.
        """
        # 1. Setup Data
        if not hasattr(self, "pop_processed"):
            self.logger.info("Preprocessing pipeline not initiated. Using raw pop data.")
            self.pop_processed = self.pop.copy()
            
        # Propensity scores live on the fitted RDMatcher state and are only
        # injected into the matching flow when explicitly requested.
        propensity_col = None
        if method == 'propensity' and hasattr(self, "propensity_logits"):
            propensity_col = kwargs.get('propensity_col', 'propensity_logit')
        propensity_score_col = kwargs.get('propensity_col', 'propensity_logit')

        # debug = kwargs.get('debug', False)
        # self.logger.info(f"Debug mode is {'on' if debug else 'off'}.")
        diags = kwargs.get('diagnostics', False)
        return_matched_data = kwargs.get('return_matched_data', False)
        # Whether to include unmatched exposed cases in the returned DataFrame
        # (does not change matching logic). Default True for backward-friendly behavior.
        include_unmatched = bool(return_unmatched_cases)
        
        # check that categorical features are not one-hot encoded if using Gower
        if distance_metric == 'gower' and self.onehot:
            self.logger.warning(
                "Gower distance is selected while one-hot encoding is enabled. "
                "Gower will use raw categorical semantics and ignore the one-hot view."
            )
        self.logger.info(f"Starting matching using method: {method}")

        # --- PS-Hybrid validation ---
        if ps_hybrid and distance_metric not in {"gower", "mahalanobis"}:
            raise ValueError("PS-Hybrid mode (ps_eligible_sets) only supports Gower or Mahalanobis distance metric.")
        if gower_sd_weights and distance_metric != "gower":
            raise ValueError("gower_sd_weights=True is only supported with distance_metric='gower'.")
        if gower_sd_weights and method != "multi":
            raise ValueError("gower_sd_weights=True is only supported with method='multi'.")
        if gower_sd_weights and kwargs.get('gower_weights') is not None:
            raise ValueError("Use either gower_sd_weights=True or explicit gower_weights, not both.")
        if gower_sd_weights and gower_sd_reference not in {"controls", "pooled"}:
            raise ValueError("gower_sd_reference must be one of {'controls', 'pooled'}.")

        # --- PS-Hybrid Caliper Computation ---
        ps_eligible_sets = None
        if ps_hybrid:
            if method != 'multi':
                raise ValueError("ps_hybrid=True requires method='multi'.")
            if not hasattr(self, 'propensity_logits') or self.propensity_logits is None:
                raise ValueError(
                    "ps_hybrid=True requires propensity scores. "
                    "Run fit_propensity_model() before rare_matching()."
                )
            # Extract logit PS for treated and controls from the fitted state.
            # Do not rely on the transient matching-view column selection,
            # because multi-covariate matching intentionally keeps propensity
            # scores out of the feature matrix to avoid leakage.
            if hasattr(self, 'pop_processed') and propensity_score_col in self.pop_processed.columns:
                pop_ps = self.pop_processed
            elif propensity_score_col in self.pop.columns:
                pop_ps = self.pop
            else:
                raise ValueError(
                    f"ps_hybrid=True requires a '{propensity_score_col}' column populated by fit_propensity_model()."
                )
            treated_mask = pop_ps[self.exposure_status] == 1
            control_mask = pop_ps[self.exposure_status] == 0

            ps_treated = pop_ps.loc[treated_mask, propensity_score_col].values
            ps_control = pop_ps.loc[control_mask, propensity_score_col].values

            # MatchIt's std.caliper=TRUE scales an unnamed distance caliper by
            # sd(distance[!discarded]). RDMatcher has no discard step here, so
            # use the full propensity-logit vector rather than treated-only SD.
            caliper_sd = np.std(pop_ps[propensity_score_col].values, ddof=1)
            caliper_width = ps_caliper * caliper_sd
            self.logger.info(
                f"PS-Hybrid caliper: {ps_caliper} × SD(logit PS) = {ps_caliper} × {caliper_sd:.4f} = {caliper_width:.4f}"
            )

            # Build eligible sets: one array of eligible control positional indices per treated unit
            # Control positional indices are 0..N_control-1 (matching Matcher.control_indices order)
            ps_eligible_sets = []
            n_empty = 0
            for i, t_ps in enumerate(ps_treated):
                eligible = np.where(np.abs(ps_control - t_ps) <= caliper_width)[0]
                if len(eligible) == 0 and not ps_caliper_strict:
                    # Relax: try 2x caliper
                    eligible = np.where(np.abs(ps_control - t_ps) <= 2 * caliper_width)[0]
                    if len(eligible) > 0:
                        self.logger.debug(f"Treated {i}: relaxed caliper 2× found {len(eligible)} eligible controls.")
                if len(eligible) == 0:
                    n_empty += 1
                ps_eligible_sets.append(eligible)

            self.logger.info(
                f"PS-Hybrid eligible sets built: {len(ps_eligible_sets)} treated units, "
                f"{n_empty} with zero eligible controls."
            )
            if n_empty > 0 and ps_caliper_strict:
                self.logger.warning(
                    f"{n_empty} treated units have zero eligible controls with strict caliper. "
                    f"These units will have no candidates within threshold."
                )

        # 2. Configure Weights & Metric based on Method
        if method == 'propensity':
            if propensity_col is None:
                raise ValueError("Propensity scores are required for 'propensity' matching method.")

            # A propensity-only comparison is one-dimensional nearest-neighbor
            # matching on the *raw* linear predictor.  Do not let Gower's
            # range normalization silently change the MatchIt scale.
            if distance_metric != 'euclidean':
                raise ValueError(
                    "Propensity-only matching requires distance_metric='euclidean' "
                    "to use raw absolute logit differences."
                )
            if propensity_caliper is not None:
                if propensity_caliper < 0:
                    raise ValueError("propensity_caliper must be non-negative.")
                full_logits = self.pop[propensity_col].to_numpy(dtype=float)
                full_sd = float(np.std(full_logits, ddof=1))
                if not np.isfinite(full_sd):
                    raise RuntimeError("Cannot compute propensity caliper from non-finite logits.")
                threshold = float(propensity_caliper) * full_sd
                self.propensity_matching_metadata_ = {
                    'distance': 'raw_absolute_logit_difference',
                    'caliper': float(propensity_caliper),
                    'logit_sd_full_cohort': full_sd,
                    'caliper_width': threshold,
                    'replacement': bool(replacement),
                }
                self.logger.info(
                    f"PSM caliper: {propensity_caliper} × SD(full-cohort logit) "
                    f"= {propensity_caliper} × {full_sd:.4f} = {threshold:.4f}"
                )

            # For propensity matching: Weight Propensity = 1.0, Covariates = 0.0
            weight_propensity = 1.0
            weight_numeric = 0.0
            # Ensure the propensity column exists in pop; user must have run calculate_propensity_logits()
            if propensity_col not in self.pop.columns:
                raise ValueError(f"Propensity column '{propensity_col}' not found in population. Run calculate_propensity_logits() first.")
            
        elif method == 'multi':
            # Default to Gower unless overridden
            weight_propensity = kwargs.get('weight_propensity', 1.0)
            weight_numeric = kwargs.get('weight_numeric', 1.0)
            
        else:
            raise ValueError(f"Method '{method}' is not supported. Use 'multi' or 'propensity'.")

        # 3. Instantiate Matcher (Fits the model once)
        # We assume Matcher is imported as: from .matcher import Matcher
        if method == 'propensity':
            # Build matching data that contains only the propensity column
            propensity_col = kwargs.get('propensity_col', 'propensity_logit')
            matching_features = [propensity_col]
            matching_data = self.pop[[propensity_col, self.patient_id_col, self.exposure_status]].copy()
            self.logger.info(f"Instantiating Matcher for propensity-only matching with {len(matching_data)} records and feature: {matching_features}")
            self.matching_data_ = matching_data.copy()

            # Build minimal feature maps so gower weight resolver can operate on the propensity column
            processed_to_original_map = {propensity_col: propensity_col}
            original_to_processed_map = {propensity_col: [propensity_col]}

            # default gower weight for propensity is 1.0 unless user overrides
            gower_weights = kwargs.get('gower_weights', {propensity_col: 1.0})

            matcher = Matcher(
                df=matching_data[[propensity_col, self.patient_id_col, self.exposure_status]],
                exposure_status=self.exposure_status,
                patient_id=self.patient_id_col,
                distance_metric=distance_metric,
                threshold=threshold,
                n_neighbors=n_neighbors,
                # Weights
                weight_numeric=0.0,
                weight_propensity=1.0,
                propensity_col=propensity_col,
                gower_weights=gower_weights,
                gower_cat_features=kwargs.get('gower_cat_features'),
                feature_name_map_processed_to_original=processed_to_original_map,
                feature_name_map_original_to_processed=original_to_processed_map,
                original_categorical_features=[],
                # Advanced options
                pca_filter=kwargs.get('pca_filter', False),
                n_jobs=n_jobs,
                parallel_chunk_size=parallel_chunk_size,
                streaming=streaming,
                stream_block_size=stream_block_size,
                stream_threshold_gb=stream_threshold_gb,
                memory_limit_gb=memory_limit_gb,
                mahalanobis_neighbor_backend=kwargs.get('mahalanobis_neighbor_backend', 'cdist'),
                mahalanobis_algorithm=kwargs.get('mahalanobis_algorithm', 'auto'),
                logger=self.logger,
            )
        else:
            matching_data, match_features, native_processed_to_original, native_original_to_processed = \
                self._build_multi_matching_view(distance_metric)
            self.logger.info(f"Instantiating Matcher with {len(matching_data)} records and features: {match_features}")
            self.matching_data_ = matching_data.copy()
            if distance_metric == 'gower':
                default_gower_cat_features = [
                    f for f in self.features_categorical
                    if f in matching_data.columns
                ]
                gower_cat_features = kwargs["gower_cat_features"] if "gower_cat_features" in kwargs else default_gower_cat_features
                if gower_cat_features is not None:
                    missing_gower_cat_features = sorted(set(gower_cat_features).difference(match_features))
                    if missing_gower_cat_features:
                        raise ValueError(
                            f"gower_cat_features contains columns not used for matching: {missing_gower_cat_features}"
                        )
            else:
                gower_cat_features = kwargs.get('gower_cat_features')
            gower_weights = kwargs.get('gower_weights')
            if gower_sd_weights:
                gower_weights = self._build_gower_sd_weights(
                    matching_data=matching_data,
                    match_features=match_features,
                    gower_sd_reference=gower_sd_reference,
                    gower_sd_weights_mult=gower_sd_weights_mult,
                )
            self.gower_weights_ = gower_weights
            self.gower_sd_reference_ = gower_sd_reference if gower_sd_weights else None

            matcher = Matcher(
                df=matching_data,
                exposure_status=self.exposure_status,
                patient_id=self.patient_id_col,
                distance_metric=distance_metric,
                threshold=threshold,
                n_neighbors=n_neighbors,
                # Weights
                weight_numeric=weight_numeric,
                weight_propensity=weight_propensity,
                propensity_col=propensity_col,
                gower_weights=gower_weights,
                gower_cat_features=gower_cat_features,
                feature_name_map_processed_to_original=native_processed_to_original,
                feature_name_map_original_to_processed=native_original_to_processed,
                original_categorical_features=list(self.features_categorical),
                # PS-Hybrid
                ps_eligible_sets=ps_eligible_sets,
                # Advanced options
                pca_filter=kwargs.get('pca_filter', False),
                n_jobs=n_jobs,
                parallel_chunk_size=parallel_chunk_size,
                streaming=streaming,
                stream_block_size=stream_block_size,
                stream_threshold_gb=stream_threshold_gb,
                memory_limit_gb=memory_limit_gb,
                mahalanobis_neighbor_backend=kwargs.get('mahalanobis_neighbor_backend', 'cdist'),
                mahalanobis_algorithm=kwargs.get('mahalanobis_algorithm', 'auto'),
                logger=self.logger,
            )

        # 4. Execute Match
        self.matched_data = matcher.match(
            k_candidates=k_candidates,
            global_optimal=global_optimal,
            replacement=replacement,
            competitive_match=competitive_match,
            safe_matches=kwargs.get('safe_matches', n_neighbors),
            fuzzy_threshold=kwargs.get('fuzzy_threshold', False),
            fuzzy_threshold_limit=kwargs.get('fuzzy_threshold_limit'),
            mcf=kwargs.get('mcf', False),
            batch_size=kwargs.get('batch_size', 1024)
        )
        self.matching_candidate_diagnostics = getattr(matcher, 'ps_hybrid_diagnostics_', None)

        self.logger.info(f"Matching complete. {len(set(self.matched_data[self.patient_id_col]))} patients matched.")

        # 5. Link & Diagnostics
        self._link_matched_data()

        self.logger.info("Running matching diagnostics.")
        # plot_features is true if diagnostics is true
        plot_features = diags
        self.summary_table = matching_diagnostics(
            self.pop_matched, 
            self.exposure_status, 
            features_numeric=self.features_numeric, 
            features_categorical=self.features_categorical,
            plot_features=plot_features
        )

        # Optionally include or exclude unmatched exposed cases in the returned DataFrame
        try:
            if include_unmatched:
                # Ensure unmatched_exposed rows are present in pop_matched. In many
                # matching flows matched_data already contains unmatched exposures
                # with n_matches == 0; only append if they are missing.
                if hasattr(self, 'unmatched_exposed') and not self.unmatched_exposed.empty:
                    unmatched = self.unmatched_exposed.copy()
                    if 'match_group' not in unmatched.columns:
                        unmatched['match_group'] = np.nan
                    if 'n_matches' not in unmatched.columns:
                        unmatched['n_matches'] = 0
                    if 'match_distance' not in unmatched.columns:
                        unmatched['match_distance'] = np.nan
                    if 'propensity_logit' in self.pop_matched.columns and 'propensity_logit' not in unmatched.columns:
                        if 'propensity_logit' in self.pop.columns:
                            unmatched = unmatched.merge(self.pop[['patient_id', 'propensity_logit']], on='patient_id', how='left')
                        else:
                            unmatched['propensity_logit'] = np.nan
                    existing_ids = set(self.pop_matched[self.patient_id_col].astype(object).tolist())
                    to_add = unmatched[~unmatched[self.patient_id_col].isin(existing_ids)]
                    if not to_add.empty:
                        self.pop_matched = pd.concat([self.pop_matched, to_add], ignore_index=True, sort=False)
            else:
                # Exclude unmatched exposures: keep only rows with at least one match
                if 'n_matches' in self.pop_matched.columns:
                    self.pop_matched = self.pop_matched[self.pop_matched['n_matches'] > 0].reset_index(drop=True)
                else:
                    # Fallback: drop rows where match_group is null
                    if 'match_group' in self.pop_matched.columns:
                        self.pop_matched = self.pop_matched[self.pop_matched['match_group'].notnull()].reset_index(drop=True)
        except Exception:
            self.logger.exception("Failed to adjust returned DataFrame for unmatched cases; returning matched-only data.")

        if return_matched_data:
            return self.pop_matched
        self.logger.info("rare_matching() finished. Data available in self.pop_matched.")


    def _link_matched_data(self):
        """
        Link matched data with the original population data.
        
        Returns:
        --------
        pd.DataFrame
            A DataFrame containing the matched patients, match_group, and n_matches with their original data.
        """

        if not hasattr(self, "matched_data"):
            raise ValueError("Matched data not found. Please run rare_matching() first.")
        
        # Create a copy of the matched data
        if self.matched_data is None:
            raise ValueError("Matched data is not available. Please run rare_matching() before calling this method.")
        
        # check if the precursor includes propensity_logit column or not and only keep it if it does
        if "propensity_logit" not in self.matched_data.columns:
            matched_data = self.matched_data.copy()[[self.patient_id_col, self.exposure_status, "match_group", "n_matches", "match_distance"]]
            # Merge with the original population data, keeping only match_group and n_matches from matched_data
            matched_data = matched_data.merge(
                hide_columns(self.pop, [self.exposure_status]),
                on=self.patient_id_col,
                how='left',
                suffixes=('', '_original')
            )
        else:
            matched_data = self.matched_data.copy()[[self.patient_id_col, self.exposure_status, "propensity_logit", "match_group", "n_matches", "match_distance"]]
            # Merge with the original population data, keeping only match_group and n_matches from matched_data
            matched_data = matched_data.merge(
                hide_columns(self.pop, [self.exposure_status, "propensity_logit"]),
                on=self.patient_id_col,
                how='left',
                suffixes=('', '_original')
            )


        # assign the matched data to self.pop_matched
        self.pop_matched = matched_data.copy()

        self.matched_exposed = self.pop_matched[self.pop_matched[self.exposure_status] == 1].copy()
        self.matched_control = self.pop_matched[self.pop_matched[self.exposure_status] == 0].copy()
        # assigned the unmatched_exposed as the exposed population that is not in matched_exposed
        self.unmatched_exposed = self.pop[self.pop[self.exposure_status] == 1].copy()
        self.unmatched_exposed = self.unmatched_exposed[~self.unmatched_exposed[self.patient_id_col].isin(self.matched_exposed[self.patient_id_col])].copy()
