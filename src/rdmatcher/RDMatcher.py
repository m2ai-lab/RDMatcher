from .options import MatchRequest, MatchingOptions, ExecutionOptions
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Literal, Optional
import logging
import os

from .processing import build_preprocessing_pipeline, apply_preprocessing_pipeline
from .train import propensity_logits_simple
from .formula import parse_formula
from .matcher import Matcher
from .matching import matching_diagnostics
from .candidate_graph import CandidateGraph
from .utils import hide_columns




class RDMatcher:
    @classmethod
    def from_distance_matrix(
        cls,
        pop_df,
        patient_id_col,
        exposure_status,
        distance_matrix,
        *,
        case_ids=None,
        control_ids=None,
        debug=False,
        log_file: Optional[str] = None,
        log_to_console: bool = False,
    ):
        """Build an RDMatcher from authoritative case-by-control costs.

        This is equivalent to constructing ``RDMatcher`` with empty feature
        lists and the supplied matrix metadata.
        """
        return cls(
            pop_df, patient_id_col, exposure_status, [], [],
            debug=debug, log_file=log_file, log_to_console=log_to_console,
            distance_matrix=distance_matrix, case_ids=case_ids,
            control_ids=control_ids,
        )

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
        log_file: Optional[str] = None,
        log_to_console: bool = False,
        *,
        distance_matrix=None,
        case_ids=None,
        control_ids=None
    ):

        self._precomputed_distances = None
        if distance_matrix is None and (case_ids is not None or control_ids is not None):
            raise ValueError("case_ids/control_ids require distance_matrix")
        if distance_matrix is not None and (process_features or features_datetime or features_log
                                            or features_bin or onehot or onehot_scalar):
            raise ValueError("Feature transforms cannot be used with precomputed distances")
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

        self._configure_logger(debug, log_file=log_file, console=log_to_console)
        self.logger.info("Initializing RDMatcher with provided control and exposure populations.")

        self.logger.info("Validating inputs and combining control and exposure populations.")
        self._validate_inputs_and_combine()
        self.logger.info("Inputs validated and populations combined successfully.")
        if distance_matrix is not None:
            from ._precomputed import PrecomputedDistances
            if not self.pop_df[self.exposure_status].isin([0, 1]).all():
                raise ValueError("Precomputed matching requires binary exposure status (0 or 1)")
            self._precomputed_distances = PrecomputedDistances(
                distance_matrix, case_ids, control_ids,
                self.exposure_pop[self.patient_id_col], self.control_pop[self.patient_id_col])

        if self.features_datetime:
            self.pop = self._convert_datetime_to_numeric()
            self.features_numeric += [f"{col}_days" for col in self.features_datetime]
            self.all_features += [f"{col}_days" for col in self.features_datetime]
            self.logger.info("Datetime features converted to numeric and added to features_numeric.")

        if not process_features:
            self.logger.info("Skipping feature preprocessing as process_features is set to False.")
            self.pop_processed = self.pop.copy()
        else:
            if onehot:
                self.logger.info("One-hot encoding is enabled and will be applied to non-Gower matching paths; Gower will keep raw categorical semantics.")
            else:
                self.logger.info("One-hot encoding disabled. Recommended for Gower distance.")
            self._process_features()


        self.matched_data = None
        self.pop_matched = None
        self.matched_exposed = None
        self.matched_control = None
        self.unmatched_exposed = None


        self.logger.info("Dataset loaded. Raw data available in self.pop.")

        from .feature_map import build_feature_name_maps

        processed_cols = [col for col in getattr(self, 'pop_processed', pd.DataFrame()).columns if col not in [self.patient_id_col, self.exposure_status]]
        p2o, o2p = build_feature_name_maps(processed_cols, self.features_numeric, self.features_categorical, self.features_datetime)
        self._processed_to_original = p2o
        self._original_to_processed = o2p


    def _build_feature_name_maps(self):
        from .feature_map import build_feature_name_maps

        processed_cols = [col for col in getattr(self, 'pop_processed', pd.DataFrame()).columns if col not in [self.patient_id_col, self.exposure_status]]
        p2o, o2p = build_feature_name_maps(processed_cols, self.features_numeric, self.features_categorical, self.features_datetime)
        self._processed_to_original = p2o
        self._original_to_processed = o2p



    def _configure_logger(self, debug: bool, log_file, console: bool = False):
        self.logger = logging.getLogger('rdmatcher')
        enabled = bool(console or log_file)
        level = logging.DEBUG if debug else logging.INFO
        if not enabled:
            for handler in self.logger.handlers[:]:
                self.logger.removeHandler(handler)
            self.logger.addHandler(logging.NullHandler())
            self.logger.setLevel(logging.CRITICAL + 1)
            self.logger.propagate = False
            return

        self.logger.setLevel(level)

        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        if console:
            console_h = logging.StreamHandler()
            console_h.setLevel(level)
            console_h.setFormatter(fmt)
            self.logger.addHandler(console_h)

        if log_file:
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)

            file_h = logging.FileHandler(log_file)
            file_h.setLevel(level)
            file_h.setFormatter(fmt)
            self.logger.addHandler(file_h)

        self.logger.propagate = False

        self.logger.debug(
            f"Logger configured. Level={level}, Handlers={len(self.logger.handlers)} "
            f"(Console={console}, File={log_file})"
        )


    def _validate_inputs_and_combine(self):

        if not isinstance(self.pop_df, pd.DataFrame):
            raise ValueError("pop_df should be a pandas DataFrame.")

        self.control_pop = self.pop_df[self.pop_df[self.exposure_status] == 0].copy()
        self.logger.info(f"Control population size: {len(self.control_pop)}")
        self.exposure_pop = self.pop_df[self.pop_df[self.exposure_status] == 1].copy()
        self.logger.info(f"Exposure population size: {len(self.exposure_pop)}")

        if self.features_log:
            for feature in self.features_log:
                if feature not in self.features_numeric:
                    raise ValueError(f"Feature {feature} is in features_log but not in features_numeric.")
        if self.features_bin:
            for feature in self.features_bin:
                if feature not in self.features_numeric:
                    raise ValueError(f"Feature {feature} is in features_bin but not in features_numeric.")

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

        for col in [self.exposure_status, self.patient_id_col]:
            if col not in self.control_pop.columns or col not in self.exposure_pop.columns:
                raise ValueError(f"{col} is missing in control_pop or exposure_pop.")

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
            if not pd.api.types.is_datetime64_any_dtype(pop_copy[col]):
                pop_copy[col] = pd.to_datetime(pop_copy[col], errors='coerce')
            min_date = pop_copy[col].min()
            pop_copy[f"{col}_days"] = (pop_copy[col] - min_date).dt.total_seconds() / (24 * 60 * 60)
        return pop_copy


    def _process_features(self):
        """Preprocess features by converting datetime to numeric values, binning (optional),"""
        if hasattr(self, '_is_processed') and self._is_processed:
            self.logger.info("Data has already been preprocessed. Skipping redundant processing.")
            return

        df = self.pop.copy()

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

        self.logger.info("Applying preprocessing pipeline to the dataset.")
        self.pop_processed = apply_preprocessing_pipeline(df, preprocessor, self.patient_id_col, self.exposure_status)
        self._is_processed = True
        self.all_features = [col for col in self.pop_processed.columns if col not in [self.patient_id_col, self.exposure_status]]
        self.logger.info("Preprocessing complete. Data available in self.pop_processed.")


    def process_features(self):
        """
        Public method to preprocess features if not already done.
        """
        self._process_features()

    def _build_processed_to_original(self, local_processed, original_to_processed):
        """Map each processed column back to its original feature name."""
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
    ):
        """Build Gower weights that make SD-scale numeric shifts comparable."""
        if gower_sd_weights_mult <= 0:
            raise ValueError("gower_sd_weights_mult must be positive.")
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

        if numeric_features:
            numeric = reference[numeric_features].astype(float)
            ranges = numeric.max(skipna=True) - numeric.min(skipna=True)
            sd = numeric.std(skipna=True, ddof=1)
            raw = ranges / (float(gower_sd_weights_mult) * sd)
            raw = raw.replace([np.inf, -np.inf], np.nan).fillna(1.0)
            raw = raw.clip(lower=1e-8)
            for feature in numeric_features:
                weights[feature] = float(raw[feature])

        if categorical_features:
            weights.update({feature: 1.0 for feature in categorical_features})

        missing = [feature for feature in match_features if feature not in weights]
        for feature in missing:
            weights[feature] = 0.0

        self.logger.info(
            f"Built Gower SD weights using {reference_label} reference pool "
            f"({len(reference)} rows): "
            f"mult={gower_sd_weights_mult}, per_feature_calibration=True, weights={weights}"
        )
        return weights

    def _build_multi_matching_view(self, distance_metric: str):
        """Build the cohort view used by multi-covariate matching."""
        match_features = list(dict.fromkeys(self.features_numeric + self.features_categorical))
        base_cols = match_features + [self.patient_id_col, self.exposure_status]

        if distance_metric == 'gower':
            matching_data = self.pop[base_cols].copy()
            processed_to_original_map = {f: f for f in match_features}
            original_to_processed_map = {f: [f] for f in match_features}
            return matching_data, match_features, processed_to_original_map, original_to_processed_map

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

        self._processed_to_original = processed_to_original_map
        self._original_to_processed = original_to_processed_map
        self._matching_processed_cols = processed_cols

        return matching_data, processed_cols, processed_to_original_map, original_to_processed_map

    def fit_propensity_model(self, formula: Optional[str]=None, random_state=404, **kwargs):
        """Fit logistic propensity scores and store scores, logits and model metadata.

        formula specifies model terms; None includes all numeric and categorical
        features as main effects. Additional arguments configure the estimator."""

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
        from .feature_map import build_feature_name_maps
        processed_cols = [c for c in local_processed.columns if c not in [self.patient_id_col, self.exposure_status]]
        _, original_to_processed = build_feature_name_maps(processed_cols, self.features_numeric, self.features_categorical, self.features_datetime)

        if formula is None:
            self.logger.info("No formula provided for propensity modeling; using all original features as main effects.")
            formula_terms = [(f,) for f in (self.features_numeric + self.features_categorical)]
        else:
            formula_terms = parse_formula(formula)

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

        if 'propensity_logit' in psm_df.columns:
            prop_map = psm_df[[self.patient_id_col, 'propensity_logit']].set_index(self.patient_id_col)
            if 'propensity_score' in psm_df.columns:
                prop_map = psm_df[[self.patient_id_col, 'propensity_score', 'propensity_logit']].set_index(self.patient_id_col)
            if psm_df[self.patient_id_col].duplicated().any() or self.pop[self.patient_id_col].duplicated().any():
                raise ValueError("Patient IDs must be unique for propensity-score alignment.")
            score_columns = ['propensity_score', 'propensity_logit']
            self.pop = self.pop.drop(columns=[c for c in score_columns if c in self.pop], errors='ignore')
            self.pop = self.pop.merge(
                prop_map.reset_index(), on=self.patient_id_col, how='left', validate='one_to_one'
            )
            if self.pop[['propensity_score', 'propensity_logit']].isna().any().any():
                raise RuntimeError("Propensity-score ID merge left unmatched patients.")
            if hasattr(self, 'pop_processed') and self.pop_processed is not None:
                self.pop_processed = self.pop_processed.drop(
                    columns=[c for c in score_columns if c in self.pop_processed], errors='ignore'
                ).merge(prop_map.reset_index(), on=self.patient_id_col, how='left', validate='one_to_one')
                if self.pop_processed[['propensity_score', 'propensity_logit']].isna().any().any():
                    raise RuntimeError("Propensity-score ID merge left unmatched processed patients.")

            if 'propensity_logit' not in self.all_features:
                self.all_features.append('propensity_logit')

            try:
                self._build_feature_name_maps()
            except Exception:
                self.logger.warning("Failed to rebuild feature name maps after adding propensity_logit.")

        self._propensity_feature_map = {
            'formula': formula,
            'parsed_terms': formula_terms,
            'meta': psm_meta,
            'original_to_processed_psm': original_to_processed,
            'design_matrix_columns': psm_meta['design_matrix_columns'],
            'categorical_reference_levels': categorical_references,
        }
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
        if not hasattr(self, "propensity_logits") and not hasattr(self, "pop"):
            raise ValueError("Propensity scores not found. Run fit_propensity_model() first.")

        if compare_matching and not hasattr(self, "pop_matched"):
            raise ValueError("Matched data not found. Run match() first.")

        n_axes = 2 if compare_matching else 1
        fig, axes = plt.subplots(1, n_axes, figsize=figsize)
        if n_axes == 1:
            axes = [axes]

        exposure_status_order = sorted(self.pop[self.exposure_status].unique(), reverse=True)

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



    @staticmethod
    def _resolve_ratio(ratio, n_neighbors):
        if n_neighbors is None:
            return ratio
        if ratio != 1:
            raise TypeError("Specify either ratio or legacy n_neighbors, not both")
        return n_neighbors

    def return_graph(self, threshold: float | MatchRequest | None = None, ratio=1, *,
                     n_neighbors=None, distance_metric=None, matching=None,
                     execution=None, request: MatchRequest | None = None) -> CandidateGraph:
        """Return the certified candidate graph before allocation.

        ``request=MatchRequest(...)`` uses the request's threshold, ratio,
        distance metric, matching, and execution settings. Its allocation
        setting is intentionally irrelevant because no allocation is run.
        """
        if isinstance(threshold, MatchRequest):
            if request is not None:
                raise TypeError("Pass a MatchRequest either positionally or by request=, not both")
            request = threshold
            threshold = None
        if request is not None:
            if (threshold is not None or ratio != 1 or n_neighbors is not None
                    or distance_metric is not None or matching is not None
                    or execution is not None):
                raise TypeError("MatchRequest cannot be combined with individual graph arguments")
            return self._run_matching(
                request.threshold, request.ratio,
                distance_metric=request.distance_metric, matching=request.matching,
                execution=request.execution, return_candidate_graph=True,
            )
        if threshold is None:
            raise TypeError("threshold is required unless request=MatchRequest(...) is provided")
        ratio = self._resolve_ratio(ratio, n_neighbors)
        return self._run_matching(threshold, ratio, distance_metric=distance_metric,
                                  matching=matching, execution=execution, return_candidate_graph=True)

    def propensity_eligible_sets(self, ps_caliper=0.2, ps_caliper_strict=True,
                                 propensity_score_col="propensity_logit"):
        """Return control-position neighborhoods used by native PS-hybrid matching."""
        if (not hasattr(self, 'propensity_logits') or self.propensity_logits is None) and self._precomputed_distances is None:
            raise ValueError("Fit the propensity model before constructing neighborhoods")
        if hasattr(self, 'pop_processed') and propensity_score_col in self.pop_processed.columns:
            pop_ps = self.pop_processed
        elif propensity_score_col in self.pop.columns:
            pop_ps = self.pop
        else:
            raise ValueError(
                f"Propensity eligibility requires a supplied or fitted '{propensity_score_col}' column."
            )
        if not np.isfinite(pop_ps[propensity_score_col].to_numpy(dtype=float)).all():
            raise ValueError("Propensity scores must be finite for every patient")
        treated_mask = pop_ps[self.exposure_status] == 1
        control_mask = pop_ps[self.exposure_status] == 0

        ps_treated = pop_ps.loc[treated_mask, propensity_score_col].values
        ps_control = pop_ps.loc[control_mask, propensity_score_col].values

        caliper_sd = np.std(pop_ps[propensity_score_col].values, ddof=1)
        caliper_width = ps_caliper * caliper_sd
        self.logger.info(
            f"PS-Hybrid caliper: {ps_caliper} × SD(logit PS) = {ps_caliper} × {caliper_sd:.4f} = {caliper_width:.4f}"
        )

        ps_eligible_sets = []
        n_empty = 0
        for i, t_ps in enumerate(ps_treated):
            eligible = np.where(np.abs(ps_control - t_ps) <= caliper_width)[0]
            if len(eligible) == 0 and not ps_caliper_strict:
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
            self.logger.debug(
                f"{n_empty} treated units have zero eligible controls with strict caliper. "
                f"These units will have no candidates within threshold."
            )
        self.propensity_eligibility_diagnostics_ = {
            "propensity_caliper_width": float(caliper_width),
            "propensity_caliper_strict": bool(ps_caliper_strict),
            "propensity_empty_neighborhoods": int(n_empty),
            "eligibility_rule": "RDMatcher native 0.2 pooled sample SD logit caliper" if ps_caliper == 0.2 else "RDMatcher native pooled sample SD logit caliper",
        }
        return ps_eligible_sets

    def match(self, threshold: float | MatchRequest | None = None, ratio: int = 1, *,
              n_neighbors=None, distance_metric=None, allocation='optimal',
              matching: MatchingOptions | None = None,
              execution: ExecutionOptions | None = None,
              request: MatchRequest | None = None) -> pd.DataFrame:
        """Match cases without replacement and return population records.

        Prefer ``ratio`` for controls per case. ``n_neighbors`` remains a
        compatibility alias. Pass ``request=MatchRequest(...)`` to reuse a
        complete matching configuration.
        """
        if isinstance(threshold, MatchRequest):
            if request is not None:
                raise TypeError("Pass a MatchRequest either positionally or by request=, not both")
            request = threshold
            threshold = None
        if request is not None:
            if (threshold is not None or ratio != 1 or n_neighbors is not None
                    or distance_metric is not None or allocation != 'optimal'
                    or matching is not None or execution is not None):
                raise TypeError("MatchRequest cannot be combined with individual matching arguments")
            return self._run_matching(
                request.threshold, request.ratio,
                distance_metric=request.distance_metric, allocation=request.allocation,
                matching=request.matching, execution=request.execution,
            )
        if threshold is None:
            raise TypeError("threshold is required unless request=MatchRequest(...) is provided")
        ratio = self._resolve_ratio(ratio, n_neighbors)
        return self._run_matching(threshold, ratio, distance_metric=distance_metric,
                                  allocation=allocation, matching=matching, execution=execution)

    def rare_matching(self, threshold: float, ratio: int = 1, *,
                      n_neighbors=None,
                      distance_metric=None, allocation='optimal',
                      matching: MatchingOptions | None = None,
                      execution: ExecutionOptions | None = None) -> pd.DataFrame:
        """Compatibility alias for :meth:`match`."""
        return self.match(
            threshold, ratio, n_neighbors=n_neighbors,
            distance_metric=distance_metric, allocation=allocation,
            matching=matching, execution=execution,
        )

    def _run_matching(self, threshold, n_neighbors, *, distance_metric=None,
                      allocation='optimal', matching=None, execution=None,
                      return_candidate_graph=False):
        precomputed = self._precomputed_distances is not None
        if distance_metric is None:
            distance_metric = 'precomputed' if precomputed else 'gower'
        if precomputed and distance_metric != 'precomputed':
            raise ValueError("distance_matrix conflicts with computed distance_metric")
        if distance_metric == 'precomputed' and not precomputed:
            raise ValueError("distance_metric='precomputed' requires distance_matrix")
        if allocation not in {'optimal', 'hybrid'}:
            raise ValueError("allocation must be 'optimal' or 'hybrid'")
        if isinstance(n_neighbors, bool) or not isinstance(n_neighbors, (int, np.integer)) or n_neighbors < 1:
            raise ValueError('n_neighbors must be a positive integer')
        if precomputed and int(n_neighbors) * len(self.exposure_pop) > np.iinfo(np.int32).max:
            raise ValueError('Requested case-slots exceed the supported int32 assignment-index range')
        if np.isnan(threshold) or threshold < 0:
            raise ValueError('threshold must be non-negative')
        matching = MatchingOptions() if matching is None else matching
        execution = ExecutionOptions() if execution is None else execution
        if not isinstance(matching, MatchingOptions) or not isinstance(execution, ExecutionOptions):
            raise TypeError('Use MatchingOptions and ExecutionOptions for settings')
        if precomputed and (matching.method != 'multi' or matching.gower_weights is not None
                            or matching.gower_sd_weights or matching.pca_filter
                            or matching.weight_numeric != 1.0 or matching.weight_propensity != 1.0
                            or matching.gower_sd_reference != 'controls'
                            or matching.gower_sd_weights_mult != 1.96):
            raise ValueError("Distance transforms and propensity-only costs conflict with precomputed distances")
        method = matching.method
        ps_hybrid, ps_caliper = matching.ps_hybrid, matching.ps_caliper
        propensity_caliper = matching.propensity_caliper
        ps_caliper_strict = matching.ps_caliper_strict
        gower_sd_weights = matching.gower_sd_weights
        gower_sd_reference = matching.gower_sd_reference
        gower_sd_weights_mult = matching.gower_sd_weights_mult
        return_unmatched_cases = matching.return_unmatched_cases
        prioritize_coverage = matching.prioritize_coverage
        n_jobs, parallel_chunk_size = execution.n_jobs, execution.parallel_chunk_size
        streaming, stream_block_size = execution.streaming, execution.stream_block_size
        stream_threshold_gb, memory_limit_gb = execution.stream_threshold_gb, execution.memory_limit_gb
        solver = execution.solver
        global_optimal, competitive_match, replacement = True, allocation == 'hybrid', False
        k_candidates = None
        if not hasattr(self, "pop_processed"):
            self.logger.info("Preprocessing pipeline not initiated. Using raw pop data.")
            self.pop_processed = self.pop.copy()

        propensity_col = None
        if method == 'propensity' and hasattr(self, "propensity_logits"):
            propensity_col = matching.propensity_col
        propensity_score_col = matching.propensity_col

        include_unmatched = bool(return_unmatched_cases)

        if distance_metric == 'gower' and self.onehot:
            self.logger.warning(
                "Gower distance is selected while one-hot encoding is enabled. "
                "Gower will use raw categorical semantics and ignore the one-hot view."
            )
        self.logger.info(f"Starting matching using method: {method}")

        if ps_hybrid and distance_metric not in {"gower", "mahalanobis", "precomputed"}:
            raise ValueError("PS-Hybrid mode (ps_eligible_sets) only supports Gower or Mahalanobis distance metric.")
        if gower_sd_weights and distance_metric != "gower":
            raise ValueError("gower_sd_weights=True is only supported with distance_metric='gower'.")
        if gower_sd_weights and method != "multi":
            raise ValueError("gower_sd_weights=True is only supported with method='multi'.")
        if gower_sd_weights and matching.gower_weights is not None:
            raise ValueError("Use either gower_sd_weights=True or explicit gower_weights, not both.")
        if gower_sd_weights and gower_sd_reference not in {"controls", "pooled"}:
            raise ValueError("gower_sd_reference must be one of {'controls', 'pooled'}.")

        ps_eligible_sets = None
        if ps_hybrid or (precomputed and propensity_caliper is not None):
            if method != 'multi':
                raise ValueError("ps_hybrid=True requires method='multi'.")
            if (not hasattr(self, 'propensity_logits') or self.propensity_logits is None) and not precomputed:
                raise ValueError(
                    "ps_hybrid=True requires propensity scores. "
                    "Run fit_propensity_model() before match()."
                )
            ps_eligible_sets = self.propensity_eligible_sets(
                ps_caliper=propensity_caliper if precomputed and propensity_caliper is not None else ps_caliper, ps_caliper_strict=ps_caliper_strict,
                propensity_score_col=propensity_score_col)

        if method == 'propensity':
            if propensity_col is None:
                raise ValueError("Propensity scores are required for 'propensity' matching method.")

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

            weight_propensity = 1.0
            weight_numeric = 0.0
            if propensity_col not in self.pop.columns:
                raise ValueError(f"Propensity column '{propensity_col}' not found in population. Run fit_propensity_model() first.")

        elif method == 'multi':
            weight_propensity = matching.weight_propensity
            weight_numeric = matching.weight_numeric

        else:
            raise ValueError(f"Method '{method}' is not supported. Use 'multi' or 'propensity'.")

        if precomputed:
            matching_data = self.pop[[self.patient_id_col, self.exposure_status]].copy()
            self.matching_data_ = matching_data.copy()
            matcher = Matcher(
                df=matching_data, exposure_status=self.exposure_status,
                patient_id=self.patient_id_col, distance_metric='precomputed',
                threshold=threshold, n_neighbors=n_neighbors,
                precomputed_distances=self._precomputed_distances,
                ps_eligible_sets=ps_eligible_sets, logger=self.logger,
                n_jobs=n_jobs)
        elif method == 'propensity':
            propensity_col = matching.propensity_col
            matching_features = [propensity_col]
            matching_data = self.pop[[propensity_col, self.patient_id_col, self.exposure_status]].copy()
            self.logger.info(f"Instantiating Matcher for propensity-only matching with {len(matching_data)} records and feature: {matching_features}")
            self.matching_data_ = matching_data.copy()

            processed_to_original_map = {propensity_col: propensity_col}
            original_to_processed_map = {propensity_col: [propensity_col]}

            gower_weights = matching.gower_weights

            matcher = Matcher(
                df=matching_data[[propensity_col, self.patient_id_col, self.exposure_status]],
                exposure_status=self.exposure_status,
                patient_id=self.patient_id_col,
                distance_metric=distance_metric,
                threshold=threshold,
                n_neighbors=n_neighbors,
                weight_numeric=0.0,
                weight_propensity=1.0,
                propensity_col=propensity_col,
                gower_weights=gower_weights,
                gower_cat_features=None,
                feature_name_map_processed_to_original=processed_to_original_map,
                feature_name_map_original_to_processed=original_to_processed_map,
                original_categorical_features=[],
                pca_filter=matching.pca_filter,
                n_jobs=n_jobs,
                parallel_chunk_size=parallel_chunk_size,
                streaming=streaming,
                stream_block_size=stream_block_size,
                stream_threshold_gb=stream_threshold_gb,
                memory_limit_gb=memory_limit_gb,
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
                gower_cat_features = default_gower_cat_features
                if gower_cat_features is not None:
                    missing_gower_cat_features = sorted(set(gower_cat_features).difference(match_features))
                    if missing_gower_cat_features:
                        raise ValueError(
                            f"gower_cat_features contains columns not used for matching: {missing_gower_cat_features}"
                        )
            else:
                gower_cat_features = None
            gower_weights = matching.gower_weights
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
                weight_numeric=weight_numeric,
                weight_propensity=weight_propensity,
                propensity_col=propensity_col,
                gower_weights=gower_weights,
                gower_cat_features=gower_cat_features,
                feature_name_map_processed_to_original=native_processed_to_original,
                feature_name_map_original_to_processed=native_original_to_processed,
                original_categorical_features=list(self.features_categorical),
                ps_eligible_sets=ps_eligible_sets,
                pca_filter=matching.pca_filter,
                n_jobs=n_jobs,
                parallel_chunk_size=parallel_chunk_size,
                streaming=streaming,
                stream_block_size=stream_block_size,
                stream_threshold_gb=stream_threshold_gb,
                memory_limit_gb=memory_limit_gb,
                logger=self.logger,
            )

        match_result = matcher.match(
            k_candidates=k_candidates,
            global_optimal=global_optimal,
            replacement=replacement,
            competitive_match=competitive_match,
            safe_matches=n_neighbors,
            fuzzy_threshold=False,
            fuzzy_threshold_limit=None,
            solver=solver,
            return_candidate_graph=return_candidate_graph,
            prioritize_coverage=prioritize_coverage,
            batch_size=execution.batch_size,
            log_matching_summary=False,
        )
        self.matching_candidate_diagnostics = getattr(matcher, 'ps_hybrid_diagnostics_', None)
        self.assignment_diagnostics_ = dict(getattr(matcher, 'assignment_diagnostics_', {}))

        if return_candidate_graph:
            self.candidate_graph_ = match_result
            return match_result

        self.matched_data = match_result

        self.logger.info(f"Matching complete. {len(set(self.matched_data[self.patient_id_col]))} patients matched.")

        self._link_matched_data()

        self.logger.info("Running matching diagnostics.")
        plot_features = False
        self.summary_table = matching_diagnostics(
            self.pop_matched,
            self.exposure_status,
            features_numeric=self.features_numeric,
            features_categorical=self.features_categorical,
            plot_features=plot_features
        )

        if include_unmatched:
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
                        unmatched = unmatched.merge(self.pop[[self.patient_id_col, 'propensity_logit']], on=self.patient_id_col, how='left')
                    else:
                        unmatched['propensity_logit'] = np.nan
                existing_ids = set(self.pop_matched[self.patient_id_col].astype(object).tolist())
                to_add = unmatched[~unmatched[self.patient_id_col].isin(existing_ids)]
                if not to_add.empty:
                    self.pop_matched = pd.concat([self.pop_matched, to_add], ignore_index=True, sort=False)
        else:
            if 'n_matches' in self.pop_matched.columns:
                self.pop_matched = self.pop_matched[self.pop_matched['n_matches'] > 0].reset_index(drop=True)
            else:
                if 'match_group' in self.pop_matched.columns:
                    self.pop_matched = self.pop_matched[self.pop_matched['match_group'].notnull()].reset_index(drop=True)

        return self.pop_matched


    def _link_matched_data(self):
        """Link matched data with the original population data."""

        if not hasattr(self, "matched_data"):
            raise ValueError("Matched data not found. Run match() first.")

        if self.matched_data is None:
            raise ValueError("Matched data is not available. Run match() before calling this method.")

        if "propensity_logit" not in self.matched_data.columns:
            matched_data = self.matched_data.copy()[[self.patient_id_col, self.exposure_status, "match_group", "n_matches", "match_distance"]]
            matched_data = matched_data.merge(
                hide_columns(self.pop, [self.exposure_status]),
                on=self.patient_id_col,
                how='left',
                suffixes=('', '_original')
            )
        else:
            matched_data = self.matched_data.copy()[[self.patient_id_col, self.exposure_status, "propensity_logit", "match_group", "n_matches", "match_distance"]]
            matched_data = matched_data.merge(
                hide_columns(self.pop, [self.exposure_status, "propensity_logit"]),
                on=self.patient_id_col,
                how='left',
                suffixes=('', '_original')
            )


        self.pop_matched = matched_data.copy()

        self.matched_exposed = self.pop_matched[self.pop_matched[self.exposure_status] == 1].copy()
        self.matched_control = self.pop_matched[self.pop_matched[self.exposure_status] == 0].copy()
        self.unmatched_exposed = self.pop[self.pop[self.exposure_status] == 1].copy()
        self.unmatched_exposed = self.unmatched_exposed[~self.unmatched_exposed[self.patient_id_col].isin(self.matched_exposed[self.patient_id_col])].copy()
