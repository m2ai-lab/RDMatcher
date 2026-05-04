import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Literal
import logging

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.utils.class_weight import compute_class_weight

import matplotlib.pyplot as plt
import seaborn as sns

from processing import build_preprocessing_pipeline, apply_preprocessing_pipeline
from train import propensity_scores_simple, propensity_scores_full
from matching import simple_matching, multi_covariate_adjusted_matching, matching_diagnostics
from utils import hide_columns
from logger import epilogger



logger = epilogger(__name__, level='INFO')


class RDMatcher:
    def __init__(
        self,
        control_pop,
        exposure_pop,
        exposure_status,
        features_numeric,
        features_categorical,
        features_datetime=None,
        features_log=None,
        features_bin=None,
        patient_id_col="patient_id",
        process_features=False,
        bin_method="binned_scaler",
        bin_width=10,
        debug=False
    ):
        self.control_pop = control_pop
        self.exposure_pop = exposure_pop
        self.exposure_status = exposure_status
        self.features_numeric = features_numeric
        self.features_categorical = features_categorical
        self.all_features = features_numeric + features_categorical
        self.features_datetime = features_datetime if features_datetime is not None else []
        self.features_log = features_log if features_log is not None else []
        self.features_bin = features_bin if features_bin is not None else []
        self.patient_id_col = patient_id_col
        self.bin_method = bin_method
        self.bin_width = bin_width
        debug = debug

        if debug:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)
        logger.info("Initializing RDMatcher with provided control and exposure populations.")
        
        # Validate inputs and concatenate dataframes
        logger.info("Validating inputs and combining control and exposure populations.")
        self._validate_inputs_and_combine()
        logger.info("Inputs validated and populations combined successfully.")

        # Convert datetime features to numeric and add them to features_numeric
        if self.features_datetime:
            self.pop = self._convert_datetime_to_numeric()
            self.features_numeric += [f"{col}_days" for col in self.features_datetime]
            self.all_features += [f"{col}_days" for col in self.features_datetime]
        logger.info("Datetime features converted to numeric and added to features_numeric.")

        # Preprocess features
        if not process_features:
            logger.info("Skipping feature preprocessing as process_features is set to False.")
        else:
            self._preprocess_features()
        
        # initialize as None
        self.matched_data = None
        self.pop_matched = None
        self.matched_exposed = None
        self.matched_control = None
        self.unmatched_exposed = None
        
        
        logger.info("Dataset loaded. Raw data available in self.pop.")


    def _validate_inputs_and_combine(self):
        # Ensure control_pop and exposure_pop are DataFrames
        if not isinstance(self.control_pop, pd.DataFrame) or not isinstance(self.exposure_pop, pd.DataFrame):
            raise ValueError("Both control_pop and exposure_pop should be pandas DataFrames.")
        
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
            logger.debug(f"Converting datetime feature '{col}' to numeric days since the first global occurrence.")
            # Convert to datetime if not already
            if not pd.api.types.is_datetime64_any_dtype(pop_copy[col]):
                pop_copy[col] = pd.to_datetime(pop_copy[col], errors='coerce')
            min_date = pop_copy[col].min()
            pop_copy[f"{col}_days"] = (pop_copy[col] - min_date).dt.total_seconds() / (24 * 60 * 60)
        return pop_copy

    
    def _preprocess_features(self):
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
            logger.info("Data has already been preprocessed. Skipping redundant processing.")
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
            bin_width=self.bin_width
        )

        # Apply preprocessing and get a final DataFrame.
        self.pop_processed = apply_preprocessing_pipeline(df, preprocessor, self.patient_id_col, self.exposure_status)
        self._is_processed = True
        logger.info("Preprocessing complete. Data available in self.pop_processed.")


    # calculate propensity scores
    def calculate_propensity_scores(self, method='simple', random_state=404):
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
                                column 'propensity_score' containing the calculated 
                                propensity scores.
                - pd.Series: A Series containing the propensity scores.
        """
        # Check that the preprocessed data is available.
        if not hasattr(self, "pop_processed"):
            raise ValueError("Preprocessed features not found. Please run _preprocess_features() first.")

        if method == 'simple':
            self.pop_processed, self.propensity_scores = propensity_scores_simple(
                self.pop_processed,
                self.exposure_status,
                self.patient_id_col,
                random_state=random_state
            )
        elif method == 'full':
            self.pop_processed, self.propensity_scores = propensity_scores_full(
                self.pop_processed,
                self.exposure_status,
                random_state=random_state,
                penalty='l1',
                solver='saga'
            )
        else:
            raise ValueError(f"Method '{method}' is not supported. Use 'simple'.")
        # Add propensity scores to the original population data
        self.pop["propensity_score"] = self.pop_processed["propensity_score"].values
        
        logger.info("Propensity scores calculated. Data available in self.pop_processed.")


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
        if not hasattr(self, "propensity_scores") and not hasattr(self, "pop"):
            raise ValueError("Propensity scores not found. Please run calculate_propensity_scores() first.")
        
        if compare_matching and not hasattr(self, "matched_data"):
            raise ValueError("Matched data not found. Please run ps_matching() first.")
        
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
            x='propensity_score',
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
                data=self.matched_data,
                x='propensity_score',
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
        
        return fig, axes



    #-------------------------------
    # Matching
    #-------------------------------
    def rare_matching(self, threshold, n_neighbors, k_candidates=500, **kwargs):
        # verbose=True, method: Literal['multi', 'simple'] = 'multi', weight_propensity=1.0, weight_numeric=1.0
        """
        Perform propensity score matching on the preprocessed population data.
        This function matches patients based on their propensity scores and other features.
        It supports both simple matching and multi-covariate adjusted matching methods.
        Parameters:
        ----------
        threshold : float
            The maximum allowable difference in multidimensional space across covariates for matching.
        n_neighbors : int
            The number of nearest neighbors to consider for matching.
        k_candidates : int, optional
            The number of candidates to consider for matching in multi-covariate adjusted matching.
        **kwargs : dict
            Additional keyword arguments to pass to the matching functions, such as:
            - method: str, default 'multi'
                The matching method to use ('multi' for multi-covariate adjusted matching, 'simple' for simple matching).
            - propensity_col: str, default 'propensity_score'
                The column name for the propensity scores in the DataFrame.
            - global_optimal: bool, default True
                Whether to use global optimal matching in multi-covariate adjusted matching.
            - replacement: bool, default False
                Whether to allow replacement in multi-covariate adjusted matching.
        Optionally Returns:
        -------
        pd.DataFrame
            A DataFrame containing the matched patients, match_group, and n_matches with their original data.
            Only returned if `return_matched_data` is set to True in kwargs.
        """
        # Check that the preprocessed data and propensity scores are available.
        if not hasattr(self, "pop_processed"):
            raise ValueError("Preprocessed features not found. Please run _preprocess_features() first.")
        if not hasattr(self, "propensity_scores"):
            raise ValueError("Propensity scores not found. Please run calculate_propensity_scores() first.")
        
        debug = kwargs.get('debug', False)
        diags = kwargs.get('diagnostics', False)
        return_matched_data = kwargs.get('return_matched_data', False)
        
        # add missing kwargs
        method = kwargs.get('method', 'multi')  # default to 'multi' if not specified
        propensity_col = kwargs.get('propensity_col', 'propensity_score')  # default to 'propensity_score' if not specified
        if 'verbose' not in kwargs: # if kwargs does not contain verbose, set it to True
            kwargs['verbose'] = True

        if method == 'simple':
            matched_data = simple_matching(
                df=self.pop_processed,
                exposure_status=self.exposure_status,
                threshold=threshold,
                n_neighbors=n_neighbors,
                **kwargs
            )
        
        elif method == 'multi':
            # check required kwargs and error out if not
            if 'global_optimal' not in kwargs:
                raise ValueError("global_optimal must be specified in kwargs for multi-covariate adjusted matching.")
            if 'replacement' not in kwargs:
                raise ValueError("replacement must be specified in kwargs for multi-covariate adjusted matching.")
            matched_data = multi_covariate_adjusted_matching(
                df=self.pop_processed,
                exposure_status=self.exposure_status,
                propensity_col=propensity_col,
                patient_id=self.patient_id_col,
                threshold=threshold,
                n_neighbors=n_neighbors,
                k_candidates=k_candidates,
                features_numeric=self.features_numeric,
                features_categorical=self.features_categorical,
                **kwargs,
            )
        else:
            raise ValueError(f"Method '{method}' is not supported. Use 'multi' or 'simple'.")

        self.matched_data = matched_data

        # print completion message
        logger.info(f"Matching complete. {len(set(matched_data[self.patient_id_col]))} patients matched.")

        # link matched data with the original population data
        self._link_matched_data()

        if debug or diags:
            logger.info("Running matching diagnostics.")
            matching_diagnostics(self.pop_matched, self.exposure_status, features_numeric=self.features_numeric, features_categorical=self.features_categorical,)
            logger.info("Matched data linked with original population data.")

        # return the matched data if requested
        if return_matched_data:
            return self.pop_matched


    def _link_matched_data(self):
        """
        Link matched data with the original population data.
        
        Returns:
        --------
        pd.DataFrame
            A DataFrame containing the matched patients, match_group, and n_matches with their original data.
        """

        if not hasattr(self, "matched_data"):
            raise ValueError("Matched data not found. Please run ps_matching() first.")
        
        # Create a copy of the matched data
        if self.matched_data is None:
            raise ValueError("Matched data is not available. Please run ps_matching() before calling this method.")
        
        matched_data = self.matched_data.copy()[[self.patient_id_col, self.exposure_status, "propensity_score", "match_group", "n_matches"]]

        # Merge with the original population data, keeping only match_group and n_matches from matched_data
        matched_data = matched_data.merge(
            hide_columns(self.pop, [self.exposure_status, "propensity_score"]),
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
