import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.utils.class_weight import compute_class_weight

import matplotlib.pyplot as plt
import seaborn as sns


class EpiMatcher:
    def __init__(
        self,
        control_pop,
        exposure_pop,
        exposure_status,
        numeric_features,
        categorical_features,
        datetime_features=None,
        patient_id_col="patient_id",
        process_features=False
    ):
        self.control_pop = control_pop
        self.exposure_pop = exposure_pop
        self.exposure_status = exposure_status
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.all_features = numeric_features + categorical_features
        self.datetime_features = datetime_features if datetime_features is not None else []
        self.patient_id_col = patient_id_col

            # Validate inputs and concatenate dataframes
        self._validate_inputs_and_combine()
    
            # Convert datetime columns to datetime type during initialization
        if self.datetime_features:
            self.pop = self._convert_to_datetime()

        # Preprocess features (scaling and encoding)
        if process_features:
            # Preprocess features (standardize numeric, one-hot encode categorical).
            self._preprocess_features()
        # else:
        #     # If features are not preprocessed, ensure the processed DataFrame is set.
        #     if not hasattr(self, 'pop_processed'):
        #         raise ValueError("Features have not been preprocessed. Please manually run _preprocess_features() first or set process_features=True.")
    


    def _validate_inputs_and_combine(self):
        # Ensure control_pop and exposure_pop are DataFrames
        if not isinstance(self.control_pop, pd.DataFrame) or not isinstance(self.exposure_pop, pd.DataFrame):
            raise ValueError("Both control_pop and exposure_pop should be pandas DataFrames.")

        # Check for missing features in control and exposure populations
        for feature_set, name in [
            (self.numeric_features, "numeric_features"),
            (self.categorical_features, "categorical_features"),
            (self.datetime_features, "datetime_features"),
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
            raise ValueError("Missing treatment values found in the combined dataset.")


    def _convert_to_datetime(self):
        df = self.pop.copy()
        for col in self.datetime_features:
            df[col] = pd.to_datetime(df[col])
        return df
    
    def _convert_datetime_to_numeric(self):
        pop_copy = self.pop.copy()
        for col in self.datetime_features:
            min_date = pop_copy[col].min()
            pop_copy[f"{col}_days"] = (pop_copy[col] - min_date).dt.total_seconds() / (24 * 60 * 60)
        return pop_copy

    
    # make a method to scale the features and one-hot encode the categorical features
    def _preprocess_features(self, bin_features=None, binning_denominator=10):
        # check if the features are already preprocessed
        if hasattr(self, 'pop_processed'):
            # if the features are preprocessed, escape the function
            return
        else:
            if self.datetime_features:
                self.pop = self._convert_datetime_to_numeric(self.pop)
                # add the new datetime features to the numeric features
                self.numeric_features += [f"{col}_days" for col in self.datetime_features]
            # Handle binning if specified
            if bin_features:
                def bin_function(col):
                    return (col - 1) // binning_denominator + 1
                for feature in bin_features:
                    self.pop[feature] = self.pop[feature].apply(bin_function)
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), self.numeric_features),
                    ('cat', OneHotEncoder(drop='first', sparse_output=False), self.categorical_features)
                ],
                verbose_feature_names_out=False
            )
            

            X_features = self.pop[self.numeric_features + self.categorical_features]
            X_transformed = preprocessor.fit_transform(X_features)
            # reset all_features to the new features
            self.all_features = preprocessor.get_feature_names_out()
            
            self.pop_processed = pd.DataFrame(X_transformed, columns=preprocessor.get_feature_names_out())
            self.pop_processed[self.exposure_status] = self.pop[self.exposure_status].values
            self.pop_processed[self.patient_id_col] = self.pop[self.patient_id_col].values

            # set the order of the columns to be the same as the original dataframe
            self.pop_processed = self.pop_processed[[self.patient_id_col, self.exposure_status] + list(self.all_features)]






    def calculate_propensity_scores(
        self,
        n_iterations=50,
        n_splits=5,
        eta0=0.01,
        plot=False,
        early_stopping=False,
        patience=5,
        tol=1e-4,
        random_state=404,
    ):
        """
        Estimate propensity scores using logistic regression with cross validation,
        tracking training and validation log losses (and AUC) at each iteration.

        This function uses an SGDClassifier with partial_fit to simulate the 
        training of a regularized logistic regression. Cross validation (with a 
        stratified split) is used to monitor the performance and identify the 
        optimum number of iterations if early stopping is enabled. After cross-validation,
        a final model is fit on the entire dataset.

        Parameters
        ----------
        n_iterations : int, default=50
            Maximum number of training iterations (epochs) per cross validation fold.
        n_splits : int, default=5
            Number of folds for stratified cross validation.
        eta0 : float, default=0.01
            The initial learning rate.
        plot : bool, default=False
            If True, plot the training and validation losses over iterations.
        early_stopping : bool, default=False
            Whether to use early stopping if validation loss does not improve.
        patience : int, default=5
            Number of iterations to wait after no improvement before stopping.
        tol : float, default=1e-4
            Tolerance for measuring the minimum change in validation loss for early stopping.
        random_state : int, default=404
            Random seed for reproducibility.

        Results stored in the instance:
        - self.avg_train_losses: Average training loss at each iteration.
        - self.avg_val_losses: Average validation loss at each iteration.
        - self.avg_auc_score: Average AUC score (across folds) obtained in the final iteration.
        - self.propensity_scores: Final propensity scores for each observation.
        """

        # Check if the features are preprocessed and use the correct population
        if not hasattr(self, "pop_processed"):
            X_transformed = self.pop[self.all_features].copy()
            y = self.pop[self.exposure_status].copy()
        else:
            X_transformed = self.pop_processed[self.all_features].copy()
            y = self.pop_processed[self.exposure_status].copy()

        # Compute balanced class weights
        classes = np.unique(y)
        class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
        # Map each observation to its class weight:
        sample_weights = np.array([class_weights[cls] for cls in y])

        # To store per-iteration losses across folds.
        all_train_losses = []
        all_val_losses = []
        auc_scores_folds = []

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        # For each CV fold, we perform incremental training and record losses.
        for fold, (train_index, val_index) in enumerate(skf.split(X_transformed, y), start=1):
            X_train, X_val = X_transformed.iloc[train_index], X_transformed.iloc[val_index]
            y_train, y_val = y.iloc[train_index], y.iloc[val_index]
            sample_weights_train = sample_weights[train_index]

            # Initialize the SGDClassifier with only one epoch per call; warm_start=True allows continuation.
            sgd_logistic = SGDClassifier(
                loss="log_loss",
                random_state=random_state,
                max_iter=1,
                warm_start=True,
                learning_rate="constant",
                eta0=eta0,
            )

            fold_train_losses = []
            fold_val_losses = []
            best_val_loss = np.inf
            no_improve_iter = 0

            # Iterate n_iterations, tracking loss along the way.
            for it in range(n_iterations):
                # Use partial_fit to update the model
                if it == 0:
                    sgd_logistic.partial_fit(
                        X_train, y_train, classes=classes, sample_weight=sample_weights_train
                    )
                else:
                    sgd_logistic.partial_fit(X_train, y_train, sample_weight=sample_weights_train)

                # Get predicted probabilities for the training and validation sets
                y_train_pred_proba = sgd_logistic.predict_proba(X_train)[:, 1]
                y_val_pred_proba = sgd_logistic.predict_proba(X_val)[:, 1]

                # Calculate the log losses
                train_loss = log_loss(y_train, y_train_pred_proba)
                val_loss = log_loss(y_val, y_val_pred_proba)

                fold_train_losses.append(train_loss)
                fold_val_losses.append(val_loss)

                # Optionally use early stopping if there is no improvement.
                if early_stopping:
                    if val_loss < best_val_loss - tol:
                        best_val_loss = val_loss
                        no_improve_iter = 0
                    else:
                        no_improve_iter += 1
                        if no_improve_iter >= patience:
                            print(
                                f"Early stopping in fold {fold} after {it+1} iterations. "
                                f"Best validation loss: {best_val_loss:.4f}"
                            )
                            break  # End training for this fold

            # Record the losses for the current fold.
            all_train_losses.append(fold_train_losses)
            all_val_losses.append(fold_val_losses)
            # Evaluate AUC on the validation set at the final iteration of this fold.
            final_auc_fold = roc_auc_score(y_val, y_val_pred_proba)
            auc_scores_folds.append(final_auc_fold)

        # Determine the minimum number of iterations that all folds achieved.
        min_iters = min(len(losses) for losses in all_train_losses)

        # Average the losses at each iteration (only up to the minimum number of epochs) across folds.
        avg_train_losses = np.mean([losses[:min_iters] for losses in all_train_losses], axis=0)
        avg_val_losses = np.mean([losses[:min_iters] for losses in all_val_losses], axis=0)
        avg_auc_score = np.mean(auc_scores_folds)

        # Print CV performance summary.
        print(f"Average AUC across folds at final iteration: {avg_auc_score:.3f}")
        print(f"Completed iterations (per fold): {min_iters}")

        # Fit the final logistic model on the entire dataset for propensity score estimation.
        final_model = SGDClassifier(
            loss="log_loss",
            random_state=random_state,
            max_iter=min_iters,
            learning_rate="constant",
            eta0=eta0,
        )
        final_model.fit(X_transformed, y, sample_weight=sample_weights)
        propensity_scores = final_model.predict_proba(X_transformed)[:, 1]

        # Save metrics in the instance.
        self.avg_train_losses = avg_train_losses
        self.avg_val_losses = avg_val_losses
        self.avg_auc_score = avg_auc_score
        self.propensity_scores = propensity_scores
        self.propensity_model = final_model
        self.pop_processed["propensity_score"] = propensity_scores

        # Optionally, plot the training and validation losses.
        if plot:
            self.plot_ps_losses()


    def plot_ps_losses(self):
        # check to be sure the losses are not empty in self.avg_train_losses and self.avg_val_losses
        if not hasattr(self, 'avg_auc_score') or not hasattr(self, 'avg_train_losses') or not hasattr(self, 'avg_val_losses'):
            raise ValueError("Propensity scores have not been calculated. Please run calculate_propensity_scores() first.")
        
        sns.set_theme(style="whitegrid")
        set1_colors = sns.color_palette("Set1")
        plt.figure(figsize=(10, 6))
        sns.lineplot(x=range(len(self.avg_train_losses)), y=self.avg_train_losses, label='Training Loss', color=set1_colors[0])
        sns.lineplot(x=range(len(self.avg_val_losses)), y=self.avg_val_losses, label='Validation Loss', color=set1_colors[1])
        plt.xlabel('Iteration')
        plt.ylabel('Log Loss')
        plt.title(f'Training and Validation Loss Over Time (AUC: {self.avg_auc_score:.3f})')
        plt.legend()
        plt.show()


    def plot_psm_coverage(self, matched_data=None, figsize=(10, 6), save_path=None):
        """
        Plot the propensity score distributions for exposed and control groups,
        before and after matching if matched_data is provided.

        Parameters:
        -----------
        matched_data : pandas.DataFrame, optional
            The matched dataset from simple_matching()
        figsize : tuple, default=(10, 6)
            Figure size for the plot
        save_path : str, optional
            Path to save the figure. If None, figure is only displayed

        Returns:
        --------
        fig, axes : tuple
            Matplotlib figure and axes objects
        """

        # Create figure
        fig, axes = plt.subplots(1, 2 if matched_data is not None else 1, figsize=figsize)
        if matched_data is None:
            axes = [axes]

        # Plot before matching
        exposure_mask = self.pop[self.exposure_status] == 1

        set1_colors = sns.color_palette("Set1")

        sns.kdeplot(
            data=self.propensity_scores[exposure_mask],
            ax=axes[0],
            label='1',
            fill=True,
            alpha=0.5,
            color=set1_colors[0]
        )
        sns.kdeplot(
            data=self.propensity_scores[~exposure_mask],
            ax=axes[0],
            label='0',
            fill=True,
            alpha=0.5,
            color=set1_colors[1]
        )
        axes[0].set_title('Propensity Score Distribution Before Matching')
        axes[0].set_xlabel('Propensity Score')
        axes[0].set_ylabel('Density')
        axes[0].legend(title="Exposure Status", loc="upper right")
        axes[0].set_xticks([0, 1])
        # axes[0].set_xticklabels(["Control (0)", "exposed (1)"])

        # Plot after matching if matched_data is provided
        if matched_data is not None:
            # Recalculate propensity scores for matched data
            matched_ps = self.matched_data['propensity_score']

            matched_exposure_mask = self.matched_data[self.exposure_status] == 1

            sns.kdeplot(
                data=matched_ps[matched_exposure_mask],
                ax=axes[1],
                label='1',
                fill=True,
                alpha=0.5,
                color=set1_colors[0]
            )
            sns.kdeplot(
                data=matched_ps[~matched_exposure_mask],
                ax=axes[1],
                label='0',
                fill=True,
                alpha=0.5,
                color=set1_colors[1]
            )
            axes[1].set_title('Propensity Score Distribution After Matching')
            axes[1].set_xlabel('Propensity Score')
            axes[1].set_ylabel('Density')
            axes[1].set_xticks([0, 1])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig, axes

    
    def ps_matching(self, caliper=0.05, n_neighbors=1):
        """
        Performs multi-covariate adjusted matching using both the estimated propensity scores 
        and the observed covariates. Matching is performed by first subsetting the data into 
        exposed and control groups, then using a NearestNeighbors search (restricted to controls) 
        to find matches within a specified radius (caliper).

        Parameters
        ----------
        caliper : float, default=0.05
            The maximum Euclidean distance allowed for a control to be considered a match.
        n_neighbors : int, default=1
            Number of control subjects to match to each exposed subject.

        Returns
        -------
        matched_data : pandas.DataFrame
            A DataFrame containing the matched pairs (or sets) with an additional column 'match_group'
            indicating the matching group. If no matches are found, an empty DataFrame is returned.
        """
        # Check that the preprocessed data and propensity scores are available.
        if not hasattr(self, "pop_processed"):
            raise ValueError("Preprocessed features not found. Please run _preprocess_features() first.")
        if not hasattr(self, "propensity_scores"):
            raise ValueError("Propensity scores not found. Please run calculate_propensity_scores() first.")
        
        # Split the data into exposed and control groups.
        exposed_df = self.pop_processed[self.pop_processed[self.exposure_status] == 1].copy()
        control_df = self.pop_processed[self.pop_processed[self.exposure_status] == 0].copy()
        
        # Get propensity scores for exposed and control groups.
        exposed_scores = exposed_df["propensity_score"].values.reshape(-1, 1)
        control_scores = control_df["propensity_score"].values.reshape(-1, 1)
        
        # Extract patient IDs from each group.
        exposed_ids = exposed_df[self.patient_id_col].values
        control_ids = control_df[self.patient_id_col].values
        
        # Fit NearestNeighbors model on the control propensity scores.
        nbrs = NearestNeighbors(radius=caliper, metric='euclidean').fit(control_scores)
        
        matched_control_ids = []  # List to store matched control patient IDs.
        matched_exposed_ids = []  # List to store corresponding exposed patient IDs.
        matched_group_ids = []    # Group id for each match pair.
        used_control_ids = set()  # To ensure a control patient is used only once.
        group_id = 0              # To track matching groups.

        for i, exposed_score in enumerate(exposed_scores):
            # Find all neighboring control observations within the caliper.
            distances, indices = nbrs.radius_neighbors(exposed_score.reshape(1, -1))
            indices = indices[0]
            distances = distances[0]
            mask_within_caliper = distances <= caliper

            # Filter indices based on the caliper and ensure the control hasn't been used already.
            available_neighbors = [
                idx for idx in indices[mask_within_caliper]
                if control_ids[idx] not in used_control_ids
            ]
            
            if len(available_neighbors) >= n_neighbors:
                chosen_neighbors = available_neighbors[:n_neighbors]
                for ctrl_idx in chosen_neighbors:
                    matched_control_ids.append(control_ids[ctrl_idx])
                    matched_exposed_ids.append(exposed_ids[i])
                    matched_group_ids.append(group_id)
                    used_control_ids.add(control_ids[ctrl_idx])
                group_id += 1  # Increment group ID for each successfully matched exposed patient.
            else:
                print(f"Warning: Not enough matches found for exposed patient {exposed_ids[i]}. "
                    f"Consider relaxing the caliper or reducing n_neighbors.")
        
        if not matched_control_ids or not matched_exposed_ids:
            print("Warning: No matches found. Consider relaxing the caliper or checking your covariate space.")
            return pd.DataFrame()

        # Subset the original processed DataFrame by matching on the patient IDs.
        matched_controls = control_df[control_df[self.patient_id_col].isin(matched_control_ids)].copy()
        matched_exposed = exposed_df[exposed_df[self.patient_id_col].isin(matched_exposed_ids)].copy()

        # Add the matching group information.
        matched_controls["match_group"] = matched_group_ids
        # For exposed patients, assign one unique match group per patient.
        matched_exposed = matched_exposed.reset_index(drop=True)
        matched_exposed["match_group"] = list(range(len(matched_exposed)))

        # Optionally, add the number of matches for each exposed patient as a new column.
        matched_controls["n_matches"] = None
        matched_exposed["n_matches"] = matched_exposed["match_group"].map(
            matched_controls["match_group"].value_counts()
        )
        
        matched_data = pd.concat([matched_exposed, matched_controls]).reset_index(drop=True)
        self.matched_data = matched_data
        # return a filtered version of self.pop called self.pop_matched using the patient_id_col
        self.pop_matched = self.pop[self.pop[self.patient_id_col].isin(matched_data[self.patient_id_col])].copy()
        self.pop_matched["match_group"] = matched_data["match_group"].values
        return matched_data
    


    def get_balance_metrics(self, matched_data):
        balance_metrics = {}
        
        # Handle all numeric features (including datetime-derived features)
        for feature in self.numeric_features:
            if feature not in matched_data.columns:
                continue
            treated_vals = matched_data[matched_data[self.exposure_status] == 1][feature]
            control_vals = matched_data[matched_data[self.exposure_status] == 0][feature]
            
            # Calculate standardized mean difference
            pooled_std = np.sqrt((np.var(treated_vals) + np.var(control_vals)) / 2)
            smd = (np.mean(treated_vals) - np.mean(control_vals)) / pooled_std if pooled_std != 0 else 0
            balance_metrics[feature] = smd
            
        return balance_metrics





    # -------------------------------------------------------------------------
    # Summary Statistics and Covariate Balance Methods
    # -------------------------------------------------------------------------

    def summary_stats_table(self):
        """
        Create a summary statistics table comparing the means, standard deviations, 
        and standardized mean differences (SMDs) for numeric and categorical covariates.

        Parameters
        ----------
        data : pandas.DataFrame
            The dataset on which to generate the summary statistics.
        covariates : list of str
            List of numeric covariate names.
        categorical : list of str
            List of categorical covariate names.

        Returns
        -------
        summary_df : pandas.DataFrame
            A table with summary statistics and SMDs for each covariate.
        """
        data = self.pop_matched
        covariates = self.numeric_features
        categorical = self.categorical_features
        exposure_status = self.exposure_status



        # A helper function to round numbers to 3 decimals.
        def round4(num):
            return round(num, 3)

        rows = []

        # Numeric covariates.
        for covariate in covariates:
            mean_treated = data[data[exposure_status] == 1][covariate].mean()
            std_treated = data[data[exposure_status] == 1][covariate].std()
            mean_control = data[data[exposure_status] == 0][covariate].mean()
            std_control = data[data[exposure_status] == 0][covariate].std()

            smd = self.calculate_smd(
                data[data[exposure_status] == 1],
                data[data[exposure_status] == 0],
                [covariate]
            )[covariate]

            rows.append({
                'Covariate': covariate,
                'Mean_Treated': round4(mean_treated),
                'Std_Treated': round4(std_treated),
                'Mean_Control': round4(mean_control),
                'Std_Control': round4(std_control),
                'SMD': round4(smd)
            })

        # Categorical covariates.
        for covariate in categorical:
            prop_treated = data[data[exposure_status] == 1][covariate].value_counts(normalize=True)
            prop_control = data[data[exposure_status] == 0][covariate].value_counts(normalize=True)

            smds = self.calculate_smd_categorical(prop_treated, prop_control, covariate)

            for category in prop_treated.index:
                rows.append({
                    'Covariate': f"{covariate}_{category}",
                    'Mean_Treated': round4(prop_treated[category] * 100),
                    'Std_Treated': np.nan,  # Standard deviation is not applicable for proportions.
                    'Mean_Control': round4(prop_control.get(category, 0) * 100),
                    'Std_Control': np.nan,
                    'SMD': round4(smds[category])
                })

        return pd.DataFrame(rows)

    @staticmethod
    def calculate_smd(group1, group2, var_list):
        """
        Compute standardized mean differences for a list of numeric variables.

        Parameters
        ----------
        group1 : pandas.DataFrame
            Data for group 1 (e.g., treated).
        group2 : pandas.DataFrame
            Data for group 2 (e.g., controls).
        var_list : list of str
            List of variable names to compute SMDs for.

        Returns
        -------
        smds : dict
            Dictionary where keys are variable names and values are SMDs.
        """
        smds = {}
        for var in var_list:
            mean1 = group1[var].mean()
            mean2 = group2[var].mean()
            std1 = group1[var].std()
            std2 = group2[var].std()

            # Calculate pooled standard deviation.
            pooled_std = np.sqrt((std1**2 + std2**2) / 2)
            smd = abs(mean1 - mean2) / pooled_std if pooled_std else np.nan
            smds[var] = smd
        return smds

    @staticmethod
    def calculate_smd_categorical(prop_treated, prop_control, covariate):
        """
        Compute standardized mean differences for categorical covariates on a per-category basis.

        Parameters
        ----------
        prop_treated : pandas.Series
            Proportion of each category in the treated group.
        prop_control : pandas.Series
            Proportion of each category in the control group.
        covariate : str
            The name of the covariate (used for labeling only).

        Returns
        -------
        smds : dict
            Dictionary where keys are categories and values are SMDs.
        """
        smds = {}
        for category in prop_treated.index:
            prop_treated_val = prop_treated.get(category, 0)
            prop_control_val = prop_control.get(category, 0)

            numerator = abs(prop_treated_val - prop_control_val)
            denominator = np.sqrt((prop_treated_val * (1 - prop_treated_val) + 
                                   prop_control_val * (1 - prop_control_val)) / 2)
            smd = numerator / denominator if denominator else np.nan
            smds[category] = smd
        return smds

    def plot_covariate_balance(self):
        """
        Plot the distribution of each numeric covariate separately by treatment status.

        Parameters
        ----------
        data : pandas.DataFrame
            The dataset on which to plot the covariate distributions.
        covariates : list of str
            List of covariate names to plot.
        exposure_status : str
            Column name indicating treatment status.
        """
        data = self.pop_matched
        covariates = self.numeric_features
        exposure_status = self.exposure_status

        for covariate in covariates:
            plt.figure(figsize=(8, 4))
            sns.kdeplot(data[data[exposure_status] == 1][covariate], label='Exposed', shade=True)
            sns.kdeplot(data[data[exposure_status] == 0][covariate], label='Control', shade=True)
            plt.title(f'Distribution of {covariate} by Treatment Status')
            plt.xlabel(covariate)
            plt.ylabel('Density')
            plt.legend()
            plt.show()
