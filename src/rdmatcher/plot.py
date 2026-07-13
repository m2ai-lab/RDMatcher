import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Literal, Optional

from sklearn.decomposition import PCA, IncrementalPCA



def plot_feature_balance(
    cohort_before: pd.DataFrame,
    feature: str,
    cohort_after: Optional[pd.DataFrame] = None,
    exposure_status_col: str = 'exposure_status',
    normalize: bool = True
):
    """
    Visualizes feature distribution before and optionally after matching using
    Matplotlib's constrained_layout for robust label management.

    Parameters
    ----------
    cohort_before : pd.DataFrame
        The DataFrame containing the cohort data *before* matching.
    feature : str
        The name of the feature column to plot.
    cohort_after : pd.DataFrame, optional
        The DataFrame containing the cohort data *after* matching.
    exposure_status_col : str, optional
        The column name for the exposure status. Defaults to 'exposure_status'.
    normalize : bool, optional
        Whether to normalize distributions. Defaults to True.
    """
    # Input Validation
    if feature not in cohort_before.columns:
        raise ValueError(f"Feature '{feature}' not found in the 'before' cohort.")
    if cohort_after is not None and feature not in cohort_after.columns:
        raise ValueError(f"Feature '{feature}' not found in the 'after' cohort.")

    # Plot Setup
    if cohort_after is not None:
        fig, axes = plt.subplots(
            1, 2, figsize=(16, 6), sharey=True, constrained_layout=True
        )
        cohorts_to_plot = [
            ('Before Matching', cohort_before, axes[0]),
            ('After Matching', cohort_after, axes[1])
        ]
        fig.suptitle(f"Balance for Feature: '{feature}'", fontsize=16)
    else:
        fig, ax = plt.subplots(
            1, 1, figsize=(10, 6), constrained_layout=True
        )
        cohorts_to_plot = [
            (f"Distribution of {feature}", cohort_before, ax)
        ]

    # Plotting Logic
    exposure_status_order = sorted(cohort_before[exposure_status_col].unique(), reverse=True)
    
    feature_dtype = cohort_before[feature].dtype
    if isinstance(cohort_before[feature].dtype, pd.CategoricalDtype) or pd.api.types.is_object_dtype(feature_dtype):
        y_val = 'proportion' if normalize else 'count'
        y_label = 'Proportion' if normalize else 'Count'

        for title, df, ax in cohorts_to_plot:
            temp = df.groupby([exposure_status_col, feature], observed=False).size().reset_index(name='count')
            if normalize:
                temp['proportion'] = temp.groupby(exposure_status_col)['count'].transform(lambda x: x / x.sum())

            sns.barplot(
                data=temp, x=feature, y=y_val, hue=exposure_status_col,
                hue_order=exposure_status_order, palette='Set1', ax=ax
            )
            
            ax.set_title(title)
            ax.set_ylabel(y_label)
            ax.set_xlabel(feature)
            
            # Use plt.setp to modify properties of existing tick labels
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            
            legend = ax.get_legend()
            if legend:
                legend.set_title("Exposure Status")

    elif pd.api.types.is_numeric_dtype(feature_dtype):
        stat_value = 'density' if normalize else 'count'
        y_label = 'Density' if normalize else 'Count'

        for title, df, ax in cohorts_to_plot:
            sns.histplot(
                data=df, x=feature, hue=exposure_status_col,
                hue_order=exposure_status_order, kde=True, palette='Set1',
                bins=30, multiple='layer', alpha=0.3, stat=stat_value,
                common_norm=False, ax=ax
            )
            ax.set_title(title)
            ax.set_ylabel(y_label)
            ax.set_xlabel(feature)
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            
            legend = ax.get_legend()
            if legend:
                legend.set_title("Exposure Status")
    else:
        plt.close(fig)
        print(f"Feature '{feature}' has an unsupported data type: {feature_dtype}. Cannot plot.")
        return
        
    plt.show()



def plot_propensity_support(cohort, exposure_status_col='exposure_status'):

    # check that propensity_score is in the cohort
    if 'propensity_score' not in cohort.columns:
        raise ValueError("The DataFrame must contain a 'propensity_score' column.")
    # Cumulative Distribution Function (CDF)
    exposure_status_order = sorted(cohort[exposure_status_col].unique(), reverse=True)
    
    plt.figure(figsize=(12, 6))
    ax = sns.ecdfplot(data=cohort, x='propensity_score', hue=exposure_status_col, hue_order=exposure_status_order, palette='Set1')
    plt.title('CDF of Propensity Scores by Exposure Status')
    plt.xlabel('Propensity Score')
    plt.ylabel('Cumulative Probability')
    legend = ax.get_legend()
    if legend:
        legend.set_title("Exposure Status")
    plt.show()



def plot_pca_threshold(data,
                       max_components=42,
                       plot=True,
                       return_pca=False,
                       sample_fraction=1.0,
                       incremental=False, 
                       batch_size=None, 
                       variance_threshold=0.95,
                       random_state=404):
    """
    Compute PCA on the input data and, optionally, produce an elbow plot that includes 
    a cutoff based on a given cumulative explained variance threshold.
    
    This function can be used to determine the number of principal components needed 
    to capture the desired variance. It is intended for use in your matching pipeline 
    so that you can optionally reduce the dimension of your feature space.
    
    Parameters:
   --------
    data : pandas.DataFrame or numpy.ndarray
        The feature matrix (samples x features).
    
    max_components : int, default=42
        Maximum number of principal components to compute.
    
    plot : bool, default=True
        If True, plot the elbow (scree) plot.
    
    return_pca : bool, default=True
        If True, return the fitted PCA along with the cutoff index.
    
    sample_fraction : float in (0, 1], default=1.0
        Fraction of the data to sample for PCA (for very large datasets).
    
    incremental : bool, default=False
        If True, uses IncrementalPCA to handle large datasets.
    
    batch_size : int, default=None
        Batch size for IncrementalPCA. If None, set to a default proportional to the data size.
    
    variance_threshold : float, default=0.95
        The target cumulative explained variance (e.g., 0.95 for 95%). A vertical line and 
        annotation will indicate the first component where this threshold is exceeded.
    
    random_state : int, default=404
        Random seed for reproducibility.
    
    Returns:
   -----
    If return_pca is True:
        pca : PCA or IncrementalPCA object
            The fitted PCA model.
        cutoff_idx : int
            The number of components required to reach the specified variance threshold.
    Otherwise, returns None.
    """
    # Convert to numpy array if a DataFrame is provided.
    if isinstance(data, pd.DataFrame):
        X = data.values
    else:
        X = np.asarray(data)
        
    n_samples, n_features = X.shape

    # Optionally sample the data for efficiency.
    if sample_fraction < 1.0:
        np.random.seed(random_state)
        sample_size = int(n_samples * sample_fraction)
        indices = np.random.choice(n_samples, size=sample_size, replace=False)
        X = X[indices]
        n_samples = sample_size

    # Adjust number of components if necessary.
    n_components = min(max_components, n_features)

    # Choose between standard and Incremental PCA.
    if incremental:
        if batch_size is None:
            batch_size = min(1000, max(10, n_samples // 10))
        pca = IncrementalPCA(n_components=n_components, batch_size=batch_size)
    else:
        pca = PCA(n_components=n_components, random_state=random_state)
        
    pca.fit(X)
    
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    
    # Find the component number where the cumulative variance first exceeds the threshold.
    cutoff_idx = np.argmax(cumulative_variance >= variance_threshold) + 1
    
    # Optionally plot the elbow plot.
    if plot:
        set1_colors = sns.color_palette("Set1")

        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot individual explained variance ratios as bars.
        ax.bar(range(1, n_components + 1), explained_variance, 
               label='Explained Variance Ratio', color=set1_colors[1])
        
        # Plot cumulative explained variance as a line.
        ax.plot(range(1, n_components + 1), cumulative_variance, 
                marker='o', color=set1_colors[0], label='Cumulative Explained Variance')
        
        # Draw a vertical line at the cutoff.
        ax.axvline(x=float(cutoff_idx), color=set1_colors[2], linestyle='--',
                   label=f'Threshold cutoff: {variance_threshold*100:.1f}%')
        
        ax.set_xlabel('Principal Component')
        ax.set_ylabel('Explained Variance Ratio')
        ax.set_title('PCA Explained Variance')
        ax.set_xticks(range(1, n_components + 1))
        ax.legend(loc='best')
        ax.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    if return_pca:
        return pca, cutoff_idx
