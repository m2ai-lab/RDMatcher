import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def plot_feature(cohort, feature, exposure_status_col='exposure_status', normalize=True):
    """
    Plots the distribution of a specified feature in the cohort, stratified by exposure status.
    
    For categorical features, if normalize=True, the frequencies within each exposure group
    are normalized so that they represent the proportions (summing to 1) within that exposure group.
    
    For numerical features, if normalize=True, the histogram and KDE are computed using the
    density (such that the area under each exposure group's histogram equals 1). Otherwise, absolute
    counts are shown.
    
    Parameters
    ----------
    cohort : pandas.DataFrame
        DataFrame containing the cohort data.
    feature : str
        The feature to plot the distribution for.
    exposure_status_col : str, optional
        The name of the column containing exposure status. Default is 'exposure_status'.
    normalize : bool, optional
        Whether to normalize the distributions by exposure group. For categorical features, this 
        turns counts into proportions per exposure group. For numerical features, this causes the 
        histogram and KDE to be computed as densities (area under the curve equals 1 per group).
        Default is True.
    
    Returns
    -------
    None
    """
    plt.figure(figsize=(10, 6))
    # Sort exposure statuses in reverse order (you can adjust this as needed)
    exposure_status_order = sorted(cohort[exposure_status_col].unique(), reverse=True)
    
    # Handle categorical features (categorical or object types)
    if pd.api.types.is_categorical_dtype(cohort[feature]) or pd.api.types.is_object_dtype(cohort[feature]):
        # Group by exposure and feature to get counts.
        temp = (
            cohort.groupby([exposure_status_col, feature])
            .size()
            .reset_index(name='count')
        )
        if normalize:
            # For each exposure group, convert counts to proportions.
            temp['proportion'] = temp.groupby(exposure_status_col)['count'].transform(lambda x: x / x.sum())
            y_val = 'proportion'
            y_label = 'Proportion'
        else:
            y_val = 'count'
            y_label = 'Count'

        ax = sns.barplot(
            data=temp,
            x=feature,
            y=y_val,
            hue=exposure_status_col,
            hue_order=exposure_status_order,
            palette='Set1'
        )
        plt.ylabel(y_label)
        plt.title(f'Distribution of {feature} by {exposure_status_col}')
        plt.xlabel(feature)
        plt.xticks(rotation=45, ha='right')
        legend = ax.get_legend()
        if legend:
            legend.set_title(exposure_status_col)
    
    # Handle numerical features.
    elif pd.api.types.is_numeric_dtype(cohort[feature]):
        # For numerical features, use histplot. If normalize is True, we set stat='density'
        # so the area under each exposure group's histogram equals 1. Additionally, set
        # common_norm to False so that each exposure group is normalized separately.
        stat_value = 'density' if normalize else 'count'
        common_norm_val = False if normalize else True
        
        ax = sns.histplot(
            data=cohort,
            x=feature,
            hue=exposure_status_col,
            hue_order=exposure_status_order,
            kde=True,
            palette='Set1',
            bins=30,
            multiple='layer',
            alpha=0.2,
            stat=stat_value,
            common_norm=common_norm_val
        )
        ylabel = 'Density' if normalize else 'Count'
        plt.title(f'Distribution of {feature} by {exposure_status_col}')
        plt.xlabel(feature)
        plt.ylabel(ylabel)
        plt.xticks(rotation=45, ha='right')
        legend = ax.get_legend()
        if legend:
            legend.set_title(exposure_status_col)
    
    else:
        print(f"The feature '{feature}' is neither categorical nor numeric. Unable to plot.")
        return
    
    plt.tight_layout()
    plt.show()