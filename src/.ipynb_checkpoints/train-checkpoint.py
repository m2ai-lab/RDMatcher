import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Literal
import logging

from logger import epilogger
from utils import hide_columns




logger = epilogger(__name__, level='INFO')


def propensity_scores_simple(df_in, exposure_status, patient_id_col, random_state=404, debug=False):
    """
    Calculate propensity scores using logistic regression.

    This function takes a DataFrame and calculates propensity scores for a 
    specified binary exposure variable using a logistic regression model. 
    The propensity scores represent the probability of the exposure variable 
    being 1, given the other features in the DataFrame.

    Args:
        df_in (pd.DataFrame): Input DataFrame containing the features and the 
                              binary exposure variable.
        exposure_status (str): The name of the column in the DataFrame that 
                               represents the binary exposure variable. 

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: A copy of the input DataFrame with an additional 
                            column 'propensity_score' containing the calculated 
                            propensity scores.
            - pd.Series: A Series containing the propensity scores.

    Notes:
        - The logistic regression model is initialized with a random state of 404 
          and uses class balancing to handle imbalanced datasets.
    """
    if debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    logger.info("Calculating propensity scores using simple logistic regression.")

    df = df_in.copy()
    X = hide_columns(df, columns_to_remove=[patient_id_col, exposure_status])
    y = df[exposure_status]
    logistic = LogisticRegression(random_state=random_state, class_weight='balanced')
    logistic.fit(X, y)
    df['propensity_score'] = logistic.predict_proba(X)[:, 1]
    
    logger.info("Propensity scores calculated successfully.")

    return df, df['propensity_score']




def propensity_scores_full(df_in, exposure_status, random_state=404, max_iter=200, penalty=None, solver: Literal['saga', 'liblinear'] = 'saga', debug=False):
    """
    Calculate propensity scores using logistic regression while incorporating every possible
    interaction term among the features.

    This function takes a DataFrame and calculates propensity scores for a 
    specified binary exposure variable using a logistic regression model that includes 
    all interaction terms between features. The propensity scores represent the probability 
    of the exposure variable being 1, given the other features and their interactions.

    Args:
        df_in (pd.DataFrame): Input DataFrame containing the features and the 
                              binary exposure variable.
        exposure_status (str): The name of the column in the DataFrame that 
                               represents the binary exposure variable. 

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: A copy of the input DataFrame with an additional 
                            column 'propensity_score' containing the calculated 
                            propensity scores.
            - pd.Series: A Series containing the propensity scores.

    Notes:
        - The logistic regression model is initialized with a random state of 404 
          and uses class balancing to handle imbalanced datasets.
    """

    if debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    logger.info("Calculating propensity scores using full logistic regression with interaction terms.")


    # Create a copy of the dataframe
    df = df_in.copy()
    
    # Obtain the features to be used in the model; assumes `hide_columns` returns only feature columns.
    X_orig = hide_columns(df)
    
    # Create every possible interaction term between features.
    # Setting degree=2 with interaction_only=True will produce all pairwise interactions
    # without including polynomial/quadratic (squared) terms.
    poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=True)
    X_interactions = poly.fit_transform(X_orig)
    
    # Create a DataFrame for the transformed features for better interpretability.
    X_inter_df = pd.DataFrame(
        X_interactions,
        columns=poly.get_feature_names_out(X_orig.columns),
        index=X_orig.index
    )

    logger.debug(f"Transformed features with interaction terms: {X_inter_df.columns.tolist()}")
    
    # The response variable remains the same.
    y = df[exposure_status]
    
    # Initialize and fit the logistic regression model.
    logger.info("Fitting logistic regression model with interaction terms.")
    logistic = LogisticRegression(
        penalty=penalty,
        solver=solver,
        random_state=random_state, 
        class_weight='balanced',
        max_iter=max_iter)
    logistic.fit(X_inter_df, y)
    
    # Calculate propensity scores using the fitted model.
    df['propensity_score'] = logistic.predict_proba(X_inter_df)[:, 1]

    logger.info("Propensity scores calculated successfully with interaction terms.")
    
    return df, df['propensity_score']




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