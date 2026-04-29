import logging
import numpy as np
import pandas as pd
import random
from typing import List

from .logger import rdlogger

logger = logging.getLogger('rdmatcher.utils')


# ------------------------------
# Utility function to hide specified columns from a DataFrame
# ------------------------------
def hide_columns(df, columns_to_remove: List[str]):
    # Remove specified columns from the DataFrame if they exist
    for col in columns_to_remove:
        try:
            df = df.drop(columns=[col])
        except KeyError:
            # If a column doesn't exist, just skip it and log it
            logger.warning(f"Column to remove not found in DataFrame: {col}")
            continue
    return df



# ------------------------------
# Utility functions to generate synthetic cohort data
# ------------------------------
# Define default categories
default_categories = {
    'sex': ['*Unspecified', 'Female', 'Male', 'Nonbinary', 'Unknown'],
    'race_ethnicity': [
        'Asian',
        'Black or African American',
        'Latinx',
        'Multi-Race/Ethnicity',
        'Native American or Alaska Native',
        'Native Hawaiian or Other Pacific Islander',
        'Other',
        'Southwest Asian and North African',
        'Unknown/Declined',
        'White'
    ]
}

def random_weights(n, seed=404, min_weight=0.01):
    """
    Generates n random floats that add up to 1.

    Arguments
    ---------
    n : int
        Number of random floats to generate.
    seed : int
        Random seed for reproducibility.
    min_weight : float
        Minimum weight for each element.

    Returns
    -------
    list
        List of n random floats that add up to 1.
    """
    np.random.seed(seed)
    weights = np.random.rand(n)
    weights /= weights.sum()

    # Ensure minimum weight for each element
    weights = np.maximum(weights, min_weight)

    # Normalize again to ensure the sum is 1
    weights /= weights.sum()

    # Round weights to 2 decimal places
    weights = np.round(weights, 2)

    # Ensure sum is exactly 1 by adjusting the last element
    difference = 1 - weights.sum()
    for i in range(n):
        if weights[i] + difference >= min_weight:
            weights[i] += difference
            break

    return weights.tolist()


def make_cohort_independent(npats, ncases, features, seed=None):
    """
    Generates a simulated patient cohort with specified features.
    Note: Assumes each feature is independent of the others (Naive Bayes assumption). This might not hold in real-world data.

    Arguments
    ---------
    npats : int
        Total number of patients to simulate.
    ncases : int
        The exact number of cases (exposed) to simulate.
    features : dict
        A dictionary defining the features to generate. The format is:
        {'column_name': ('type', ctrl_params, case_params)}
        - 'type' can be 'continuous' or 'categorical'.
        - For 'continuous', params are a list: [distribution, [distribution_parameters]].
            - 'distribution' can be 'normal', 'uniform', 'negative_binomial', or 'beta'
            - 'distribution_parameters' are the parameters for the distribution. Ensure to provide
              them as a list.
        - For 'categorical', params are a dict: {'category': probability}.
    seed : int, optional
        A random seed for reproducibility. Defaults to None.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the simulated cohort data, with cases and
        controls randomly distributed.
    """
    # 1. Input validation and setup
    if not all(isinstance(i, int) and i >= 0 for i in [npats, ncases]):
        raise ValueError("npats and ncases must be non-negative integers.")
    ncontrols = npats - ncases
    if ncontrols < 0:
        raise ValueError("Number of cases cannot exceed the total number of patients.")
    
    # Use NumPy's modern random number generator for better practice
    rng = np.random.default_rng(seed)

    # 2. Efficiently create and shuffle exposure status
    # This is much faster than the original sample-and-correct method.
    exposure_status = np.concatenate([
        np.ones(ncases, dtype=int),
        np.zeros(ncontrols, dtype=int)
    ])
    rng.shuffle(exposure_status)
    
    # 3. Initialize a dictionary to hold all column data
    cohort_data = {
        'patient_id': np.arange(1, npats + 1),
        'exposure_status': exposure_status
    }
    
    # Create boolean masks once for efficient assignment
    is_case = (exposure_status == 1)
    is_control = ~is_case

    # 4. Generate data for each feature
    for col, (feature_type, ctrl_params, case_params) in features.items():
        # Initialize an empty array for the feature's data
        # Using dtype=object allows mixing strings and numbers initially
        feature_array = np.empty(npats, dtype=object)

        if feature_type == 'continuous':
            # Assumes params are [distribution, [distribution_parameters]]
            # don't hard code these, try out to see if the 'distribution' is a valid np function with a 'try', then 'try' to call it with the parameters provided
            dist_ctrl, params_ctrl = ctrl_params
            dist_case, params_case = case_params
            # set the distribution function that will be used here, e.g., np.random.normal
            try:
                dist_func_ctrl = getattr(rng, dist_ctrl)
                dist_func_case = getattr(rng, dist_case)
            except AttributeError:
                raise ValueError(f"Unsupported distribution '{dist_ctrl}' or '{dist_case}' for continuous feature '{col}'.")

            feature_array[is_control] = dist_func_ctrl(*params_ctrl, size=ncontrols)
            feature_array[is_case] = dist_func_case(*params_case, size=ncases)

            # Convert to a numeric type for efficiency
            cohort_data[col] = feature_array.astype(float)

            # make anything <0 equal to 0
            cohort_data[col] = np.where(cohort_data[col] < 0, 0, cohort_data[col])

        elif feature_type == 'categorical':
            # Assumes params are a dictionary of {'category': probability}
            if not (isinstance(ctrl_params, dict) and isinstance(case_params, dict)):
                 raise TypeError(f"Parameters for categorical feature '{col}' must be dictionaries.")
            
            ctrl_cats, ctrl_probs = list(ctrl_params.keys()), list(ctrl_params.values())
            case_cats, case_probs = list(case_params.keys()), list(case_params.values())

            feature_array[is_control] = rng.choice(ctrl_cats, size=ncontrols, p=ctrl_probs)
            feature_array[is_case] = rng.choice(case_cats, size=ncases, p=case_probs)
            cohort_data[col] = feature_array
        
        else:
            raise ValueError(f"Unknown feature type '{feature_type}' for column '{col}'.")

    # 5. Create the final DataFrame from the collected data
    # Desired column order
    column_order = ['patient_id', 'exposure_status'] + list(features.keys())
    cohort = pd.DataFrame(cohort_data)[column_order]
    
    return cohort