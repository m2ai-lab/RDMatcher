# import numpy as np
# import pandas as pd


#! FIX THIS
def hide_columns(df, columns_to_remove=['description', 'status']):
    """
    Remove the columns 'is_hht' and 'patientdurablekey' from the DataFrame, if present.
    
    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.
        
    Returns
    -------
    pd.DataFrame
        A new DataFrame without the specified columns.
    """
    return df.drop(columns=[col for col in columns_to_remove if col in df.columns])