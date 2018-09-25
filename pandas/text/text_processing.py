import numpy as np
import pandas as pd
import pandas.api.types as ptypes


def basic_preprocessing_text(df, columns=None, case='lower', strip=True, return_df=False):
    """ Apply basic text processing to a list of columns. Consistent text case 
    and stripping of whitespace
    
    Parameters
    ----------
    df : Pandas DataFrame
        A dataframe containing the data to transform
    columns: list[str]
        A list of columns for which to process the text in 
    case: list,  {'lower', 'upper'}
        A list of the values to replace
    strip: bool, default True
        Boolean flag for stripping whitespace from text
    return_df: bool, default False 
        Boolean flag for returning the original data place with text 
        processed inplace.

    Returns
    ----------
    df_copy: Pandas DataFrame
        The original dataframe with the specified columns text processed
        If return_df is True, return df with text columns proccessed inplace
    """
    # Check for columns existence
    if not columns:
        raise ValueError
    # Check to make sure columns are in df 
    # and are string dtypes
    assert all(True for col in columns if col in df.columns.values.tolist())    
    assert all(ptypes.is_string_dtype(df[col]) for col in columns)
    
    # Defensive copy
    df_copy = df.copy()

    # For each column, process according to parameters
    for col in columns:
        if case == 'lower':
            df_copy[col] = df_copy[col].str.lower()
        if case == 'upper':
            df_copy[col] = df_copy[col].str.upper()
        if strip:
            df_copy[col] = df_copy[col].str.strip()

    if return_df:
        return df_copy
    else:
        return df_copy[columns]