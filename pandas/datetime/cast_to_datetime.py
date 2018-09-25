import numpy as np
import pandas as pd
import pandas.api.types as ptypes

def cast_to_dateime(df, columns=None, format=None, return_df=False):
    """ Given a list of columns, cast them to datetime
    
    Parameters
    ----------
    df : Pandas DataFrame
        A dataframe containing the data to transform
    columns: list[str]
        A list of columns for which to cast to datetime 
    format : string, default None
    strftime to parse time, eg “%d/%m/%Y”,
        Specifying this will speed up the conversion
    return_df: bool, default False
        Boolean flag for returning the original data place with text 
        processed inplace.

    Returns
    ----------
    df_copy: Pandas DataFrame
        The original dataframe with the specified columns text processed
        If return_df is True, return df with columns cast inplace
    """
    # Check for columns existence
    if not columns:
        raise ValueError
    # Check to make sure columns are in df 
    # and are string dtypes
    assert all(True for col in columns if col in df.columns.values.tolist())    
    assert all(ptypes.is_datetime64_any_dtype(df[col]) for col in columns)
    
    # Defensive copy
    df_copy = df.copy()

    # For each column, process according to parameters
    for col in columns:
            df_copy[col] = df_copy[col].apply(pd.to_datetime, format=format)

    if return_df:
        return df_copy
    else:
        return df_copy[columns]