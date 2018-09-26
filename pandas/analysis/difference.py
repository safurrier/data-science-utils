import pandas as pd
import numpy as np

def pct_change_from_col(df, anchor_col, diff_col):
    """ Given two columns of values, compute the percent change
        in vectorized manner

    Parameters
    ----------
    df : DataFrame
        Dataframe of data
    anchor_col : str
        The column name of from which to compute the percent differenc **from**
    diff_col : str
        The column name of from which to compute the percent differenc **to**
    Returns
    -------
    Series
        A Series of the percent change from the anchor column to the difference
        column

    Example
    -------
    df['Pct_Change_Jan_to_Feb'] = pct_change(df, 'Sales_Jan', 'Sales_Feb')

    """
    return (df[anchor_col] - df[diff_col]) / df[anchor_col]