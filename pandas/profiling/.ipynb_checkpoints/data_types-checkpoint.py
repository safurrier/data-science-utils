import pandas as pd
import numpy as np

def is_numeric_non_categoric(series, positive_val=1, negative_val=0, min_cardinality = 1):
    """
    Check a series to see if it is numeric and has values other two designated
    positive and negative states (default is 0 and 1). Also check for a cardinality limit.
    Will return False if number of unique values is not above this (default is 1)

    Parameters
    ----------
    series : Series
        A pandas series to check
    positive_val : int
        The positive value if series were to be binary
    negative_val : int
        The negative value if the series were to be binary
    min_cardinality : int
        A limit to the cardinality of the series. Anything below this limit will return False.
        For use when there may be integer transformed categories
            (e.g. series Fruit has categories 'Apple', 'Orange', 'Banana' mapped to 1, 2, 3)

    Returns
    -------
    Boolean
        True if column appears to be numeric, non binary with cardinality above the specified limit
    """
    # Check if series dtype is a numpy number
    is_numeric = (np.issubdtype(series.dtype, np.number))
    # Check if there are other values in a column besides the designated positive or negative class


    # If the size of the returned array is > 0, it's not binary
    try:
        is_non_binary = (np.setdiff1d(series.unique(), np.array([positive_val,negative_val])).size != 0)
    # Type Error occurs when float is compared to string, which would constitute non binary
    # Since this would only occur if there is not a match between the positive/negative values
    # Unique values of the column
    except TypeError:
        is_non_binary = False


    # Check if the number of unique values in the series is above the cardinality limit
    is_above_cardinality_minimum =  (len(series.unique()) > min_cardinality)

    return is_numeric & is_non_binary & is_above_cardinality_minimum

def is_pandas_categorical(series):
    """ Check if a series is has the pandas dtype Categorical"""
    return series.dtype.name == 'category'

def df_binary_columns_list(df):
    """ Returns a list of binary columns (unique values are either 0 or 1)"""
    binary_cols = [col for col in df if
               df[col].dropna().value_counts().index.isin([0,1]).all()]
    return binary_cols