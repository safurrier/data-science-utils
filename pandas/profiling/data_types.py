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

def categoric_or_continuous_columns(df, unique_value_to_total_value_ratio_threshold=.05, text_unique_threshold=.9,
                      exclude_strings = True, return_dict = False, return_continuous = False, return_text=False,
                                    return_categoric = True):
    """ Determine if a column in a dataframe is continous based on a ratio
    between the number of unique values in a column and the total number of values
    Low cardinality values will get cut off if above the specified ratio.
    Optionally specify return_dict to return a dictionary where values are column names
    and values are boolean True if categoric and false if continouous

    Default ratio threshold is .05

    'exclude_strings' is True by default (i.e. if a column has string values it will be marked
    as a categoric column). If looking for columns that may be numeric/continuous but
    first need to be processed, this can be set to False.

    Parameters
    ----------
    df : Pandas DataFrame
        A DataFrame to search columns within
    unique_value_to_total_value_ratio_threshold : float
        The maximum ratio of unique values in a column / total observations. Akin to a cardinality ratio.
        Default is .05, or that anyting with more than 5% of its values being unique will be considered
        non-categoric.
    exclude_strings : Boolean
        Flag to include all columns with any string values as categoric columns. Default is True.
    return_dict: Boolean
        Flag to return a dictionary of the form {column: Categoric_Boolean} where the value is True if a column
        is categoric. Default is False
    return_categoric: Boolean
        Flag to return a list of the categoric columns. Default is True.
    return_continuous: Boolean
        Flag to return a list of the continuous columns. Default is False

    Returns
    -------
    Dict/List
        A list of the column names that are categoric/continuous OR a dictionary with keys of column names and
        values True if categoric
    """
    from collections import OrderedDict
    likely_categoric = OrderedDict()
    for column in df.columns:
        likely_categoric[column] = 1.*df[column].nunique()/df[column].count() < unique_value_to_total_value_ratio_threshold

        # Check if any of the values in the column are strings.
        if exclude_strings:
            # If so, its value should be true to indicate it is categoric
            if df[column].apply(type).eq(str).any():
                likely_categoric[column] = True

    likely_text = OrderedDict()
    for column in df.columns:
        # Check for unique pct above threshold and value is string
        likely_text[column] = (1.*df[column].nunique()/df[column].count() > text_unique_threshold) #& isinstance(df[column].values[0], str)


    if return_dict:
        return likely_categoric
    if return_continuous:
        continuous_cols = [col for col, value in likely_categoric.items() if not value]
        return continuous_cols
    if return_categoric:
        categoric_cols = [col for col, value in likely_categoric.items() if value]
        return categoric_cols
    if return_text:
        text_cols = [col for col, value in likely_text.items() if value]
        return text_cols
    else:
        print('Please specify valid return option')