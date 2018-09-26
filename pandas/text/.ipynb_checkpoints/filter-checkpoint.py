import pandas as pd
import numpy as np

def filter_text_in_column(df, column=None, list_of_strings=None, include_or_exclude='exclude', case=True):
    """ Given a column and a list of strings, return a dataframe (or original column) where strings are
        filtered out or filtered to rows where strings are present.

        Caution: Strings are treated as regex inputs.

    Parameters
    ----------
    df : Pandas DataFrame
        A pandas dataframe to search
    column: str
        The name of the column to search the strings within
    list_of_strings: list
        A list of the strings to use as inputs for regex searching within column
    include_or_exclude : str --> 'exclude' or 'include'
        'exclude': Option to return dataframe without rows that match in list of strings
        'include': Option to return dataframe with rows that match in list of strings
    case : Boolean
        Flag for strings to be case sensitive or not

    Returns
    -------
    filtered_df: Pandas DataFrame
        A dataframe filtered down to records where column either matched the list of strings
        or did not match
    """
    # Assertions and argument checking
    df_copy = df.copy()
    if not column:
        return print('Designate column to search within')
    if not list_of_strings:
        return print('Designate strings to search for')
    assert isinstance(column, str)
    assert isinstance(list_of_strings, list)
    assert (include_or_exclude == 'exclude') | (include_or_exclude == 'include')
    # Force column to string type
    df_copy[column] = df_copy[column].astype(str)

    # Filter Out columns that include/ don't include strings
    if include_or_exclude == 'exclude':
        filtered_df = df_copy[~df_copy[column].str.contains('|'.join(list_of_strings), na=False, case=case)]
    elif include_or_exclude == 'include':
        filtered_df = df_copy[df_copy[column].str.contains('|'.join(list_of_strings), na=False, case=case)]

    return filtered_df
