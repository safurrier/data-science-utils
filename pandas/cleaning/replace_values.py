import pandas as pd
import numpy as np

def replace_column_values(df, col=None, values=None, replacement=None, new_col_name=None, return_df=False):
    """ Replace values in a given column with specified value. Function form of DataFrame.replace()
    Parameters
    ----------
    df : Pandas DataFrame
        A dataframe containing the data to transform
    col: str
        The name of the column to replace certain values in
    values: list
        A list of the values to replace
    replacement: object
        Replaces the matches of values with this value
    new_col_name: str
        The name of the new column which will have the original with replaced values
        If None, the original column will be replaced inplace.
    return_df: bool, default False
        Boolean flag for returning the original data place with the column value 
        processed inplace.        

    Returns
    ----------
    df_copy: Pandas DataFrame
        The original dataframe with the column's value replaced
    """
    # Copy so original is not modified
    df_copy = df.copy()
    if not values:
        raise 'Please specify values to replace'

    if not replacement:
        raise 'Please specify replacement value'


    # If  column name specified, create new column
    if new_col_name:
        df_copy[new_col_name] = df_copy[col].replace(values, replacement)
        col = new_col_name
    # Else replace old column
    else:
        df_copy[col] = df_copy[col].replace(values, replacement)
    
    # Return entire df or just replaced values series
    if return_df:
        return df_copy
    else:
        return df_copy[col]
    
def replace_df_values(df, values):
    """ Call pd.DataFrame.replace() on a dataframe and return resulting dataframe.
    Values should be in format of nested dictionaries,
    E.g., {‘a’: {‘b’: nan}}, are read as follows:
        Look in column ‘a’ for the value ‘b’ and replace it with nan
    """
    df_copy = df.copy()
    return df_copy.replace(values)    