import numpy as np
import pandas


def are_numeric_columns(df, columns=[]):
    """ Check if a list of columns are numpy.number    """
    """ 
    Given a list of column names, check to see if each of those appears
    in the columns of a dataframe filtered by dtype=np.number

    Parameters
    ----------
    df : DataFrame
        A pandas DataFrame to check 
    columns : list
        A list of columns to check
    negative_val : int
        The negative value if the series were to be binary
    -------
    List[boolean]
        A list of booleans corresponding with True if column appears to be numeric and False if otherwise
    """
    # Select numeric columns from dataframe
    numeric_cols = df.select_dtypes(include=np.number).columns.values.tolist()
    cols_in_numeric_cols = [True if col in numeric_cols else False 
                            for col in columns]
    return cols_in_numeric_cols
  