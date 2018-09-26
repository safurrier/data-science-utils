import pandas as pd
import numpy as np

def pivot_df_to_row(df, col_pivot_out=None):
    """ Take a dataframe and pivot out so that all data is in one row. Columns prefixed with original column name"""

    """
        Parameters
        ----------
        df : dataframe
            A dataframe
        col_pivot_out : list of strings
            The columns to pivot out and return
        Returns
        -------
        int
            "Wide" dataframe with columns in format old_col_name_index_name
            E.g. Polarity
            Count 1
            Would be Polarity_Count with associated value of 1 in the new row
    """
    # If no columns specified, use all
    if not col_pivot_out:
        col_pivot_out = df.columns.values.tolist()

    new_cols = []
    for column in col_pivot_out:
        new_cols.append(df.T.loc[column].add_prefix(column+'_'))

    pivoted_df = pd.DataFrame(pd.concat(new_cols)).T
    return pivoted_df
