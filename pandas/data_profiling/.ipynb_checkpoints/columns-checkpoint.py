import pandas as pd
import numpy as np

def duplicate_column_name_report(df):
    """ Returns a list of columns with duplicate column names
        and its associated count
    """
    from collections import Counter
    col_counts = Counter(df.columns).most_common()
    dupe_columns = [tup for tup in col_counts if tup[1] >1]
    if not dupe_columns:
        return 'No duplicated column names'
    else:
        return dupe_columns

def only_unique_columns(df):
    """ Returns the dateaframe with duplicated column names
        removed. Note: Doesn't check column values, only names
    """
    df_copy = df.copy()
    # Find Unique Indices
    _, i = np.unique(df.columns, return_index=True)
    unique_df = df_copy.iloc[:, i]
    return unique_df