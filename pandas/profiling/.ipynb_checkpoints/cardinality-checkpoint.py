import pandas as pd
import numpy as np

def column_cardinality(df, columns=None):
    """Given a dataframe and optionally subset of columns, return a table
    with the number of unique values associated with each feature and percent of
    total values that are unique"""
    if not columns:
        columns = df.columns.values.tolist()
    # Get number unique
    n_unique = [df[col].nunique() for col in columns]
    pct_unique = [cardinality/df.shape[0] for cardinality in n_unique]
    cardinality_df = pd.DataFrame.from_dict({
        'column':columns,
        'n_unique':n_unique,
        'pct_of_all_values_unique': pct_unique,
    },
        orient='columns').sort_values(by='pct_of_all_values_unique', ascending=False).reset_index(drop=True)
    return cardinality_df