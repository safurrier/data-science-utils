import pandas as pd
import numpy as np

def column_equality(series, col1, col2, comparison='equal', pos_return_val=1, neg_return_val=0):
    """
    Apply to a dataframe row to return a binary feature depending on equality or inequality

    E.g. df.apply(lambda s: column_match(s, 'day_of_week', 'day_of_sale'), axis=1) to for matching the two.
    Result is series of positive_return_vals and neg_return_vals. Defaults to
    """
    if comparison == 'equal':
        if series[col1] == series[col2]:
            return pos_return_val
        else:
            return neg_return_val
    if comparison == 'unequal':
        if series[col1] != series[col2]:
            return pos_return_val
        else:
            return neg_return_val