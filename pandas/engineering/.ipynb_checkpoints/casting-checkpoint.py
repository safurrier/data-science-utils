import pandas as pd 
import numpy as np

def cast_to_int_allow_nan(x):
    # If not a float or int
    if not isinstance(x, float) and not isinstance(x, int):
        # and any character is alphabetic
        if any([True for char in x if char.isalpha() or char == '-']):
               return np.nan
        else:
            return int(x)
    else:
        if np.isnan(x) or np.isinf(x) or np.isneginf(x):
            return np.nan
        return int(x)