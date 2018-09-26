import pandas as pd
import numpy as np

def scale_feature(df, feat_col, scale, value, scaled_feature_col_name=None, concat=False):
    """
    Given a dataframe, feature column name and value to check, return a scaled response
    If the value is present, multiply it by the scale multiplier. Can be used to increase or decrease
    importance of binary features
    """
    # If weighted_feature_col_name is none use this instead
    if not scaled_feature_col_name:
        scaled_feature_col_name = feat_col+'_weighted'

    def scale_value(s, value):
        "Given a series and a value, return a binary feature 1 if present and 0 if otherwise"
        if s[feat_col] == value:
            return s[feat_col] * scale
        else:
            return s[feat_col]
    # Return weighted feature series
    scaled_feature = df.apply(lambda s: scale_value(s, value), axis=1)
    # Set series name
    scaled_feature.name = weighted_feature_col_name
    if concat:
        return pandas.concat([df, scaled_feature], axis=1)
    return scaled_feature