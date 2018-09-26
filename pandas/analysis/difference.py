import pandas as pd
import numpy as np

def pct_change_from_col(df, anchor_col, diff_col):
    """ Given two columns of values, compute the percent change
        in vectorized manner

    Parameters
    ----------
    df : DataFrame
        Dataframe of data
    anchor_col : str
        The column name of from which to compute the percent differenc **from**
    diff_col : str
        The column name of from which to compute the percent differenc **to**
    Returns
    -------
    Series
        A Series of the percent change from the anchor column to the difference
        column

    Example
    -------
    df['Pct_Change_Jan_to_Feb'] = pct_change(df, 'Sales_Jan', 'Sales_Feb')

    """
    return (df[anchor_col] - df[diff_col]) / df[anchor_col]

# Useful for finding significant differences beteween the target class and null class
# for various features
def class_groupby_difference(df, target_class, features, agg_func=['mean'],
                                  difference_col_name = 'difference_between_target_class'):
    """ Group by a binary category and compute the difference between features

    Parameters
    ----------
    df : DataFrame
        Dataframe of data
    target_class : str
        The column name of the binary class which the data will be grouped by
    features : list
        A list of the columns of features to compute the differences of
    agg_func : list
        A list of the pandas groupby agg functions to apply to each feature.
        Default is mean
    difference_col_name : str
        The name of to give the column of computed differences in the returned dataframe.
        Default is difference_between_target_classes
    Returns
    -------
    DataFrame / Series
        A dataframe with one column of the feature used and agg function applied and the
        other the computed difference
    """

    # Create a dictionary of column names and function to aggregate to them in the groupby
    feature_agg_dict = {feat:agg_func for feat in features}

    # Groupby the binary class and apply function to selected features
    groupby_df = (df.groupby(by=[target_class])
              .agg(feature_agg_dict)
             )

    # Unstack columns so it's clear how features were aggregated
    unstacked_column_names = ["-".join(multi_index_column) for multi_index_column in groupby_df.columns]

    # Difference dictionary
    # Each key is a column name and each value is the difference between the feature value
    # for the aggregated feature when grouped by binary target class
    difference_dict = dict(zip(unstacked_column_names, groupby_df.diff().values[1]))

    return pd.DataFrame.from_dict({'feature':unstacked_column_names, difference_col_name: groupby_df.diff().values[1]})