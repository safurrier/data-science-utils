import warnings
import pandas as pd
import numpy as np


def groupby_agg_features(df: pd.DataFrame,
                         groupby_cols: list = None,
                         agg_cols: list = None,
                         agg_funcs: list = None,
                         print_groupby_dict: bool = False,
                         return_df: bool = True):
    """Apply a groupby to column(s) and in turn apply specified aggregation
    functions to columns specifed in 'agg_cols'. Returns a 2D dataframe
    with the new columns named with the form 'applied_column_agg_func'
    
    Parameters
    ----------
    df : df
        A pandas DataFrame
    groupby_cols : list
        A list of columns to use to group the data by. 
    agg_cols : list
        A list of columns to apply the agg functions to
    agg_funcs : int/float
        A list of aggregation functions applicable to Pandas GroupBy function. 
        E.g. sum, mean, mode, etc
    print_groupby_dict : bool, default=False
       A boolean flag for printing out the groupby dictionary which can be passed to Pandas 
       df.groupby(agg=groupby_dictionary) easily
    return_df: bool, default=True
        A boolean flag for left joining the df with the computed features using the groupby keys

    Returns
    -------
    return_df: Pandas DataFrame
        The original dataframe with the added groupby features
    groupby_features: Pandas DataFrame
        A dataframe of just the computed features
    """    
    # Catch single strings and put inside lists
    if isinstance(groupby_cols, str):
        groupby_cols=[groupby_cols]
    if isinstance(agg_cols, str):
        agg_cols=[agg_cols]
    if isinstance(agg_funcs, str):
        agg_funcs=[agg_funcs]
        
    groupby_dict = {}
    col_names = []
    for col in agg_cols:
        groupby_dict[col] = agg_funcs
        for func in agg_funcs:
            col_names.append('__'.join(groupby_cols) + '__' + col + '_' + func)
    if print_groupby_dict:
        print(groupby_dict)
    groupby_features = df.groupby(by=groupby_cols).agg(groupby_dict)
    groupby_features.columns = col_names
    if return_df:
        df_copy = df.copy()
        df_copy = pd.merge(
            df_copy,
            groupby_features,
            on=groupby_cols
        )
        if df_copy.shape[0] != df.shape[0]:
            warnings.warn(f'The new shape of the df with the groupby features is {df_copy.shape}'
                          f'but the previous shape of the df is {df.shape}. It is possible that the'
                          f'groupby columns were not unique')
        return df_copy
    else:
        return groupby_features