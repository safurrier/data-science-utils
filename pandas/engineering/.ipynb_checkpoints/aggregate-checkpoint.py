import warnings
import pandas as pd
import numpy as np


def groupby_agg_features(df: pd.DataFrame,
                         groupby_cols: list = None,
                         agg_cols: list = None,
                         agg_funcs: list = None,
                         print_groupby_dict: bool = False,
                         keep_groupby_cols_in_new_col_name=False,
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
    keep_groupby_cols_in_new_col_name: bool, default = False
        Boolean flag to keep groupby cols in the returned aggregated feature name.
        E.g if groupby col is ID, aggregating Revenue and summing, turning this on would return
        ID__Revenue__SUM rather than Revenue__SUM
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
            if keep_groupby_cols_in_new_col_name:
                col_names.append('__'.join(groupby_cols) + '__' + col + '_' + func.upper())
            else:
                col_names.append(col + '_' + func.upper())
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
    
# df_weighted_groupby depends on functions weighted_average_func and weighted_sum_func
######################################################################################
def weighted_average_func(datacol, weightscol):
    """
    From https://stackoverflow.com/a/33881167
    A function that returns a function for computing a weighted 
    groupby mean    
    """
    def wavg(group):
        dd = group[datacol]
        ww = group[weightscol] * 1.0
        return (dd * ww).sum() / ww.sum()
    return wavg

def weighted_sum_func(datacol, weightscol):
    """
    From https://stackoverflow.com/a/33881167
    A function that returns a function for computing a weighted 
    groupby sum
    """
    def wavg(group):
        dd = group[datacol]
        ww = group[weightscol] * 1.0 # To Float
        return (dd * ww).sum() 
    return wavg


def df_weighted_groupby(df, 
                        groupbycol: list = None,
                        weightscol:str = None,
                        agg_func: str = 'mean',
                        datacols:str = None, 
                        keep_weights:bool = True, 
                        rename_cols:bool = True, 
           ):
    """
    Apply a weighted groupby mean/sum

    Parameters
    ----------
    df : df
        A pandas DataFrame
    groupby_cols : list
        A list of columns to use to group the data by.
    weightscol : str
        The column to use as a weight
    datacols : int/float
        A list of the columns to apply the weighted aggregation to
    agg_func: str = 'mean',
        The aggregate function to apply. 'mean' for weighted average and 'sum' for weighted sum
    keep_weights:bool = True,
       A boolean flag return the weights column along with the aggregated columns.
    rename_cols:bool = True, 
        Boolean flag to denote automatic renaming of the columns returned
        
    Returns
    -------
    df_aggregated: Pandas DataFrame
        The grouped data with the weighted aggregation applied 
        
    Example
    -------
    df_weighted_groupby(df, 
    groupbycol='Primary_Key', 
    weightscol='Sale_Amount',
    datacols=['Transactions'],
    keep_weights = False)
    """    
    if agg_func == 'mean':
        func = weighted_average_func
    elif agg_func == 'sum':
        func = weighted_sum_func
    grouped = df.groupby(groupbycol)
    df_aggregated = grouped.agg({weightscol:sum})
    #datacols = [cc for cc in df.columns if cc not in [groupbycol, weightscol]]
    for dcol in datacols:
        try:
            wavg_f = func(dcol, weightscol)
            df_aggregated[dcol] = grouped.apply(wavg_f)
        except TypeError:  # handle non-numeric columns
            raise f'Non Numeric data encountered in colum "{dcol}"'
    # Drop weights col if specified
    if not keep_weights:
        df_aggregated.drop(labels=weightscol, axis=1, inplace=True)
    # Rename as weighted average suffix
    if rename_cols:
        renamed_cols = [
            col+f'_{weightscol}_Weighted_'+ agg_func
            if col in datacols
            else col
            for col 
            in df_aggregated.columns            
            ]   
        df_aggregated.columns = renamed_cols
    return df_aggregated    