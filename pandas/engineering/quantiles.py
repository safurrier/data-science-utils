import pandas as pd
import numpy as np

def quantile_binned_feature(df, col=None, quantiles=10, new_col_name=None,
                            attempt_smaller_quantiles_if_necessary=True,
                            rank_data=False, rank_method='average', return_df=False):
    """ Discretize a continuous feature by seperating it into specified quantiles
    
    Parameters
    ----------
    df : Pandas DataFrame
        A dataframe containing the data to transform
    col: str
        The name of the column to transform into 'quantiles' number of equal bins
    quantiles: int
        The number of quantiles to cut the features into
    new_col_name: str
        The name to give the new column of transformed data.
        Default is col + '_cut_into_'+str(quantiles)+'_equal_bins'
    attempt_smaller_quantiles_if_necessary: boolean
        Flag to specify whether to recursively try smaller numbers of quantiles until
        the feature can be split into 'quantiles' number of equal groups.
        Default is True
    rank_data: Boolean
        Create quantiles from ranks (rather than actual values) 
        of the data before computing quantiles.
    rank_method: str, default 'average'
        Must be aplicable method to pandas.DataFrame.rank(method=rank_method).
        Use 'first' to sort values and rank this way.
    return_df : boolean
        Flag to return the entire DataFrame or just the transformed data
        Default is set to False

    Returns
    ----------
    df_copy[new_col_name]: Pandas Series
        Default return is Pandas Series of transformed data
    df_copy: Pandas DataFrame
        If specified with return_df=True, A copy of the dataframe including transformed data
    """

    if not col:
        return print('Must pass column name argument')

    # Copy df to not alter original
    df_copy = df.copy(deep=True)

    # Rank data before hand if necessary
    if rank_data:
        # Special case to sort values low to high
        if rank_method == 'first':
            # Sort values low to high and then substitute with rank
            df_copy[col] = df_copy[col].sort_values().rank(method='first')
        else:
            df_copy[col] = df_copy[col].rank(method=rank_method)

    # Recursive version
    if attempt_smaller_quantiles_if_necessary:
        # Base case
        if quantiles == 0:
            print('Unable to bin column into equal groups')
            pass
        try:
            new_col = pd.qcut(df_copy[col], quantiles, labels=[i+1 for i in range(quantiles)])
            # Change default name if none specified
            if not new_col_name:
                new_col_name = col + '_cut_into_'+str(quantiles)+'_equal_bins'
                new_col.name = new_col_name

            df[new_col_name] = new_col

            # Return df or transformed data
            if return_df:
                return df
            else:
                return new_col
        # If unable to cut into equal quantiles with this few of bins, reduce by one and try again
        except ValueError:
            #print(quantiles)
            return_val = quantile_binned_feature(df_copy, col, quantiles=quantiles-1,
                                                 new_col_name=new_col_name, return_df=return_df)
            return return_val
    # Single attempt
    else:
        new_col = pd.qcut(df_copy[col], quantiles, labels=[i for i in range(quantiles)])
        # Change default name if none specified
        if not new_col_name:
            new_col_name = col + '_cut_into_'+str(quantiles)+'_equal_bins'
            new_col.name = new_col_name

        df[new_col_name] = new_col

        # Return df or transformed data
        if return_df:
            return df
        else:
            return new_col
        
        
def above_quantile_threshold(X, source_col=None, quantile_threshold=None, new_colname=None):
    """ Return an area with 1 if X[source_col] is above the specified quantile threshold.
    Percentile_threshold should in range between 0-1 (e.g. 99th percentile would be .99)

    Parameters
    ----------
    X : df
        A pandas DataFrame 
    source_col : string
        The column name from which to compute percentile threshold
    percentile_threshold : float
        A value between 0-1 that will act as the threshold for column positive value (1 not 0)
        E.g. .99 woul indicate 99th quantile. All observations with 1 in the resulting column
        would be above the 99th quantile threshold. 
    new_colname : str
       Name to give the new computed column. If none specified defaults to:
       source_col + _above_ + quantile + _percentile

    Returns
    -------
    Boolean
        True if column appears to be numeric, non binary with cardinality above the specified limit
    """
    # Create new column name if none specified
    if not new_colname:
        new_colname = source_col + '_above_' + str(quantile_threshold) + '_quantile'
    if not source_col:
        raise 'No source column to compute quantile threshold from specified'
        new_colname = source_col + '_above_' + str(quantile_threshold) + '_quantile'
    if not quantile_threshold:
        raise 'No source column to quantile threshold specified. Should be float in range 0-1, eg .75'   
        
    # New column is array with 1 where source col is above specified quantile
    new_col = np.where(X[source_col] > X[source_col].quantile(quantile_threshold), 1, 0)
    return X.assign(**{new_colname: new_col})


def quantile_cutpoints(df, column, quantiles, round_ndigits=None, as_dict=False):
    """ Given a column and a list of quantiles, return a list of cutpoints for specified quantiles

    Parameters
    ----------
    df: pandas.DataFrame
        The dataframe which holds the data under name `column` to compute cutpoints for
    column: str
        String column name to find cutpoints for
    quantiles: list
        A list of the quantiles to find values for
    round_ndigits: int
        If specified, the number of digits after the decimal to round the cutpoint values to 
    as_dict: bool
        If specified, return value is a dictionary of the form {quantile:cutpoint_value}

    Example
    -------
    quantile_cutpoints(df=df, 
    column='Revenue', 
    quantiles=[.1, .5, .9],
    round_ndigits=2)

    [2000.25, 10_000.67, 99_000.23]
    """

    value_cutpoints = []
    for quantile in quantiles:
        val = df[column].quantile(
            q=quantile)
        # Round if specified
        if round_ndigits:
            val = round(val, round_ndigits)
        
        # Add to list
        value_cutpoints.append(val)
    if as_dict:
        # Zip to dict and return
        return zip(dict(quantiles, value_cutpoints))
    else:
        return value_cutpoints
        