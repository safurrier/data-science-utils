import pandas as pd
import numpy as np

def log_plus_constant(df, data=None, constant=1, new_col_name=None, return_df=False):
    """ Apply log(x+1) where x is an array of data in df[data]
    Parameters
    ----------
    df : Pandas DataFrame
        A dataframe containing the data to transform
    constant: int/float
        The value to add to the data before taking the natural log.
        Default value is 1
    new_col_name: str
        The name to give the new column of transformed data.
        Default is Log( constant + data columnd name)
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
    import pandas as pd
    import numpy as np

    if not data:
        print('Must specify data column')
        pass
    # Default column name for transformed column
    if not new_col_name:
        new_col_name = 'Log(' + str(constant) + '+ ' + df[data].name + ')'
    # Transform the data
    df_copy = df.copy()
    df_copy[new_col_name] = np.log(df[data].values + constant)

    # Return entire dataframe or just transformed data
    if return_df:
        return df_copy
    else:
        return df_copy[new_col_name]
