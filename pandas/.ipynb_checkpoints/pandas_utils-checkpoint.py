import numpy as np
import pandas as pd

def replace_column_values(df, col=None, values=None, replacement=None, new_col_name=None):
    """ Replace values in a given column with specified value. Function form of DataFrame.replace()
    Parameters
    ----------
    df : Pandas DataFrame
        A dataframe containing the data to transform
    col: str
        The name of the column to replace certain values in
    values: list
        A list of the values to replace
    replacement: object
        Replaces the matches of values
    new_col_name: str
        The name of the new column which will have the original with replaced values
        If None, the original column will be replaced inplace.

    Returns
    ----------
    df_copy: Pandas DataFrame
        The original dataframe with the column's value replaced
    """
    # Copy so original is not modified
    df_copy = df.copy()
    if not values:
        return print('Please specify values to replace')

    if not replacement:
        return print('Please specify replacement value')


    # If  column name specified, create new column
    if new_col_name:
        df_copy[new_col_name] = df_copy[col].replace(values, replacement)
    # Else replace old column
    else:
        df_copy[col] = df_copy[col].replace(values, replacement)
    return df_copy

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
        Create quantiles from ranks of the data before computing quantiles.
    rank_method: str
        Must be apicable method to pandas.DataFrame.rank(method=rank_method).
        Default is average. Use 'first' to sort values and rank this way.
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

def log_of_x_plus_constant(df, data=None, constant=1, new_col_name=None, return_df=False):
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


def above_percentile_threshold(X, source_col=None, percentile_threshold=None, new_colname=None):
    """ Return an area with 1 if X[source_col] is above the specified percentile threshold.
    Percentile_threshold should in range between 0-1 (e.g. 99th percentile would be .99)

    Parameters
    ----------
    X : df
        A pandas DataFrame
    source_col : string
        The column name from which to compute percentile threshold
    percentile_threshold : float
        A value between 0-1 that will act as the threshold for column positive value (1 not 0)
        E.g. .99 woul indicate 99th percentile. All observations with 1 in the resulting column
        would be above the 99th percentile threshold.
    new_colname : str
       Name to give the new computed column. If none specified defaults to:
       source_col + _above_ + percentile_threshold + _percentile

    Returns
    -------
    Boolean
        True if column appears to be numeric, non binary with cardinality above the specified limit
    """
    # Create new column name if none specified
    if not new_colname:
        new_colname = source_col + '_above_' + str(percentile_threshold) + '-percentile'
    if not source_col:
        raise 'No source column to compute percentile threshold from specified'
        new_colname = source_col + '_above_' + str(percentile_threshold) + '_percentile'
    if not percentile_threshold:
        raise 'No source column to percentile threshold specified. Should be float in range 0-1, eg .75'

    # New column is array with 1 where source col is above specified quantile
    new_col = np.where(X[source_col] > X[source_col].quantile(percentile_threshold), 1, 0)
    return X.assign(**{new_colname: new_col})

def column_above_or_below_threshold(df, column=None, above_or_below='above',
                                    threshold=None, new_col_name=None, return_df=True):
    """ Return a column with 1 if df[source_col] is above/below a certain threshold.
    Parameters
    ----------
    X : df
        A pandas DataFrame
    column : string
        The column name from which to compute on
    above_or_below : str
        'above' to return 1 if value is above a certain threshold and 0 otherwise
        'below' to return 1 if value is below a certain threshold and 0 otherwise
    threshold_upper : int/float
        A value between will act as the upper threshold for column
        E.g. 45 would with 'above' denoted would return 1 for all values lower than 45
    new_col_name : str
       Name to give the new computed column.

    Returns
    -------
    thresholded_binary_column: Pandas Series
        A series with 1 where the column values were above the lower threshold and below upper threshold
    """
    # Create new column name if none specified
    if not new_col_name:
        new_col_name = column + '_'+ above_or_below +'_' + str(threshold)
    # Checking Arguments
    if not column:
        raise 'No column to threshold from'
    if (above_or_below == 'below') | (above_or_below == 'below'):
        raise 'Specify "above" or "below" in argument above_or_below'

    if above_or_below == 'above':
        # New column is array with 1 where source col is above specified threshold
        new_col = np.where(df[column] > df[column], 1, 0)
    if above_or_below == 'below':
        # New column is array with 1 where source col is above specified threshold
        new_col = np.where(df[column] > df[column], 1, 0)

    if return_df:
        return df.assign(**{new_col_name: new_col})
    else:
        new_col = pd.Series(new_col, name=new_col_name)

def identity_df(df):
    """ Return the dataframe. For use with decorators"""
    return df

def column_comparison(series, col1, col2, comparison='equal', pos_return_val=1, neg_return_val=0):
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
    if comparison == 'inequal':
        if series[col1] != series[col2]:
            return pos_return_val
        else:
            return neg_return_val

def dummies_from_bins(df, col, bins, bin_labels, col_prefix):
    """
    Given a dataframe and column to create binary features from bins, return dummy columns of said bins
    concatenated onto the end of the df
    """
    # cut the column values into bins. the labels provided are the returned values
    # bins must increase monotonically
    binned_values = pandas.cut(df[col],
                           bins=bins,
                           labels=bin_labels)

    # Create dummy variables and add prefix to col label
    dummies_cols = pandas.get_dummies(binned_values).add_prefix(col_prefix)

    # Concatenate onto end of original df
    df = pandas.concat([df, dummies_cols], axis=1)
    return df


def bin_apply(s, feature_col, min_val, max_val,binary=False):
    """
    Apply function to pandas df with axis=1 to evaluate row values and return value or binary response
    If binary=True, response values are 1 if present 0 otherwise
    Else returns the original value or a NaN
    E.g.:
    df.apply(lambda s: bin_feature_binary(s, 'hazard_rank', 0, 3), axis=1) to create a binary feature that returns
    1 if hazard group is between 0-3 and 0 if otherwise

    """
    if (s[feature_col] >= min_val) & (s[feature_col] <= max_val):
        if binary:
            return 1
        else:
            return s[feature_col]
    else:
        if binary:
            return 0
        else:
            return np.nan

def bin_df_feature(df, feature_col, min_val, max_val, binary=False):
    """
    Given a dataframe, feature column (series), bin edges, return a new series whose values are those that fit within the
    bin edges. Optionally denote if binary response (1 if present, 0 otherwise)
    """
    if binary:
        return df.apply(lambda s: bin_apply(s, feature_col, min_val, max_val, binary=True), axis=1)
    else:
        return df.apply(lambda s: bin_apply(s, feature_col, min_val, max_val, binary=False), axis=1)



def binary_feature(df, feat_col, value, binary_feature_col_name=None, concat=False):
    """
    Given a dataframe, feature column name and value to check, return a series of binary responses 1 and 0
    1 if the value in the feature column to check is present, 0 if otherwise
    binary_feature
    """
    # If binary_feature_col_name is none use this instead
    if not binary_feature_col_name:
        binary_feature_col_name = feat_col+'_is_'+str(value)

    def is_value_present(s, value):
        """
        Given a series and a value, return a binary feature 1 if present and 0 if otherwise
        """
        if s[feat_col] == value:
            return 1
        else:
            return 0
    # Return binary feature series
    binary_feature = df.apply(lambda s: is_value_present(s, value), axis=1)
    # Set series name
    binary_feature.name = binary_feature_col_name
    if concat:
        return pandas.concat([df, binary_feature], axis=1)
    return binary_feature

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

def is_numeric_non_categoric(series, positive_val=1, negative_val=0, min_cardinality = 1):
    """
    Check a series to see if it is numeric and has values other two designated
    positive and negative states (default is 0 and 1). Also check for a cardinality limit.
    Will return False if number of unique values is not above this (default is 1)

    Parameters
    ----------
    series : Series
        A pandas series to check
    positive_val : int
        The positive value if series were to be binary
    negative_val : int
        The negative value if the series were to be binary
    min_cardinality : int
        A limit to the cardinality of the series. Anything below this limit will return False.
        For use when there may be integer transformed categories
            (e.g. series Fruit has categories 'Apple', 'Orange', 'Banana' mapped to 1, 2, 3)

    Returns
    -------
    Boolean
        True if column appears to be numeric, non binary with cardinality above the specified limit
    """
    # Check if series dtype is a numpy number
    is_numeric = (np.issubdtype(series.dtype, np.number))
    # Check if there are other values in a column besides the designated positive or negative class


    # If the size of the returned array is > 0, it's not binary
    try:
        is_non_binary = (np.setdiff1d(series.unique(), np.array([positive_val,negative_val])).size != 0)
    # Type Error occurs when float is compared to string, which would constitute non binary
    # Since this would only occur if there is not a match between the positive/negative values
    # Unique values of the column
    except TypeError:
        is_non_binary = False


    # Check if the number of unique values in the series is above the cardinality limit
    is_above_cardinality_minimum =  (len(series.unique()) > min_cardinality)

    return is_numeric & is_non_binary & is_above_cardinality_minimum

def is_pandas_categorical(series):
    """ Check if a series is has the pandas dtype Categorical"""
    return series.dtype.name == 'category'

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

def column_is_value(df, column=None, value=None, new_col_name=None, return_df=True):
    """ Return a column of 1s where df[column] == value and 0s if not"""
    if not new_col_name:
        new_col_name = column+'is_not_'+str(value)
    df[new_col_name] = np.where(df[column] == value, 1, 0)
    if return_df:
        return df
    else:
        return df[new_col_name]

def column_is_not_value(df, column=None, value=None, new_col_name=None, return_df=True):
    """ Return a column of 1s where df[column] == value and 0s if not"""
    if not new_col_name:
        new_col_name = column+'is_not_'+str(value)
    df[new_col_name] = np.where(df[column] != value, 1, 0)
    if return_df:
        return df
    else:
        return df[new_col_name]

def tidy_correlation(df, columns=None, keep_identity_correlations=False, abs_val_correlations=False):
    """Given a dataframe and optionally subset of columns, compute correlations 
    between all features. Enable keep_identity_correlations for including the correlation
    between a feature and itself. Enable abs_val_correlations to add a column for absolute value of 
    correlation value.
    """
    df_copy = df.copy()
    if columns:
        df_copy = df_copy[columns]
    
    df_copy = (df_copy.corr()
               .reset_index()
               .melt(id_vars='index', var_name="Paired_Feature", value_name="Correlation")
               .rename(columns={'index':'Base_Feature'})
               .sort_values(by='Correlation', ascending=False)
              )
    if abs_val_correlations:
        df_copy['Correlation_Absolute_Value'] = df_copy['Correlation'].abs()
        df_copy = df_copy.sort_values(by='Correlation_Absolute_Value', ascending=False)
    if keep_identity_correlations:
        return df_copy
    else:
        return df_copy.query('Base_Feature != Paired_Feature')

def df_binary_columns_list(df):
    """ Returns a list of binary columns (unique values are either 0 or 1)"""
    binary_cols = [col for col in df if
               df[col].dropna().value_counts().index.isin([0,1]).all()]
    return binary_cols

def categoric_or_continuous_columns(df, unique_value_to_total_value_ratio_threshold=.05, text_unique_threshold=.9,
                      exclude_strings = True, return_dict = False, return_continuous = False, return_text=False,
                                    return_categoric = True):
    """ Determine if a column in a dataframe is continous based on a ratio
    between the number of unique values in a column and the total number of values
    Low cardinality values will get cut off if above the specified ratio.
    Optionally specify return_dict to return a dictionary where values are column names
    and values are boolean True if categoric and false if continouous

    Default ratio threshold is .05

    'exclude_strings' is True by default (i.e. if a column has string values it will be marked
    as a categoric column). If looking for columns that may be numeric/continuous but
    first need to be processed, this can be set to False.

    Parameters
    ----------
    df : Pandas DataFrame
        A DataFrame to search columns within
    unique_value_to_total_value_ratio_threshold : float
        The maximum ratio of unique values in a column / total observations. Akin to a cardinality ratio.
        Default is .05, or that anyting with more than 5% of its values being unique will be considered
        non-categoric.
    exclude_strings : Boolean
        Flag to include all columns with any string values as categoric columns. Default is True.
    return_dict: Boolean
        Flag to return a dictionary of the form {column: Categoric_Boolean} where the value is True if a column
        is categoric. Default is False
    return_categoric: Boolean
        Flag to return a list of the categoric columns. Default is True.
    return_continuous: Boolean
        Flag to return a list of the continuous columns. Default is False

    Returns
    -------
    Dict/List
        A list of the column names that are categoric/continuous OR a dictionary with keys of column names and
        values True if categoric
    """
    from collections import OrderedDict
    likely_categoric = OrderedDict()
    for column in df.columns:
        likely_categoric[column] = 1.*df[column].nunique()/df[column].count() < unique_value_to_total_value_ratio_threshold

        # Check if any of the values in the column are strings.
        if exclude_strings:
            # If so, its value should be true to indicate it is categoric
            if df[column].apply(type).eq(str).any():
                likely_categoric[column] = True

    likely_text = OrderedDict()
    for column in df.columns:
        # Check for unique pct above threshold and value is string
        likely_text[column] = (1.*df[column].nunique()/df[column].count() > text_unique_threshold) #& isinstance(df[column].values[0], str)


    if return_dict:
        return likely_categoric
    if return_continuous:
        continuous_cols = [col for col, value in likely_categoric.items() if not value]
        return continuous_cols
    if return_categoric:
        categoric_cols = [col for col, value in likely_categoric.items() if value]
        return categoric_cols
    if return_text:
        text_cols = [col for col, value in likely_text.items() if value]
        return text_cols
    else:
        print('Please specify valid return option')

def variable_mutual_information(df, columns=None,
                           auto_select_and_process_relevant_cols = False,
                           max_cardinality_ratio_for_numeric_cols = .05):
    """
    For a given dataframe's (subset) of columns, compute the mutual information between
    pairs of variables.

    Warning: Computationally expensive. Consider performing on sample of data

    Note: Dependency on function categoric_or_continuous_columns for determining whether to call
    sklearn's mutual_information_classifier or mutual_information_regression

    Parameters
    ----------
    df : Pandas DataFrame
        A pandas dataframe to check mutual information from
    columns : list
        A list of the columns to compute mutual information between
    auto_select_and_process_relevant_cols : Boolean
        Flag to select only numberic columns and fill NaNs as 0
    max_cardinality_ratio_for_numeric_cols: float
        The maximum ratio of unique values in a column / total observations. Akin to a cardinality ratio.
        Default is .05, or that anyting with more than 5% of its values being unique will be considered
        numeric.

    Returns
    -------
    mutual_information_df: Pandas DataFrame
        A dataframe in tidy format of the source column, paired column and mutual information score

    """
    from sklearn.feature_selection import mutual_info_classif
    from sklearn.feature_selection import mutual_info_regression
    from sklearn.exceptions import DataConversionWarning
    import pandas as pd
    import warnings
    warnings.filterwarnings(action='ignore', category=DataConversionWarning)


    # If not specified columns use all 
    if not columns:
        columns = df.columns.values.tolist()

    # Subset to relevant columns
    X = df[columns]
    
    # If specified, select numeric and fill NaNs    
    if auto_select_and_process_relevant_cols:
        # Autoselect takes numeric columns only and fills NaNs with 0
        X = X.select_dtypes(include=np.number).fillna(0)    


    categoric_cols = categoric_or_continuous_columns(X,
                                                     unique_value_to_total_value_ratio_threshold=max_cardinality_ratio_for_numeric_cols)
    continuous_cols = categoric_or_continuous_columns(X, return_continuous=True,
                                                      unique_value_to_total_value_ratio_threshold=max_cardinality_ratio_for_numeric_cols)
    cols = X.columns.values.tolist()

    mutual_info_features = []
    # For each categoric column
    for column in categoric_cols:
        # Try Categoric, if it doesn't work it's likey due to continuous value. Add to continouous cols
        try:
            # Compute mutual information
            mi = mutual_info_classif(X, X[column])
            # Create Dataframe of the feature, paired feature and mutual information
            mmi_df = pd.DataFrame({'feature':column, 'paired_feature':cols, 'mutual_information': mi})
            # Add df to list
            mutual_info_features.append(mmi_df)
        except ValueError:
            continuous_cols.append(column)


    for column in continuous_cols:
        try:
            # Compute mutual information
            mi = mutual_info_regression(X, X[column])
            # Create Dataframe of the feature, paired feature and mutual information
            mmi_df = pd.DataFrame({'feature':column, 'paired_feature':cols, 'mutual_information': mi})
            # Add df to list
            mutual_info_features.append(mmi_df)
        except ValueError:
            print('Unable to compute mutual information for column {}'.format(column))

    # Concat all mutual info into one df
    mutual_information_df = pd.concat(mutual_info_features).sort_values(by='mutual_information', ascending=False)
    # Remove instances where a feature was compared with itself
    mutual_information_df = mutual_information_df[mutual_information_df['feature'] != mutual_information_df['paired_feature']]
    #display(mutual_information_df)
    return mutual_information_df

def target_mutual_information(df, target, target_type='categoric', columns=None, 
                           auto_select_and_process_relevant_cols = False):
    """ 
    For a given dataframe's (subset) of columns, compute the mutual information between
    pairs of variables. 
    
    Warning: Computationally expensive. Consider performing on sample of data
    
    Note: Dependency on function categoric_or_continuous_columns for determining whether to call
    sklearn's mutual_information_classifier or mutual_information_regression

    Parameters
    ----------
    df : Pandas DataFrame
        A pandas dataframe to check mutual information from 
    target: str
        The name of the target column in the DataFrame
    target_type: str
        The type of target. Must be either 'categoric' (default) or 'continuous'
    columns : list
        A list of the columns to compute mutual information between
    auto_select_and_process_relevant_cols : Boolean
        Flag to select only numberic columns and fill NaNs as 0

    Returns
    -------
    mutual_information_df: Pandas DataFrame
        A dataframe in tidy format of the source column, paired column and mutual information score
        
    """
    from sklearn.feature_selection import mutual_info_classif
    from sklearn.feature_selection import mutual_info_regression
    from sklearn.exceptions import DataConversionWarning
    import pandas as pd
    import warnings
    warnings.filterwarnings(action='ignore', category=DataConversionWarning)

    # If not specified columns use all 
    if not columns:
        columns = df.columns.values.tolist()

    # Subset to relevant columns
    X = df[columns]
    
    # If specified, select numeric and fill NaNs    
    if auto_select_and_process_relevant_cols:
        # Autoselect takes numeric columns only and fills NaNs with 0
        X = X.select_dtypes(include=np.number).fillna(0)    
    if target_type == 'categoric':
        mi = mutual_info_classif(X, df[target])
    elif target_type == 'continuous':
        mi = mutual_info_regression(X, df[target])
    elif target_type not in ['categoric', 'continuous']:
        return print('Specify "categoric" or "continuous" for target_type argument')
    
    
    mmi_df = pd.DataFrame({'feature':target, 'paired_feature':columns, 'mutual_information': mi})
    # Concat all mutual info into one df
    mmi_df = mmi_df.sort_values(by='mutual_information', ascending=False)
    # Remove instances where a feature was compared with itself
    mmi_df = mmi_df[mmi_df['feature'] != mmi_df['paired_feature']]
    #display(mutual_information_df)
    return mmi_df 

def filter_text_in_column(df, column=None, list_of_strings=None, include_or_exclude='exclude', case=True):
    """ Given a column and a list of strings, return a dataframe (or original column) where strings are
        filtered out or filtered to rows where strings are present.

        Caution: Strings are treated as regex inputs.

    Parameters
    ----------
    df : Pandas DataFrame
        A pandas dataframe to search
    column: str
        The name of the column to search the strings within
    list_of_strings: list
        A list of the strings to use as inputs for regex searching within column
    include_or_exclude : str --> 'exclude' or 'include'
        'exclude': Option to return dataframe without rows that match in list of strings
        'include': Option to return dataframe with rows that match in list of strings
    case : Boolean
        Flag for strings to be case sensitive or not

    Returns
    -------
    filtered_df: Pandas DataFrame
        A dataframe filtered down to records where column either matched the list of strings
        or did not match
    """
    # Assertions and argument checking
    df_copy = df.copy()
    if not column:
        return print('Designate column to search within')
    if not list_of_strings:
        return print('Designate strings to search for')
    assert isinstance(column, str)
    assert isinstance(list_of_strings, list)
    assert (include_or_exclude == 'exclude') | (include_or_exclude == 'include')
    # Force column to string type
    df_copy[column] = df_copy[column].astype(str)

    # Filter Out columns that include/ don't include strings
    if include_or_exclude == 'exclude':
        filtered_df = df_copy[~df_copy[column].str.contains('|'.join(list_of_strings), na=False, case=case)]
    elif include_or_exclude == 'include':
        filtered_df = df_copy[df_copy[column].str.contains('|'.join(list_of_strings), na=False, case=case)]

    return filtered_df

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


def rename_df_columns_prefix_suffix(df, match_pattern=None, split_char=None, replace=None, suffix=True, prefix=False):
    """ Given a dataframe, a splitting character and a regex pattern to match columns, return a dataframe
        where matching columns have been renamed with their old prefix/suffix replaced by a specified prefix/suffix.

    Parameters
    ----------
    df : Pandas DataFrame
        A pandas dataframe
    match_pattern: str
        The regex pattern that will be used to get a list of columns to rename.
        All columns matched by this will be renamed.
    split_char: str
        The character to be used to split the column name into two parts
        E.g. 'Apples_6months' would have split_char '_' to seperate the Apples from 6months
    replace : str
        The prefix or suffix to replace the old prefix/suffix with
    suffix/prefix : Boolean
        Flag whether to replace to the prefix/suffix based on the split character
        Suffix by default

    Returns
    -------
    renamed_df: Pandas DataFrame
        A dataframe with renamed columns

    Examples
    -------
    Replace the suffix for '_1mo' with the suffix '_2mo'
    rename_df_columns_prefix_suffix(df, match_pattern='_1mo$', replace='_2mo', split_char='_')
    """
    # Check arguments
    if not all(var is not None for var in [match_pattern, split_char, replace]):
        return 'Must pass arguments to all of match_pattern, split_char, and replace'

    # Match the columns with the pattern we want to replace
    cols_to_replace = df.filter(regex=match_pattern).columns.values.tolist()
    # Split on the joining character (e.g. underscore)
    col_stems = [col.split(split_char) for col in cols_to_replace]

    # Check to make sure the colum name hasn't been broken into
    # more than 2 parts
    split_lengths = [len(col) for col in col_stems]
    assert sum(split_lengths) == (len(cols_to_replace) * 2), "More than one character to split on in column names"

    # Pull the correct column stem
    if suffix:
        # Pull stem from beginning of word
        col_stems = [stem[0] for stem in col_stems]
        # Add new prefix to word
        new_col_names = [stem+replace for stem in col_stems]
    if prefix:
        # Pull stem from end of word
        col_stems = [stem[1] for stem in col_stems]
        # Add new prefix to word
        new_col_names = [replace+stem for stem in col_stems]

    # Create Renaming Dictionary
    rename_dict = dict(zip(cols_to_replace, new_col_names))

    # Rename dataframe into new object
    renamed_df = df.rename(columns=rename_dict)

    return renamed_df

def extract_cols_to_nested_dict(df, col1, col2):
    """ From a df of the form col1 | col2 return a nested
        dictionary of the form {col1_val1:[col2_val_a, col2_val_b, col2_val_c],
        col1_val2:[col2_val_a, col2_val_h, col2_val_m}
    """
    nested_dict = {}
    for val in df[col1].unique():
        present_col2_vals = df[df[col1].isin([val])].loc[:, col2].values.tolist()
        nested_dict[val] = present_col2_vals
    return nested_dict
