import numpy as np
import pandas as pd

from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import FunctionTransformer, StandardScaler, RobustScaler
from sklearn.preprocessing import Imputer, MultiLabelBinarizer



###############################################################################################################
# Custom Transformers
###############################################################################################################














###############################################################################################################
# Functions used with Transformers
###############################################################################################################

def column_values_target_average(df, feature, target,
                                      sample_frequency=True,
                                      freq_weighted_average=True,
                                      min_mean_target_threshold = 0,
                                      min_sample_size = 0,
                                      min_sample_frequency = 0,
                                      min_weighted_target_threshold=0):
    """ Group by a feature and computing the average target value and sample size
    Returns a dictionary Pandas DataFrame fitting that criteria

    Parameters
    ----------
    df : Pandas DataFrame
        The dataframe where data resides
    feature : str
        Column name for which to groupby and check for average target value
    target : str
        Column name of the target to find grouped by average of
    sample_frequency: Boolean
        Flag to include sample frequency for a given feature value.
        Default is true
    freq_weighted_average: Boolean
        Flag to include the frequency weighted average for a given feature value.
        Default is true
    min_mean_target_threshold : float
        The minimum value of the average target class to use as cutoff.
        E.g. .5 would only return values whose associate with the target is
        above an average of .5
    min_sample_size: int
        The minimum value of the number of samples for a feature value
        E.g. 5 would only feature values with at least 5 observations in the data
    min_weighted_target_threshold : float
        The minimum value of the frequency weighted average target class to use as cutoff.
        E.g. .5 would only return values whose associate with the frequency weighted target
        average is above an average of .5
    min_sample_frequency: float
        The minimum value of the frequency of samples for a feature value
        E.g. .5 would only include feature values with at least 50% of the values in the column

    Returns
    -------
    grouped_mean_target_df
        DataFrame of the feature values and their asssociations
    """
    grouped_mean_target_df = (df.groupby(by=feature)
     .agg({target:['size', 'mean']})
     .loc[:, target]
     .reset_index()
     .sort_values(by='mean', ascending=False)
     .rename(columns={'mean':'avg_target', 'size':'sample_size'})
    )
    # Sum the sample sizes to get total number of samples
    total_samples = grouped_mean_target_df['sample_size'].sum()

    # Flags for adding sample frequency and frequency weighted average
    if sample_frequency:
        # Compute frequency
        grouped_mean_target_df['feature_value_frequency'] = grouped_mean_target_df['sample_size'] / total_samples
        # Filter out minimums
        grouped_mean_target_df = grouped_mean_target_df[grouped_mean_target_df['feature_value_frequency'] >= min_sample_frequency]

    if freq_weighted_average:
        # Sample frequency must be calculated for frequency weighted average
        grouped_mean_target_df['feature_value_frequency']  = grouped_mean_target_df['sample_size'] / total_samples
        grouped_mean_target_df['freq_weighted_avg_target'] = grouped_mean_target_df['feature_value_frequency']  * grouped_mean_target_df['avg_target']
        grouped_mean_target_df = grouped_mean_target_df[(grouped_mean_target_df['feature_value_frequency'] >= min_sample_frequency)
                                                       & (grouped_mean_target_df['freq_weighted_avg_target'] >= min_weighted_target_threshold)
                                                       ]

        # If sample frequency not included, drop the column
        if not sample_frequency:
            grouped_mean_target_df.drop(labels=['feature_value_frequency'], axis=1, inplace=True)

    # Filter out minimum metrics
    grouped_mean_target_df = grouped_mean_target_df[
        (grouped_mean_target_df['avg_target'] >= min_mean_target_threshold)
        & (grouped_mean_target_df['sample_size'] >= min_sample_size)]




    return grouped_mean_target_df



def df_feature_values_target_average(df, target,
                                                           include=None,
                                                           exclude=None,
                                      min_mean_target_threshold = 0,
                                      min_sample_size = 0,
                                      min_sample_frequency = 0,
                                      min_weighted_target_threshold=0):

    """ For a given dataframe and a target column, groupby each column and compute
    for each column value the the average target value, feature value sample size,
    feature value frequency, and frequency weighted average target value

    Parameters
    ----------
    df : Pandas DataFrame
        The dataframe where data resides
    target : str
        Column name of the target to find grouped by average of
    sample_frequency: Boolean
        Flag to include sample frequency for a given feature value.
        Default is true
    include: list
        A list of columns to include when computing
    exclude: list
        A list of columns to exclude when computing
    freq_weighted_average: Boolean
        Flag to include the frequency weighted average for a given feature value.
        Default is true
    min_mean_target_threshold : float
        The minimum value of the average target class to use as cutoff.
        E.g. .5 would only return values whose associate with the target is
        above an average of .5
    min_sample_size: int
        The minimum value of the number of samples for a feature value
        E.g. 5 would only feature values with at least 5 observations in the data
    min_weighted_target_threshold : float
        The minimum value of the frequency weighted average target class to use as cutoff.
        E.g. .5 would only return values whose associate with the frequency weighted target
        average is above an average of .5
    min_sample_frequency: float
        The minimum value of the frequency of samples for a feature value
        E.g. .5 would only include feature values with at least 50% of the values in the column

    Returns
    -------
    feature_values_target_average_df
        DataFrame of the feature values and their asssociations
    """

    # Start with all columns and filter out/include desired columns
    columns_to_check = df.columns.values.tolist()
    if include:
        columns_to_check = [col for col in columns_to_check if col in include]
    if exclude:
        columns_to_check = [col for col in columns_to_check if col not in exclude]

    # Compute for all specified columns in dataframe
    dataframe_lists = [column_values_target_average(df, column, target,
                                      min_mean_target_threshold = min_mean_target_threshold,
                                      min_sample_size = min_sample_size,
                                      min_sample_frequency = min_sample_frequency,
                                      min_weighted_target_threshold = min_weighted_target_threshold)
                     .rename(columns={column:'feature_value'}).assign(feature = column)
            for column in columns_to_check if column != target]

    feature_values_target_average_df = pd.concat(dataframe_lists)

    return feature_values_target_average_df

def feature_vals_target_association_dict(df, feature, target,
                                      min_mean_target_threshold = 0,
                                      min_sample_size = 0,
                                      min_sample_frequency = 0,
                                      min_weighted_target_threshold=0,
                                         ignore_binary=True):
    """Return a dictionary of the form column_name:[list of values] for values in a
       feature that have an above certain threshold for feature value mean target value,
       feature value sample size, feature value sample frequency and feature value frequency
       weighted mean target value

    """
    if ignore_binary:
        # Check to see if only values are 1 and 0. If so, don't compute rest
        if df[feature].dropna().value_counts().index.isin([0,1]).all():
            return {feature: []}

    grouped_mean_target = column_values_target_average(df, feature, target,
                                      min_mean_target_threshold = min_mean_target_threshold,
                                      min_sample_size = min_sample_size,
                                      min_sample_frequency = min_sample_frequency,
                                      min_weighted_target_threshold = min_weighted_target_threshold)

    return {feature: grouped_mean_target[feature].values.tolist()}


def df_feature_vals_target_association_dict(df, target,
                                                           include=None,
                                                           exclude=None,
                                      min_mean_target_threshold = 0,
                                      min_sample_size = 0,
                                      min_sample_frequency = 0,
                                      min_weighted_target_threshold=0,
                                           ignore_binary=True):


    columns_to_check = df.columns.values.tolist()
    if include:
        columns_to_check = [col for col in columns_to_check if col in include]
    if exclude:
        columns_to_check = [col for col in columns_to_check if col not in exclude]

    # Compute for all specified columns in dataframe
    list_of_dicts = [feature_vals_target_association_dict(df, column, target,
                                       min_mean_target_threshold = min_mean_target_threshold,
                                      min_sample_size = min_sample_size,
                                      min_sample_frequency = min_sample_frequency,
                                      min_weighted_target_threshold = min_weighted_target_threshold,
                                                         ignore_binary=ignore_binary)
            for column in columns_to_check if column != target]

    # Combine into single dictionary if there are any values
    # that fit the minimum thresholds
    combined_dict = {}
    for dictionary in list_of_dicts:
        # Check it see if any values in list
        feat_vals = list(dictionary.values())
        if len(feat_vals[0]) >=1:
            combined_dict.update(dictionary)
    return combined_dict


def get_specific_dummies(df, col_map=None, prefix=None, suffix=None, return_df=True):
    """ Given a mapping of column_name: list of values, one hot the values
    in the column and concat to dataframe. Optional arguments to add prefixes
    and/or suffixes to created column names.

    Example col_map: {'foo':['bar', 'zero']} would create one hot columns
    for the values bar and zero that appear in the column foo"""
    one_hot_cols = []
    for column, value in col_map.items():
        for val in value:
            # Create one hot encoded arrays for each value specified in key column
            one_hot_column = pd.Series(np.where(df[column] == val, 1, 0))
            # Set descriptive name
            one_hot_column.name = column+'_==_'+str(val)
            # add to list of one hot columns
            one_hot_cols.append(one_hot_column)
    # Concatenate all created arrays together
    one_hot_cols = pd.concat(one_hot_cols, axis=1)
    if prefix:
        one_hot_cols = one_hot_cols.add_prefix(prefix)
    if suffix:
        one_hot_cols = one_hot_cols.add_suffix(suffix)
    if return_df:
        return pd.concat([df, one_hot_cols], axis=1)
    else:
        return one_hot_cols

def one_hot_column_text_match(df, column, text_phrases, case=False):
    """Given a dataframe, text column to search and a list of text phrases, return a binary
       column with 1s when text is present and 0 otherwise
    """
    # Ignore regex group match warning
    import warnings
    warnings.filterwarnings("ignore", 'This pattern has match groups')

    # Create regex pattern to match any phrase in list

    # The first phrase will be placed in its own groups
    regex_pattern = '({})'.format(text_phrases[0])

    # If there's more than one phrase
    # Each phrase is placed in its own group () with an OR operand in front of it |
    # and added to the original phrase

    if len(text_phrases) > 1:
        subsquent_phrases = "".join(['|({})'.format(phrase) for phrase in text_phrases[1:]])
        regex_pattern += subsquent_phrases

    # Cast to string to ensure .str methods work
    df_copy = df.copy()
    df_copy[column] = df_copy[column].astype(str)

    matches = df_copy[column].str.contains(regex_pattern, na=False, case=case).astype(int)

    # One hot where match is True (must use == otherwise NaNs throw error)
    #one_hot = np.where(matches==True, 1, 0 )

    return matches

def get_text_specific_dummies(df, col_map=None, case=False, prefix=None, suffix=None, return_df=True):
    """ Given a mapping of column_name: list of values, search for text matches
    for the phrases in the list. Optional arguments to add prefixes
    and/or suffixes to created column names.

    Example col_map: {'foo':['bar', 'zero']} would search the text in the values of
    'foo' for any matches of 'bar' OR 'zero' the result is a one hot encoded
    column of matches"""
    one_hot_cols = []
    for column, value in col_map.items():
        # Create one hot encoded arrays for each value specified in key column
        one_hot_column = pd.Series(one_hot_column_text_match(df, column, value, case=case))
        # Check if column already exists in df
        if column+'_match_for: '+str(value)[1:-1].replace(r"'", "") in df.columns.values.tolist():
            one_hot_column.name = column+'_supplementary_match_for: '+str(value)[1:-1].replace(r"'", "")
        else:
            # Set descriptive name
            one_hot_column.name = column+'_match_for: '+str(value)[1:-1].replace(r"'", "")
        # add to list of one hot columns
        one_hot_cols.append(one_hot_column)
    # Concatenate all created arrays together
    one_hot_cols = pd.concat(one_hot_cols, axis=1)
    if prefix:
        one_hot_cols = one_hot_cols.add_prefix(prefix)
    if suffix:
        one_hot_cols = one_hot_cols.add_suffix(suffix)
    if return_df:
        return pd.concat([df, one_hot_cols], axis=1)
    else:
        return one_hot_cols


# Functions for use with FunctionTransformer
def replace_column_values(df, col=None, values=None, replacement=None, new_col_name=None):
    """ Discretize a continuous feature by seperating it into specified quantiles
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

def replace_df_values(df, values):
    """ Call pd.DataFrame.replace() on a dataframe and return resulting dataframe.
    Values should be in format of nested dictionaries,
    E.g., {‘a’: {‘b’: nan}}, are read as follows:
        Look in column ‘a’ for the value ‘b’ and replace it with nan
    """
    df_copy = df.copy()
    return df_copy.replace(values)
