import numpy as np
import pandas as pd

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
    
def one_hot_column_text_match(df, column=None, text_phrases=None, new_col_name=None, return_df=False, case=False):
    """Given a dataframe, text column to search and a list of text phrases, return a binary
       column with 1s when text is present and 0 otherwise
    """
    # Ignore regex group match warning
    import warnings
    warnings.filterwarnings("ignore", 'This pattern has match groups')
    
    # Check params
    assert column in df.columns.values.tolist(), print(f"Column {column} not found in df columns")
    assert text_phrases, print(f"Must specify 'text_phrases' as a list of strings")

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
    
    ## Alter name
    if not new_col_name:
        # If none provided use column name and values matched
        new_col_name = column+'_match_for: '+str(text_phrases)[1:-1].replace(r"'", "")
    matches.name = new_col_name
    
    if return_df:
        df_copy = df.copy()
        df_copy[new_col_name] = matches
        return df_copy
    else:
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
    
    
def inverse_sample_weights(df, target_col, weight_col, 
                    new_col_name=None, min_class_weight = .01,
                    return_df = True):
    """ Given a target class an column to use to derive training weights, 
        create a column of weights where the negative class is the inverse
        of the weights column. 
        
        E.g. weight column of 'Price' would use Price value for positive class
        (where target == 1) and 1/Price for the negative class. 
    """
    df_copy = df.copy()
    pos_class_weights = np.where(df[target_col] == 1 , # Where class is positive
                                       df[weight_col], # Use this val as weight
                                       0) # Else 0
    neg_class_weights_inverse = np.where(df[target_col] == 0 , # Where class is neg
                                       1/df[weight_col], # Use inverse of this
                                       0) # Else 0
    # Handle Edge Case where dividing by 0 results in undefined
    neg_class_weights_inverse = np.where(neg_class_weights_inverse == np.inf , # Where weight is inf (divided by 0)
                                       min_class_weight, # Replace with smallest weighting
                                       neg_class_weights_inverse) # Otherwise keep it
    # Combine weights
    combined_weights_inverse = np.where(pos_class_weights == 0, # Where negative classes 
                                    neg_class_weights_inverse, # Place the inverse as negative weights
                                    pos_class_weights) # Else keep the positive weights
    if not new_col_name:
        new_col_name = 'Sample_Inverse_Weights'
        
    df_copy[new_col_name] = combined_weights_inverse
    
    if return_df:
        return df_copy
    else:
        return pd.Series(combined_weights_inverse, name=new_col_name)   
    
def even_sample_weights(df, target_col, weight_col, 
                    new_col_name=None, 
                    return_df = True):
    """ Given a target class an column to use to derive training weights, 
        create a column of weights where the negative class is the inverse
        of the weights column. 
        
        E.g. weight column of 'Price' would use Price value for positive class
        (where target == 1) and 1/Price for the negative class. 
    """
    df_copy = df.copy()
    pos_class_weights = np.where(df[target_col] == 1 , # Where class is positive
                                       df[weight_col], # Use this val as weight
                                       0) # Else 0
    neg_class_even_weights = np.where(df[target_col] == 0, # Where class is neg
          (df[target_col] == 0).sum()/(df[target_col] == 0).shape[0] , # Create even weighting
                                       0) 
    # Combine weights
    combined_weights = np.where(pos_class_weights == 0, # Where negative classes 
                                    neg_class_even_weights, # Place the inverse as negative weights
                                    pos_class_weights) # Else keep the positive weights
    if not new_col_name:
        new_col_name = 'Sample_Even_Weights'
        
    df_copy[new_col_name] = combined_weights
    
    if return_df:
        return df_copy
    else:
        return pd.Series(combined_weights, name=new_col_name) 
    
def df_group_one_hot(df, cols_to_group, how ='any', new_col_name=None, return_df=False):
    """Given a list of columns, find the intersection of their one hot values
    and return as a Series
    
    Parameters
    ----------
    df : Pandas DataFrame
        A dataframe from which to pull the one hot columns
    cols_to_group : list[str]
        A list of column names for which to group.
    how: str
        Denotes whether to return columns intersection ('all') or union ('any')
        Default is any
        Must be either 'any' or 'all'
    new_col_name: str
        A string name for the returned column. Optional         
    return_df : bool
        Optional flag to return the df with the computed column.
    
    Returns
    -------
    grouped_one_hot_col: Pandas Series (default)
        A series of the computed intersection of the one hot columns 
    df_copy: Pandas DataFrame
        A copied dataframe with the computed column in it
        
    
    Examples
    df_group_one_hot(df, ['flag1', 'flag2', 'flag3'])
    
    0         0
    1         0
    2         0
    3         1
    4         0
    5         1
    Name: flag1_and_flag2_and_flag3
    
    1s indicate intersection where original columns ['flag1', 'flag2', 'flag3']
    all had 1 
    
    """
    assert how in ['any', 'all'], print(f'argument "how" must be either "any" or "all" and not "{how}"')
    
    # Group one hot columns
    # Create boolean mask where columns are 1
    mask = df[cols_to_group] == 1
    if how == 'all':
        # Create column name if necessary
        if not new_col_name:
            new_col_name = '_AND_'.join(cols_to_group)
        grouped_one_hot_col = pd.Series(mask 
                                        .all(axis=1) # Return True where entire row is all True
                                        .astype(int),  # Cast to int
                                        name = new_col_name 
                                       )
    elif how =='any':
        # Create column name if necessary
        if not new_col_name:
            new_col_name = '_OR_'.join(cols_to_group)        
        grouped_one_hot_col = pd.Series(mask # Create boolean mask where columns are 1
                                        .any(axis=1) # Return True where any element of row is True
                                        .astype(int),  # Cast to int
                                        name = new_col_name 
                                       )
        
    if return_df:
        df_copy = df.copy()
        df_copy[new_col_name] = grouped_one_hot_col
        return df_copy
    else:
        return grouped_one_hot_col 
    
   