import pandas as pd
import numpy as np
import re as witchcraft
import warnings


def feature_value_match_dict_from_column_names(column_names, 
                                               prefix=None, suffix=None,
                                               col_name_to_feature_vals_delimiter='_match_for_____',
                                               feature_vals_delimiter='_'):
    """Given a list of column names of the form COLUMN_NAME_match_for: FEATURE_VALUE1
    FEATURE_VALUE2, FEATURE_VALUE3 return a dictionary map of the form {COLUMN_NAME:
    [FEATURE_VALUE1, FEATURE_VALUE2, FEATURE_VALUE3]}. Optional arguments for column 
    prefix, suffix, col_to_feature_delimiter and feature_value_delimeter
    
    Parameters
    ----------
    column_names: list[str]
        A list of string column names for which to extract to a dictionary
        of the form {column_name:[list of feature values]}
    prefix: str
        A string prefix to remove from the created columns
    suffix: str
        A string suffix to from the created columns
    col_name_to_feature_vals_delimiter : str, Default = '_match_for: '
        The string delimiter that seperates the column features values from 
        the column name
    feature_vals_delimiter: str, Default = ', '
        The string delimiter that seperates the features values from 
        each other
        
    Example
    ---------
    feature_value_match_dict_from_column_names([
    'Sparse_feature_aggregation_Claim Classification_1mo_match_for: Catastrophic',
    'Sparse_feature_aggregation_Injury Type_1mo_match_for: Permanent Partial-Unscheduled',
    'Sparse_feature_aggregation_riss_match_for: 14.0, 17.0, 12.0, 13.0'],
    
    prefix='Sparse_feature_aggregation_')
    
    >>> {'Claim Classification_1mo': ['Catastrophic'], 
    'Injury Type_1mo': ['Permanent Partial-Unscheduled'],
    'riss': ['14.0', '17.0', '12.0', '13.0']}
    """
    # If single string column name passed, 
    # turn it into a list
    if isinstance(column_names, str):
        column_names = [column_names]
    # Remove prefix/suffix if specified
    if prefix:
        column_names = [col.replace(prefix, "")
                  for col 
                  in column_names]
    if suffix:
        column_names = [col.replace(suffix, "")
                  for col 
                  in column_names]    # Create empty map
    match_value_map = {}    
    # Iterate through column list        
    for column in column_names:
        # Split the into column name and feature value
        column_name, match_values = column.split(col_name_to_feature_vals_delimiter)
        # Extract just the feature values
        match_values_list = match_values.split(feature_vals_delimiter)
        # Add to feature value map
        match_value_map[column_name] = match_values_list
    return match_value_map     


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
    
def text_match_one_hot(df, column=None, text_phrases=None, new_col_name=None, return_df=False, case=False,
                      supress_warnings: bool=False):
    """Given a dataframe, text column to search and a list of text phrases, return a binary
       column with 1s when text is present and 0 otherwise
    """
    # Ignore regex group match warning
    warnings.filterwarnings("ignore", 'This pattern has match groups')
    
    # Check params
    assert text_phrases, print(f"Must specify 'text_phrases' as a list of strings")
    if (column not in df.columns.values.tolist()):
        if not suppress_warnings:
            warnings.warn(f'Column "{column}" not found in dataframe. No matches attempted')
            return

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
    
def text_match_one_hot_from_map(df, col_map=None, case=False, prefix=None, suffix=None, return_df=True):
    """ Given a mapping of column_name: list of values, search for text matches
    for the phrases in the list. Optional arguments to add prefixes 
    and/or suffixes to created column names.
    
    Example col_map: {'foo':['bar', 'zero']} would search the text in the values of
    'foo' for any matches of 'bar' OR 'zero' the result is a one hot encoded
    column of matches"""
    # For naming columns
    def legalize_string(string, illegal_char_replacement):
        "Turn a string into a valid callable variable name"
        # Remove invalid characters
        string = witchcraft.sub('[^0-9a-zA-Z_]', illegal_char_replacement, string)

        # Remove leading characters until we find a letter or underscore
        string = witchcraft.sub('^[^a-zA-Z_]+', illegal_char_replacement, string)
        return string
    
    one_hot_cols = [] 
    for column, value in col_map.items():
        # Set descriptive name
        new_col_name = column+'_match_for_____'+str(value)[1:-1].replace(r"'", "").replace(r", ", "_")
        new_col_name = legalize_string(new_col_name, '__')
        
        # Create one hot encoded arrays for each value specified in key column
        one_hot_column = pd.Series(one_hot_column_text_match(df, column, value, case=case, suppress_warnings=suppress_warnings))
        
        # Check if column already exists in df
        if new_col_name in df.columns.values.tolist():
            new_col_name = column+'_supplementary_match_for_____'+str(value)[1:-1].replace(r"'", "").replace(r", ", "_")
            new_col_name = legalize_string(new_col_name, '__')
            one_hot_column.name = new_col_name        
        else:
            one_hot_column.name = new_col_name
            
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