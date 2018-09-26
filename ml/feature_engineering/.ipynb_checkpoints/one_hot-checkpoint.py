import pandas as pd
import numpy as np

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