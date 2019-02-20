import pandas as pd
import numpy as np
import re as witchcraft

def legal_column_names(df, join_char='_', replace_illegal_chars=True, illegal_char_replacement='',
                      camelcase=False, lowercase=False, uppercase=False, verbose=0):
    """Given a dataframe renamed the columns to remove spaces and remove illegal characters
    so that column names can be used as proper python variables. Optionally define case of 
    column names (camelcase, lower, upper)"""
    # Start updates as old columns
    new_columns = df.columns.values.tolist()
    # Add case formatting if so desired
    if camelcase:
        new_columns = [column.title() for column in new_columns]
    if lowercase:
        new_columns = [column.lower() for column in new_columns]   
    if uppercase:  
        new_columns = [column.upper() for column in new_columns]        
    new_columns = [join_char.join(column.split(' ')) for column in new_columns]
    if replace_illegal_chars:
        def legalize_string(string, illegal_char_replacement):
            "Turn a string into a valid callable variable name"
            # Remove invalid characters
            string = witchcraft.sub('[^0-9a-zA-Z_]', illegal_char_replacement, string)

            # Remove leading characters until we find a letter or underscore
            string = witchcraft.sub('^[^a-zA-Z_]+', illegal_char_replacement, string)
            return string
        new_columns = [legalize_string(column, illegal_char_replacement=illegal_char_replacement)
                       for column in new_columns]

    renaming_dict = dict(zip(df.columns.values.tolist(), new_columns))
    if verbose > 0:
        new_name_dict = {key: value for key, value in renaming_dict if key!=value}
        print(f'The following columns will be renamed:\n{new_name_dict}')

    renamed_df = df.rename(columns=renaming_dict)
    return renamed_df

def duplicate_column_name_report(df):
    """ Returns a list of columns with duplicate column names
        and its associated count
    """
    from collections import Counter
    col_counts = Counter(df.columns).most_common()
    dupe_columns = [tup for tup in col_counts if tup[1] >1]
    if not dupe_columns:
        return 'No duplicated column names'
    else:
        return dupe_columns

def only_unique_columns(df):
    """ Returns the dateaframe with duplicated column names
        removed. Note: Doesn't check column values, only names
    """
    df_copy = df.copy()
    # Find Unique Indices
    _, i = np.unique(df.columns, return_index=True)
    unique_df = df_copy.iloc[:, i]
    return unique_df

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

