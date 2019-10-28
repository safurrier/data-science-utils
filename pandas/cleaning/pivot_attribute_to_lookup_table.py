import pandas as pd

def pivot_attribute_to_lookup_table(df: pd.DataFrame=None, 
                                    key: str=None, 
                                    pivot_field: str=None,
                                    pivot_field_prefix: str=None) -> pd.DataFrame:
    """
    Pivot out a dataframe from a provided a key and a field. 
    New data frame takes the form:
    Key | Pivot_Field_Value_1 | Pivot_Field_Value_2 | etc 

    Parameters
    ----------
    df : Pandas DataFrame
        A dataframe containing table to pivot out
    key : str
        The string column name to use as a key
        for pivoting out the values of arg 'pivot_field'
    pivot_field : str
        The string column name pivot out with each succesive 
        value associated with the key as its own column
    pivot_field_prefix : str
        A string prefix to add to the field value prefix
        Default field names will be pivot_field + _ + int
    fillna_val: 
    Returns
    -------
    Pandas DataFrame
        "Wide" dataframe with columns in format:
        Key | Pivot_Field_Value_1 | Pivot_Field_Value_2 | etc
    """
    tmp_df = df.copy(deep=True)
    # Getting a consecutive count by Policy ID Number
    tmp_df['ID'] = tmp_df.groupby(key).cumcount()
    # Adding one so that it's 0 indexed
    tmp_df['ID'] += 1

    # Pivot to wide format
    wide_format_df = (tmp_df.pivot(key, 'ID', pivot_field)
    )

    # Use more descriptive names
    col_names = [pivot_field+'_'+str(col) for col in wide_format_df.columns]
    wide_format_df.columns = col_names

    # Add prefix if specified
    if pivot_field_prefix:
        wide_format_df = wide_format_df.add_prefix(pivot_field_prefix)
    
    # Reset so that the key is a column again
    wide_format_df = wide_format_df.reset_index()

    return wide_format_df