import pandas as pd
import numpy as np

def find_null_columns(df):
    """
    Return a list of columns with null values

    Args:
    df - dataframe - Dataframe to check columns of

    Returns:
    list of null columns
    """
    return df.columns[df.isnull().any()].tolist()


def null_column_report(df, total=True, percent=True, ):
    """
    Print each null column column in a dataframe that is null as well as percent null

    Args:
    df - pandas dataframe
    total - boolean - Flag to indicate whether to print total null records per column
    percent - boolean - Flag to indicate whether to print percent of column that is null

    Returns:
    None
    """
    num_null_columns = df.shape[1] - df.dropna(axis=1).shape[1]
    print('Number of columns with null values:\n{}\n'.format(num_null_columns))
    null_columns = find_null_columns(df)
    for col in null_columns:
        total_null_records = df[col].isnull().sum()
        print('Column:')
        print(col)
        if total:
            print('Total Nulls:')
            print(total_null_records)
        if percent:
            print('Percent Null:')
            print(round(total_null_records/df.shape[0], 2))
            print()

def null_column_report_df(df):
    """
    Searches a dataframe for null columns and returns a dataframe of the format
    Column | Total Nulls | Percent Nulls
    """
    num_null_columns = df.shape[1] - df.dropna(axis=1).shape[1]
    print('Number of columns with null values:\n{}\n'.format(num_null_columns))
    null_columns = df.columns[df.isnull().any()].tolist()
    null_info_records = []
    for col in null_columns:
        total_null_records = df[col].isnull().sum()
        percent_null_records = round(total_null_records/df.shape[0], 2)
        null_info_records.append({
            'Column':col,
            'Total_Null_Records':total_null_records,
            'Percent_Null_Records':percent_null_records
        })
    return pd.DataFrame(null_info_records)