import pandas as pd
def read_data(fpath: str, key: list, keep_cols: list, file_format='csv', reader_kwargs: dict = {}, ):
    """Read in a file with a key column and subsequent associated data.
    Parameters for reading in added as dict to reader_kwargs param
    """
    # Check for file format
    assert file_format in ['csv', 'pickle', 'excel'], 'File Formats supported are one of "csv", "pickle" or "excel"'
    # If key is a string put it into a list
    if isinstance(key, str):
        key = [key]
    # Determine input file type and pandas reader
    if file_format == 'csv':
        reader = pd.read_csv
    elif file_format == 'pickle':
        reader = pd.read_pickle
    elif file_format == 'excel':
        reader = pd.read_excel

    # Read in filepath:
    try:
        data = reader(fpath, **reader_kwargs)
    except FileNotFoundError as e:
        print(f'Unable to read in {fpath}. Check the file_format and file path')
        raise e

    # Subset to desired columns
    data = data[key + keep_cols]

    return data