import pandas as pd

def read_data(fpath: str, keep_cols: list=None, file_format='csv', reader_kwargs: dict = {}, ):
    """Read in a file with a key column and subsequent associated data.
    Parameters for reading in added as dict to reader_kwargs param
    """
    # Check for file format
    assert file_format in ['csv', 'pickle', 'pkl', 'xlsx', 'excel'], \
    'File Formats supported are one of "csv", "pickle" or "excel"'
    # Check if extension in supported file format
    extension = fpath.split('.')[1]
    if extension not in ['csv', 'pickle', 'pkl', 'xlsx', 'excel']:
        print('Warning, are you sure the specified file format is supported?')
    
    # Determine input file type and pandas reader
    if file_format == 'csv':
        reader = pd.read_csv
    elif (file_format == 'pickle') | (file_format == 'pkl'):
        reader = pd.read_pickle
    elif (file_format == 'excel') | (file_format == 'xlsx'):
        reader = pd.read_excel

    # Read in filepath:
    try:
        data = reader(fpath, **reader_kwargs)
    except FileNotFoundError as e:
        print(f'Unable to read in {fpath}. Check the file_format and file path')
        raise e
    
    missing_cols = []
    # If specificed, look for subset of columns
    if keep_cols:
        keep_cols = [col
                     if col in data.columns.values.tolist()
                     else missing_cols.append(col)
                     for col
                     in keep_cols]
    else:
        keep_cols = data.columns.values.tolist()
    
    assert keep_cols != [None], 'No columns from keep_cols param found in data' 
    # Print missing columns if present
    if missing_cols:
        print(f'WARNING: These columns were not found in the data: {missing_cols}')

    # Subset to desired columns
    data = data[keep_cols]

    return data