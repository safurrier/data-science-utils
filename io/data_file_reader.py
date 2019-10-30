import pandas as pd
import pathlib
from typing import List

def read_data(fpath: str, keep_cols: List[str]=None, reader_kwargs: dict = {}, ) -> pd.DataFrame:
    """Read in a file with a key column and subsequent associated data.
    Parameters for reading in added as dict to reader_kwargs param
    """
    # Check if extension in supported file format
    fpath = pathlib.Path(fpath)
    file_format = fpath.suffix
    if file_format not in ['.csv', '.pickle', '.pkl', '.xlsx', '.hdf']:
        print(f'File format "{file_format}" does not appear to be a supported file format')
        raise ValueError
    
    # Determine input file type and pandas reader
    if file_format == '.csv':
        reader = pd.read_csv
    elif (file_format == '.pickle') | (file_format == '.pkl'):
        reader = pd.read_pickle
    elif (file_format == '.xlsx'):
        reader = pd.read_excel
    elif (file_format == '.hdf'):
        reader = pd.read_excel        

    # Read in filepath:
    try:
        data = reader(fpath.as_posix(), **reader_kwargs)
    except FileNotFoundError as e:
        print(f'Unable to read in {fpath.as_posix()}. Check the file_format and file path')
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