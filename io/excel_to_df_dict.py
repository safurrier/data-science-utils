import pandas as pd
import pathlib
def excel_to_df_dict(excel_path: str = None, 
                     include_sheets: list = None,
                     exclude_sheets: list = None,
                     parse_kwds : dict = None):
    """Given a path to an excel file, read in the sheets to dataframes and return a dictionary
    of the form {sheet_name: dataframe}
    """
    if not include_sheets:
        include_sheets = []
    if not exclude_sheets:
        exclude_sheets = []        
    if not parse_kwds:
        parse_kwds = {}
    excel_path = pathlib.Path(excel_path)
    
    # Read in Excel
    xl = pd.ExcelFile(excel_path.as_posix())
    # If specified load data in df_dict
    df_dict = {}
    for sheet_name in xl.sheet_names:
        # Filter down to exclusive set of sheets desired
        if (sheet_name in include_sheets) | (sheet_name not in exclude_sheets):
            # Parse data
            data = xl.parse(sheet_name, **parse_kwds)
            # Add to dictionary
            df_dict[sheet_name] = data
    return df_dict