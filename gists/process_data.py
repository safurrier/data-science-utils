# coding: utf-8
import os
import warnings
import pandas as pd
import numpy as np
import click
import pathlib
from src.utils.io.get_absolute_fpath import get_absolute_fpath
from src.utils.io.python_config_dict import config_dict_from_python_fpath

# Silence C dtype mapping warnings
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
# Silence Deprecation Warning for click using importlib
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Change to root dir
os.chdir(get_absolute_fpath())

# Script Purpose
#################

# 1) Load in data from a source file and table (e.g. SQL, HDF, cloud etc)
# 2) Load in an sklearn pipeline for ETL transformation via the config file
# specified at --config_path which should be a python file holding necessary 
# variables, specifically the ETL pipeline in variable PIPELINE
# 3) Export data to a db and table. If none is specified, use the import variable ones


@click.command()
@click.option('--config_path', default=None)
@click.option('--db_import_path', default=None)
@click.option('--db_input_table', default=None)
@click.option('--db_export_path', default=None)
@click.option('--db_export_table', default=None)
@click.option('--verbose', default=1)
def process_data(config_path: str = None,
                 db_import_path: str = None,
                 db_input_table: str = None,
                 db_export_path: str = None,
                 db_export_table: str = None,
                 verbose: int = None
                 ):
    """
    Process the data
    """
	# Set export variables if none specified to be the same as import (e.g. inplace)
	if verbose:
		if (not db_export_path) | (not db_export_table):
			click.echo('No export variables set. Data ETL will be done in place at the'
			f'db and table specified by db at "{db_export_path}" and table "{db_export_table}"')
	if not db_export_path:
		db_export_path = db_import_path
	if not db_export_table:
		db_export_table = db_input_table
    # Turn string paths into pathlib Paths
    db_import_path = pathlib.Path(db_import_path)
    db_export_path = pathlib.Path(db_export_path)
    config = config_dict_from_python_fpath(config_path)

    # Establish database connection
    # conn = sqlite3.connect(db_import_path.as_posix())
    # Read  Data
    # df = pd.read_sql(sql=f'SELECT * FROM {db_input_table}', con=conn)
    df = pd.read_hdf(db_import_path.as_posix(), db_input_table)
    if verbose:
        click.echo(f'Reading in data from table {db_input_table} at {db_import_path.as_posix()}')
    # Process data with pipeline
    pipeline = config['PIPELINE']
    transformed_df = pipeline.fit_transform(df)
    # Write to DB
    # Write each DF into the database, replacing the table if it previously existed
    if verbose:
        click.echo(f'Final datashape for transformed data is {transformed_df.shape}')
        click.echo(f'Compare to original shapes of: {df.shape}')
    if verbose:
        click.echo(f'Placing data into table "{db_export_table}" in the db at {db_export_path.as_posix()}')
    # transformed_df.to_sql(db_export_table, con=conn, if_exists='replace', index=False)
    transformed_df.to_hdf(db_export_path.as_posix(), db_export_table)


if __name__ == '__main__':
    process_data()
