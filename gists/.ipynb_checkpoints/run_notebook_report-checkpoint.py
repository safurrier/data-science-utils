# coding: utf-8
import os
import warnings
import pandas as pd
import click
import pathlib
import subprocess
import papermill as pm
from src.utils.io.get_absolute_fpath import get_absolute_fpath
from src.utils.io.python_config_dict import config_dict_from_python_fpath

# Silence C dtype mapping warnings
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
# Silence Deprecation Warning for click using importlib
warnings.filterwarnings("ignore", category=DeprecationWarning)
pd.options.mode.chained_assignment = None  # default='warn'


# Change to root dir
os.chdir(get_absolute_fpath())


@click.command()
@click.option('--notebook_import_path', default=None)
@click.option('--config_path', default=None)
@click.option('--parameterized_notebook_export_path', default=None)
@click.option('--report_export_path', default=None)
@click.option('--verbose', default=1)
def run_notebook_report(notebook_import_path: str = None,
                        config_path: str = None,
                        parameterized_notebook_export_path: str = None,
                        report_export_path: str = None,
                        verbose: int = None
                        ):
    """
    Exectute a Jupyter Notebook with specific configuration settings
    and output as a codeless html file

    Utitlizes Papermill to execute notebooks and Subprocess to call
    NBconvert on the executed notebook

    Parameters
    ----------
    notebook_import_path: str
        A string path to the notebook to execute initially with papermill. 
        Should be a notebook with references to a config file that is 
        specified in a cell with a 'parameter tag'

    config_path: str
        A string path to the configuration file that parameterizes the notebook at
        `notebook_import_path`
    parameterized_notebook_export_path: str
        A string path designating the location to place the parameterized notebook at. 
        Will be parameterized with papermill and exectued using nbconvert.      
    report_export_path: str
        A string path designating the location to place executed html file at 
    verbose: int

    # TODO
    #######
    # Options for 
    # code cell execution present or not in report
    # Removing the parameterized notebook after executing it with nbconvert

    Example
    -------
    python run_notebook_report.py --notebook_import_path='notebooks/monthly_viz.ipnyb' \
    --config_path='config/monthly_viz_config.py' \
    --parameterized_notebook_export_path='reports/monthly_viz/november.ipnyb'\
    --report_export_path='reports/monthly_viz/november.html'
    """

    # Turn string paths into pathlib Paths
    notebook_import_path = pathlib.Path(notebook_import_path)
    config_path = pathlib.Path(config_path)
    parameterized_notebook_export_path = pathlib.Path(
        parameterized_notebook_export_path)
    report_export_path = pathlib.Path(report_export_path)

    if verbose:
        click.echo(
            f'Attempting to execute template notebook at {notebook_import_path.as_posix()}'
            f' using papermill with paremeters set in config file at {config_path.as_posix()}\n'
        )
    # Execute notebook using papermill
    pm.execute_notebook(
        notebook_import_path.as_posix(),
        parameterized_notebook_export_path.as_posix(),
        # Run according to params
        parameters=dict(
            config_path=config_path.as_posix())
    )
    if verbose:
        click.echo(
            f'Outputted executed notebook at {parameterized_notebook_export_path.as_posix()}\n'
        )
    # %% [markdown]
    # Convert to html report
    resp = subprocess.run(
        f"jupyter nbconvert {parameterized_notebook_export_path.as_posix()} --to=html --TemplateExporter.exclude_input=True")

    # If there was an error print
    if resp.check_returncode():
        click.echo(
            f'Error converting {parameterized_notebook_export_path.as_posix()} to html report \n')
    # Otherwise move the outputted report and alert the use
    else:
        # # Remove old report if present
        # if os.path.isfile(report_export_path.as_posix()):
        #     os.remove(report_export_path.as_posix())
        # Move the outputted file to where the script specifies
        os.rename(parameterized_notebook_export_path.as_posix().replace(
            "ipynb", "html"), report_export_path.as_posix())

        if verbose:
            click.echo(
                f'Output notebook at html without code at {report_export_path.as_posix()} \n')


if __name__ == '__main__':
    run_notebook_report()
