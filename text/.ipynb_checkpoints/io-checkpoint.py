import pandas as pd
import textacy
import textblob
import en_core_web_sm
nlp = en_core_web_sm.load()


# Multiprocessing Imports
from dask import dataframe as dd
from dask.multiprocessing import get
from multiprocessing import cpu_count

def dask_df_textacy_apply(df, text_col, textacy_col_name='textacy_doc', ncores=None, inplace=False):
    """
    Use dask to parallelize apply textacy Doc object creation from a dataframe

    Parameters
    ----------
    df : DataFrame
        Dataframe which holds the text
    text_col : str
        The name of the text column in the df
    textacy_col_name : str
        The name to give to the column with the textacy doc objects
    ncores : int
        Number of cores to use for multiprocessing. Defaults to all cores in cpu minus one.
    inplace : bool
        Whether to return the entire df with the textacy doc series concatenated
        or only textacy doc series.
        Default is False
    Returns
    -------
    DataFrame / Series
        Either the dataframe passed as arg with the textacy series as last column or
        just the textacy column
    """
    # If no number of cores to work with, default to max
    if not ncores:
        nCores = cpu_count() - 1
        nCores

    # Partition dask dataframe and map textacy doc apply
	# Sometimes this fails because it can't infer the dtypes correctly
	# meta=pd.Series(name=0, dtype='object') is a start
	# This is also a start https://stackoverflow.com/questions/40019905/how-to-map-a-column-with-dask?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
    # Possibly both the inner lambda apply and outer lambda df both need metadata?
    textacy_series = dd.from_pandas(df, npartitions=nCores).map_partitions(
      lambda df : df[text_col].apply(lambda x : textacy.doc.Doc(x, lang=nlp))).compute(get=get)

    # Name the series
    textacy_series.name = textacy_col_name

    # If inplace return the dataframe and textacy Series
    if inplace:
        return pd.concat([df, textacy_series], axis=1)
    # Else return just the Textacy series
    else:
        return textacy_series

def load_textacy_corpus(df, text_col, metadata=True, metadata_columns=None):
    # Fill text columns nulls with empty strings
    df[text_col] = df[text_col].fillna('')
    if metadata:
        # Default to metadata columns being every column except the text column
        metadata_cols = list(df.columns)
        # If list is provided use those
        if metadata_columns:
            metadata_cols = metadata_columns

        # Add text column to metadata columns
        # These will constitute all the information held in the textacy corpus
            metadata_columns.append(text_col)
        # Subset to these
        df = df[metadata_cols]

        # Convert to nested dict of records
        records = df.to_dict(orient='records')
        # Split into text and metadata stream
        text_stream, metadata_stream = textacy.io.split_records(records, text_col)

        # Create Corpus
        return textacy.corpus.Corpus(lang='en', texts=text_stream, metadatas=metadata_stream)
    # With no metadata
    else:
        text_stream = (text for text in df[text_col].values)
        return textacy.corpus.Corpus(lang='en', texts=text_stream)