import pandas as pd
import textacy
import textblob
import en_core_web_sm
nlp = en_core_web_sm.load()


# Multiprocessing Imports
from dask import dataframe as dd
from dask.multiprocessing import get
from multiprocessing import cpu_count

# Sentiment Imports
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Local Imports
from src.utils.pandas_utils import pivot_df_to_row

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

# Entity Extraction

def corpus_entity_counts(corpus, include=None, exclude=None):
    """
    Given a textacy corpus, return a dataframe of entities and their respective counts.

    Parameters
    ----------
    corpus : int
        Description of arg1
    include : str or Set[str]
        Remove named entities whose type IS NOT in this param;
        if “NUMERIC”, all numeric entity types (“DATE”, “MONEY”, “ORDINAL”, etc.) are included
    exclude : str or Set[str]
        remove named entities whose type IS in this param; if “NUMERIC”,
        all numeric entity types (“DATE”, “MONEY”, “ORDINAL”, etc.) are excluded

    Returns
    -------
    Dataframe
        A pandas dataframe with entities and their respective counts, sorted by highest count

    """
    from collections import Counter

    # Extract all entities
    entities = [list(textacy.extract.named_entities(doc, include_types=include, exclude_types=exclude))
                for doc in
                corpus]
    # Pull all non-null entities to flattened list
    non_null_entities = []
    for entity in entities:
        if entity:
            non_null_entities.extend(entity)
    # Change dtype to string so counter can distinguish
    non_null_entities = [str(x) for x in non_null_entities]

    # Count entities
    entity_counts = Counter(non_null_entities)

    # Entity Dataframe
    df = (pd.DataFrame.from_dict(entity_counts, orient='index')
          .reset_index()
          .rename(columns={'index':'Entity', 0:'Count'})
          .sort_values(by='Count', ascending=False)
          .reset_index(drop=True))

    return df



def entity_statements(doc, entity, ignore_entity_case=True,
                      min_n_words=1, max_n_words=300, return_entity=False):
    """
    Extract sentences with a specified entity present in it
    Modified from source code of Textacy's textacy.extract.semistructured_statements()

    Args:
        doc (``textacy.Doc`` or ``spacy.Doc``)
        entity (str): a noun or noun phrase of some sort (e.g. "President Obama",
            "global warming", "Python")
        ignore_entity_case (bool): if True, entity matching is case-independent
        min_n_words (int): min number of tokens allowed in a matching fragment
        max_n_words (int): max number of tokens allowed in a matching fragment

    Yields:
        (``spacy.Span`` or ``spacy.Token``) or (``spacy.Span`` or ``spacy.Token``, ``spacy.Span`` or ``spacy.Token``):
        dependin on if return_entity is enabled or not


    Notes:
        Inspired by N. Diakopoulos, A. Zhang, A. Salway. Visual Analytics of
        Media Frames in Online News and Blogs. IEEE InfoVis Workshop on Text
        Visualization. October, 2013.

        Which itself was inspired by by Salway, A.; Kelly, L.; Skadiņa, I.; and
        Jones, G. 2010. Portable Extraction of Partially Structured Facts from
        the Web. In Proc. ICETAL 2010, LNAI 6233, 345-356. Heidelberg, Springer.
    """
    if ignore_entity_case is True:
        entity_toks = entity.lower().split(' ')
        get_tok_text = lambda x: x.lower_
    else:
        entity_toks = entity.split(' ')
        get_tok_text = lambda x: x.text

    first_entity_tok = entity_toks[0]
    n_entity_toks = len(entity_toks)
    #cue = cue.lower()
    #cue_toks = cue.split(' ')
    #n_cue_toks = len(cue_toks)

    def is_good_last_tok(tok):
        if tok.is_punct:
            return False
        if tok.pos in {CONJ, DET}:
            return False
        return True

    for sent in doc.sents:
        for tok in sent:

            # filter by entity
            if get_tok_text(tok) != first_entity_tok:
                continue
            if n_entity_toks == 1:
                the_entity = tok
                the_entity_root = the_entity

            elif all(get_tok_text(tok.nbor(i=i + 1)) == et for i, et in enumerate(entity_toks[1:])):
                the_entity = doc[tok.i: tok.i + n_entity_toks]
                the_entity_root = the_entity.root
            else:
                continue
            if return_entity:
                yield (the_entity, sent.orth_)
            else:
                yield (sent.orth_)
            break

def list_of_entity_statements(corpus, entity):
    """
    Given an entity and a textacy corpus, return a list of all the sentences in which this entity occurs

    Parameters
    ----------
    corpus : textacy Corpus object
    entity : str
        The entity for which to search all the sentences within the corpus
    Returns
    -------
    entity_sentences
        A list of strings, each being a sentence which contains the entity search
    """

    entity_sentences = [list(entity_statements(doc, entity=entity))
                        for doc
                        in corpus
                        if list(entity_statements(doc, entity=entity))] # If statement that removes null sentences

    entity_sentences = [item for sublist in entity_sentences for item in sublist]

    return entity_sentences

# Entity Sentiment extractions
def vader_entity_sentiment(df,
                              textacy_col,
                              entity,
                              inplace=True,
                              vader_sent_types=['neg', 'neu', 'pos', 'compound'],
                              keep_stats=['count', 'mean', 'min', '25%', '50%', '75%', 'max']):
    """
    Pull the descriptive sentiment stats of text sentence with a specified entity in it.

    Parameters
    ----------
    df : DataFrame
        Dataframe which holds the text
    textacy_col : str
        The name to give to the column with the textacy doc objects
    entity : str
        The entity to search the textacy Doc object for
    inplace : bool
        Whether to return the entire df with the sentiment info or the sentiment info alone
        Default is False
    vader_sent_types : list
        The type of sentiment to extract. neg: negative, pos: positive, neu: neutral, compound is
        comination of all three types of all
    keep_stats : list
        A list of the summary statistics to keep. Default is all returned by pandas DataFrame.describe() method

    Returns
    -------
    DataFrame
        Either the dataframe passed as arg with the sentiment info as trailing columns
        or the sentiment descriptive stats by itself
    """
    vader_analyzer = SentimentIntensityAnalyzer()

    sentiment_rows = []
    for text in df[textacy_col].values:
        text_entities = list(entity_statements(text, entity))


         # Iterate through all sentences and get sentiment analysis
        entity_sentiment_info = [vader_analyzer.polarity_scores(sentence)
                                for
                                sentence
                                in
                                text_entities]

        # After taking sentiments, turn into a dataframe and describe
        try:
            # Indices and columns to keep
            keep_stats = keep_stats
            keep_cols = vader_sent_types

            # Describe those columns
            summary_stats = pd.DataFrame(entity_sentiment_info).describe().loc[keep_stats, keep_cols]

            # Add row to list
            sentiment_rows.append(pivot_df_to_row(summary_stats))

        # If there's nothing to describe
        except ValueError as e:
            # Create a summary stats with nulls
            summary_stats = pd.DataFrame(index=keep_stats, columns=keep_cols)

            # Add to list of rows
            sentiment_rows.append(pivot_df_to_row(summary_stats))
    # Concatenate All rows together into one dataframe
    sentiment_df = pd.concat(sentiment_rows).add_prefix(entity+'_')

    if not inplace:
        return sentiment_df.reset_index(drop=True)
    else:
        # Return original df with new sentiment attached
        return pd.concat([df, sentiment_df], axis=1)


def textblob_entity_sentiment(df,
                              textacy_col,
                              entity,
                              inplace=True,
                              subjectivity=False,
                              keep_stats=['count', 'mean', 'min', '25%', '50%', '75%', 'max']):
    """
    Pull the descriptive sentiment stats of text sentence with a specified entity in it.

    Parameters
    ----------
    df : DataFrame
        Dataframe which holds the text
    textacy_col : str
        The name to give to the column with the textacy doc objects
    entity : str
        The entity to search the textacy Doc object for
    inplace : bool
        Whether to return the entire df with the sentiment info or the sentiment info alone
        Default is False
    subjectivity : bool
        Whether to include the subjectivity of the sentiment. Defaults to False.
    keep_stats : list
        A list of the summary statistics to keep. Default is all returned by pandas DataFrame.describe() method

    Returns
    -------
    DataFrame
        Either the dataframe passed as arg with the sentiment info as trailing columns
        or the sentiment descriptive stats by itself
    """
    sentiment_rows = []
    for text in df[textacy_col].values:
        text_entities = list(entity_statements(text, entity))

         # Iterate through all sentences and get sentiment analysis
        entity_sentiment_info = [textblob.TextBlob(sentence).sentiment_assessments
                                for
                                sentence
                                in
                                text_entities]

        # After taking sentiments, turn into a dataframe and describe
        try:
            # Indices and columns to keep
            #keep_stats = ['count', 'mean', 'min', '25%', '50%', '75%', 'max']
            keep_cols = ['polarity']

            # If subjectivity is set to true, values for it will also be captured
            if subjectivity:
                keep_cols.append('subjectivity')

            # Describe those columns
            summary_stats = pd.DataFrame(entity_sentiment_info).describe().loc[keep_stats, keep_cols]

            # Add row to list
            sentiment_rows.append(pivot_df_to_row(summary_stats))

        # If there's nothing to describe
        except ValueError as e:
            # Create a summary stats with nulls
            summary_stats = pd.DataFrame(index=keep_stats, columns=keep_cols)

            # Add to list of rows
            sentiment_rows.append(pivot_df_to_row(summary_stats))
    # Concatenate All rows together into one dataframe
    sentiment_df = pd.concat(sentiment_rows).add_prefix(entity+'_')

    if not inplace:
        return sentiment_df.reset_index(drop=True)
    else:
        # Return original df with new sentiment attached
        return pd.concat([df, sentiment_df], axis=1)
