import pandas as pd
import textacy
import textblob
import en_core_web_sm
nlp = en_core_web_sm.load()

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
          .rename(columns={'index': 'Entity', 0: 'Count'})
          .sort_values(by='Count', ascending=False)
          .reset_index(drop=True))

    return df


def entity_statements(doc, entity, ignore_entity_case=True,
                      min_n_words=1, max_n_words=300, return_entity=False):
    """
    Extract sentences with a specified entity present in it


    Args:
        doc (``textacy.Doc`` or ``spacy.Doc``)
        entity (str): a noun or noun phrase of some sort (e.g. "President Obama",
            "global warming", "Python")
        ignore_entity_case (bool): if True, entity matching is case-independent
        min_n_words (int): min number of tokens allowed in a matching fragment
        max_n_words (int): max number of tokens allowed in a matching fragment

    Yields:
        (``spacy.Span`` or ``spacy.Token``) or (``spacy.Span`` or ``spacy.Token``, ``spacy.Span`` or ``spacy.Token``):
        depending on if return_entity is enabled or not


    Notes:
        Modified from source code of Textacy's textacy.extract.semistructured_statements()

        Inspired by N. Diakopoulos, A. Zhang, A. Salway. Visual Analytics of
        Media Frames in Online News and Blogs. IEEE InfoVis Workshop on Text
        Visualization. October, 2013.

        Which itself was inspired by by Salway, A.; Kelly, L.; Skadiņa, I.; and
        Jones, G. 2010. Portable Extraction of Partially Structured Facts from
        the Web. In Proc. ICETAL 2010, LNAI 6233, 345-356. Heidelberg, Springer.
    """
    if ignore_entity_case is True:
        entity_toks = entity.lower().split(' ')
        def get_tok_text(x): return x.lower_
    else:
        entity_toks = entity.split(' ')
        def get_tok_text(x): return x.text

    first_entity_tok = entity_toks[0]
    n_entity_toks = len(entity_toks)

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
                        if list(entity_statements(doc, entity=entity))]  # If statement that removes null sentences

    entity_sentences = [
        item for sublist in entity_sentences for item in sublist]

    return entity_sentences
