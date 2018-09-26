import pandas as pd
import numpy as np

def feature_list_log(selected_features_df, method=None, notes = None, split_char='---',
                     export=False, export_fpath=None, previous_features_df_fpath=None):
    """ Given a DataFrame, create a record of its columns as selected features.
        Optionally specify feature selection method or notes
        Can specify a filepath of previous feature list logs to add to
        Exports file to specified path, or defaults to 'datetime-featurelist.csv'
        
    Parameters
    ----------
    selected_features_df : Pandas DataFrame
        A dataframe whose columns are the selected features
    method : str
        A feature selection method name. 
    notes: str
        A string of notes to add to the log         
    export_fpath : str
        The file path to export the log to. If specified, features_log will export there.
    previous_features_df_fpath : str
        The file path to of previous feature list logs to add to  
    split_char : str
        The string character pattern to join the features together on.
        Default is '---' 
    
    Returns
    -------
    features_log: Pandas DataFrame
        A dataframe of feature lists log
    """    
    import datetime
    # Time as YYY:MM:DD HH:MM
    timestamp = str(datetime.datetime.today())[:-10]
    if not notes:
        notes = ''
    if not method:
        method = 'Not_Specified'
    # Create record of features
    selected_features_record = {
        'Date':timestamp,
        'Method':method,
        'Number_of_Features':selected_features_df.shape[1],
        'Notes': notes,
        'Features':split_char.join(selected_features_df.columns.values.tolist())
        }
    if previous_features_df_fpath:
        # Load in previous log of features
        previous_feature_selections_df = pd.read_csv(previous_features_df_fpath)
    # Create record
    features_log = pd.DataFrame(selected_features_record, index=[0])
    # Add to old records
    if previous_features_df_fpath:
        features_log = pd.concat([previous_feature_selections_df, features_log], ignore_index=True)
        features_log.reset_index(drop=True)
    # Export
    if export_fpath:
        print(f'Exported featurelist log df to {export_fpath}')
        features_log.to_csv(export_fpath, index=False)
    return features_log

def ranked_flists(flist_log, split_char='---'):
    """Given a feature list log, return an unpacked dataframe with each
       method and ranking of features (assuming features are listed in order
       of importance)
    """
    ranked_flist_dfs = []
    # iterrows is slow but featurelist log should rarely if ever
    # be large
    for index, row in flist_log.iterrows():
        method = row['Method']
        flist = row.Features.split(split_char)
        
        # Get ranking from list order
        ranked_feats = [ranked_feature for ranked_feature in enumerate(flist)]
        ranks = [ranked_feat[0]+1 for ranked_feat in ranked_feats]
        feats = [ranked_feat[1] for ranked_feat in ranked_feats]
        
        # Record
        ranked_features = pd.DataFrame({
            'Method':method,
            'Rank':ranks,
            'Feature':feats
        })
        ranked_flist_dfs.append(ranked_features)
    
    ranked_flists_df = pd.concat(ranked_flist_dfs, ignore_index=True).reset_index(drop=True)
    return ranked_flists_df
        
        
def feature_list_latest(feature_selection_df, method=None, max_features=None, 
                          notes_contains=None, return_df = False, split_char='---'):
    """Given a log of feature lists, return the latest feature list.
        Optionally filter by method, max number, note regex search.
        Assumes feature list is contained in column 'Features' and joined by specific
        string character pattern
        Order of filtering goes: method --> max number of features --> notes
    Parameters
    ----------
    feature_selection_df : Pandas DataFrame
        A dataframe log of feature selection lists.
        Must have columns: ['Date', 'Method', 'Number_of_Features', 'Notes', 'Features']
    method : str
        A feature selection method name. Searches df column 'Method' for a match
    max_features : int
        The maximum number of features to have in a feature list
    notes_contains: str
        A string to search the notes for when filtering
    return_df: Boolean
        Flag to return the feature list df rather than latest feature list
        Default is False        
    split_char : str
        The string character pattern the feature list is joined together on.
    Returns
    -------
    latest_flist: list
        A list of the latest feature list matching the filtering criteria
    feature_selection_df: Pandas DataFrame
        A dataframe of feature lists filtered by the given criteria
    """
    if method:
        ## Filter by method
        feature_selection_df = feature_selection_df[feature_selection_df.Method == method]
    if max_features:
        ## Filter by max number of features
        feature_selection_df = feature_selection_df[feature_selection_df.Number_of_Features <= max_features]
    if notes_contains:
        ## Filter by notes containing
        feature_selection_df = feature_selection_df[feature_selection_df.Notes.str.contains(notes_contains)]  
    ## If the dataframe isn't empty
    if feature_selection_df.shape[0] != 0:
        if return_df:
            return feature_selection_df
        else:
            # Return the last appearing feature list in the df and split features by split character
            latest_flist = feature_selection_df.iloc[feature_selection_df.shape[0]-1, :].Features.split(split_char)
            return latest_flist
    else:
        return print('No feature list with specified filters found')  
    
def interaction_terms(flist, unique=True):
    """ Compute a list of the interaction terms seperated by '_*_' if 
        scaled or '_+_' if log-additive
        Optionally specify unique=False interaction terms 
        to keep redundant interaction terms
    """
    # Seperate into base feature, interaction feature, method
    multiplicative_features = []
    additive_features = []
    for feat in flist:
        if len(feat.split('_*_')) >1:
            interaction_term = feat.split('_*_')
            multiplicative_features.append((interaction_term[0], interaction_term[1], 'multiplicative'))
        if len(feat.split('_+_')) >1:
            interaction_term = feat.split('_+_')
            additive_features.append((interaction_term[0][4:-1], interaction_term[1][4:-1], 'log-additive'))        
    interaction_terms = multiplicative_features + additive_features
    ### Remove duplicates of Base|Interacting that are == Interacting|Base
    interaction_df = pd.DataFrame(interaction_terms, columns=['Base_Feature', 'Interaction_Term', 'Method'])
    if unique:
        dupes = interaction_df.T.apply(sorted).T.duplicated()
        interaction_df = interaction_df[~dupes]
    return interaction_df   

def interaction_features_to_dict(df):
    """ From a df of the form base feature | Interaction Term return a nested 
        dictionary of the form {basefeature:[feature1, feature2, feature3]}
    """
    interactions_dict = {}
    for base_feature in df.Base_Feature.unique():
        int_terms = df[df.Base_Feature.isin([base_feature])].Interaction_Term.values.tolist()
        interactions_dict[base_feature] = int_terms
    return interactions_dict
    