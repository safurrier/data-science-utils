import pandas as pd
import numpy as np

def text_vectorize_and_cluster(text, df=None, vectorizer=None, clusterer=None, 
                               vector_params=None, clusterer_params=None,
                               outlier_scores=False, one_hot_labels=False, return_df=False,
                               return_type='clusters'):
    """ Given processed text, vectorize and cluster it. Return cluster labels or cluster labels
    along with fitted vectorizer and clusterer.
    
    Parameters
    ----------
    text : object
        Object which contains text that will be passed to the transformer's .fit_transform() method
        As such, text must already be processed and in correct format.
    df : Pandas DataFrame
        Optional dataframe attach clustering results to        
    vectorizer: object
        Class for text vectorization. Must follow sklearn transformer convention and 
        implement .fit_transform() method
        E.g. CountVectorizer from sklearn
    vector_params: dict[str:obj]
        Dictionary to pass to vectorizer as parameters
    clusterer: object
        Class for clustering. Must follow sklearn estimator convention and 
        implement .fit_predict() method for implementing cluster assignment
    clusterer_params: dict[str:obj]
        Dictionary to pass to clusterer as parameters
    outlier_scores: boolean
        Flag to indicate outlier scores computed by clusterer. Accessed 
        from clusterer.outlier_scores_ attribute
    one_hot_labels: boolean
        Flag to indicate if cluster labels should be one hot encoded
        instead of returns as a one dimensional array of ordinal 
        integer labels
    return_df: boolean
        Flag to indicate if results should be returned concatenated
        with the dataframe passed to 'df' kword arg
    return_type: str  in ['clusters', 'all', ]
        String indicating return type. Must be on of ['clusters', 'all', 'df']
        
        clusters: Return the cluster results as a one dimensional array of ordinal 
        integer labels or concatenated to dataframe if return_df=True
        
        all: Return the fitted vectorizer, clusterer and cluster label results

    Returns
    -------
    clusters: pd.Series or pd.DataFrame
        Return the cluster results as a one dimensional array of ordinal 
        integer labels or concatenated to dataframe if return_df=True
    clusters, vectorizer, clusterer: object, object, pd.Series or pd.DataFrame
        Return the fitted vectorizer, clusterer and cluster label results
    """
    # Check vectorizer and clusterer for correct methods
    assert "fit_transform" in dir(vectorizer), "vectorizer has no 'fit_transform' method"
    assert "fit_predict" in dir(clusterer), "clusterer has no 'fit_predict' method"
    if return_df:
        assert isinstance(df, pd.DataFrame), "If specifying 'return_df', data must be passed to argument 'df'"
    # Instantiate vectorizer with params if specified
    if vector_params:
        vectorizer = vectorizer(**vector_params)
    # Else instantiate the vectorizer
    elif vectorizer:
        vectorizer = vectorizer()
    # Fit and trasnform text to vectors
    vectors = vectorizer.fit_transform(text) 

    # Instantiate vectorizer with params if specified
    if clusterer_params:
        clusterer = clusterer(**clusterer_params)
    elif clusterer:
        clusterer = clusterer()        
    # Fit and trasnform vectors to clusters
    cluster_labels = clusterer.fit_predict(vectors)

    if len(set(clusterer.labels_)) <= 1:
        return print('Clusterer could not find any meaningful labels. All data would fall under one cluster')

    # Create DataFrame of Cluster Labels
    results = pd.DataFrame(cluster_labels, columns=['Cluster_Label'])
    
    # Add Outlier Score if specified
    if outlier_scores:
        results['Outlier_Score'] = clusterer.outlier_scores_

    # Add labels as dummy variables
    if one_hot_labels:
        one_hot_cols = pd.get_dummies(results['Cluster_Label'], prefix='Cluster_Label')
        one_hot_col_names = one_hot_cols.columns.values.tolist()
        results = pd.merge(results, one_hot_cols, left_index=True, right_index=True)
    
    # Attach to data if specified
    if return_df:
        results = pd.merge(df, results, left_index=True, right_index=True)
        
    # Return all or just cluster results
    if return_type == 'all':
        return results, vectorizer, clusterer 
    elif return_type == 'clusters':
        return results
