import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import mutual_info_regression
from sklearn.exceptions import DataConversionWarning
import warnings

def tidy_correlation(df, columns=None, keep_identity_correlations=False, abs_val_correlations=False):
    """Given a dataframe and optionally subset of columns, compute correlations 
    between all features. Enable keep_identity_correlations for including the correlation
    between a feature and itself. Enable abs_val_correlations to add a column for absolute value of 
    correlation value.
    """
    df_copy = df.copy()
    if columns:
        df_copy = df_copy[columns]
    
    df_copy = (df_copy.corr()
               .reset_index()
               .melt(id_vars='index', var_name="Paired_Feature", value_name="Correlation")
               .rename(columns={'index':'Base_Feature'})
               .sort_values(by='Correlation', ascending=False)
              )
    if abs_val_correlations:
        df_copy['Correlation_Absolute_Value'] = df_copy['Correlation'].abs()
        df_copy = df_copy.sort_values(by='Correlation_Absolute_Value', ascending=False)
    if keep_identity_correlations:
        return df_copy
    else:
        return df_copy.query('Base_Feature != Paired_Feature')

    
## There are two functions for variable_mutual_information    
## One of them works when subsetting to columns and one of them doesn't
## Need to see which is which 
def variable_mutual_information(df, columns=None,
                           auto_select_and_process_relevant_cols = False,
                           max_cardinality_ratio_for_numeric_cols = .05):
    """
    For a given dataframe's (subset) of columns, compute the mutual information between
    pairs of variables.

    Warning: Computationally expensive. Consider performing on sample of data

    Note: Dependency on function categoric_or_continuous_columns for determining whether to call
    sklearn's mutual_information_classifier or mutual_information_regression

    Parameters
    ----------
    df : Pandas DataFrame
        A pandas dataframe to check mutual information from
    columns : list
        A list of the columns to compute mutual information between
    auto_select_and_process_relevant_cols : Boolean
        Flag to select only numberic columns and fill NaNs as 0
    max_cardinality_ratio_for_numeric_cols: float
        The maximum ratio of unique values in a column / total observations. Akin to a cardinality ratio.
        Default is .05, or that anyting with more than 5% of its values being unique will be considered
        numeric.

    Returns
    -------
    mutual_information_df: Pandas DataFrame
        A dataframe in tidy format of the source column, paired column and mutual information score

    """
    warnings.filterwarnings(action='ignore', category=DataConversionWarning)


    # If not specified columns use all 
    if not columns:
        columns = df.columns.values.tolist()

    # Subset to relevant columns
    X = df[columns]
    
    # If specified, select numeric and fill NaNs    
    if auto_select_and_process_relevant_cols:
        # Autoselect takes numeric columns only and fills NaNs with 0
        X = X.select_dtypes(include=np.number).fillna(0)    


    categoric_cols = categoric_or_continuous_columns(X,
                                                     unique_value_to_total_value_ratio_threshold=max_cardinality_ratio_for_numeric_cols)
    continuous_cols = categoric_or_continuous_columns(X, return_continuous=True,
                                                      unique_value_to_total_value_ratio_threshold=max_cardinality_ratio_for_numeric_cols)
    cols = X.columns.values.tolist()

    mutual_info_features = []
    # For each categoric column
    for column in categoric_cols:
        # Try Categoric, if it doesn't work it's likey due to continuous value. Add to continouous cols
        try:
            # Compute mutual information
            mi = mutual_info_classif(X, X[column])
            # Create Dataframe of the feature, paired feature and mutual information
            mmi_df = pd.DataFrame({'feature':column, 'paired_feature':cols, 'mutual_information': mi})
            # Add df to list
            mutual_info_features.append(mmi_df)
        except ValueError:
            continuous_cols.append(column)


    for column in continuous_cols:
        try:
            # Compute mutual information
            mi = mutual_info_regression(X, X[column])
            # Create Dataframe of the feature, paired feature and mutual information
            mmi_df = pd.DataFrame({'feature':column, 'paired_feature':cols, 'mutual_information': mi})
            # Add df to list
            mutual_info_features.append(mmi_df)
        except ValueError:
            print('Unable to compute mutual information for column {}'.format(column))

    # Concat all mutual info into one df
    mutual_information_df = pd.concat(mutual_info_features).sort_values(by='mutual_information', ascending=False)
    # Remove instances where a feature was compared with itself
    mutual_information_df = mutual_information_df[mutual_information_df['feature'] != mutual_information_df['paired_feature']]
    #display(mutual_information_df)
    return mutual_information_df



## There are two functions for variable_mutual_information    
## One of them works when subsetting to columns and one of them doesn't
## Need to see which is which 
def variable_mutual_information(df, columns=None, 
                           auto_select_and_process_relevant_cols = False, 
                           max_cardinality_ratio_for_numeric_cols = .05):
    """ 
    For a given dataframe's (subset) of columns, compute the mutual information between
    pairs of variables. 
    
    Warning: Computationally expensive. Consider performing on sample of data
    
    Note: Dependency on function categoric_or_continuous_columns for determining whether to call
    sklearn's mutual_information_classifier or mutual_information_regression

    Parameters
    ----------
    df : Pandas DataFrame
        A pandas dataframe to check mutual information from 
    columns : list
        A list of the columns to compute mutual information between
    auto_select_and_process_relevant_cols : Boolean
        Flag to select only numberic columns and fill NaNs as 0
    max_cardinality_ratio_for_numeric_cols: float 
        The maximum ratio of unique values in a column / total observations. Akin to a cardinality ratio.
        Default is .05, or that anyting with more than 5% of its values being unique will be considered 
        numeric.

    Returns
    -------
    mutual_information_df: Pandas DataFrame
        A dataframe in tidy format of the source column, paired column and mutual information score
        
    """
    from sklearn.feature_selection import mutual_info_classif
    from sklearn.feature_selection import mutual_info_regression
    # Subset to relevant columns
    if columns:
        df = df[columns]
    # Determine which columns to compute mutual information for
    if auto_select_and_process_relevant_cols:
        # Autoselect takes numeric columns only and fills NaNs with 0
        X = df.select_dtypes(include=np.number).fillna(0)
    # Otherwise use provided data
    else:
        X = df
        
        
    categoric_cols = categoric_or_continuous_columns(X, 
                                                     unique_value_to_total_value_ratio_threshold=max_cardinality_ratio_for_numeric_cols)
    continuous_cols = categoric_or_continuous_columns(X, return_continuous=True, 
                                                      unique_value_to_total_value_ratio_threshold=max_cardinality_ratio_for_numeric_cols)
    cols = X.columns.values.tolist()

    mutual_info_features = []
    # For each categoric column
    for column in categoric_cols:
        # Compute mutual information
        mi = mutual_info_classif(X, X[column])
        # Create Dataframe of the feature, paired feature and mutual information
        mmi_df = pd.DataFrame({'feature':column, 'paired_feature':cols, 'mutual_information': mi})
        # Add df to list 
        mutual_info_features.append(mmi_df)
    
    for column in continuous_cols:
        # Compute mutual information
        mi = mutual_info_regression(X, X[column])
        # Create Dataframe of the feature, paired feature and mutual information
        mmi_df = pd.DataFrame({'feature':column, 'paired_feature':cols, 'mutual_information': mi})
        # Add df to list 
        mutual_info_features.append(mmi_df)
        
    # Concat all mutual info into one df
    mutual_information_df = pd.concat(mutual_info_features).sort_values(by='mutual_information', ascending=False)
    # Remove instances where a feature was compared with itself
    mutual_information_df = mutual_information_df[mutual_information_df['feature'] != mutual_information_df['paired_feature']]
    #display(mutual_information_df)
    return mutual_information_df

def target_mutual_information(df, target, target_type='categoric', columns=None, 
                           auto_select_and_process_relevant_cols = False):
    """ 
    For a given dataframe's (subset) of columns, compute the mutual information between
    pairs of variables. 
    
    Warning: Computationally expensive. Consider performing on sample of data
    
    Note: Dependency on function categoric_or_continuous_columns for determining whether to call
    sklearn's mutual_information_classifier or mutual_information_regression

    Parameters
    ----------
    df : Pandas DataFrame
        A pandas dataframe to check mutual information from 
    target: str
        The name of the target column in the DataFrame
    target_type: str
        The type of target. Must be either 'categoric' (default) or 'continuous'
    columns : list
        A list of the columns to compute mutual information between
    auto_select_and_process_relevant_cols : Boolean
        Flag to select only numberic columns and fill NaNs as 0

    Returns
    -------
    mutual_information_df: Pandas DataFrame
        A dataframe in tidy format of the source column, paired column and mutual information score
        
    """

    warnings.filterwarnings(action='ignore', category=DataConversionWarning)

    # If not specified columns use all 
    if not columns:
        columns = df.columns.values.tolist()

    # Subset to relevant columns
    X = df[columns]
    
    # If specified, select numeric and fill NaNs    
    if auto_select_and_process_relevant_cols:
        # Autoselect takes numeric columns only and fills NaNs with 0
        X = X.select_dtypes(include=np.number).fillna(0)    
    if target_type == 'categoric':
        mi = mutual_info_classif(X, df[target])
    elif target_type == 'continuous':
        mi = mutual_info_regression(X, df[target])
    elif target_type not in ['categoric', 'continuous']:
        return print('Specify "categoric" or "continuous" for target_type argument')
    
    
    mmi_df = pd.DataFrame({'feature':target, 'paired_feature':columns, 'mutual_information': mi})
    # Concat all mutual info into one df
    mmi_df = mmi_df.sort_values(by='mutual_information', ascending=False)
    # Remove instances where a feature was compared with itself
    mmi_df = mmi_df[mmi_df['feature'] != mmi_df['paired_feature']]
    #display(mutual_information_df)
    return mmi_df     

def mahalanobis(x=None, data=None, cov=None):
    """Compute the Mahalanobis Distance between each row of x and the data  
    x    : vector or matrix of data with, say, p columns.
    data : ndarray of the distribution from which Mahalanobis distance of each observation of x is to be computed.
    cov  : covariance matrix (p x p) of the distribution. If None, will be computed from data.
    
    Source: https://www.machinelearningplus.com/statistics/mahalanobis-distance/
    
    E.g.
    df_x = df[['carat', 'depth', 'price']].head(500)
    df_x['mahala'] = mahalanobis(x=df_x, data=df[['carat', 'depth', 'price']])
    df_x.head()
    """
    x_minus_mu = x - np.mean(data)
    if not cov:
        cov = np.cov(data.values.T)
    inv_covmat = sp.linalg.inv(cov)
    left_term = np.dot(x_minus_mu, inv_covmat)
    mahal = np.dot(left_term, x_minus_mu.T)
    return mahal.diagonal()

