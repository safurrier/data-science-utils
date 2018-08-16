# -*- coding: utf-8 -*-
import pandas as pd

# Useful for finding significant differences beteween the target class and null class
# for various features
def class_groupby_difference(df, target_class, features, agg_func=['mean'],
                                  difference_col_name = 'difference_between_target_class'):
    """ Group by a binary category and compute the difference between features

    Parameters
    ----------
    df : DataFrame
        Dataframe of data
    target_class : str
        The column name of the binary class which the data will be grouped by
    features : list
        A list of the columns of features to compute the differences of
    agg_func : list
        A list of the pandas groupby agg functions to apply to each feature.
        Default is mean
    difference_col_name : str
        The name of to give the column of computed differences in the returned dataframe.
        Default is difference_between_target_classes
    Returns
    -------
    DataFrame / Series
        A dataframe with one column of the feature used and agg function applied and the
        other the computed difference
    """

    # Create a dictionary of column names and function to aggregate to them in the groupby
    feature_agg_dict = {feat:agg_func for feat in features}

    # Groupby the binary class and apply function to selected features
    groupby_df = (df.groupby(by=[target_class])
              .agg(feature_agg_dict)
             )

    # Unstack columns so it's clear how features were aggregated
    unstacked_column_names = ["-".join(multi_index_column) for multi_index_column in groupby_df.columns]

    # Difference dictionary
    # Each key is a column name and each value is the difference between the feature value
    # for the aggregated feature when grouped by binary target class
    difference_dict = dict(zip(unstacked_column_names, groupby_df.diff().values[1]))

    return pd.DataFrame.from_dict({'feature':unstacked_column_names, difference_col_name: groupby_df.diff().values[1]})

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
