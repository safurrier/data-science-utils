import numpy as np
import pandas as pd

def column_values_target_average(df, feature, target,
                                      sample_frequency=True,
                                      freq_weighted_average=True,
                                      min_mean_target_threshold = 0, 
                                      min_sample_size = 0,
                                      min_sample_frequency = 0,
                                      min_weighted_target_threshold=0):
    """ Group by a feature and computing the average target value and sample size
    Returns a dictionary Pandas DataFrame fitting that criteria

    Parameters
    ----------
    df : Pandas DataFrame
        The dataframe where data resides
    feature : str
        Column name for which to groupby and check for average target value
    target : str
        Column name of the target to find grouped by average of
    sample_frequency: Boolean
        Flag to include sample frequency for a given feature value.
        Default is true
    freq_weighted_average: Boolean
        Flag to include the frequency weighted average for a given feature value.
        Default is true        
    min_mean_target_threshold : float
        The minimum value of the average target class to use as cutoff.
        E.g. .5 would only return values whose associate with the target is 
        above an average of .5
    min_sample_size: int
        The minimum value of the number of samples for a feature value
        E.g. 5 would only feature values with at least 5 observations in the data
    min_weighted_target_threshold : float
        The minimum value of the frequency weighted average target class to use as cutoff.
        E.g. .5 would only return values whose associate with the frequency weighted target 
        average is above an average of .5
    min_sample_frequency: float
        The minimum value of the frequency of samples for a feature value
        E.g. .5 would only include feature values with at least 50% of the values in the column         

    Returns
    -------
    grouped_mean_target_df
        DataFrame of the feature values and their asssociations
    """
    grouped_mean_target_df = (df.groupby(by=feature)
     .agg({target:['size', 'mean']})
     .loc[:, target]
     .reset_index()
     .sort_values(by='mean', ascending=False)
     .rename(columns={'mean':'avg_target', 'size':'sample_size'})
    )
    # Sum the sample sizes to get total number of samples
    total_samples = grouped_mean_target_df['sample_size'].sum()
    
    # Flags for adding sample frequency and frequency weighted average
    if sample_frequency:
        # Compute frequency
        grouped_mean_target_df['feature_value_frequency'] = grouped_mean_target_df['sample_size'] / total_samples
        # Filter out minimums
        grouped_mean_target_df = grouped_mean_target_df[grouped_mean_target_df['feature_value_frequency'] >= min_sample_frequency]
        
    if freq_weighted_average:
        # Sample frequency must be calculated for frequency weighted average
        grouped_mean_target_df['feature_value_frequency']  = grouped_mean_target_df['sample_size'] / total_samples 
        grouped_mean_target_df['freq_weighted_avg_target'] = grouped_mean_target_df['feature_value_frequency']  * grouped_mean_target_df['avg_target']
        grouped_mean_target_df = grouped_mean_target_df[(grouped_mean_target_df['feature_value_frequency'] >= min_sample_frequency)
                                                       & (grouped_mean_target_df['freq_weighted_avg_target'] >= min_weighted_target_threshold)
                                                       ]
        
        # If sample frequency not included, drop the column
        if not sample_frequency:
            grouped_mean_target_df.drop(labels=['feature_value_frequency'], axis=1, inplace=True)
    
    # Filter out minimum metrics
    grouped_mean_target_df = grouped_mean_target_df[
        (grouped_mean_target_df['avg_target'] >= min_mean_target_threshold) 
        & (grouped_mean_target_df['sample_size'] >= min_sample_size)]
    

    
    
    return grouped_mean_target_df



def df_feature_values_target_average(df,                                             
                                     target,
                                     include=None,
                                     exclude=None,
                                     min_mean_target_threshold = 0, 
                                     min_sample_size = 0,
                                     min_sample_frequency = 0,
                                     min_weighted_target_threshold=0,
                                     ignore_binary=True):

    
    """ For a given dataframe and a target column, groupby each column and compute 
    for each column value the the average target value, feature value sample size,
    feature value frequency, and frequency weighted average target value

    Parameters
    ----------
    df : Pandas DataFrame
        The dataframe where data resides
    target : str
        Column name of the target to find grouped by average of
    sample_frequency: Boolean
        Flag to include sample frequency for a given feature value.
        Default is true
    include: list
        A list of columns to include when computing
    exclude: list
        A list of columns to exclude when computing        
    freq_weighted_average: Boolean
        Flag to include the frequency weighted average for a given feature value.
        Default is true        
    min_mean_target_threshold : float
        The minimum value of the average target class to use as cutoff.
        E.g. .5 would only return values whose associate with the target is 
        above an average of .5
    min_sample_size: int
        The minimum value of the number of samples for a feature value
        E.g. 5 would only feature values with at least 5 observations in the data
    min_weighted_target_threshold : float
        The minimum value of the frequency weighted average target class to use as cutoff.
        E.g. .5 would only return values whose associate with the frequency weighted target 
        average is above an average of .5
    min_sample_frequency: float
        The minimum value of the frequency of samples for a feature value
        E.g. .5 would only include feature values with at least 50% of the values in the column         

    Returns
    -------
    feature_values_target_average_df
        DataFrame of the feature values and their asssociations
    """
    
    # Start with all columns and filter out/include desired columns
    columns_to_check = df.columns.values.tolist()
    if include:
        columns_to_check = [col for col in columns_to_check if col in include]
    if exclude:
        columns_to_check = [col for col in columns_to_check if col not in exclude]
        
    # Compute for all specified columns in dataframe
    dataframe_lists = [column_values_target_average(df, column, target,  
                                      min_mean_target_threshold = min_mean_target_threshold, 
                                      min_sample_size = min_sample_size,
                                      min_sample_frequency = min_sample_frequency,
                                      min_weighted_target_threshold = min_weighted_target_threshold)
                     .rename(columns={column:'feature_value'}).assign(feature = column)
            for column in columns_to_check if column != target] 
    
    feature_values_target_average_df = pd.concat(dataframe_lists)
    
    return feature_values_target_average_df

def feature_vals_target_association_dict(df, 
                                         feature,                                             
                                         target,
                                            include=None,
                                            exclude=None,
                                            min_mean_target_threshold = 0, 
                                            min_sample_size = 0,
                                            min_sample_frequency = 0,
                                            min_weighted_target_threshold=0,
                                           ignore_binary=True):

    """Return a dictionary of the form column_name:[list of values] for values in a
       feature that have an above certain threshold for feature value mean target value,
       feature value sample size, feature value sample frequency and feature value frequency
       weighted mean target value
       
    """
    if ignore_binary:
        # Check to see if only values are 1 and 0. If so, don't compute rest
        if df[feature].dropna().value_counts().index.isin([0,1]).all():
            return {feature: []}
        
    grouped_mean_target = column_values_target_average(df, feature, target,  
                                      min_mean_target_threshold = min_mean_target_threshold, 
                                      min_sample_size = min_sample_size,
                                      min_sample_frequency = min_sample_frequency,
                                      min_weighted_target_threshold = min_weighted_target_threshold)
    
    return {feature: grouped_mean_target[feature].values.tolist()}
    

def df_feature_vals_target_association_dict(df, 
                                            target,
                                            include=None,
                                            exclude=None,
                                            min_mean_target_threshold = 0, 
                                            min_sample_size = 0,
                                            min_sample_frequency = 0,
                                            min_weighted_target_threshold=0,
                                           ignore_binary=True):

    
    columns_to_check = df.columns.values.tolist()
    if include:
        columns_to_check = [col for col in columns_to_check if col in include]
    if exclude:
        columns_to_check = [col for col in columns_to_check if col not in exclude]
        
    # Compute for all specified columns in dataframe
    list_of_dicts = [feature_vals_target_association_dict(df, column, target,  
                                       min_mean_target_threshold = min_mean_target_threshold, 
                                      min_sample_size = min_sample_size,
                                      min_sample_frequency = min_sample_frequency,
                                      min_weighted_target_threshold = min_weighted_target_threshold,
                                                         ignore_binary=ignore_binary)
            for column in columns_to_check if column != target]

    # Combine into single dictionary if there are any values
    # that fit the minimum thresholds
    combined_dict = {}
    for dictionary in list_of_dicts:
        # Check it see if any values in list
        feat_vals = list(dictionary.values())
        if len(feat_vals[0]) >=1:
            combined_dict.update(dictionary)
    return combined_dict