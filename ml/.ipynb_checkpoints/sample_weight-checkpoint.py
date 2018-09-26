import pandas as pd
import numpy as np

def inverse_sample_weights(df, target_col, weight_col, 
                    new_col_name=None, min_class_weight = .01,
                    return_df = True):
    """ Given a target class an column to use to derive training weights, 
        create a column of weights where the negative class is the inverse
        of the weights column. 
        
        E.g. weight column of 'Price' would use Price value for positive class
        (where target == 1) and 1/Price for the negative class. 
    """
    df_copy = df.copy()
    pos_class_weights = np.where(df[target_col] == 1 , # Where class is positive
                                       df[weight_col], # Use this val as weight
                                       0) # Else 0
    neg_class_weights_inverse = np.where(df[target_col] == 0 , # Where class is neg
                                       1/df[weight_col], # Use inverse of this
                                       0) # Else 0
    # Handle Edge Case where dividing by 0 results in undefined
    neg_class_weights_inverse = np.where(neg_class_weights_inverse == np.inf , # Where weight is inf (divided by 0)
                                       min_class_weight, # Replace with smallest weighting
                                       neg_class_weights_inverse) # Otherwise keep it
    # Combine weights
    combined_weights_inverse = np.where(pos_class_weights == 0, # Where negative classes 
                                    neg_class_weights_inverse, # Place the inverse as negative weights
                                    pos_class_weights) # Else keep the positive weights
    if not new_col_name:
        new_col_name = 'Sample_Inverse_Weights'
        
    df_copy[new_col_name] = combined_weights_inverse
    
    if return_df:
        return df_copy
    else:
        return pd.Series(combined_weights_inverse, name=new_col_name)   
    
def even_sample_weights(df, target_col, weight_col, 
                    new_col_name=None, 
                    return_df = True):
    """ Given a target class an column to use to derive training weights, 
        create a column of weights where the negative class is the inverse
        of the weights column. 
        
        E.g. weight column of 'Price' would use Price value for positive class
        (where target == 1) and 1/Price for the negative class. 
    """
    df_copy = df.copy()
    pos_class_weights = np.where(df[target_col] == 1 , # Where class is positive
                                       df[weight_col], # Use this val as weight
                                       0) # Else 0
    neg_class_even_weights = np.where(df[target_col] == 0, # Where class is neg
          (df[target_col] == 0).sum()/(df[target_col] == 0).shape[0] , # Create even weighting
                                       0) 
    # Combine weights
    combined_weights = np.where(pos_class_weights == 0, # Where negative classes 
                                    neg_class_even_weights, # Place the inverse as negative weights
                                    pos_class_weights) # Else keep the positive weights
    if not new_col_name:
        new_col_name = 'Sample_Even_Weights'
        
    df_copy[new_col_name] = combined_weights
    
    if return_df:
        return df_copy
    else:
        return pd.Series(combined_weights, name=new_col_name) 