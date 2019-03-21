import pandas as pd
import numpy as np
# Bivariate Lift DF
# For original data shape (to merge back onto existing value)
# set return_melted=False

## Example Altair Lift Chart using the returned lift_df
# lift_df = bivariate_lift(df, 'column1', , 'column2')    
#
# Altair Lift Chart
#
# alt.Chart(lift_df).mark_line().encode(
#     x='Quantile:Q', 
#     y='Value:Q',
#     color='Field:N'
# ) 
# alt.Chart(lift_df).mark_line().encode(
#     x='Quantile:Q', 
#     y='Value:Q',
#     color='Field:N'
# ) 


def bivariate_lift(df, col1, col2, return_melted=True, round_place=2):
    """Compute bivariate lift of two variables"""
    # Get the step size to compute for quantiles
    if round_place & return_melted:
        quantiles = np.round(np.arange(0, 1, step=10 / df.shape[0]), round_place)
    # Don't round if specified or if returning data in original shape 
    # So that the values can be used to merge back on
    else:
        quantiles = np.arange(0, 1, step=10 / df.shape[0])
    
    # Compute for each column
    col1_quantiles = df[col1].quantile(quantiles).sort_values().to_frame()
    col2_quantiles = df[col2].quantile(quantiles).sort_values().to_frame()
    # Outer merge together on index
    # Outer b/c if quantile/value is 0 the index will be missing
    qq_df = pd.merge(col1_quantiles,
         col2_quantiles,
         how='outer',
         left_index=True,
         right_index=True)
    if not return_melted:
        qq_df = qq_df.reset_index().rename(columns={'index': f'{col1}_Quantile'})
        qq_df[f'{col2}_Quantile'] = qq_df[f'{col1}_Quantile']  
        return qq_df
    # Melt down to tidy data format
    lift_df = (qq_df.reset_index()
               .rename(columns={'index': 'Quantile'}) # Index (Quantile) Rename
               .melt(id_vars='Quantile',
                     value_vars=[col1, col2],
                     var_name='Field',
                     value_name='Value',)
              )
    return lift_df