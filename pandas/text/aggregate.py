import pandas as pd
import numpy as np


def concat_text_by_key(df, key=None, text_col=None, join_char='   ', new_col_name=None, return_df=True):
    """Given a set of keys and a text column concatenate all associated texts together into a single field

    Ex:
    concat_text_by_key(df, 
                  key='User_ID',
                  text_col='Notes')

    Will return a dataframe with field 'Combined_Notes' with all associated notes for each User_ID joined together
    by join_char kwarg
    """
    if not key:
        raise 'Must pass a value to "key" to aggregate records by in order to combine text'
    if not text_col:
        raise 'Must pass a field name to "text_col" in order to combine text'
    if not new_col_name:
        new_col_name = 'Combined_' + text_col

    df = df.assign(**{new_col_name:
                      (df.groupby(by=key, as_index=False)
                       [text_col].apply(
                           lambda x:
                          '' if x.dtype == 'float'  # Return blank if a numeric dtype
                          else join_char.join(
                              [str(x) for x in x]
                          )
                      )  # Join together all
                          .reset_index(drop=True)
                      )
                      }
                   )
    if return_df:
        return df
    else:
        return df[new_col_name]
