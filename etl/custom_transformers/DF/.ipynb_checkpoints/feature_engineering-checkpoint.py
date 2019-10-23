import pandas as pd
import numpy as np
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import TransformerMixin, BaseEstimator
from ....ml.feature_engineering.target_mean_aggregate import df_feature_vals_target_association_dict
from ....ml.feature_engineering.one_hot import get_specific_dummies
from ....ml.feature_engineering.one_hot import text_match_one_hot_from_map


class DFLookupTable(TransformerMixin):
    """ Given a feature column and path to lookup table, left join the the data
    and lookup values

    Parameters
    ----------
    feature : list/str
        A list of the column name(s) of the feature to look up. This is what the lookup table will
        be left joined on.
    lookup_key: list
        A list of the column name(s) the lookup table will be joined on. MUST be in same order
        as 'feature' arg. If none specified will default to the 'feature' argument.
    table_lookup_keep_cols: list
        A list of the columns which should be kept from the joined tables. Default is
        to keep all columns from table
    table_lookup_exclude_cols: list
        A list of the columns which should removed from the joined tables. 
        E.g. if joined table should include all columns from the lookup eable
        except column 'col45', pass no kwarg to table_lookup_keep_cols and
        ['col45'] to table_lookup_exclude_cols
    table_path : str
        The path to the table to use as a lookup
    table_format : str
        The type of file the lookup table is. Currently only accepts
        'csv' (default), 'pickle', 'hdf'
    table_key : str
        The key to pass for reading in the table. Currently used only when table format
        is 'hdf'
    merge_type: str 
        The type of merge to use. Default is 'left' merge to base data on 'feature' arg
        to preserve all rows in the original data even if the key is missing.
        For options check pandas merge method
    merge_as_string: Boolean
        Force the key columns to be cast to string before joining
    merge_dtype: int, str, float, etc
        The dtype to explicitly cast the join keys to before merging
    add_prefix: str
        A prefix to add onto the new columns added from the lookup table
    add_suffix: str
        A suffix to add onto the new columns added from the lookup table

    """
    import pandas as pd

    def __init__(self, feature=None, lookup_key=None,
                 table_lookup_keep_cols=None, 
                 table_lookup_exclude_cols=None,
                 table_path=None,
                 table_format='csv',
                 table_key=None,
                 merge_type='left',
                 merge_as_string=False, 
                 merge_dtype=None,
                add_prefix=None, add_suffix=None):
        # If argument is a single string, put it in a 
        # list with one value
        if isinstance(feature, str):
            feature = [feature]
        self.feature = feature
        self.table_path = table_path
        self.table_key = table_key
        self.table_format = table_format
        self.merge_as_string = merge_as_string
        self.merge_dtype = merge_dtype
        self.merge_type = merge_type        
        self.add_prefix = add_prefix
        self.add_suffix = add_suffix        
        # Determine which column to left join the lookup table on
        if not lookup_key:
            # If none specified use 'feature'
            self.lookup_key = self.feature
        else:
            # If argument is a single string, put it in a 
            # list with one value
            if isinstance(lookup_key, str):
                lookup_key = [lookup_key]
            self.lookup_key = lookup_key
            
        # Load Lookup Table
        if self.table_format == 'csv':
            self.lookup_table = pd.read_csv(self.table_path)
        elif self.table_format == 'pickle':
            self.lookup_table =  pd.read_pickle(self.table_path) 
        elif self.table_format == 'hdf':
            self.lookup_table =  pd.read_hdf(self.table_path, key=self.table_key)             
            
        # If no keep columns specified, keep all columns from lookup table
        if table_lookup_keep_cols:
            # If argument is a single string, put it in a 
            # list with one value
            if isinstance(table_lookup_keep_cols, str):
                table_lookup_keep_cols = [table_lookup_keep_cols]
        if not table_lookup_keep_cols:
            table_lookup_keep_cols = self.lookup_table.columns.values.tolist()
        # If specified remove specific columns
        if table_lookup_exclude_cols:
            # If argument is a single string, put it in a 
            # list with one value
            if isinstance(table_lookup_exclude_cols, str):
                table_lookup_exclude_cols = [table_lookup_exclude_cols]
            table_lookup_keep_cols = [col for col in table_lookup_keep_cols if col not in table_lookup_exclude_cols]
        self.table_lookup_keep_cols = table_lookup_keep_cols
            
        # Remove key columns from keep columns if present
        # Avoids duplicate columns later on when joining
        for feat in self.feature:
            if feat in self.table_lookup_keep_cols:
                self.table_lookup_keep_cols = [name for name in self.table_lookup_keep_cols
                                          if name != feat]


    def fit(self, X, y=None):
        # If transformer has already been fit
        # and has fitted_data attribute
        if hasattr(self, 'fitted_data'):
            # Reload the lookup table because column
            # names may have been changd
            if self.table_format == 'csv':
                self.lookup_table = pd.read_csv(self.table_path)
            elif self.table_format == 'pickle':
                self.lookup_table =  pd.read_pickle(self.table_path)
            elif self.table_format == 'hdf':
                self.lookup_table =  pd.read_hdf(self.table_path, key=self.table_key)             


        # Cast dtypes to string if specified
        if self.merge_as_string:
            X[self.feature] = X[self.feature].astype(str)
            self.lookup_table[self.lookup_key] = \
            self.lookup_table[self.lookup_key].astype(str)

        if self.merge_dtype:
            X[self.feature] = X[self.feature].astype(self.merge_dtype)
            self.lookup_table[self.lookup_key] = \
            self.lookup_table[self.lookup_key].astype(self.merge_dtype)            

        # Determine which columns to keep from lookup table
        # If none specified use all the columns in the lookup table
        if not self.table_lookup_keep_cols:
            all_cols = self.lookup_table.columns.values.tolist()
            all_cols = [col for col in all_cols if col not in self.feature]
            self.table_lookup_keep_cols = all_cols

        # Reduce to only desired columns before merge
        # Rename lookup_key to the same as self.feature
        keep_cols = self.table_lookup_keep_cols + self.lookup_key
        self.lookup_table = self.lookup_table[keep_cols]

        # Renaming dict
        # Creating a renaming dict to rename the lookup
        # table keys to the 'feature' keys
        renaming_dict = dict(zip(self.lookup_key, self.feature))
        self.lookup_table = self.lookup_table.rename(columns=renaming_dict)


        if self.add_prefix:
            # Don't add oprefix to ORIGINAL table lookup key columns
            keep_cols = [col for col in keep_cols if col not in self.lookup_key]
            # Concat the renamed columns (with prefix added) and the key column
            self.lookup_table = pd.concat([self.lookup_table[keep_cols].add_prefix(self.add_prefix),
                                           self.lookup_table[self.feature]], axis=1)
            # Update keep_cols in case adding a suffix also
            keep_cols = self.lookup_table.columns.values.tolist()
            # Remove the key column which is now the SAME as SELF.FEATURE
            keep_cols = [col for col in keep_cols if col not in self.feature]


        if self.add_suffix:
            # Don't add suffix to key columns
            # Remove the key column which is now the SAME as SELF.FEATURE
            keep_cols = [col for col in keep_cols if col not in self.feature]
            # If a prefix has already been added, remove the updated key column
            # that will have the prefix on it.

            # Concat the renamed columns (with suffix added) and the key column
            self.lookup_table = pd.concat([self.lookup_table[keep_cols].add_suffix(self.add_suffix),
                                           self.lookup_table[self.feature]], axis=1)
        
        # Merge join the two
        new_df = pd.merge(X,
                self.lookup_table,
                on=self.feature,
                          how=self.merge_type)

        # Assign to attribute
        self.fitted_data = new_df

        return self

    def transform(self, X, y=None):
        if hasattr(self, 'fitted_data'):
            return self.fitted_data
        else:
            print('Transformer has not been fit yet')

    def fit_transform(self, X, y=None):
        if hasattr(self, 'fitted_data'):
            # Reload the lookup table because column
            # names may have been changd
            if self.table_format == 'csv':
                self.lookup_table = pd.read_csv(self.table_path)
            elif self.table_format == 'pickle':
                self.lookup_table =  pd.read_pickle(self.table_path)
            elif self.table_format == 'hdf':
                self.lookup_table =  pd.read_hdf(self.table_path, key=self.table_key)             

        return self.fit(X, y=y).fitted_data

class TargetAssociatedFeatureValueAggregator(TransformerMixin):
    """ Given a dataframe, a set of columns and associative target thresholds,
        mine feature values associated with the target class that meet said thresholds.
        Aggregate those features values together and create a one hot encode feature when
        a given observation matches any of the mined features' values.

        Currently only works for binary classifier problems where the positive class is
        1 and negative is 0

        Example:

    """

    def __init__(self, include=None, exclude=None, prefix=None, suffix=None,
                 aggregation_method='aggregate',
                 min_mean_target_threshold=0, min_sample_size=0,
                 min_sample_frequency=0, min_weighted_target_threshold=0,
                 ignore_binary=True, verbose=True):

        self.aggregation_method = aggregation_method
        self.include = include
        self.exclude = exclude
        self.prefix = prefix
        self.suffix = suffix
        self.min_mean_target_threshold = min_mean_target_threshold
        self.min_sample_size = min_sample_size
        self.min_sample_frequency = min_sample_frequency
        self.min_weighted_target_threshold = min_weighted_target_threshold
        self.ignore_binary = ignore_binary
        """
        Parameters
        ----------
        aggregation_method: str --> Default = 'aggregate'
            Option flag.
            'aggregate'
                Aggregate feature values from specific a specific feature.
                Aggregate those features values together and create a one hot encode
                feature when a given observation matches any of the mined features' values.
            'one_hot'
                Create one hot encoded variable for every feature value that meets the
                specified thresholds.
                Warning: Can greatly increase dimensions of data
        include: list
            A list of columns to include when computing
        exclude: list
            A list of columns to exclude when computing
        prefix: str
            A string prefix to add to the created columns
        suffix: str
            A string suffix to add to the created columns
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
        ignore_binary: boolean
            Flag to ignore include feature values in columns with binary values [0 or 1] as this is
            redundant to aggregate.
            Default is True
        verbose: boolean, default is True
            Verbose output to print status of transformer
        """

    def fit(self, X, y, verbose=1):
        """
        Parameters
        ----------
        X: Pandas DataFrame
            A dat
        y: str/Pandas Series of Target
            The STRING column name of the target in dataframe X
        """
        self.X = X

        # Check if y is a string (column name) or Pandas Series (target values)
        if isinstance(y, str):
            self.y = y
        if isinstance(y, pd.Series):
            self.y = y.name
        self.one_hot_dict = df_feature_vals_target_association_dict(X, y,
                                include=self.include, exclude=self.exclude,
                                min_mean_target_threshold=self.min_mean_target_threshold,
                                min_sample_size=self.min_sample_size,
                                min_sample_frequency=self.min_sample_frequency,
                                min_weighted_target_threshold=self.min_weighted_target_threshold,
                                ignore_binary=self.ignore_binary
                                )
        # Check to make sure the one hot dict isn't empty
        if verbose:
            if not self.one_hot_dict:
                print('WARNING:one_hot_dict attribute empty is empty')
        return self

    def transform(self, X, y=None):
        if not hasattr(self, 'one_hot_dict'):
            return f'{self} has not been fitted yet. Please fit before transforming'
        if self.aggregation_method == 'one_hot':
            return get_specific_dummies(col_map = self.one_hot_dict,
                                        prefix=self.prefix, suffix=self.suffix)
        else:
            assert self.aggregation_method == 'aggregate'
            return get_text_specific_dummies(X, col_map = self.one_hot_dict,
                                             prefix=self.prefix, suffix=self.suffix)


class DFDummyMapTransformer(TransformerMixin):
    """
    From a dictionary mapping of {column:[list of feature values]}, create dummy columns
    for each pair of column:feature value. Return binary column columns_feature_value where
    1 indicates presence of feature value and 0 its absence
    """

    def __init__(self, dummy_map=None, remove_original=True, join_char='_', verbose=1):
        """
        Parameters
        ----------
        dummy_map: dict
            Nested dictionary of the form {basefeature:[feature1, feature2, feature3]}
        remove_original: bool
            Boolean flag to remove the columns that will be one hot encoded.
            Default is True. False will one hot encode but keep the original columns as well
        join_char: str, default '_'
            The string pattern to join the column name and feature value on to describe the 
            dummy encoded column. E.g. column Fruit with feature value Apple would be 'Fruit_Apple'
        verbose: int
            Verbosity of output. Default is 1, which prints out base features and interacting
            terms in interactions_dict_map that are not present in the data. Set to None to
            ignore this.
        """
        self.dummy_map = dummy_map
        self.verbose = verbose
        self.remove_original = remove_original
        self.join_char = join_char

    def fit(self, X, y=None):
        ## Check which base features are present in the data
        base_feats = [key for key in self.dummy_map.keys()]
        present_base_feats = [key for key in self.dummy_map.keys() if key in X.columns.values.tolist()]
        non_present_base_feats = list(set(base_feats).difference(present_base_feats))

        if self.verbose == 1:
            if non_present_base_feats:
                print(f'Warning:\nThe following base features are not present in the data: {non_present_base_feats}')
        ## Set attributes
        self.present_base_feats = present_base_feats
        self.non_present_base_feats = non_present_base_feats
        
        # Set private attributes for float and int columns 
        # To later check for potential matching errors
        self._float_cols = X.select_dtypes(include=[np.float]).columns.values.tolist()
        self._int_cols = X.select_dtypes(include=['int']).columns.values.tolist()

        return self

    def transform(self, X):
        try:
            self.one_hot_encoded_cols = [] 
            # Iterate through the dict's keys (df columns) and values (column feature values to match)
            for base_feature, feature_values in self.dummy_map.items():
                # Flag for printing one hot encoded values
                print_value_counts = False
                if base_feature in self.present_base_feats:
                   # Iterate through feature values, and turn it into a set 
                    # So there are no duplicates
                    for feature_value in set(feature_values):
                        # Check for data type mismatch
                        if self.verbose == 1:
                            if (isinstance(feature_value, int)) & (base_feature in self._float_cols):
                                print(f"""Warning: Feature Value "{feature_value}" is an integer in float column "{base_feature}". 
Consider checking if it matched correctly""")
                                print_value_counts = True
                            if (isinstance(feature_value, float)) & (base_feature in self._int_cols):
                                print(f"""Warning: Feature Value "{feature_value}" is a float in integer column "{base_feature}". 
Consider checking if it matched correctly""")     
                                print_value_counts = True
                        
                        # Cast to string if not already a string
                        if not isinstance(feature_value, str):
                            feature_value = str(feature_value)
                        # Create name based on self.join_charjoin_char
                        one_hot_feature_name = self.join_char.join([base_feature, feature_value])
                        # Match using np.where
                        one_hot_col = pd.Series(np.where(X[base_feature].astype(str) == feature_value, 1, 0),
                                               name=one_hot_feature_name)
                        # Warning
                        if self.verbose == 1:
                            # If no positive matches
                            if one_hot_col.sum() == 0:
                                print(f'Warning: Dummy Column {one_hot_feature_name} returning all 0s') 
                            # If data type mismatch, print value counts of one hot encoded col
                            if print_value_counts:
                                print('This is the one_hot_encoded column value counts:')
                                print(one_hot_col.value_counts())    
                        self.one_hot_encoded_cols.append(one_hot_col)
            # Concat all one hot encoded columns together            
            self.one_hot_encoded_cols = pd.concat(self.one_hot_encoded_cols, axis=1)

            if self.remove_original:
                # Remove the encoded columns from original
                keep_cols = list(set(X.columns.values.tolist()).difference(self.present_base_feats))
                # Put in original ordering
                ordered_keep_cols = [column for column in X.columns.values.tolist() if column in keep_cols]
                X_transform = X[ordered_keep_cols]
            else:
                # Else keep all columns
                X_transform = X
            # Merge encoded cols back onto data
            X_transform = pd.merge(X_transform, self.one_hot_encoded_cols, left_index=True, right_index=True)
            return X_transform
        except AttributeError:
            print('Must use .fit() method before transforming')
            
            
class DFInteractionsTransformer(BaseEstimator, TransformerMixin):
    """ Given a nested dictionary of the form {basefeature:[feature1, feature2, feature3]} and
        a method (either 'scale' or 'log-additive'), compute the interactions between the base
        feature and interacting terms. Add on to existing dataframe
    """
    def __init__(self, interactions_dict_map, method='scale', fillna_val=None, verbose=1):
        """
        Parameters
        ----------
        interactions_dict_map: dict
            Nested dictionary of the form {basefeature:[feature1, feature2, feature3]}
        method: str --> 'scale' or 'log-additive'
            A list of columns to include when computing
        fillna_val: float
            Optional fill value for NaN results in interaction terms
        verbose: int
            Verbosity of output. Default is 1, which prints out base features and interacting
            terms in interactions_dict_map that are not present in the data
        """
        self.interactions_dict_map = interactions_dict_map
        self.method = method
        self.fillna_val = fillna_val
        self.verbose = verbose

    def fit(self, X, y=None):
        ## Check which base features are present in the data
        base_feats = [key for key in self.interactions_dict_map.keys()]
        present_base_feats = [key for key in self.interactions_dict_map.keys() if key in X.columns.values.tolist()]
        non_present_base_feats = list(set(base_feats).difference(present_base_feats))

        interacting_feats = list(set([column for value in self.interactions_dict_map.values() for column in value]))
        present_interacting_feats = [feat for feat in interacting_feats if feat in X.columns.values.tolist()]
        non_present_interacting_feats = list(set(interacting_feats).difference(present_interacting_feats))
        if self.verbose == 1:
            if non_present_base_feats:
                print(f'Warning:\nThe following base features are not present in the data: {non_present_base_feats}')
            if non_present_interacting_feats:
                print(f'Warning:\nThe following interacting features are not present in the data: {non_present_interacting_feats}')

        ## Set attributes
        self.present_base_feats = present_base_feats
        self.non_present_base_feats = non_present_base_feats
        self.present_interacting_feats = present_interacting_feats
        self.non_present_interacting_feats = non_present_interacting_feats
        self._alter_index = False
        # Check to make sure all terms are numeric
        is_number = np.vectorize(lambda x: np.issubdtype(x, np.number))
        if not all(is_number(X[self.present_base_feats+self.present_interacting_feats].dtypes)):
            self.non_numeric_cols = [col for col in self.present_base_feats+self.present_interacting_feats if not \
                                is_number(X[col].dtype)]
            raise ValueError(f"""Base Features and Interaction Terms must be numeric.
            The following columns are non-numeric: {self.non_numeric_cols}""")

        return self

    def transform(self, X, y=None):
        ### Check to see if index is sorted and in order
        if any(X.reset_index().index.values - X.index.values) != 0:
            # If it's not in order, reset it but keep the old index
            # to return to later
            X = X.reset_index().rename(columns={'index':'original_index'})
            self._alter_index = True
        # Container for computed interaction terms
        interaction_terms_df = []
        if self.method == 'scale':
            for base_feature, interaction_terms in self.interactions_dict_map.items():
                ## Check to make sure base feature and interaction terms present in df
                if base_feature in self.present_base_feats:
                    tmp_interaction_terms = [feat for feat in interaction_terms if feat in self.present_interacting_feats]
                    # Pull vector to scale by
                    scaling_vector = X[base_feature].values.reshape(X.shape[0], 1)
                    # Reshape and multiply
                    try:
                        interaction_terms_scaled =   scaling_vector * X[tmp_interaction_terms].values
                    except TypeError:
                        display(X[tmp_interaction_terms].dtypes)
                    # Return Column names
                    col_names = [base_feature+"_TIMES_"+ int_term for int_term in tmp_interaction_terms]
                    # Turn into DataFrame
                    interaction_terms_scaled_df = pd.DataFrame(interaction_terms_scaled, columns=col_names)
                    # Add to collection
                    interaction_terms_df.append(interaction_terms_scaled_df)
        if self.method == 'log-additive':
            X_copy = X.copy()
            # Check to make sure column is positive
            # If not, add minimum value as constant and +1 to get to positive domain
            for col in self.present_interacting_feats+self.present_base_feats:
                while X_copy[col].min() <= 0:
                    # If minimum is negative value, reverse sign and add 1 then add to column
                    if np.sign(X_copy[col].min()) == -1:
                         X_copy[col] +=  -1*X_copy[col].min() + 1
                    # Else add by the minimum value +1 to get to all positive values
                    else:
                        X_copy[col] +=  X_copy[col].min() + 1

            for base_feature, interaction_terms in self.interactions_dict_map.items():
                if base_feature in self.present_base_feats:
                    tmp_interaction_terms = [feat for feat in interaction_terms if feat in self.present_interacting_feats]
                    # Pull vector to log add by
                    base_vector = np.log(X_copy[base_feature].values)
                    # Take the log of the rest of the data
                    log_interaction_features = np.log(X_copy[tmp_interaction_terms].values)
                    # Add together
                    interaction_terms_log_added = base_vector.reshape(base_vector.shape[0], 1) \
                    + log_interaction_features
                    # Return Column names
                    col_names = ['LOG_'+base_feature+"_PLUS_LOG_"+int_term for int_term in tmp_interaction_terms]
                    # Turn into DataFrame
                    interaction_terms_log_added_df = pd.DataFrame(interaction_terms_log_added, columns=col_names)
                    # Add to collection
                    interaction_terms_df.append(interaction_terms_log_added_df)

        # Concat together
        interaction_terms_df = pd.concat(interaction_terms_df, axis=1)

        # Fill NaNs if specified
        if self.fillna_val:
            interaction_terms_df = interaction_terms_df.fillna(self.fillna_val)
        # (or edge case where fill val is 0 which normally evaluates to False)
        if self.fillna_val == 0:
            interaction_terms_df = interaction_terms_df.fillna(self.fillna_val)

        # Add to self attributes
        self.interaction_terms_df = interaction_terms_df

        # Merge data back together
        try:
            # Merge on index
            X_transformed = pd.merge(X, self.interaction_terms_df, left_index=True, right_index=True)
            if self._alter_index:
                X_transformed = X_transformed.set_index('original_index')
                X_transformed.index.name = None
            return X_transformed
        except AttributeError:
            print('Must use .fit() method before transforming')            
            
class Log1pTransformer(TransformerMixin):
    
    def __init__(self, columns: list=None):
        self.columns = columns
        self.present_cols = None

    def fit(self, X, y=None, columns=None):
        if columns:
            self.columns = columns
        if self.columns:
            present_cols = [col for col in self.columns if col in X.columns.values]
            missing_cols = list(set(present_cols).difference(X.columns.values))
            if missing_cols:
                print(f'WARNING: The following columns were passed to be log transformed by are missing: {missing_cols}')
            self.present_cols = present_cols
        return self

    def transform(self, X):
        # assumes X is a DataFrame
        # Subset to columns
        if self.present_cols:
            X_subset = X[self.present_cols]
        else:
            X_subset = X
        # Apply log + 1 transform
        X_log = np.log1p(X_subset)
        
        # Get new col names
        new_col_names = [
            'LOG_'+col
            for col 
            in X_log.columns.values.tolist()
        ]
        X_log.columns = new_col_names
        
        # Return to orignal frame
        X_new =  pd.merge(X, X_log, left_index=True, right_index=True)
        return X_new            