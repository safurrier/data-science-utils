import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import FunctionTransformer, StandardScaler, RobustScaler
from sklearn.preprocessing import Imputer, MultiLabelBinarizer
from functools import reduce



###############################################################################################################
# Custom Transformers
###############################################################################################################


class NumericStringCharacterRemover(TransformerMixin):
    """Take a string column and remove common formatting characters
    (e.g. 1,000 to 1000 and $10 to 10) and transform to numeric

    Parameters
    ----------
    columns : list
        A list of the columns for which to apply this transformation to
    return_numeric : boolean
        Flag to return the comma-removed data as a numeric column.
        Default is set to true
    """
    def __init__(self, columns, return_numeric=True):
        self.columns = columns
        self.return_numeric = return_numeric

    def fit(self, X, y=None):
        # Check for which specified columns are in the X dataframe
        self.columns_present = [col for col in X.columns.values.tolist() if col in self.columns]
        # Within the selected columns, regex search for commas and replace
        # with whitespace
        self.no_commas = X[self.columns_present].replace(to_replace={',','\$', '-'}, value='', regex=True)
        return self

    def transform(self, X, y=None):
        # Replace original columns with new columns
        X[self.columns_present] = self.no_commas
        # If numeric, apply pd.to_numeric
        if self.return_numeric:
            X[self.columns_present] = X[self.columns_present].apply(pd.to_numeric)
        return X

    def fit_transform(self, X, y=None):
        """Convenience function performing both fit and transform"""
        return self.fit(X).transform(X)

class ColumnNameFormatter(TransformerMixin):
    """ Rename a dataframe's column to underscore, lowercased column names

    Parameters
    ----------
    rename_cols : list
        A list of the columns which should be renamed. If not specified, all columns
        are renamed.
    export_mapped_names : boolean
        Flag to return export a csv with the mapping between the old column names
        and new ones
        Default is set to False
    export_mapped_names_path : str
        String export path for csv with the mapping between the old column names
        and new ones. If left null, will export to current working drive with
        filename ColumnNameFormatter_Name_Map.csv
    """

    def __init__(self, rename_cols=None,
                 export_mapped_names=False,
                 export_mapped_names_path=None):
        self.rename_cols = rename_cols
        self.export_mapped_names = export_mapped_names
        self.export_mapped_names_path = export_mapped_names_path

    def _set_renaming_dict(self, X, rename_cols):
        """ Create a dictionary with keys the columns to rename and
            values their renamed versions"""

        # If no specific columns to rename are specified, use all
        # in the data
        if not rename_cols:
            rename_cols = X.columns.values.tolist()

        import re
        # Create Regex to remove illegal characters
        illegal_chars = r"[\\()~#%&*{}/:<>?|\"-]"
        illegal_chars_regex = re.compile(illegal_chars)

        # New columns are single strings joined by underscores where
        # spaces/illegal characters were
        new_columns = ["_".join(re.sub(illegal_chars_regex, " ", col).split(" ")).lower()
                                for col
                                in rename_cols]
        renaming_dict = dict(zip(rename_cols, new_columns))
        return renaming_dict

    def fit(self, X, y=None):
        """ Check the logic of the renaming dict property"""
        self.renaming_dict = self._set_renaming_dict(X, self.rename_cols)
        # Assert that the renaming dict exists and is a dictionary
        assert(isinstance(self.renaming_dict, dict))
        # Assert that all columns are in the renaming_dict
        #assert all([column in self.renaming_dict.keys() for column in rename_cols])
        return self

    def transform(self, X, y=None):
        """ Rename the columns of the dataframe"""
        if self.export_mapped_names:
            # Create mapping of old column names to new ones
            column_name_df = (pd.DataFrame.from_dict(self.renaming_dict, orient='index')
             .reset_index()
             .rename(columns={'index':'original_column_name', 0:'new_column_name'}))

            # If no path specified, export to working directory name with this filename
            if not self.export_mapped_names_path:
                self.export_path = 'ColumnNameFormatter_Name_Map.csv'
            column_name_df.to_csv(self.export_mapped_names_path, index=False)

        # Return X with renamed columns
        return X.rename(columns=self.renaming_dict)

    def fit_transform(self, X, y=None):
        """Convenience function performing both fit and transform"""
        return self.fit(X).transform(X)

class FeatureLookupTable(TransformerMixin):
    """ Given a feature column and path to lookup table, left join the the data
    and lookup values

    Parameters
    ----------
    feature : list
        A list of the column name(s) of the feature to look up. This is what the lookup table will
        be left joined on.
    lookup_key: list
        A list of the column name(s) the lookup table will be joined on. MUST be in same order
        as 'feature' arg. If none specified will default to the 'feature' argument.
    table_lookup_keep_cols: list
        A list of the columns which should be kept from the joined tables. Default is
        to keep all columns from table
    table_path : str
        The path to the table to use as a lookup
    table_format : str
        The type of file the lookup table is. Currently only accepts
        'csv' (default) or 'pickle'
    merge_as_string: Boolean
        Force the key columns to be cast to string before joining
    add_prefix: str
        A prefix to add onto the new columns added from the lookup table
    add_suffix: str
        A suffix to add onto the new columns added from the lookup table

    """
    import pandas as pd

    def __init__(self, feature=None, lookup_key=None,
                 table_lookup_keep_cols=None, table_path=None,
                 table_format='csv', merge_as_string=False,
                add_prefix=None, add_suffix=None):
        self.feature = feature
        self.table_lookup_keep_cols = table_lookup_keep_cols
        self.table_path = table_path
        self.table_format = table_format
        self.merge_as_string = merge_as_string
        self.add_prefix = add_prefix
        self.add_suffix = add_suffix

        # Determine which column to left join the lookup table on
        if not lookup_key:
            # If none specified use 'feature'
            self.lookup_key = self.feature
        else:
            self.lookup_key = lookup_key

        if self.table_format == 'csv':
            self.lookup_table = pd.read_csv(self.table_path)
        elif self.table_format == 'pickle':
            self.lookup_table =  pd.read_pickle(self.table_path)


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

        # Cast dtypes to string if specified
        if self.merge_as_string:
            X[self.feature] = X[self.feature].astype(str)
            self.lookup_table[self.lookup_key] = \
            self.lookup_table[self.lookup_key].astype(str)

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

        # Left join the two
        new_df = pd.merge(X,
                self.lookup_table,
                on=self.feature,
                          how='left')

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
                 ignore_binary=True):

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
        """

    def fit(self, X, y):
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

    def __init__(self, dummy_map=None, remove_original=True, verbose=1):
        """
        Parameters
        ----------
        dummy_map: dict
            Nested dictionary of the form {basefeature:[feature1, feature2, feature3]}
        remove_original: bool
            Boolean flag to remove the columns that will be one hot encoded.
            Default is True. False will one hot encode but keep the original columns as well
        verbose: int
            Verbosity of output. Default is 1, which prints out base features and interacting
            terms in interactions_dict_map that are not present in the data. Set to None to
            ignore this.
        """
        self.dummy_map = dummy_map
        self.verbose = verbose
        self.remove_original = remove_original

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

        # Extract one hot columns to keep
        # By joining each key column with each feature value on "_"
        _one_hot_cols_to_keep = []
        for key, values in self.dummy_map.items():
            for value in values:
                _one_hot_cols_to_keep.append("_".join([key, value]))
        self._one_hot_cols_to_keep = _one_hot_cols_to_keep

        return self

    def transform(self, X):
        try:
            # Extract dummy columns
            self.one_hot_encoded_cols = pd.get_dummies(X[self.present_base_feats])
            # Only keep those specified in the map
            # Might throw key error here
            self.one_hot_encoded_cols = self.one_hot_encoded_cols[self._one_hot_cols_to_keep]

            if self.remove_original:
                # Remove the encoded columns from original
                keep_cols = list(set(X.columns.values.tolist()).difference(self.present_base_feats))
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

class DFNullMapFill(BaseEstimator, TransformerMixin):
    """ Given a dataframe and a dictionary mapping {column:null_fill_value}, replace the
        column values where null with specified value

        Example:
        df = pd.DataFrame([[np.nan, 2, np.nan, 0],
                           [3, 4, np.nan, 1],
                           [np.nan, np.nan, np.nan, 5],
                           [np.nan, 3, np.nan, 4]],
                           columns=list('ABCD'))
         df
             A    B   C  D
        0  NaN  2.0 NaN  0
        1  3.0  4.0 NaN  1
        2  NaN  NaN NaN  5
        3  NaN  3.0 NaN  4

        filler = DataFrameNullFill(null_column_fill_map={'A': 0, 'B': 1, 'C': 2, 'D': 3})

        filler.fit(df)

        df = filler.transform(df)

            A   B   C   D
        0   0.0 2.0 2.0 0
        1   3.0 4.0 2.0 1
        2   0.0 1.0 2.0 5
        3   0.0 3.0 2.0 4
    """

    def __init__(self, null_column_fill_map=None):
        """
        Parameters
        ----------
        null_column_fill_map: dict
            A dictionary mapping {column name: nan fill value}
            E.g. {'A': 0, 'B': 1, 'C': 2, 'D': 3})

        """
        if not null_column_fill_map:
            print('Specify a null_column_fill_map')
        assert isinstance(null_column_fill_map, dict)
        self.null_column_fill_map = null_column_fill_map

    def fit(self, X, y=None):
        original_set_with_copy_setting = pd.options.mode.chained_assignment
        # Disable SettingWithCopy Warning
        pd.options.mode.chained_assignment = None
        assert isinstance(X, pd.DataFrame)
        # First, if there are any Categorical Dtypes, add the fill value
        # to the categories
        X_categorical = X.select_dtypes(include='category')
        # Get list of categorical columns
        categorical_cols = X_categorical.columns.values.tolist()

        # Find out which categorical columns have a null fill specified
        categorical_fill_map = {key:value for key, value in self.null_column_fill_map.items()
                                if key in categorical_cols}

        # For each of those, add the fill value as a category
        for column in list(categorical_fill_map.keys()):
            # If fill value is not in categories, add it
            if categorical_fill_map[column] not in X_categorical[column].cat.categories.values.tolist():
                X_categorical[column] = X_categorical[column].cat.add_categories([categorical_fill_map[column]])

        # Replace the updated columns
        X[categorical_cols] = X_categorical[categorical_cols]
        # Set state and return to original SettingWithCopy Warning setting
        pd.options.mode.chained_assignment = original_set_with_copy_setting
        self.is_fit=True
        return self

    def transform(self, X, y=None):
        X_copy = X
        X_transformed = X_copy.fillna(value=self.null_column_fill_map)
        return X_transformed

class DFDummyTransformer(TransformerMixin):
    # Transforms dummy variables from a list of columns

    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):
        # Assumes no columns provided, in which case all columns will be transformed
        if not self.columns:
            self.already_binary_cols = df_binary_columns_list(X)
            self.cols_to_transform = list(set(X.columns.values.tolist()).difference(self.already_binary_cols))
            # Encode the rest of the columns
            self.dummy_encoded_cols = pd.get_dummies(X[self.cols_to_transform])
        if self.columns:
            self.cols_to_transform = self.columns
            # Encode the rest of the columns
            self.dummy_encoded_cols = pd.get_dummies(X[self.cols_to_transform])
        return self

    def transform(self, X):
        # assumes X is a DataFrame
        # Remove the encoded columns from original
        X_transform = X[list(set(X.columns.values.tolist()).difference(self.cols_to_transform))]
        # Merge on encoded cols
        X_transform = pd.merge(X_transform, self.dummy_encoded_cols, left_index=True, right_index=True)

        return X_transform

    def df_binary_columns_list(df):
        """ Returns a list of binary columns (unique values are either 0 or 1)"""
        binary_cols = [col for col in df if
               df[col].dropna().value_counts().index.isin([0,1]).all()]
        return binary_cols

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
                    col_names = [base_feature+"_*_"+ int_term for int_term in tmp_interaction_terms]
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
                    col_names = ['Log('+base_feature+")_+_Log("+int_term+')' for int_term in tmp_interaction_terms]
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

###############################################################################################################
# Custom Transformers from PyData Seattle 2017 Talk
###############################################################################################################

# Reference
# http://zacstewart.com/2014/08/05/pipelines-of-featureunions-of-pipelines.html
# https://github.com/jem1031/pandas-pipelines-custom-transformers

class DFFunctionTransformer(TransformerMixin):
    # FunctionTransformer but for pandas DataFrames

    def __init__(self, *args, **kwargs):
        self.ft = FunctionTransformer(*args, **kwargs)

    def fit(self, X, y=None):
        # stateless transformer
        return self

    def transform(self, X):
        Xt = self.ft.transform(X)
        Xt = pd.DataFrame(Xt, index=X.index, columns=X.columns)
        return Xt


class DFFeatureUnion(BaseEstimator, TransformerMixin):
    # FeatureUnion but for pandas DataFrames

    def __init__(self, transformer_list):
        self.transformer_list = transformer_list

    def fit(self, X, y=None):
        for (name, t) in self.transformer_list:
            t.fit(X, y)
        return self

    def transform(self, X):
        # assumes X is a DataFrame
        Xts = [t.transform(X) for _, t in self.transformer_list]
        Xunion = reduce(lambda X1, X2: pd.merge(X1, X2, left_index=True, right_index=True), Xts)
        return Xunion


class DFImputer(TransformerMixin):
    # Imputer but for pandas DataFrames

    def __init__(self, strategy='mean'):
        self.strategy = strategy
        self.imp = None
        self.statistics_ = None

    def fit(self, X, y=None):
        self.imp = Imputer(strategy=self.strategy)
        self.imp.fit(X)
        self.statistics_ = pd.Series(self.imp.statistics_, index=X.columns)
        return self

    def transform(self, X):
        # assumes X is a DataFrame
        Ximp = self.imp.transform(X)
        Xfilled = pd.DataFrame(Ximp, index=X.index, columns=X.columns)
        return Xfilled


class DFStandardScaler(BaseEstimator, TransformerMixin):
    # StandardScaler but for pandas DataFrames

    def __init__(self, cols=None):
        self.ss = None
        self.mean_ = None
        self.scale_ = None
        self.cols = cols

    def fit(self, X, y=None):
        if not self.cols:
            self.cols = X.select_dtypes(include=np.number).columns.values.tolist()
        self.ss = StandardScaler()
        self.ss.fit(X[self.cols])
        self.mean_ = pd.Series(self.ss.mean_, index=X[self.cols].columns)
        self.scale_ = pd.Series(self.ss.scale_, index=X[self.cols].columns)
        return self

    def transform(self, X):
        # assumes X is a DataFrame
        # Scale the specified columns
        Xss = self.ss.transform(X[self.cols])
        Xscaled = pd.DataFrame(Xss, index=X.index, columns=X[self.cols].columns)
        # Merge back onto the dataframe
        Xscaled = pd.merge(X[[col for col in X.columns if col not in self.cols]],
                           Xscaled, left_index=True, right_index=True)
        return Xscaled


class DFRobustScaler(TransformerMixin):
    # RobustScaler but for pandas DataFrames

    def __init__(self):
        self.rs = None
        self.center_ = None
        self.scale_ = None

    def fit(self, X, y=None):
        self.rs = RobustScaler()
        self.rs.fit(X)
        self.center_ = pd.Series(self.rs.center_, index=X.columns)
        self.scale_ = pd.Series(self.rs.scale_, index=X.columns)
        return self

    def transform(self, X):
        # assumes X is a DataFrame
        Xrs = self.rs.transform(X)
        Xscaled = pd.DataFrame(Xrs, index=X.index, columns=X.columns)
        return Xscaled


class ColumnExtractor(BaseEstimator, TransformerMixin):
    """ Given a list of columns and optionally a list of columns to include/exclude,
        filter a dataframe down to the selected columns.
    """

    def __init__(self, cols=None, include=None, exclude=None):
        """
        Parameters
        ----------
        cols: list
            A list of string column names to subset the data to
        exclude: list
            A list of string columns to exclude from the dataframe
        include: list
            A list of string columns to include in the dataframe
        """
        self.cols = cols
        self.include = include
        self.exclude = exclude

    def fit(self, X, y=None):
        ## Default to all columns if none were passed
        if not self.cols:
            self.cols = X.columns.values.tolist()
        # Filter out unwanted columns
        if self.exclude:
            self.cols = [col for col in self.cols if col not in self.exclude]
        # Filter down to subset of desired columns
        if self.include:
            self.cols = [col for col in self.cols if col in self.include]
        return self

    def transform(self, X):
        # assumes X is a DataFrame
        Xcols = X[self.cols]
        return Xcols


class ZeroFillTransformer(TransformerMixin):

    def fit(self, X, y=None):
        # stateless transformer
        return self

    def transform(self, X):
        # assumes X is a DataFrame
        Xz = X.fillna(value=0)
        return Xz


class Log1pTransformer(TransformerMixin):

    def fit(self, X, y=None):
        # stateless transformer
        return self

    def transform(self, X):
        # assumes X is a DataFrame
        Xlog = np.log1p(X)
        return Xlog


class DateFormatter(TransformerMixin):

    def fit(self, X, y=None):
        # stateless transformer
        return self

    def transform(self, X):
        # assumes X is a DataFrame
        Xdate = X.apply(pd.to_datetime)
        return Xdate


class DateDiffer(TransformerMixin):

    def fit(self, X, y=None):
        # stateless transformer
        return self

    def transform(self, X):
        # assumes X is a DataFrame
        beg_cols = X.columns[:-1]
        end_cols = X.columns[1:]
        Xbeg = X[beg_cols].as_matrix()
        Xend = X[end_cols].as_matrix()
        Xd = (Xend - Xbeg) / np.timedelta64(1, 'D')
        diff_cols = ['->'.join(pair) for pair in zip(beg_cols, end_cols)]
        Xdiff = pd.DataFrame(Xd, index=X.index, columns=diff_cols)
        return Xdiff


class DummyTransformer(TransformerMixin):

    def __init__(self):
        self.dv = None

    def fit(self, X, y=None):
        # assumes all columns of X are strings
        Xdict = X.to_dict('records')
        self.dv = DictVectorizer(sparse=False)
        self.dv.fit(Xdict)
        return self

    def transform(self, X):
        # assumes X is a DataFrame
        Xdict = X.to_dict('records')
        Xt = self.dv.transform(Xdict)
        cols = self.dv.get_feature_names()
        Xdum = pd.DataFrame(Xt, index=X.index, columns=cols)
        # drop column indicating NaNs
        nan_cols = [c for c in cols if '=' not in c]
        Xdum = Xdum.drop(nan_cols, axis=1)
        return Xdum


class MultiEncoder(TransformerMixin):
    # Multiple-column MultiLabelBinarizer for pandas DataFrames

    def __init__(self, sep=','):
        self.sep = sep
        self.mlbs = None

    def _col_transform(self, x, mlb):
        cols = [''.join([x.name, '=', c]) for c in mlb.classes_]
        xmlb = mlb.transform(x)
        xdf = pd.DataFrame(xmlb, index=x.index, columns=cols)
        return xdf

    def fit(self, X, y=None):
        Xsplit = X.applymap(lambda x: x.split(self.sep))
        self.mlbs = [MultiLabelBinarizer().fit(Xsplit[c]) for c in X.columns]
        return self

    def transform(self, X):
        # assumes X is a DataFrame
        Xsplit = X.applymap(lambda x: x.split(self.sep))
        Xmlbs = [self._col_transform(Xsplit[c], self.mlbs[i])
                 for i, c in enumerate(X.columns)]
        Xunion = reduce(lambda X1, X2: pd.merge(X1, X2, left_index=True, right_index=True), Xmlbs)
        return Xunion


class StringTransformer(TransformerMixin):

    def fit(self, X, y=None):
        # stateless transformer
        return self

    def transform(self, X):
        # assumes X is a DataFrame
        Xstr = X.applymap(str)
        return Xstr


class ClipTransformer(TransformerMixin):

    def __init__(self, a_min, a_max):
        self.a_min = a_min
        self.a_max = a_max

    def fit(self, X, y=None):
        # stateless transformer
        return self

    def transform(self, X):
        # assumes X is a DataFrame
        Xclip = np.clip(X, self.a_min, self.a_max)
        return Xclip


class AddConstantTransformer(TransformerMixin):

    def __init__(self, c=1):
        self.c = c

    def fit(self, X, y=None):
        # stateless transformer
        return self

    def transform(self, X):
        # assumes X is a DataFrame
        Xc = X + self.c
        return Xc

###############################################################################################################
# Functions used with Transformers
###############################################################################################################

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



def df_feature_values_target_average(df, target,
                                                           include=None,
                                                           exclude=None,
                                      min_mean_target_threshold = 0,
                                      min_sample_size = 0,
                                      min_sample_frequency = 0,
                                      min_weighted_target_threshold=0):

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

def feature_vals_target_association_dict(df, feature, target,
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


def df_feature_vals_target_association_dict(df, target,
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


def get_specific_dummies(df, col_map=None, prefix=None, suffix=None, return_df=True):
    """ Given a mapping of column_name: list of values, one hot the values
    in the column and concat to dataframe. Optional arguments to add prefixes
    and/or suffixes to created column names.

    Example col_map: {'foo':['bar', 'zero']} would create one hot columns
    for the values bar and zero that appear in the column foo"""
    one_hot_cols = []
    for column, value in col_map.items():
        for val in value:
            # Create one hot encoded arrays for each value specified in key column
            one_hot_column = pd.Series(np.where(df[column] == val, 1, 0))
            # Set descriptive name
            one_hot_column.name = column+'_==_'+str(val)
            # add to list of one hot columns
            one_hot_cols.append(one_hot_column)
    # Concatenate all created arrays together
    one_hot_cols = pd.concat(one_hot_cols, axis=1)
    if prefix:
        one_hot_cols = one_hot_cols.add_prefix(prefix)
    if suffix:
        one_hot_cols = one_hot_cols.add_suffix(suffix)
    if return_df:
        return pd.concat([df, one_hot_cols], axis=1)
    else:
        return one_hot_cols

def one_hot_column_text_match(df, column, text_phrases, case=False):
    """Given a dataframe, text column to search and a list of text phrases, return a binary
       column with 1s when text is present and 0 otherwise
    """
    # Ignore regex group match warning
    import warnings
    warnings.filterwarnings("ignore", 'This pattern has match groups')

    # Create regex pattern to match any phrase in list

    # The first phrase will be placed in its own groups
    regex_pattern = '({})'.format(text_phrases[0])

    # If there's more than one phrase
    # Each phrase is placed in its own group () with an OR operand in front of it |
    # and added to the original phrase

    if len(text_phrases) > 1:
        subsquent_phrases = "".join(['|({})'.format(phrase) for phrase in text_phrases[1:]])
        regex_pattern += subsquent_phrases

    # Cast to string to ensure .str methods work
    df_copy = df.copy()
    df_copy[column] = df_copy[column].astype(str)

    matches = df_copy[column].str.contains(regex_pattern, na=False, case=case).astype(int)

    # One hot where match is True (must use == otherwise NaNs throw error)
    #one_hot = np.where(matches==True, 1, 0 )

    return matches

def get_text_specific_dummies(df, col_map=None, case=False, prefix=None, suffix=None, return_df=True):
    """ Given a mapping of column_name: list of values, search for text matches
    for the phrases in the list. Optional arguments to add prefixes
    and/or suffixes to created column names.

    Example col_map: {'foo':['bar', 'zero']} would search the text in the values of
    'foo' for any matches of 'bar' OR 'zero' the result is a one hot encoded
    column of matches"""
    one_hot_cols = []
    for column, value in col_map.items():
        # Create one hot encoded arrays for each value specified in key column
        one_hot_column = pd.Series(one_hot_column_text_match(df, column, value, case=case))
        # Check if column already exists in df
        if column+'_match_for: '+str(value)[1:-1].replace(r"'", "") in df.columns.values.tolist():
            one_hot_column.name = column+'_supplementary_match_for: '+str(value)[1:-1].replace(r"'", "")
        else:
            # Set descriptive name
            one_hot_column.name = column+'_match_for: '+str(value)[1:-1].replace(r"'", "")
        # add to list of one hot columns
        one_hot_cols.append(one_hot_column)
    # Concatenate all created arrays together
    one_hot_cols = pd.concat(one_hot_cols, axis=1)
    if prefix:
        one_hot_cols = one_hot_cols.add_prefix(prefix)
    if suffix:
        one_hot_cols = one_hot_cols.add_suffix(suffix)
    if return_df:
        return pd.concat([df, one_hot_cols], axis=1)
    else:
        return one_hot_cols


# Functions for use with FunctionTransformer
def replace_column_values(df, col=None, values=None, replacement=None, new_col_name=None):
    """ Discretize a continuous feature by seperating it into specified quantiles
    Parameters
    ----------
    df : Pandas DataFrame
        A dataframe containing the data to transform
    col: str
        The name of the column to replace certain values in
    values: list
        A list of the values to replace
    replacement: object
        Replaces the matches of values
    new_col_name: str
        The name of the new column which will have the original with replaced values
        If None, the original column will be replaced inplace.

    Returns
    ----------
    df_copy: Pandas DataFrame
        The original dataframe with the column's value replaced
    """
    # Copy so original is not modified
    df_copy = df.copy()
    if not values:
        return print('Please specify values to replace')

    if not replacement:
        return print('Please specify replacement value')


    # If  column name specified, create new column
    if new_col_name:
        df_copy[new_col_name] = df_copy[col].replace(values, replacement)
    # Else replace old column
    else:
        df_copy[col] = df_copy[col].replace(values, replacement)
    return df_copy

def replace_df_values(df, values):
    """ Call pd.DataFrame.replace() on a dataframe and return resulting dataframe.
    Values should be in format of nested dictionaries,
    E.g., {‘a’: {‘b’: nan}}, are read as follows:
        Look in column ‘a’ for the value ‘b’ and replace it with nan
    """
    df_copy = df.copy()
    return df_copy.replace(values)
