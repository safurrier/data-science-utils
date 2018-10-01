import pandas as pd
import numpy as np
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import TransformerMixin, BaseEstimator

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

