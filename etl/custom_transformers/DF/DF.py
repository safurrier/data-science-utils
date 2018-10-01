import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import FunctionTransformer, StandardScaler, RobustScaler
from sklearn.preprocessing import Imputer, MultiLabelBinarizer

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