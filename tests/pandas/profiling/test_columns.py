import pandas as pd
import numpy as np

from src.utils.pandas.profiling.columns import (
    flatten_multiindex_columns,
)


def test_flatten_multiindex_columns():
    multiIndex_df = pd.DataFrame(np.random.random((4, 4)))
    multiIndex_df.columns = pd.MultiIndex.from_product([[1, 2], ['A', 'B']])
    assert (flatten_multiindex_columns(multiIndex_df).columns ==
            ['1_A', '1_B', '2_A', '2_B']).all()
