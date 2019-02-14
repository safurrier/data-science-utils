import pandas as pd
import numpy as np


def data_dictionary(df: pd.DataFrame, include_db_storage_type=False, number_of_examples: int = 3,
                    col_desc_map: dict = None):
    """Given a dataframe iterate through the columns compile a basic data dictionary
    :param df:
    :type df: Pandas DataFrame
    Pandas DataFrame of the data to create a data dictionary from
    :param include_db_storage_type:
    :type include_db_storage_type: Boolean
    Flag for including the SQL Dtype of a field
    :param number_of_examples:
    :type number_of_examples: int
    Integer number of examples to pull from. Default is 3
    :param col_desc_map:
    :type col_desc_map: dict
    Dictionary mapping column name : description
    """
    # For pulling the correct higher level
    # data type from subtype
    dtype_group_dict = {
        'bool': 'Boolean',
        'object': 'Object',
        'float16': 'Numeric',
        'float64': 'Numeric',
        'float32': 'Numeric',
        'int64': 'Numeric',
        'int32': 'Numeric',
        'int8': 'Numeric',
        'int16': 'Numeric',
        'uint64': 'Numeric',
        'uint32': 'Numeric',
        'uint8': 'Numeric',
        'uint16': 'Numeric',
        'datetime64[ns]': 'Time',
        'timedelta[ns]': 'Time',
        'category': 'Object'
    }

    row_count = df.shape[0]
    records = []
    for column in df.columns.values.tolist():
        # Add a description based on the passed dictionary {'column':'description'}
        if col_desc_map:
            try:
                description = col_desc_map[column]
            except KeyError:
                description = ''
        else:
            description = ''
        # Some computed fields
        try:
            most_common_value = df[column].value_counts().index.values.tolist()[0]
            most_common_value_pct = df[column].value_counts().values[0] / df.shape[0]
        except IndexError:
            # No values present results in an empty list which throws
            # An index error
            # Replace the would be value with a null
            most_common_value = np.nan
            most_common_value_pct = np.nan
        # Check for constants
        if most_common_value_pct == 1:
            constant_value = True
        else:
            constant_value = False

        # Look for dtype
        column_dtype = df[column].dtype.name

        # Check for Boolean
        if df[column].nunique() == 2:
            potential_boolean = True
            boolean_test_values = np.array([1, 0, 1.0, 0.0, '1', '0', True, False, 'Y', 'N', 'Yes', 'No'])
            if np.isin(df[column].unique().tolist(), boolean_test_values).all():
                column_dtype = 'bool'
        else:
            potential_boolean = False
        # Cardinality computations
        cardinality = df[column].nunique()
        pct_unique = cardinality / row_count
        # Check for potential Primary Key
        # Heuristic based on if >98% unique and if dtype is object or time
        if (pct_unique > .98) & (dtype_group_dict[column_dtype] in ['Object', 'Time']):
            potential_key = True
        else:
            potential_key = False
        number_of_nulls = df[column].isnull().sum()
        # Create a record with basic info
        record = {
            'Field': column,
            'Description': description,
            'Dtype': dtype_group_dict[column_dtype],
            'Memory_Type': column_dtype,
            'Cardinality': cardinality,
            'Percent_Null': number_of_nulls / row_count,
            'Percent_Unique': pct_unique,
            'Number_of_Nulls': number_of_nulls,
            'Most_Common_Value': most_common_value,
            'Most_Common_Value_Percent_of_Field': most_common_value_pct,
            'Constant': constant_value,
            'Potential_Boolean': potential_boolean,
            'Potential_Key': potential_key,
        }
        if include_db_storage_type:
            db_storage_dict = {
                'Numeric': 'NUMERIC',
                'Object': 'TEXT',
                'Boolean': 'NUMERIC',
            }
            record['DB_Storage_Type'] = db_storage_dict[record['Dtype']]

        column_order = list(record.keys())
        # Pull examples based on the most frequent values
        ex_values = df[column].value_counts().index.values[:number_of_examples].tolist()
        ex_labels = [f'Example_{i + 1}' for i in range(number_of_examples)]
        ex_dict = dict(zip(ex_labels, ex_values))

        # Update the Dictionary
        record = {**record, **ex_dict}
        # Add to list of records
        records.append(record)
    data_dictionary_df = pd.DataFrame.from_records(records, columns=column_order + ex_labels)
    return data_dictionary_df