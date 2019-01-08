import pandas as pd

def data_dictionary(df: pd.DataFrame, number_of_examples: int=3):
    """Given a dataframe iterate through the columns and extract the datatype as well as specified number of examples"""
    # For pulling the correct higher level
    # data type from subtype
    dtype_group_dict = {
        'bool':'Boolean',
        'object':'Object',
        'float16':'Numeric',
        'float64':'Numeric',
        'float32':'Numeric',
        'int64':'Numeric',
        'int32':'Numeric',
        'int8':'Numeric',
        'int16':'Numeric',
        'uint64':'Numeric',
        'uint32':'Numeric',
        'uint8':'Numeric',
        'uint16':'Numeric',      
        'datetime64[ns]': 'Time',
        'timedelta[ns]': 'Time',
    }
    
    records = []
    for column in df.columns.values.tolist():
        # Some computed fields
        try:
            most_common_value = df[column].value_counts().index.values.tolist()[0]
            most_common_value_pct = df[column].value_counts().values[0]/df.shape[0]
        except IndexError: 
            # No values present results in an empty list which throws
            # An index error
            # Replace the would be value with a null
            most_common_value = np.nan
            most_common_value_pct = np.nan
        number_of_nulls = df[column].isnull().sum()
        # Create a record with basic info
        record = {
            'Field':column,
            'Dtype':dtype_group_dict[df[column].dtype.name],
            'Memory_Type':df[column].dtype.name,
            'Cardinality':df[column].nunique(),
            'Percent_Null':number_of_nulls/df.shape[0],   
            'Number_of_Nulls':number_of_nulls, 
            'Most_Common_Value':most_common_value,
            'Most_Common_Value_Percent_of_Field':most_common_value_pct,
        }
        column_order = list(record.keys())
        # Pull examples based on the most frequent values
        ex_values = policy_center_df[column].value_counts().index.values[:number_of_examples].tolist()
        ex_labels = [f'Example_{i+1}' for i in range(number_of_examples)]
        ex_dict = dict(zip(ex_labels, ex_values))

        # Update the Dictionary
        record = {**record, **ex_dict}
        # Add to list of records
        records.append(record)
    data_dictionary = pd.DataFrame.from_records(records, columns=column_order+ex_labels)
    return data_dictionary