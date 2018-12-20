import pandas as pd

# Much of this code is adapted from https://www.dataquest.io/blog/pandas-big-data/


def mem_usage(pandas_obj, verbose=0):
    """Return the MBs of memory from a pandas object"""
    if isinstance(pandas_obj, pd.DataFrame):
        usage_b = pandas_obj.memory_usage(deep=True).sum()
    else:  # we assume if not a df it's a series
        usage_b = pandas_obj.memory_usage(deep=True)
    usage_mb = usage_b / 1024 ** 2  # convert bytes to megabytes
    if verbose:
        print("{:03.2f} MB".format(usage_mb))
    return round(usage_mb)


def column_dtype_dict(df):
    """Return a dictionary mapping column names to dtypes. 
    Can be used to read in data to inital dataframe with optimized dtypes"""
    dtypes = df.dtypes
    # Pull the column name
    dtypes_col = dtypes.index
    # And the dtype
    dtypes_type = [i.name for i in dtypes.values]
    # Create Dictionary
    column_types = dict(zip(dtypes_col, dtypes_type))
    return column_types


def memory_usage_by_dtype(df, verbose=0):
    """Return the MBs of memory summary for each dtype in a pandas object"""
    dtypes = df.dtypes
    dtypes_type = list(set([i.name for i in dtypes.values]))
    if verbose:
        print(f'Number of dtypes: {len(dtypes_type)}')
        print(f'Dtypes present: {dtypes_type}')
    memory_usage = []
    for dtype in dtypes_type:
        # Select Dtype
        selected_dtype = df.select_dtypes(include=[dtype])
        # Compute mean mb memory, count of dtype and total
        mean_usage_b = selected_dtype.memory_usage(deep=True).mean()
        mean_usage_mb = mean_usage_b / 1024 ** 2
        count_dtype = selected_dtype.memory_usage(deep=True).count()
        total_mb = count_dtype * mean_usage_mb
        memory_usage.append((dtype, mean_usage_mb, count_dtype, total_mb))
    # Create DF report of usage by dtype
    memory_usage_df = pd.DataFrame(memory_usage, columns=['dtype',
                                                          'Avg_Memory_Usage_MB',
                                                          'Count',
                                                          'Total_Memory_Usage_MB'])
    return memory_usage_df


def optimize_categoric(df, unique_threshold=.5, verbose=0):
    """For a given DataFrame convert the object dtype columns to categorical 
    if the % of unique values is less than specific threshold (default is 50%)
    """
    df_data_objects = df.select_dtypes(include=['object']).copy()
    converted_obj = pd.DataFrame()
    for col in df_data_objects.columns:
        num_unique_values = len(df_data_objects[col].unique())
        num_total_values = len(df_data_objects[col])

        # if pct of unique values is less than threshold of total values
        if num_unique_values / num_total_values < unique_threshold:
            # Convert to categorical 
            converted_obj.loc[:, col] = df_data_objects[col].astype('category')
        # Else keep column as original object dtype
        else:
            converted_obj.loc[:, col] = df_data_objects[col]
    categorical_data_converted = df.copy()
    categorical_data_converted[converted_obj.columns] = converted_obj
    if verbose:
        mem_usage_original_data = mem_usage(df)
        mem_usage_optimized = mem_usage(categorical_data_converted)
        print(f'Before Category Conversion: {mem_usage_original_data} MB')
        print(f'After Category Conversion: {mem_usage_optimized} MB')
        mem_compressed = round(1 - mem_usage_optimized / mem_usage_original_data, 2) * 100
        print(f'Compressed Memory by: {mem_compressed}%')
    return categorical_data_converted


def optimize_numeric(df, verbose=0):
    """Optimize Numeric Dtypes by downcasting to most appropriate dtype"""
    # Downcast ints to smallest unsigned integers possible
    df_int = df.select_dtypes(include=['int'])
    converted_int = df_int.apply(pd.to_numeric, downcast='unsigned')

    df_float = df.select_dtypes(include=['float'])
    converted_float = df_float.apply(pd.to_numeric, downcast='float')

    # Return the memory optimized version
    optimized_df = df.copy()
    optimized_df[converted_int.columns] = converted_int
    optimized_df[converted_float.columns] = converted_float

    if verbose:
        mem_usage_original_data = mem_usage(df)
        mem_usage_optimized = mem_usage(optimized_df)
        print(f'Before Numeric Downcasting: {mem_usage_original_data} MB')
        print(f'After Numeric Downcasting: {mem_usage_optimized} MB')
        mem_compressed = round(1 - mem_usage_optimized / mem_usage_original_data, 2) * 100
        print(f'Compressed Memory by: {mem_compressed}%')

    return optimized_df


def optimize_df(df, unique_threshold=.5, verbose=0):
    """Given a DataFrame, convert object dtype to categoric if the % of unique values is less 
    than specific threshold (default is 50%) and downcast numeric columns
    for memory optimization"""
    optimized_df = df.copy()
    # Optmize Categoric
    optimized_df = optimize_categoric(optimized_df, unique_threshold=unique_threshold)
    # Optmize Numeric
    optimized_df = optimize_numeric(optimized_df)

    if verbose:
        mem_usage_original_data = mem_usage(df)
        mem_usage_optimized = mem_usage(optimized_df)
        print(f'Before Optimization: {mem_usage_original_data} MB')
        print(f'After Optimization: {mem_usage_optimized} MB')
        mem_compressed = round(1 - mem_usage_optimized / mem_usage_original_data, 2) * 100
        print(f'Compressed Memory by: {mem_compressed}%')

    return optimized_df
