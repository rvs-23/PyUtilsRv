import itertools
from collections.abc import Iterable
from functools import reduce
from typing import Any

import numpy as np
import pandas as pd


def find_problematic_cols_df(
    df: pd.DataFrame,
    check_constant: bool = True,
    check_all_null: bool = True,
    ignore_cols: list[str] | None = None,
) -> dict[str, list[str]]:
    """
    Identifies columns in a DataFrame that are either entirely constant,
    entirely null, or both.

    Args:
        df: pd.DataFrame
            The input DataFrame to analyze.
        check_constant: bool, default True
            If True, identifies columns where all non-null values are the same.
        check_all_null: bool, default True
            If True, identifies columns where all values are null.
        ignore_cols: Optional[List[str]], default None
            A list of column names to exclude from the analysis.

    Returns:
        Dict[str, List[str]]
            A dictionary with keys 'constant_cols' and 'all_null_cols',
            where each key maps to a list of column names satisfying
            the respective condition.

    Raises:
        ValueError:
            If `ignore_cols` contains column names not present in `df`.

    Example:
        >>> data = {'A': [1, 1, 1], 'B': [np.nan, np.nan, np.nan],
        ...         'C': [1, 2, 3], 'D': ['x', 'x', np.nan]}
        >>> df = pd.DataFrame(data)
        >>> find_problematic_cols_df(df)
        {'constant_cols': ['A', 'D'], 'all_null_cols': ['B']}
        >>> find_problematic_cols_df(df, ignore_cols=['A'])
        {'constant_cols': ['D'], 'all_null_cols': ['B']}
        >>> find_problematic_cols_df(df, check_constant=False)
        {'constant_cols': [], 'all_null_cols': ['B']}

    Note:
        - A column with a single unique non-null value (e.g., [1, 1, np.nan, 1])
          is considered constant.
        - A column with only null values (e.g., [np.nan, np.nan]) is an all-null column.
    """
    columns_to_consider = df.columns
    if ignore_cols:
        if not all(col in df.columns for col in ignore_cols):
            missing = [col for col in ignore_cols if col not in df.columns]
            raise ValueError(f"Columns in 'ignore_cols' not found in DataFrame: {missing}")
        # More efficient way to exclude ignored columns
        columns_to_consider = df.columns.difference(ignore_cols)

    if columns_to_consider.empty:  # If all columns were ignored or df was empty
        return {"constant_cols": [], "all_null_cols": []}

    # df.nunique(dropna=True) is key:
    # - If a column has only NaNs, nunique() is 0.
    # - If a column has one unique non-NaN value (and possibly NaNs), nunique() is 1.
    df_nuniq = df[columns_to_consider].nunique(dropna=True)

    constant_cols = []
    if check_constant:
        constant_cols = list(df_nuniq[df_nuniq == 1].index)

    all_null_cols = []
    if check_all_null:
        all_null_cols = list(df_nuniq[df_nuniq == 0].index)

    return {"constant_cols": constant_cols, "all_null_cols": all_null_cols}


##############################


def get_df_info(
    df: pd.DataFrame,
    subset_cols: list[str] | None = None,
    include_numerical_stats: bool = False,
) -> pd.DataFrame:
    """
    Generates a DataFrame containing comprehensive information about each column
    in the input DataFrame, such as data types, null counts, unique value counts,
    and optionally, descriptive statistics for numerical columns.

    Args:
        df: pd.DataFrame
            The input Pandas DataFrame to analyze.
        subset_cols: Optional[List[str]], default None
            A list of column names to include in the analysis. If None, all
            columns in the DataFrame are analyzed.
        include_numerical_stats: bool, default False
            If True, additional descriptive statistics (min, max, median,
            skewness, kurtosis) are calculated and included for numerical columns.

    Returns:
        pd.DataFrame
            A DataFrame where each row corresponds to a column from the input
            DataFrame, and columns contain various pieces of information and
            statistics about that column.

    Raises:
        ValueError:
            If `subset_cols` contains column names not present in `df`.

    Example:
        >>> data = {'col1': [1, 2, 3, 2, np.nan],
        ...         'col2': ['a', 'b', 'a', 'c', 'b'],
        ...         'col3': pd.to_datetime(['2023-01-01', np.nan, '2023-01-03', '2023-01-01', '2023-01-04'])}
        >>> df = pd.DataFrame(data)
        >>> df_info = get_df_info(df, include_numerical_stats=True)
        >>> print(df_info[['Column', 'Dtype_pandas', 'Null_count', 'Unique_count', 'Min', 'Mode']])
          Column Dtype_pandas  Null_count  Unique_count  Min Mode
        0   col1      float64           1             3  1.0  [2.0]
        1   col2       object           0             3  NaN  [a, b] # Mode can have multiple values
        2   col3datetime64[ns]           1             3  NaT  [2023-01-01 00:00:00]

    Notes:
        - 'Dtype_python' attempts to determine the Python type of a sample non-null
          value from the column. This can be useful for understanding the underlying
          data representation beyond the Pandas dtype (e.g., distinguishing
          between int, float, str within an 'object' dtype column).
        - If a column contains all NaN values, its 'Dtype_python' will be
          `<class 'float'>` because NaN is typically a float, and 'Mode' will be an empty list.
        - For numerical stats, columns with 'object' or 'datetime64[ns]' dtypes
          are excluded.
    """

    if subset_cols is None:
        columns_to_analyze = list(df.columns)
    else:
        # Validate that all specified subset_cols are in the DataFrame
        if not all(col in df.columns for col in subset_cols):
            missing = [col for col in subset_cols if col not in df.columns]
            raise ValueError(f"Columns in 'subset_cols' not found in DataFrame: {missing}")
        columns_to_analyze = subset_cols

    # Handle case where df might be empty or subset_cols results in empty list
    if not columns_to_analyze:
        return pd.DataFrame()

    num_rows, _ = df.shape

    # --- Basic Information Gathering ---
    info_records: list[dict[str, object]] = []
    for col_name in columns_to_analyze:
        column_series = df[col_name]
        non_null_series = column_series.dropna()

        # Handling Dtype_python for all-NaN or empty columns
        if not non_null_series.empty:
            # Take the first non-null value to determine Python type
            # .sample() could still be problematic if only one non-null value exists and it's complex
            # Using .iloc[0] on non_null_series is more direct.
            python_type = type(non_null_series.iloc[0])
        elif not column_series.empty:  # Column exists but all values are NaN
            python_type = type(np.nan)  # Typically <class 'float'>
        else:  # Should not happen if columns_to_analyze is derived from df.columns
            python_type = None

        null_count = column_series.isnull().sum()
        unique_values = column_series.unique()  # Includes NaN if present

        # nunique() by default does not count NaN. To count NaN as a unique entity:
        # unique_count = column_series.nunique(dropna=False)
        # However, usually we want count of non-NaN unique values.
        unique_count = column_series.nunique()

        # Mode can return multiple values if they have the same highest frequency
        mode_values = list(column_series.mode().to_numpy())
        # Handle case where mode is empty (e.g., all NaN column)
        if not mode_values and column_series.isnull().all():
            mode_values = []  # Or [np.nan] depending on desired representation

        record: dict[str, object] = {
            "column": col_name,
            "dtype_pandas": column_series.dtype,
            "dtype_python": python_type,
            "null_count": null_count,
            "null%": (null_count / num_rows) * 100 if num_rows > 0 else 0,
            "unique_count": unique_count,
            "unique%": (unique_count / num_rows) * 100 if num_rows > 0 else 0,
            "mode": mode_values,
            # Storing all unique values can be memory intensive for high cardinality columns.
            # Potential problem: Theunique_values variable is an ndarray of tuples with unknown length
            # (ndarray[tuple[int, ...], Unknown]), and we cannot directly slice it with [:10].
            # The problem is that the type checker doesn't know the length of the tuples in the ndarray,
            # so it can't guarantee that the slice operation will work correctly.
            # To fix this issue, we must use the tolist() method to convert the ndarray to a list
            "unique_values_sample": unique_values.tolist()[:10],  # Show a sample
        }
        # Add the record to the main list
        info_records.append(record)

    df_info = pd.DataFrame(info_records)

    if not df_info.empty and include_numerical_stats:
        # --- Numerical Specific Information ---
        # Select numeric columns more robustly using select_dtypes
        numeric_df = df[columns_to_analyze].select_dtypes(include=np.number)

        if not numeric_df.empty:
            # Calculate stats only for columns present in numeric_df
            numeric_cols_present = [col for col in df_info["column"] if col in numeric_df.columns]

            if numeric_cols_present:  # Ensure there are numeric columns to process
                stats_data = {
                    "min": numeric_df[numeric_cols_present].min(),
                    "max": numeric_df[numeric_cols_present].max(),
                    "median": numeric_df[numeric_cols_present].median(),
                    "mean": numeric_df[numeric_cols_present].mean(),
                    "std_Dev": numeric_df[numeric_cols_present].std(),
                    "skewness": numeric_df[numeric_cols_present].skew(),
                    "kurtosis": numeric_df[numeric_cols_present].kurtosis(),
                }
                # Convert stats_data dictionary of Series to DataFrame
                stats_df_for_merge = pd.DataFrame(stats_data)

                # Merge these stats into the main df_info DataFrame
                # Set 'Column' as index for df_info temporarily for merging
                df_info = df_info.set_index("column").join(stats_df_for_merge, how="left").reset_index()

            # Fill NaN for non-numeric columns in the newly added stat columns
            # This is implicitly handled by the left join if stats_df_for_merge only contains numeric cols.
            # If join created new columns that weren't in stats_df_for_merge, they'd be all NaN.
            # Ensure all stat columns exist even if no numeric columns were found
            stat_cols_to_ensure = [
                "min",
                "max",
                "median",
                "mean",
                "std_Dev",
                "skewness",
                "kurtosis",
            ]
            for stat_col in stat_cols_to_ensure:
                if stat_col not in df_info.columns:
                    df_info[stat_col] = np.nan

    # The not df_info.empty here acts as a safety check. While it's unlikely df_info would become
    # empty after the first block if it wasn't empty before, it ensures that the reordering
    # logic (which involves accessing df_info.columns) doesn't attempt to operate on an empty DataFrame, which
    # could lead to errors or unexpected behavior.
    if not df_info.empty and include_numerical_stats:
        # --- Final Formatting ---
        # Define preferred column order
        leading_cols = [
            "column",
            "dtype_pandas",
            "dtype_python",
            "null_count",
            "null%",
            "unique_count",
            "unique%",
        ]
        numerical_stat_cols = [
            "min",
            "max",
            "median",
            "mean",
            "std_Dev",
            "skewness",
            "kurtosis",
        ]
        trailing_cols = ["mode", "unique_values_sample"]

        # Filter out columns that might not exist (e.g. if no numeric stats were computed)
        existing_numerical_stat_cols = [col for col in numerical_stat_cols if col in df_info.columns]

        # Construct the final column order
        final_col_order = leading_cols + existing_numerical_stat_cols + trailing_cols

        # Defensive: Ensure all existing columns are included, even if not in the predefined order
        # This handles cases where some columns might be unexpectedly missing or added
        current_cols_set = set(df_info.columns)
        final_col_order_existing = [col for col in final_col_order if col in current_cols_set]

        # Add any remaining columns that were not in the defined order
        remaining_cols = list(current_cols_set - set(final_col_order_existing))
        df_info = df_info[final_col_order_existing + remaining_cols]

    return df_info


##############################


def get_common_elements(*iterables: Iterable[Any]) -> set[Any]:
    """
    Finds common elements present in two or more iterables.

    This function takes a variable number of iterables (e.g., lists, tuples, sets)
    and returns a set containing only those elements that are found in all
    input iterables.

    Args:
        *iterables: Iterable[Any]
            A variable number of iterables. Each iterable can contain
            elements of any type that can be added to a set.

    Returns:
        Set[Any]
            A set of elements common to all input iterables. If no common
            elements are found, or if no iterables are provided,
            an empty set is returned.

    Example:
        >>> get_common_elements([1, 2, 3], [2, 3, 4], [3, 4, 5])
        {3}
        >>> get_common_elements(['a', 'b'], ['b', 'c'])
        {'b'}
        >>> get_common_elements([1, 2], ['a', 'b'])
        set()
        >>> get_common_elements([10, 20, np.nan], [20, 30, np.nan])
        {20, nan} # np.nan is treated as a distinct, hashable element by sets.
        >>> get_common_elements() # No iterables provided
        set()

    Note:
        - The order of elements in the returned set is not guaranteed.
        - If `np.nan` is present in all input iterables, it will be included
          in the result set because `np.nan` is a hashable object and
          `set.intersection` treats distinct `np.nan` objects as equivalent
          for membership testing within sets.
    """
    if not iterables:
        return set()
    # Map each iterable to a set, then find the intersection of all these sets.
    # The first set is taken as the initial value for reduce.
    return reduce(lambda x, y: x.intersection(y), map(set, iterables))


##############################


def common_cols_by_name_bw_dfs(
    df_dictionary: dict[str, pd.DataFrame], comb_size: int = 2
) -> dict[tuple[str, ...], set[str]]:
    """
    Identifies common column names among combinations of DataFrames.

    Given a dictionary of DataFrames, this function finds combinations of
    `comb_size` DataFrames and determines the set of column names that are
    common to all DataFrames within each combination.

    Args:
        df_dictionary: Dict[str, pd.DataFrame]
            A dictionary where keys are descriptive names (strings) for DataFrames
            and values are the Pandas DataFrame objects themselves.
        comb_size: int, default 2
            The number of DataFrames to include in each combination for comparison.
            For example, if `comb_size` is 2, it compares all pairs of DataFrames.
            If `comb_size` is 3, it compares all triplets.

    Returns:
        Dict[Tuple[str, ...], Set[str]]
            A dictionary where:
                - Keys are tuples of DataFrame names representing a combination.
                - Values are sets of common column names found in that combination.
            Only combinations with at least one common column are included.

    Raises:
        ValueError:
            If `comb_size` is less than 2 or greater than the number of
            DataFrames in `df_dictionary`.
        TypeError:
            If `df_dictionary` is not a dictionary or its values are not
            Pandas DataFrames.

    Example:
        >>> df1 = pd.DataFrame(columns=['A', 'B', 'C'])
        >>> df2 = pd.DataFrame(columns=['B', 'C', 'D'])
        >>> df3 = pd.DataFrame(columns=['C', 'D', 'E'])
        >>> dfs = {'df_alpha': df1, 'df_beta': df2, 'df_gamma': df3}
        >>> common_cols_by_name_bw_dfs(dfs, comb_size=2)
        {('df_alpha', 'df_beta'): {'B', 'C'},
         ('df_alpha', 'df_gamma'): {'C'},
         ('df_beta', 'df_gamma'): {'C', 'D'}}
        >>> common_cols_by_name_bw_dfs(dfs, comb_size=3)
        {('df_alpha', 'df_beta', 'df_gamma'): {'C'}}
    """
    if not isinstance(df_dictionary, dict):
        raise TypeError("Input 'df_dictionary' must be a dictionary.")
    if not all(isinstance(df, pd.DataFrame) for df in df_dictionary.values()):
        raise TypeError("All values in 'df_dictionary' must be Pandas DataFrames.")

    num_dfs = len(df_dictionary)
    if comb_size < 2:
        raise ValueError("comb_size must be 2 or greater.")
    if comb_size > num_dfs:
        raise ValueError(
            f"comb_size ({comb_size}) cannot be greater than the number of DataFrames provided ({num_dfs})."
        )

    common_columns_results: dict[tuple[str, ...], set[str]] = {}
    df_names = list(df_dictionary.keys())
    for df_name_combination in itertools.combinations(df_names, comb_size):
        # Extract the DataFrames corresponding to the current combination of names
        list_of_column_sets = [set(df_dictionary[name].columns) for name in df_name_combination]

        if not list_of_column_sets:  # Should not happen with comb_size >= 2
            continue

        # Using get_common_elements helper function
        common_cols_set = get_common_elements(*list_of_column_sets)

        if common_cols_set:
            common_columns_results[df_name_combination] = common_cols_set

    return common_columns_results


##############################
