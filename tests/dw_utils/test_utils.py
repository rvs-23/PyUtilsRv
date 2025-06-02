# tests/dw_utils/test_utils.py

import numpy as np
import pandas as pd
import pytest

from utilities.dw_utils.utils import (
    common_cols_by_name_bw_dfs,
    find_problematic_cols_df,
    get_common_elements,
    get_df_info,
)


def assert_results_equal(result, expected_constant, expected_null):  # type: ignore
    """
    Helper function to compare the result of find_problematic_cols_df with expected outcomes.

    Args:
        result (dict): The result of find_problematic_cols_df.
        expected_constant (list): The list of expected constant columns.
        expected_null (list): The list of expected all-null columns.
    """
    assert sorted(result.get("constant_cols", [])) == sorted(expected_constant)  # type: ignore
    assert sorted(result.get("all_null_cols", [])) == sorted(expected_null)  # type: ignore


def test_basic_scenario():
    """Test with a mix of constant, null, and normal columns."""
    data = {
        "A": [1, 1, 1, 1],
        "B": [np.nan, np.nan, np.nan, np.nan],
        "C": [1, 2, 3, 4],
        "D": ["x", "x", np.nan, "x"],  # Constant with a NaN
        "E": [True, True, True, True],
    }
    df = pd.DataFrame(data)
    result = find_problematic_cols_df(df)
    assert_results_equal(result, expected_constant=["A", "D", "E"], expected_null=["B"])


def test_only_constant_columns():
    """Test when all columns are constant."""
    data = {"A": [5, 5, 5], "B": ["z", "z", "z"]}
    df = pd.DataFrame(data)
    result = find_problematic_cols_df(df)
    assert_results_equal(result, expected_constant=["A", "B"], expected_null=[])


def test_only_null_columns():
    """Test when all columns are null."""
    data = {"A": [np.nan, np.nan], "B": [None, None]}  # None also treated as NaN by pandas
    df = pd.DataFrame(data)
    result = find_problematic_cols_df(df)
    assert_results_equal(result, expected_constant=[], expected_null=["A", "B"])


def test_no_problematic_columns():
    """Test with a DataFrame having no constant or all-null columns."""
    data = {"A": [1, 2, 3], "B": ["x", "y", "z"], "C": [1.1, 2.2, np.nan]}
    df = pd.DataFrame(data)
    result = find_problematic_cols_df(df)
    assert_results_equal(result, expected_constant=[], expected_null=[])


def test_empty_dataframe():
    """Test with an empty DataFrame."""
    df = pd.DataFrame()
    result = find_problematic_cols_df(df)
    assert_results_equal(result, expected_constant=[], expected_null=[])


def test_ignore_cols_functional():
    """Test the ignore_cols parameter functionality."""
    data = {"A": [1, 1], "B": [np.nan, np.nan], "C": [1, 2]}
    df = pd.DataFrame(data)
    result = find_problematic_cols_df(df, ignore_cols=["A", "B"])
    assert_results_equal(result, expected_constant=[], expected_null=[])

    result_ignore_one_const = find_problematic_cols_df(df, ignore_cols=["A"])
    assert_results_equal(result_ignore_one_const, expected_constant=[], expected_null=["B"])

    result_ignore_normal = find_problematic_cols_df(df, ignore_cols=["C"])
    assert_results_equal(result_ignore_normal, expected_constant=["A"], expected_null=["B"])


def test_ignore_cols_all_columns_ignored():
    """Test when all columns are in ignore_cols."""
    data = {"A": [1, 1], "B": [np.nan, np.nan]}
    df = pd.DataFrame(data)
    result = find_problematic_cols_df(df, ignore_cols=["A", "B"])
    assert_results_equal(result, expected_constant=[], expected_null=[])


def test_ignore_cols_raises_value_error():
    """Test that ignore_cols raises ValueError for non-existent columns."""
    data = {"A": [1, 1]}
    df = pd.DataFrame(data)
    with pytest.raises(ValueError) as excinfo:
        find_problematic_cols_df(df, ignore_cols=["Z"])
    assert "Columns in 'ignore_cols' not found in DataFrame: ['Z']" in str(excinfo.value)


def test_check_constant_false():
    """Test when check_constant is False."""
    data = {"A": [1, 1], "B": [np.nan, np.nan], "C": [1, 2]}
    df = pd.DataFrame(data)
    result = find_problematic_cols_df(df, check_constant=False)
    assert_results_equal(result, expected_constant=[], expected_null=["B"])


def test_check_all_null_false():
    """Test when check_all_null is False."""
    data = {"A": [1, 1], "B": [np.nan, np.nan], "C": [1, 2]}
    df = pd.DataFrame(data)
    result = find_problematic_cols_df(df, check_all_null=False)
    assert_results_equal(result, expected_constant=["A"], expected_null=[])


def test_both_checks_false():
    """Test when both check_constant and check_all_null are False."""
    data = {"A": [1, 1], "B": [np.nan, np.nan], "C": [1, 2]}
    df = pd.DataFrame(data)
    result = find_problematic_cols_df(df, check_constant=False, check_all_null=False)
    assert_results_equal(result, expected_constant=[], expected_null=[])


def test_constant_col_with_nans():
    """Test a column that is constant among non-NaN values but also contains NaNs."""
    data = {"A": [10, np.nan, 10, 10, np.nan]}
    df = pd.DataFrame(data)
    result = find_problematic_cols_df(df)
    # Such a column IS considered constant by the nunique(dropna=True) == 1 logic
    assert_results_equal(result, expected_constant=["A"], expected_null=[])


def test_dataframe_with_single_row():
    """Test DataFrame with a single row (all columns will appear constant)."""
    data = {"A": [1], "B": ["hello"], "C": [np.nan]}
    df = pd.DataFrame(data)
    result = find_problematic_cols_df(df)
    assert_results_equal(result, expected_constant=["A", "B"], expected_null=["C"])


def test_dataframe_with_only_one_column_constant():
    data = {"A": [1, 1, 1]}
    df = pd.DataFrame(data)
    result = find_problematic_cols_df(df)
    assert_results_equal(result, expected_constant=["A"], expected_null=[])


def test_dataframe_with_only_one_column_null():
    data = {"A": [np.nan, np.nan, np.nan]}
    df = pd.DataFrame(data)
    result = find_problematic_cols_df(df)
    assert_results_equal(result, expected_constant=[], expected_null=["A"])


# --- Tests for get_df_info ---
@pytest.fixture
def sample_df_for_info():
    """Provides a sample DataFrame for get_df_info tests."""
    data = {
        "col_int": [1, 2, 3, 2, None, 1],
        "col_float": [1.0, 2.5, np.nan, 3.5, 2.5, 1.0],
        "col_str": ["a", "b", "a", "c", "b", "d"],
        "col_bool": [True, False, True, True, False, None],  # Will become object due to None
        "col_all_nan": [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        "col_datetime": pd.to_datetime(
            ["2023-01-01", np.nan, "2023-01-03", "2023-01-01", "2023-01-04", "2023-01-05"]
        ),
    }
    return pd.DataFrame(data)


def test_get_df_info_basic_structure(sample_df_for_info):
    """Test basic structure and content of the output DataFrame."""
    df_info = get_df_info(sample_df_for_info)
    assert isinstance(df_info, pd.DataFrame)
    assert len(df_info) == len(sample_df_for_info.columns)  # One row per input column
    expected_cols = [
        "column",
        "dtype_pandas",
        "dtype_python",
        "null_count",
        "null%",
        "unique_count",
        "unique%",
        "mode",
        "unique_values_sample",
    ]
    for col in expected_cols:
        assert col in df_info.columns
    assert df_info["column"].tolist() == sample_df_for_info.columns.tolist()


def test_get_df_info_subset_cols(sample_df_for_info):
    """Test the subset_cols parameter."""
    subset = ["col_int", "col_str"]
    df_info = get_df_info(sample_df_for_info, subset_cols=subset)
    assert len(df_info) == len(subset)
    assert df_info["column"].tolist() == subset

    # Test subset_cols with an empty list
    df_info_empty_subset = get_df_info(sample_df_for_info, subset_cols=[])
    assert df_info_empty_subset.empty


def test_get_df_info_subset_cols_invalid(sample_df_for_info):
    """Test subset_cols with invalid column names raises ValueError."""
    with pytest.raises(ValueError) as excinfo:
        get_df_info(sample_df_for_info, subset_cols=["col_int", "non_existent_col"])
    assert "Columns in 'subset_cols' not found in DataFrame: ['non_existent_col']" in str(excinfo.value)


def test_get_df_info_numerical_stats_false(sample_df_for_info):
    """Test when include_numerical_stats is False (default)."""
    df_info = get_df_info(sample_df_for_info, include_numerical_stats=False)
    numerical_stat_cols = ["min", "max", "median", "mean", "std_Dev", "skewness", "kurtosis"]
    for stat_col in numerical_stat_cols:
        assert stat_col not in df_info.columns


def test_get_df_info_numerical_stats_true(sample_df_for_info):
    """Test when include_numerical_stats is True."""
    df_info = get_df_info(sample_df_for_info, include_numerical_stats=True)
    numerical_stat_cols = ["min", "max", "median", "mean", "std_Dev", "skewness", "kurtosis"]
    for stat_col in numerical_stat_cols:
        assert stat_col in df_info.columns

    # Check stats for a numeric column (col_int)
    col_int_info = df_info[df_info["column"] == "col_int"].iloc[0]
    assert pd.notna(col_int_info["min"]) and col_int_info["min"] == 1.0
    assert pd.notna(col_int_info["max"]) and col_int_info["max"] == 3.0
    assert pd.notna(col_int_info["mean"])
    assert col_int_info["null_count"] == 1
    assert col_int_info["unique_count"] == 3  # 1, 2, 3
    assert col_int_info["dtype_pandas"] == sample_df_for_info["col_int"].dtype  # float64 because of None
    assert col_int_info["dtype_python"] is type(
        sample_df_for_info["col_int"].dropna().iloc[0]
    )  # should be int or float

    # Check stats for a non-numeric column (col_str) - numerical stats should be NaN
    col_str_info = df_info[df_info["column"] == "col_str"].iloc[0]
    for stat_col in numerical_stat_cols:
        assert pd.isna(col_str_info[stat_col])
    assert col_str_info["null_count"] == 0
    assert col_str_info["unique_count"] == 4  # a, b, c, d
    assert col_str_info["dtype_python"] is str

    # Check all_nan column
    col_all_nan_info = df_info[df_info["column"] == "col_all_nan"].iloc[0]
    assert col_all_nan_info["null_count"] == len(sample_df_for_info)
    assert col_all_nan_info["unique_count"] == 0
    assert col_all_nan_info["dtype_python"] is float
    assert col_all_nan_info["mode"] == []
    for stat_col in numerical_stat_cols:  # Numerical stats for all NaN column
        assert pd.isna(col_all_nan_info[stat_col])


def test_get_df_info_empty_input_df():
    """Test get_df_info with an empty DataFrame."""
    df_empty = pd.DataFrame()
    df_info = get_df_info(df_empty)
    assert df_info.empty


def test_get_df_info_column_order(sample_df_for_info):
    """Test the expected column order when numerical stats are included."""
    df_info = get_df_info(sample_df_for_info, include_numerical_stats=True)
    expected_leading_cols = [
        "column",
        "dtype_pandas",
        "dtype_python",
        "null_count",
        "null%",
        "unique_count",
        "unique%",
    ]
    expected_stat_cols = ["min", "max", "median", "mean", "std_Dev", "skewness", "kurtosis"]
    expected_trailing_cols = ["mode", "unique_values_sample"]

    actual_cols = df_info.columns.tolist()

    # Check that leading columns are in order at the beginning
    assert actual_cols[: len(expected_leading_cols)] == expected_leading_cols

    # Check that all expected stat columns are present (order among them is already tested by their names)
    for col in expected_stat_cols:
        assert col in actual_cols

    # Check that trailing columns are at the end (relative order among them)
    # This is a bit more complex due to potential extra columns not in our lists.
    # A simpler check: ensure the known trailing cols are indeed trailing.
    assert expected_trailing_cols[0] in actual_cols
    assert expected_trailing_cols[1] in actual_cols
    # For more precise trailing order, you might need to know the exact final list.
    # A basic check:
    assert actual_cols.index(expected_trailing_cols[0]) > actual_cols.index(
        expected_stat_cols[-1]
    )  # if stats exist
    assert actual_cols.index(expected_trailing_cols[1]) > actual_cols.index(expected_trailing_cols[0])


# --- Tests for get_common_elements ---
def test_get_common_elements_basic():
    """Test basic functionality with common and unique elements."""
    list1 = [1, 2, 3, 4, 5]
    list2 = (4, 5, 6, 7, 8)  # Use a tuple
    set1 = {5, 8, 9, 10}
    assert get_common_elements(list1, list2, set1) == {5}


def test_get_common_elements_no_common():
    """Test with no common elements."""
    assert get_common_elements([1, 2, 3], [4, 5, 6], [7, 8, 9]) == set()


def test_get_common_elements_all_common():
    """Test when all elements are common."""
    assert get_common_elements([10, 20], [10, 20], (10, 20)) == {10, 20}


def test_get_common_elements_empty_inputs():
    """Test with empty iterables."""
    assert get_common_elements([], [1, 2], []) == set()  # Intersection with empty set is empty
    assert get_common_elements([], []) == set()


def test_get_common_elements_no_inputs():
    """Test with no arguments passed."""
    assert get_common_elements() == set()


def test_get_common_elements_with_nan():
    """Test handling of np.nan."""
    # np.nan is tricky as np.nan != np.nan, but set inclusion works
    # because they are the same object if from the same source, or hash to same value.
    nan_val = np.nan
    assert get_common_elements([1, nan_val, 2], [nan_val, 2, 3], (nan_val, 4, 2)) == {2, nan_val}


def test_get_common_elements_mixed_types():
    """Test with mixed data types in iterables."""
    assert get_common_elements([1, "a", True], ["a", True, 3], (True, "a", 5.0)) == {"a", True}


# --- Tests for common_cols_by_name_bw_dfs ---
@pytest.fixture
def sample_dfs_for_common_cols():
    """Provides sample DataFrames for common_cols_by_name_bw_dfs tests."""
    df1 = pd.DataFrame(columns=["A", "B", "C", "X"])
    df2 = pd.DataFrame(columns=["B", "C", "D", "Y"])
    df3 = pd.DataFrame(columns=["C", "D", "E", "Z"])
    df_empty = pd.DataFrame()
    return {"df1": df1, "df2": df2, "df3": df3, "df_empty": df_empty}


def test_common_cols_basic_pairs(sample_dfs_for_common_cols):
    """Test basic pair-wise common column identification."""
    dfs = {k: sample_dfs_for_common_cols[k] for k in ["df1", "df2", "df3"]}
    result = common_cols_by_name_bw_dfs(dfs, comb_size=2)

    expected = {("df1", "df2"): {"B", "C"}, ("df1", "df3"): {"C"}, ("df2", "df3"): {"C", "D"}}
    # Compare sets within the dictionary values
    assert result.keys() == expected.keys()
    for key in expected:
        assert result[key] == expected[key]


def test_common_cols_triplets(sample_dfs_for_common_cols):
    """Test common columns for combinations of 3."""
    dfs = {k: sample_dfs_for_common_cols[k] for k in ["df1", "df2", "df3"]}
    result = common_cols_by_name_bw_dfs(dfs, comb_size=3)
    expected = {("df1", "df2", "df3"): {"C"}}
    assert result == expected


def test_common_cols_no_common(sample_dfs_for_common_cols):
    """Test when no columns are common in a combination."""
    df_no_common = pd.DataFrame(columns=["P", "Q"])
    dfs = {"df1": sample_dfs_for_common_cols["df1"], "df_no_common": df_no_common}
    result = common_cols_by_name_bw_dfs(dfs, comb_size=2)
    assert result == {}  # Expect empty dict as no common columns


def test_common_cols_with_empty_df(sample_dfs_for_common_cols):
    """Test behavior with empty DataFrames."""
    dfs = {"df1": sample_dfs_for_common_cols["df1"], "df_empty": sample_dfs_for_common_cols["df_empty"]}
    result = common_cols_by_name_bw_dfs(dfs, comb_size=2)
    # Common columns with an empty df will be an empty set, so the combination won't be added
    assert result == {}


def test_common_cols_invalid_comb_size(sample_dfs_for_common_cols):
    """Test raises ValueError for comb_size too small or too large."""
    dfs = {k: sample_dfs_for_common_cols[k] for k in ["df1", "df2"]}
    with pytest.raises(ValueError, match="comb_size must be 2 or greater"):
        common_cols_by_name_bw_dfs(dfs, comb_size=1)
    with pytest.raises(ValueError, match="cannot be greater than the number of DataFrames"):
        common_cols_by_name_bw_dfs(dfs, comb_size=3)


def test_common_cols_invalid_input_type():
    """Test raises TypeError for invalid df_dictionary input."""
    with pytest.raises(TypeError, match="Input 'df_dictionary' must be a dictionary"):
        common_cols_by_name_bw_dfs("not_a_dict")  # type: ignore

    df1 = pd.DataFrame(columns=["A"])
    with pytest.raises(TypeError, match="All values in 'df_dictionary' must be Pandas DataFrames"):
        common_cols_by_name_bw_dfs({"df1": df1, "df2": "not_a_df"})  # type: ignore
