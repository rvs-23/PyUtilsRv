import matplotlib.pyplot as plt  # Important for tests involving plots
import numpy as np
import pandas as pd
import pytest
from matplotlib.lines import Line2D
from pandas.api.types import is_datetime64_any_dtype

from utilities.viz_utils.generic_plots import plot_hist_distribution, plot_line_chart


@pytest.fixture
def sample_df_for_viz_tests():
    """Provides a sample DataFrame with consistent column lengths for visualization tests."""
    n_samples = 100  # Define a consistent number of samples for all columns
    data = {
        "col_int": np.random.poisson(5, n_samples) + 1,
        "col_float": np.random.normal(0, 1, n_samples),
        "col_for_hist": np.concatenate(
            [
                np.random.normal(0, 1, n_samples // 2),  # First half
                np.random.normal(5, 1, n_samples - (n_samples // 2)),  # Second half, handles odd n_samples
            ]
        ),
        "col_all_nan": [np.nan] * n_samples,
        "col_datetime": pd.to_datetime(pd.date_range(start="2023-01-01", periods=n_samples, freq="D")),
    }
    return pd.DataFrame(data)


# --- Tests for plot_hist_distribution ---
def test_hist_distribution_runs_without_error(sample_df_for_viz_tests):
    """Smoke test: Check if the function executes without raising an error."""
    df_to_use = sample_df_for_viz_tests
    feature_to_test = "col_for_hist"
    try:
        fig, ax = plot_hist_distribution(df_to_use, feature_to_test)
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
    finally:
        if "fig" in locals():  # Ensure fig exists before trying to close
            plt.close(fig)


def test_plot_hist_distribution_invalid_feature(sample_df_for_viz_tests):
    """Test that it raises ValueError for a non-existent feature."""
    with pytest.raises(ValueError, match="Feature 'non_existent_feature' not found"):
        fig, ax = plot_hist_distribution(sample_df_for_viz_tests, "non_existent_feature")
        plt.close(fig)  # Should close even if an error occurs before return


def test_plot_hist_distribution_titles_and_labels(sample_df_for_viz_tests):
    """Test default and custom titles/labels."""
    feature = "col_float"
    df_to_use = sample_df_for_viz_tests

    # Test default title and labels
    fig_default, ax_default = plot_hist_distribution(df_to_use, feature)
    try:
        assert ax_default.get_title() == f"Distribution of {feature}"
        assert ax_default.get_xlabel() == feature
        assert ax_default.get_ylabel() == f"Frequency of {feature}"
    finally:
        plt.close(fig_default)

    # Test custom title and labels
    custom_title = "My Custom Title"
    custom_xlabel = "My X-Axis"
    custom_ylabel = "My Y-Axis"
    fig_custom, ax_custom = plot_hist_distribution(
        df_to_use, feature, title=custom_title, xlabel=custom_xlabel, ylabel=custom_ylabel
    )
    try:
        assert ax_custom.get_title() == custom_title
        assert ax_custom.get_xlabel() == custom_xlabel
        assert ax_custom.get_ylabel() == custom_ylabel
    finally:
        plt.close(fig_custom)


# --- Tests for plot_line_chart ---
def test_raw_numeric_plot(monkeypatch):
    # Create a small numeric DataFrame
    df = pd.DataFrame({"x": [1, 2, 3, 4], "y": [10, 15, 5, 20]})
    fig, ax = plot_line_chart(df, x_col="x", y_col="y", verbosity=1)
    # Ensure figure and axes are returned
    assert fig is not None
    assert ax is not None
    # There should be one Line2D
    lines = ax.get_lines()
    assert len(lines) == 1
    line: Line2D = lines[0]
    # Check x and y data
    np.testing.assert_array_equal(line.get_xdata(), np.array([1, 2, 3, 4]))
    np.testing.assert_array_equal(line.get_ydata(), np.array([10, 15, 5, 20]))


def test_datetime_exact_format(monkeypatch):
    # Create a DataFrame with known date strings
    dates = ["2025-01-01", "2025-01-02", "2025-01-03"]
    df = pd.DataFrame({"date_str": dates, "val": [100, 200, 150]})
    fig, ax = plot_line_chart(df, x_col="date_str", y_col="val", datetime_format="%Y-%m-%d", verbosity=1)
    lines = ax.get_lines()
    assert len(lines) == 1
    x_data = lines[0].get_xdata()
    # Check that x_data are datetime64
    assert is_datetime64_any_dtype(pd.Series(x_data))
    # Compare exact dates
    expected = pd.to_datetime(dates, format="%Y-%m-%d")
    np.testing.assert_array_equal(x_data, expected.values)


def test_mixed_format_with_resample():
    # Mixed x values, but only parseable rows survive resample
    df = pd.DataFrame(
        {"x_mixed": ["2025-01-01", "foo", "2025-01-01", "bar", "2025-01-02"], "val": [10, 20, 30, 40, 50]}
    )
    # Resample daily; parseable dates are 2025-01-01 (two rows) and 2025-01-02
    fig, ax = plot_line_chart(
        df,
        x_col="x_mixed",
        y_col="val",
        datetime_format="mixed",
        resample_rule="D",
        agg_func="sum",
        verbosity=1,
    )
    lines = ax.get_lines()
    assert len(lines) == 1
    x_data = lines[0].get_xdata()
    y_data = lines[0].get_ydata()
    # After resampling, expect two days: 2025-01-01 sum=40, 2025-01-02 sum=50
    expected_index = pd.to_datetime(["2025-01-01", "2025-01-02"])
    np.testing.assert_array_equal(x_data, expected_index.values)
    np.testing.assert_array_equal(y_data, np.array([40, 50]))


def test_resample_single_bucket_adds_marker():
    # All parseable dates fall into the same month
    dates = ["2025-05-01", "2025-05-15", "2025-05-20"]
    df = pd.DataFrame({"timestamp": dates, "val": [5, 10, 15]})
    fig, ax = plot_line_chart(
        df,
        x_col="timestamp",
        y_col="val",
        datetime_format="%Y-%m-%d",
        resample_rule="ME",
        agg_func="mean",
        verbosity=1,
    )
    lines = ax.get_lines()
    assert len(lines) == 1
    line = lines[0]
    x_data = line.get_xdata()
    y_data = line.get_ydata()
    # Resampled by month → single point: 2025-05-31
    expected_date = pd.to_datetime(["2025-05-31"])
    np.testing.assert_array_equal(x_data, expected_date.values)
    # Mean of [5,10,15] = 10
    np.testing.assert_array_equal(y_data, np.array([10]))
    # Since only one point, marker should be 'o'
    assert line.get_marker() == "o"


def test_binning_numeric():
    # Numeric x and y
    arr_x = np.array([0.1, 0.4, 0.5, 0.8, 1.2, 1.5])
    arr_y = np.array([10, 20, 30, 40, 50, 60])
    df = pd.DataFrame({"x_num": arr_x, "y_num": arr_y})
    # Create 2 bins: [0.1–0.8), [0.8–1.5]
    fig, ax = plot_line_chart(df, x_col="x_num", y_col="y_num", bins=2, agg_func="mean", verbosity=1)
    lines = ax.get_lines()
    assert len(lines) == 1
    x_data = lines[0].get_xdata()
    y_data = lines[0].get_ydata()
    # Compute expected bin centers and means
    # Bin edges would be approximately [0.1, 0.8] and [0.8, 1.5]
    # First bin values: 0.1, 0.4, 0.5 => mean y = (10+20+30)/3 = 20
    # Second bin values: 0.8, 1.2, 1.5 => mean y = (40+50+60,)/3 = 50
    expected_centers = np.array([(0.1 + 0.8) / 2, (0.8 + 1.5) / 2])
    expected_means = np.array([20.0, 50.0])
    np.testing.assert_allclose(x_data, expected_centers, rtol=1e-2)
    np.testing.assert_allclose(y_data, expected_means, rtol=1e-2)


def test_invalid_columns_raise():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    with pytest.raises(ValueError):
        plot_line_chart(df, x_col="not_a", y_col="b")
    with pytest.raises(ValueError):
        plot_line_chart(df, x_col="a", y_col="not_b")


def test_conflicting_resample_and_bins():
    df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    with pytest.raises(ValueError):
        plot_line_chart(df, x_col="x", y_col="y", resample_rule="D", bins=3)
