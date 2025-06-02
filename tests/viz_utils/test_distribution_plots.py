import matplotlib.pyplot as plt  # Important for tests involving plots
import numpy as np
import pandas as pd
import pytest

from utilities.viz_utils.distribution_plots import hist_distribution


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


# --- Tests for hist_distribution ---
def test_hist_distribution_runs_without_error(sample_df_for_viz_tests):
    """Smoke test: Check if the function executes without raising an error."""
    df_to_use = sample_df_for_viz_tests
    feature_to_test = "col_for_hist"
    try:
        fig, ax = hist_distribution(df_to_use, feature_to_test)
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
    finally:
        if "fig" in locals():  # Ensure fig exists before trying to close
            plt.close(fig)


def test_hist_distribution_invalid_feature(sample_df_for_viz_tests):
    """Test that it raises ValueError for a non-existent feature."""
    with pytest.raises(ValueError, match="Feature 'non_existent_feature' not found"):
        fig, ax = hist_distribution(sample_df_for_viz_tests, "non_existent_feature")
        plt.close(fig)  # Should close even if an error occurs before return


def test_hist_distribution_titles_and_labels(sample_df_for_viz_tests):
    """Test default and custom titles/labels."""
    feature = "col_float"
    df_to_use = sample_df_for_viz_tests

    # Test default title and labels
    fig_default, ax_default = hist_distribution(df_to_use, feature)
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
    fig_custom, ax_custom = hist_distribution(
        df_to_use, feature, title=custom_title, xlabel=custom_xlabel, ylabel=custom_ylabel
    )
    try:
        assert ax_custom.get_title() == custom_title
        assert ax_custom.get_xlabel() == custom_xlabel
        assert ax_custom.get_ylabel() == custom_ylabel
    finally:
        plt.close(fig_custom)
