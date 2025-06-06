import matplotlib.pyplot as plt
import pandas as pd

# Define Complex Types
FigAxTuple = tuple[plt.Figure, plt.Axes]
OptionalTupleLimits = tuple[float | None, float | None] | None


def hist_distribution(
    df: pd.DataFrame,
    feature: str,
    figsize: tuple[int, int] = (10, 6),
    bins: int = 50,
    color: str = "teal",  # Histogram color
    plot_central_tendencies: bool = True,
    log_scale_y: bool = False,
    scientific_notation_x: bool = False,
    grid: bool = True,  # Default to True as grids are often helpful
    x_limits: OptionalTupleLimits = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
) -> FigAxTuple:
    """
    Generates and returns a histogram for a specified feature in a DataFrame.

    Args:
        df: Input Pandas DataFrame.
        feature: The name of the column (feature) to plot.
        figsize: Size of the figure (width, height) in inches.
        bins: Number of bins for the histogram.
        color: Color of the histogram bars.
        plot_central_tendencies: If True, plots vertical lines for mean, median, and mode.
        log_scale_y: If True, sets the y-axis to a logarithmic scale.
        scientific_notation_x: If True, allows scientific notation on the x-axis.
                               If False (default), forces plain style.
        grid: If True, displays a grid on the plot.
        x_limits: Tuple of (min_x, max_x) to set x-axis limits.
                  Either value can be None to set only one limit.
                  If None (default), x-axis limits are auto-scaled.
        title: Custom title for the plot. If None, a default title is generated.
        xlabel: Custom label for the x-axis. If None, defaults to the feature name.
        ylabel: Custom label for the y-axis. If None, a default label is generated.

    Returns:
        A tuple containing the matplotlib Figure and Axes objects (fig, ax).

    Raises:
        ValueError: If `log_scale_y` and `scientific_notation_x` are used in a conflicting way
                    (though their direct conflict was removed, good to keep error checks).
                    If `feature` is not in `df.columns`.

    Example:
        >>> data = {
        ...     'sample_data': np.concatenate([
        ...         np.random.normal(0, 1, 500),
        ...         np.random.normal(5, 1, 500)
        ...     ])
        ... }
        >>> sample_df = pd.DataFrame(data)
        >>> fig, ax = hist_distribution(sample_df, 'sample_data', bins=30)
        >>> # To show the plot (in an interactive environment): plt.show()
        >>> # To save the plot: fig.savefig('histogram.png')

        >>> # Example with log scale and custom title
        >>> fig2, ax2 = hist_distribution(
        ...     sample_df, 'sample_data',
        ...     log_scale_y=True,
        ...     title="Log Scale Distribution of Sample Data",
        ...     x_limits=(-3, 8)
        ... )
    """
    if feature not in df.columns:
        raise ValueError(f"Feature '{feature}' not found in DataFrame columns.")

    fig, ax = plt.subplots(figsize=figsize)

    # Plot histogram
    ax.hist(df[feature].dropna(), bins=bins, color=color, log=log_scale_y)  # Add .dropna() for robustness

    if plot_central_tendencies:
        # Calculate central tendency values for non-NaN data
        valid_data = df[feature].dropna()
        if not valid_data.empty:
            central_vals = {
                "Mean": valid_data.mean(),
                "Median": valid_data.median(),
            }
            # Mode can be multi-modal or empty if all values are unique after dropna
            modes = valid_data.mode()
            if not modes.empty:
                central_vals["Mode"] = modes[0]  # Take the first mode if multiple

            # Define distinct colors for the central tendency lines
            # Using a small, clear palette.
            line_colors = ["orangered", "forestgreen", "mediumblue"]

            idx = 0
            for stat_name, stat_value in central_vals.items():
                if pd.notna(stat_value):  # Ensure stat_value itself is not NaN
                    ax.axvline(
                        x=stat_value,
                        label=f"{stat_name}: {stat_value:.2f}",
                        linestyle="--",  # Common shorthand for dashed
                        color=line_colors[idx % len(line_colors)],
                        linewidth=1.5,
                    )
                    idx += 1
        else:
            # Handle case where after dropna, data is empty (e.g. all NaN column)
            # No central tendencies to plot in this case.
            pass

    # Apply formatting
    if not scientific_notation_x:
        ax.ticklabel_format(useOffset=False, style="plain", axis="x")

    # Use user-provided title/labels if available, otherwise generate defaults
    ax.set_title(title if title is not None else f"Distribution of {feature}")
    ax.set_xlabel(xlabel if xlabel is not None else feature)
    ax.set_ylabel(ylabel if ylabel is not None else f"Frequency of {feature}")

    ax.grid(grid)  # Apply grid based on parameter

    # Set x-axis limits if provided
    if x_limits is not None:
        ax.set_xlim(left=x_limits[0], right=x_limits[1])  # More explicit for clarity

    # Add legend if there are labeled items (like central tendency lines)
    # Check if there are any handles and labels to avoid empty legend warning
    handles, labels = ax.get_legend_handles_labels()
    if handles:  # Only show legend if there's something to label
        ax.legend()

    return fig, ax


def draw_boxplot(
    df: pd.DataFrame,
    feature: str,
    figsize: tuple[int, int] = (16, 6),
    whisker_range: float = 1.5,
    showfliers: float = True,
    percentile_limits: tuple[float, float] | None = None,
    fontsize: int = 12,
    box_color: str = "lightblue",
    median_color: str = "red",
    whisker_color: str = "navy",
    grid: bool = False,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
):
    """
    Generates and returns a horizontal boxplot for a specified feature in a DataFrame,
    with options to control whiskers, outliers, and zoomed‐in views.

    Args:
        df: pandas.DataFrame
            Input DataFrame containing the data.
        feature: str
            The name of the numeric column (feature) to plot.
        figsize: tuple (width, height)
            Size of the figure (in inches). Defaults to (16, 6).
        whisker_range: float
            The “whis” parameter in boxplot (how far the whiskers extend beyond IQR).
            Defaults to 1.5.
        showfliers: bool
            Whether to draw individual outlier points. If False, only whiskers and box
            are shown. Defaults to True.
        percentile_limits: tuple (low_pct, high_pct) or None
            If specified (each between 0 and 1), clamps the x‐axis to the feature’s values
            at those percentiles (e.g., (0.05, 0.95)). Defaults to None.
        fontsize: int or float
            Base font size (in points) for title, axis labels, and tick labels. Defaults to 12.
        box_color: str or matplotlib color
            Fill color for the IQR box. Defaults to "lightblue".
        median_color: str or matplotlib color
            Color for the median line and dashed median indicator. Defaults to "red".
        whisker_color: str or matplotlib color
            Color for whiskers and caps. Defaults to "navy".
        grid: bool
            If True, displays a background grid. Defaults to False.
        title: str or None
            Custom title for the plot. If None, a default title is generated.
        xlabel: str or None
            Custom label for the x‐axis. If None, defaults to the feature name.
        ylabel: str or None
            Custom label for the y‐axis. If None, defaults to an empty string.

    Returns:
        A tuple containing the matplotlib Figure and Axes objects (fig, ax).

    Raises:
        ValueError: If `percentile_limits` is provided but not in [(0 ≤ low_pct < high_pct ≤ 1)].
                    If `feature` is not found in `df.columns`.

    Example:
        >>> import pandas as pd
        >>> import numpy as np
        >>> data = {
        ...     'values': np.concatenate([
        ...         np.random.normal(0, 1, 500),
        ...         np.random.normal(5, 1, 500)
        ...     ])
        ... }
        >>> df = pd.DataFrame(data)
        >>> fig, ax = draw_boxplot(
        ...     df, 'values',
        ...     showfliers=False,
        ...     percentile_limits=(0.05, 0.95),
        ...     fontsize=14,
        ...     box_color='lightgreen',
        ...     median_color='purple',
        ...     grid=True,
        ...     title="Trimmed Boxplot of Values",
        ...     xlabel="Value",
        ...     ylabel=""
        ... )
        >>> # To show: plt.show()
        >>> # To save: fig.savefig('boxplot.png')
    """
    if feature not in df.columns:
        raise ValueError(f"Feature '{feature}' not found in DataFrame columns.")

    data = df[feature].dropna()

    # Create figure and axes
    fig, ax = plt.subplots(figsize=figsize)

    # Compute quartiles for potential xlim or annotation
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_whisker = Q1 - whisker_range * IQR
    upper_whisker = Q3 + whisker_range * IQR

    ax.boxplot(
        x=data,
        vert=False,
        whis=whisker_range,
        showfliers=showfliers,
        patch_artist=True,
        boxprops=dict(facecolor=box_color, edgecolor=whisker_color),
        medianprops=dict(color=median_color, linewidth=2),
        whiskerprops=dict(color=whisker_color, linewidth=1.5),
        capprops=dict(color=whisker_color, linewidth=1.5),
        flierprops=dict(marker="o", markerfacecolor=whisker_color, markersize=4, alpha=0.6),
    )

    # Title and labels
    plot_title = title if title is not None else f"Boxplot of '{feature}'"
    ax.set_title(plot_title, fontsize=fontsize + 2, pad=12)
    ax.set_xlabel(xlabel if xlabel is not None else feature, fontsize=fontsize)
    ax.set_ylabel(ylabel if ylabel is not None else "", fontsize=fontsize)

    # Tick label font size
    ax.tick_params(axis="both", which="major", labelsize=fontsize - 1)

    # Optionally zoom in to a percentile window:
    if percentile_limits is not None:
        low_pct, high_pct = percentile_limits
        if not (0 <= low_pct < high_pct <= 1):
            raise ValueError("percentile_limits must be between 0 and 1, e.g. (0.05, 0.95).")
        lower_clamp = data.quantile(low_pct)
        upper_clamp = data.quantile(high_pct)
        ax.set_xlim(lower_clamp, upper_clamp)
        # Annotate percentile bounds
        ax.annotate(
            f"{int(low_pct * 100)}th pct →",
            xy=(lower_clamp, 1),
            xytext=(lower_clamp, 1.1),
            textcoords="data",
            arrowprops=dict(arrowstyle="->", color="gray"),
            color="gray",
            fontsize=fontsize - 2,
            va="center",
        )
        ax.annotate(
            f"← {int(high_pct * 100)}th pct",
            xy=(upper_clamp, 1),
            xytext=(upper_clamp, 1.1),
            textcoords="data",
            arrowprops=dict(arrowstyle="->", color="gray"),
            color="gray",
            fontsize=fontsize - 2,
            va="center",
        )
    else:
        # If not using percentile limits and outliers are hidden, zoom to whiskers
        if not showfliers:
            ax.set_xlim(lower_whisker, upper_whisker)

    # Dashed vertical line at the true median
    median_val = data.median()
    ax.axvline(median_val, color=median_color, linestyle="--", linewidth=1)

    # Optionally display grid
    if grid:
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

    return fig, ax
