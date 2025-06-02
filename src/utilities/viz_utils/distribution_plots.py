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
