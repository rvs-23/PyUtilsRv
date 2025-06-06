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
