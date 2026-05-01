"""Correlation matrix computation across aligned factor return series."""
import pandas as pd


def correlation_matrix(wide_df: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    """Compute pairwise correlation on a wide DataFrame (date index, factor columns).
    Drops rows where all values are NaN before computing."""
    df = wide_df.dropna(how="all")
    return df.corr(method=method)


def rolling_correlation(
    s1: pd.Series,
    s2: pd.Series,
    window: int = 36,
) -> pd.Series:
    """36-month rolling correlation between two return series."""
    aligned = pd.concat([s1, s2], axis=1).dropna()
    if aligned.shape[1] < 2:
        return pd.Series(dtype=float)
    return aligned.iloc[:, 0].rolling(window).corr(aligned.iloc[:, 1])
