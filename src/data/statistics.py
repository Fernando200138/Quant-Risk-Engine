# src/data/statistics.py

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute log returns from price data.

    Parameters
    ----------
    prices : pd.DataFrame
        Price data with datetime index and assets as columns.

    Returns
    -------
    pd.DataFrame
        Log returns.
    """
    return np.log(prices / prices.shift(1)).dropna()


def annualized_return(log_returns: pd.DataFrame,
                      trading_days: int = 252) -> pd.Series:
    """
    Annualized mean return from daily log returns.
    """
    return log_returns.mean() * trading_days


def annualized_volatility(log_returns: pd.DataFrame,
                          trading_days: int = 252) -> pd.Series:
    """
    Annualized volatility.
    """
    return log_returns.std() * np.sqrt(trading_days)


def covariance_matrix(log_returns: pd.DataFrame,
                      trading_days: int = 252) -> pd.DataFrame:
    """
    Annualized covariance matrix.
    """
    return log_returns.cov() * trading_days


def correlation_matrix(log_returns: pd.DataFrame) -> pd.DataFrame:
    """
    Correlation matrix.
    """
    return log_returns.corr()


def rolling_volatility(log_returns: pd.DataFrame,
                       window: int = 30,
                       trading_days: int = 252) -> pd.DataFrame:
    """
    Rolling annualized volatility.
    """
    return (
        log_returns
        .rolling(window)
        .std()
        * np.sqrt(trading_days)
    )


def skewness(log_returns: pd.DataFrame) -> pd.Series:
    """
    Sample skewness for each asset.
    """
    return log_returns.apply(
        lambda x: skew(x.dropna(), bias=False)
    )


def excess_kurtosis(log_returns: pd.DataFrame) -> pd.Series:
    """
    Excess kurtosis.

    Normal distribution -> 0
    Fat tails -> positive values
    """
    return log_returns.apply(
        lambda x: kurtosis(x.dropna(),
                           fisher=True,
                           bias=False)
    )


def summary_statistics(log_returns: pd.DataFrame,
                       trading_days: int = 252) -> pd.DataFrame:
    """
    Convenient summary table.
    """
    stats = pd.DataFrame({
        "Annual Return": annualized_return(
            log_returns,
            trading_days
        ),
        "Annual Volatility": annualized_volatility(
            log_returns,
            trading_days
        ),
        "Skewness": skewness(log_returns),
        "Excess Kurtosis": excess_kurtosis(log_returns)
    })

    return stats.round(4)