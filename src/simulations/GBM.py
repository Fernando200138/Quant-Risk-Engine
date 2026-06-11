from dataclasses import dataclass
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.features.returns import compute_log_returns

import numpy as np
import pandas as pd


@dataclass
class GBMParameters:
    """
    Parameters of a Geometric Brownian Motion model.

    Attributes
    ----------
    drift : float
        Annualized expected return (μ).

    volatility : float
        Annualized volatility (σ).
    """

    drift: float
    volatility: float
@dataclass
class GBMCorrelationMatrix:
    """
    Correlation matrix of a Correlated brownian motion
    """
    corr_matrix: pd.DataFrame

def estimate_Correlation_matrix(
        log_returns:pd.DataFrame
        )->GBMCorrelationMatrix:
    return log_returns.corr()


def estimate_parameters(
    log_returns: pd.Series,
    trading_days: int = 252,
) -> GBMParameters:
    """
    Estimate GBM parameters from historical log returns.

    Parameters
    ----------
    log_returns : pd.Series
        Historical daily log returns.

    trading_days : int, default=252
        Number of trading days in a year.

    Returns
    -------
    GBMParameters
        Annualized drift and volatility.
    """

    mu_daily = float(log_returns.mean())
    sigma_daily = float(log_returns.std())

    mu_annual = mu_daily * trading_days
    sigma_annual = sigma_daily * np.sqrt(trading_days)

    return GBMParameters(
        drift=mu_annual,
        volatility=sigma_annual,
    )


def simulate_path(
    s0: float,
    params: GBMParameters,
    years: float = 1.0,
    steps_per_year: int = 252,
    random_seed: int | None = None,
) -> np.ndarray:
    """
    Simulate a single GBM price path.

    Parameters
    ----------
    s0 : float
        Initial asset price.

    params : GBMParameters
        GBM drift and volatility.

    years : float, default=1.0
        Simulation horizon in years.

    steps_per_year : int, default=252
        Number of simulation steps per year.

    random_seed : int | None
        Optional seed for reproducibility.

    Returns
    -------
    np.ndarray
        Simulated price path.
    """

    if random_seed is not None:
        np.random.seed(random_seed)

    n_steps = int(years * steps_per_year)
    dt = 1 / steps_per_year

    z = np.random.normal(
        loc=0,
        scale=1,
        size=n_steps,
    )

    prices = np.zeros(n_steps + 1)
    prices[0] = s0

    for t in range(1, n_steps + 1):

        prices[t] = (
            prices[t - 1]
            * np.exp(
                (
                    params.drift
                    - 0.5 * params.volatility**2
                )
                * dt
                + params.volatility
                * np.sqrt(dt)
                * z[t - 1]
            )
        )

    return prices


def simulate_paths(
    s0: float,
    params: GBMParameters,
    n_paths: int = 1000,
    years: float = 1.0,
    steps_per_year: int = 252,
    random_seed: int | None = None,
) -> np.ndarray:
    """
    Simulate many GBM price paths at once.

    Parameters
    ----------
    s0 : float
        Initial asset price.

    params : GBMParameters
        GBM parameters.

    n_paths : int, default=1000
        Number of Monte Carlo paths.

    years : float, default=1.0
        Simulation horizon.

    steps_per_year : int, default=252
        Time discretization.

    random_seed : int | None
        Optional seed.

    Returns
    -------
    np.ndarray
        Shape = (n_steps + 1, n_paths)
    """

    if random_seed is not None:
        np.random.seed(random_seed)

    n_steps = int(years * steps_per_year)
    dt = 1 / steps_per_year

    z = np.random.normal(
        loc=0,
        scale=1,
        size=(n_steps, n_paths),
    )

    drift_term = (
        params.drift
        - 0.5 * params.volatility**2
    ) * dt

    diffusion_term = (
        params.volatility
        * np.sqrt(dt)
        * z
    )

    log_returns = drift_term + diffusion_term

    cumulative_log_returns = np.cumsum(
        log_returns,
        axis=0,
    )

    paths = np.vstack(
        [
            np.zeros(n_paths),
            cumulative_log_returns,
        ]
    )

    prices = s0 * np.exp(paths)

    return prices

def simulate_paths2(
    s0: float,
    params: GBMParameters,
    n_paths: int = 1000,
    years: float = 1.0,
    steps_per_year: int = 252,
    random_seed: int | None = None,
) -> np.ndarray:
    if random_seed is not None:
        np.random.seed(random_seed)

    n_steps = int(years * steps_per_year)
    dt = 1 / steps_per_year
    prices = np.zeros((n_steps + 1, n_paths))

    for i in range(n_paths):

        z = np.random.normal(size=n_steps)

        prices[0, i] = s0

        for t in range(1, n_steps + 1):

            prices[t, i] = (
                prices[t - 1, i]
                * np.exp(
                    (params.drift - 0.5 * params.volatility**2) * dt
                    + params.volatility * np.sqrt(dt) * z[t - 1]
                )
            )
    return prices

def simulate_correlated_GBM(
        corr_matrix:GBMCorrelationMatrix,
        prices:pd.DataFrame,
        returns:pd.DataFrame,
        n_paths:int,
        n_assets:int,
        years:int,
        steps_per_year:int,
)->np.ndarray:
    L=np.linalg.cholesky(corr_matrix)
    S0 = prices.iloc[-1].to_numpy()
    n_steps=years * steps_per_year
    mu = (returns.mean() * 252).to_numpy()
    sigma = (returns.std() * np.sqrt(252)).to_numpy()
    paths = np.zeros((n_steps, n_assets, n_paths))

    paths[0] = S0[:, None]

    dt = 1/252

    for t in range(1, n_steps):

        Z = np.random.normal(size=(n_assets, n_paths))
        Z_corr = L @ Z

        paths[t] = (
            paths[t-1]
            * np.exp(
                (mu[:, None] - 0.5 * sigma[:, None]**2) * dt
                + sigma[:, None] * np.sqrt(dt) * Z_corr
            )
        )
    return paths

        