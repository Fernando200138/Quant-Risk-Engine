# src/data/plots.py

"""
Visualization module for statistics.py.

Each public function accepts the output of the corresponding
statistics helper and returns a ``matplotlib.figure.Figure``
so the caller controls whether to save, show, or embed it.

Typical usage
-------------
>>> import pandas as pd
>>> from statistics import compute_log_returns, summary_statistics
>>> from plots import (
...     plot_summary_statistics,
...     plot_rolling_volatility,
...     plot_correlation_matrix,
...     plot_return_vs_volatility,
...     plot_return_distributions,
... )
>>>
>>> prices = pd.read_csv("prices.csv", index_col=0, parse_dates=True)
>>> log_returns = compute_log_returns(prices)
>>>
>>> plot_summary_statistics(summary_statistics(log_returns)).savefig("summary.png", dpi=150)
>>> plot_rolling_volatility(log_returns).savefig("rolling_vol.png", dpi=150)
"""

from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.figure import Figure

from src.data.statistics import (
    annualized_return,
    annualized_volatility,
    rolling_volatility,
    correlation_matrix,
    summary_statistics,
    skewness,
    excess_kurtosis,
)

# ---------------------------------------------------------------------------
# Style defaults
# ---------------------------------------------------------------------------

_PALETTE = [
    "#378ADD", "#1D9E75", "#D85A30", "#D4537E",
    "#7F77DD", "#BA7517", "#639922", "#E24B4A",
]

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#cccccc",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#eeeeee",
    "grid.linewidth": 0.8,
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "medium",
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "legend.frameon": False,
})

_PCT = mticker.FuncFormatter(lambda x, _: f"{x * 100:.1f}%")


# ---------------------------------------------------------------------------
# 1. Summary statistics bar chart
# ---------------------------------------------------------------------------

def plot_summary_statistics(
    stats: pd.DataFrame,
    trading_days: int = 252,
) -> Figure:
    """
    Four-panel bar chart of annualized return, annualized volatility,
    skewness, and excess kurtosis.

    Parameters
    ----------
    stats : pd.DataFrame
        Output of ``summary_statistics()``.
    trading_days : int
        Passed through for axis-label context only.

    Returns
    -------
    Figure
    """
    metrics = ["Annual Return", "Annual Volatility", "Skewness", "Excess Kurtosis"]
    assets = stats.index.tolist()
    colors = _PALETTE[: len(assets)]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    fig.suptitle("Summary statistics", fontsize=14, fontweight="medium", y=1.01)
    axes = axes.flatten()

    pct_metrics = {"Annual Return", "Annual Volatility"}

    for ax, metric in zip(axes, metrics):
        values = stats[metric]
        bars = ax.bar(assets, values, color=colors, width=0.55, zorder=3)
        ax.set_title(metric)
        ax.set_xticks(range(len(assets)))
        ax.set_xticklabels(assets)
        ax.axhline(0, color="#aaaaaa", linewidth=0.8, zorder=2)

        if metric in pct_metrics:
            ax.yaxis.set_major_formatter(_PCT)

        for bar, val in zip(bars, values):
            offset = max(abs(values)) * 0.04
            va = "bottom" if val >= 0 else "top"
            y = val + offset if val >= 0 else val - offset
            label = f"{val * 100:.2f}%" if metric in pct_metrics else f"{val:.3f}"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y, label,
                ha="center", va=va, fontsize=9, color="#444444",
            )

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 2. Rolling volatility
# ---------------------------------------------------------------------------
 
def plot_rolling_volatility(
    log_returns: pd.DataFrame,
    window: int = 30,
    trading_days: int = 252,
    dates: pd.Series | None = None,
) -> Figure:
    """
    Line chart of rolling annualized volatility per asset with a
    monthly x-axis.
 
    Parameters
    ----------
    log_returns : pd.DataFrame
        Daily log returns.
    window : int
        Rolling window in days (default 30).
    trading_days : int
        Annualization factor (default 252).
    dates : pd.Series, optional
        Series of date strings (``YYYY-MM-DD``) or datetime values
        aligned with *log_returns*. When provided its values become
        the x-axis. If omitted the DataFrame index is used as-is.
 
    Returns
    -------
    Figure
 
    Examples
    --------
    >>> plot_rolling_volatility(log_returns, dates=df["index"])
    """
    import matplotlib.dates as mdates
 
    returns = log_returns.copy()
 
    if dates is not None:
        dates_aligned = pd.to_datetime(dates.values[-len(returns):])
        returns.index = dates_aligned
    elif not isinstance(returns.index, pd.DatetimeIndex):
        returns.index = pd.to_datetime(returns.index)
 
    rv = rolling_volatility(returns, window=window, trading_days=trading_days)
 
    fig, ax = plt.subplots(figsize=(11, 4))
    linestyles = ["-", "--", "-.", ":"]
 
    for i, col in enumerate(rv.columns):
        ax.plot(
            rv.index, rv[col],
            label=col,
            color=_PALETTE[i % len(_PALETTE)],
            linestyle=linestyles[i % len(linestyles)],
            linewidth=1.5,
            alpha=0.9,
        )
 
    n_months = len(pd.period_range(rv.index.min(), rv.index.max(), freq="M"))
    if n_months <= 24:
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    else:
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
 
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=9)
 
    ax.yaxis.set_major_formatter(_PCT)
    ax.set_title(f"Rolling {window}-day annualized volatility")
    ax.set_xlabel("")
    ax.set_ylabel("Volatility (annualized)")
    ax.legend(loc="upper left", ncol=min(len(rv.columns), 4))
    fig.tight_layout()
    return fig

# ---------------------------------------------------------------------------
# 3. Correlation matrix heat-map
# ---------------------------------------------------------------------------

def plot_correlation_matrix(log_returns: pd.DataFrame) -> Figure:
    """
    Annotated heat-map of the pairwise correlation matrix.

    Parameters
    ----------
    log_returns : pd.DataFrame
        Daily log returns.

    Returns
    -------
    Figure
    """
    corr = correlation_matrix(log_returns)
    n = len(corr)

    fig, ax = plt.subplots(figsize=(max(5, n * 1.1), max(4, n * 1.0)))

    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson ρ", fontsize=10)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.index)

    for i in range(n):
        for j in range(n):
            val = corr.iloc[i, j]
            text_color = "white" if abs(val) > 0.6 else "#333333"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=9, color=text_color)

    ax.set_title("Correlation matrix")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 4. Return vs. volatility scatter (risk-return space)
# ---------------------------------------------------------------------------

def plot_return_vs_volatility(
    log_returns: pd.DataFrame,
    trading_days: int = 252,
) -> Figure:
    """
    Scatter plot of annualized return versus annualized volatility.

    Parameters
    ----------
    log_returns : pd.DataFrame
        Daily log returns.
    trading_days : int
        Annualization factor (default 252).

    Returns
    -------
    Figure
    """
    ret = annualized_return(log_returns, trading_days)
    vol = annualized_volatility(log_returns, trading_days)

    fig, ax = plt.subplots(figsize=(7, 5))

    for i, asset in enumerate(log_returns.columns):
        ax.scatter(
            vol[asset], ret[asset],
            color=_PALETTE[i % len(_PALETTE)],
            s=120, zorder=5, edgecolors="white", linewidths=0.8,
        )
        ax.annotate(
            asset,
            xy=(vol[asset], ret[asset]),
            xytext=(6, 4), textcoords="offset points",
            fontsize=10, color="#333333",
        )

    ax.axhline(0, color="#aaaaaa", linewidth=0.8)
    ax.xaxis.set_major_formatter(_PCT)
    ax.yaxis.set_major_formatter(_PCT)
    ax.set_xlabel("Annualized volatility")
    ax.set_ylabel("Annualized return")
    ax.set_title("Return vs. volatility")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 5. Return distributions (histograms + moments)
# ---------------------------------------------------------------------------

def plot_return_distributions(
    log_returns: pd.DataFrame,
    bins: int = 50,
) -> Figure:
    """
    One subplot per asset showing the daily log-return histogram
    annotated with skewness and excess kurtosis.

    Parameters
    ----------
    log_returns : pd.DataFrame
        Daily log returns.
    bins : int
        Number of histogram bins (default 50).

    Returns
    -------
    Figure
    """
    assets = log_returns.columns.tolist()
    n = len(assets)
    ncols = min(n, 3)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, nrows * 3.5))
    axes = np.array(axes).flatten()

    sk = skewness(log_returns)
    ku = excess_kurtosis(log_returns)

    for i, asset in enumerate(assets):
        ax = axes[i]
        data = log_returns[asset].dropna()
        color = _PALETTE[i % len(_PALETTE)]

        ax.hist(data, bins=bins, color=color, alpha=0.75,
                edgecolor="white", linewidth=0.4, zorder=3)

        mu, sigma = data.mean(), data.std()
        x = np.linspace(data.min(), data.max(), 300)
        normal_y = (
            np.exp(-0.5 * ((x - mu) / sigma) ** 2)
            / (sigma * np.sqrt(2 * np.pi))
        )
        scale = len(data) * (data.max() - data.min()) / bins
        ax.plot(x, normal_y * scale, color="#555555",
                linewidth=1.2, linestyle="--", label="Normal", zorder=4)

        ax.axvline(mu, color=color, linewidth=1, linestyle=":")

        stats_text = f"skew {sk[asset]:.3f}\nkurt {ku[asset]:.3f}"
        ax.text(0.97, 0.96, stats_text, transform=ax.transAxes,
                ha="right", va="top", fontsize=9,
                color="#444444", linespacing=1.6)

        ax.set_title(asset)
        ax.set_xlabel("Daily log return")
        ax.set_ylabel("Frequency")
        ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{x * 100:.1f}%")
        )
        ax.legend(fontsize=8)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Return distributions", fontsize=14, fontweight="medium", y=1.01)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Convenience: save all plots at once
# ---------------------------------------------------------------------------

def save_all(
    log_returns: pd.DataFrame,
    output_dir: str = ".",
    fmt: str = "png",
    dpi: int = 150,
    trading_days: int = 252,
    rolling_window: int = 30,
) -> None:
    """
    Generate and save every plot to *output_dir*.

    Parameters
    ----------
    log_returns : pd.DataFrame
        Daily log returns from ``compute_log_returns()``.
    output_dir : str
        Directory for output files (default current directory).
    fmt : str
        File format passed to ``savefig`` (default ``"png"``).
    dpi : int
        Resolution (default 150).
    trading_days : int
        Annualization factor (default 252).
    rolling_window : int
        Window for rolling volatility (default 30).
    """
    import os

    os.makedirs(output_dir, exist_ok=True)

    stats = summary_statistics(log_returns, trading_days)
    plots = {
        "summary_statistics": plot_summary_statistics(stats, trading_days),
        "rolling_volatility": plot_rolling_volatility(
            log_returns, rolling_window, trading_days
        ),
        "correlation_matrix": plot_correlation_matrix(log_returns),
        "return_vs_volatility": plot_return_vs_volatility(log_returns, trading_days),
        "return_distributions": plot_return_distributions(log_returns),
    }

    for name, fig in plots.items():
        path = os.path.join(output_dir, f"{name}.{fmt}")
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {path}")
