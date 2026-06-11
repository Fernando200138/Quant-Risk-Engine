import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.simulations.GBM import (
    estimate_parameters,
    simulate_paths,
    simulate_paths2,
)
from src.data.statistics import compute_log_returns
# Assuming returns is your log-return dataframe
df=pd.read_parquet(r"D:\Documentos\Vida_profesional\Coding\Projects\Quant_Risk_Engine\src\data\raw\AAPL.parquet")

prices = pd.DataFrame({
    "AAPL": df["adj_close"]
})
returns = compute_log_returns(prices)

params = estimate_parameters(log_returns=returns)

# Generate Monte Carlo simulations
simulate=False
if simulate:
    paths = simulate_paths(
        s0=200,
        params=params,
        n_paths=5000,
        years=1,
        random_seed=42,
    )

    # Terminal prices (last trading day)
    terminal_prices = paths[-1]

    # Create figure
    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(14, 6),
    )

    # -------------------------
    # Left: Simulated paths
    # -------------------------
    ax1.plot(paths[:, :100])
    ax1.set_title("GBM Monte Carlo Paths")
    ax1.set_xlabel("Trading Days")
    ax1.set_ylabel("Price")

    # -------------------------
    # Right: Terminal distribution
    # -------------------------
    ax2.hist(
        terminal_prices,
        bins=50,
    )

    ax2.axvline(
        np.mean(terminal_prices),
        linestyle="--",
        label=f"Mean = {np.mean(terminal_prices):.2f}",
    )

    ax2.axvline(
        np.median(terminal_prices),
        linestyle=":",
        label=f"Median = {np.median(terminal_prices):.2f}",
    )

    ax2.set_title("Distribution of Final Prices")
    ax2.set_xlabel("Price at T = 1 year")
    ax2.set_ylabel("Frequency")
    ax2.legend()

    plt.tight_layout()
    plt.show()
paths = simulate_paths(
    s0=200,
    params=params,
    n_paths=5000,
    years=1,
    random_seed=42,
)

terminal_prices = paths[-1]

# Terminal log returns
log_returns = np.log(terminal_prices / 200)

# Theoretical values
theoretical_mean = (
    params.drift
    - 0.5 * params.volatility**2
)

theoretical_volatility = params.volatility

# Simulated values
simulated_mean = np.mean(log_returns)
simulated_volatility = np.std(log_returns)

print(
    f"Theoretical mean log-return: {theoretical_mean:.6f}"
)
print(
    f"Simulated mean log-return:   {simulated_mean:.6f}"
)

print()

print(
    f"Theoretical volatility:      {theoretical_volatility:.6f}"
)
print(
    f"Simulated volatility:        {simulated_volatility:.6f}"
)
estimated_sigma = np.std(log_returns)

estimated_mu = (
    np.mean(log_returns)
    + 0.5 * estimated_sigma**2
)

print(
    f"Real drift (mu):      {params.drift:.6f}"
)
print(
    f"Estimated drift:      {estimated_mu:.6f}"
)

print()

print(
    f"Real volatility:      {params.volatility:.6f}"
)
print(
    f"Estimated volatility: {estimated_sigma:.6f}"
)

log_paths = np.log(paths / 200)

# variance across paths at each time
variances = np.var(log_paths, axis=1)

# time axis in years
n_steps = paths.shape[0] - 1
time = np.arange(n_steps + 1) / 252
theoretical_variance = params.volatility**2 * time
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))

plt.plot(
    time,
    variances,
    label="Simulated variance",
)

plt.plot(
    time,
    theoretical_variance,
    "--",
    label=r"Theoretical $\sigma^2 t$",
)

plt.xlabel("Time (years)")
plt.ylabel("Variance")
plt.title("Variance of log returns through time")
plt.legend()
plt.grid(True)

plt.show()