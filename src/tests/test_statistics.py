import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
# tests/test_statistics.py

import pandas as pd

from src.data.statistics import (
    compute_log_returns,
    summary_statistics,
    correlation_matrix,
    rolling_volatility,
)

# Load data
df = pd.read_parquet(r"D:\Documentos\Vida_profesional\Coding\Projects\Quant_Risk_Engine\src\data\raw\AAPL.parquet")

#print(df.head())
#print(df.columns)
prices = pd.DataFrame({
    "AAPL": df["adj_close"]
})
returns = compute_log_returns(prices)

print(returns.head())
stats = summary_statistics(returns)

print(stats)
corr = correlation_matrix(returns)

print(corr)
rolling_vol = rolling_volatility(
    returns,
    window=30
)

print(rolling_vol.tail())