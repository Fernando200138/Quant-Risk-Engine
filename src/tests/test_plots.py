import sys
import os
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))
data_dir = project_root / "src" / "data"/"raw"
from src.data.statistics import compute_log_returns, summary_statistics
from src.Plots.plots import (
     plot_summary_statistics,
     plot_rolling_volatility,plot_correlation_matrix,plot_return_vs_volatility,plot_return_distributions)
#prices = pd.read_csv("prices.csv", index_col=0, parse_dates=True)
df = pd.read_parquet(data_dir/"AAPL.parquet")

#print(df.head())
#print(df.columns)
prices = pd.DataFrame({
    "AAPL": df["adj_close"]
})
log_returns = compute_log_returns(prices)
print(df)

#plot_summary_statistics(summary_statistics(log_returns))
plot_rolling_volatility(log_returns, dates=df['Date'])
#plot_return_vs_volatility(log_returns)
plot_return_distributions(log_returns)
plt.show()