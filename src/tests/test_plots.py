import sys
import os
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import pandas as pd
from src.data.statistics import compute_log_returns, summary_statistics
from src.Plots.plots import (
     plot_summary_statistics,
     plot_rolling_volatility,plot_correlation_matrix,plot_return_vs_volatility,plot_return_distributions)
#prices = pd.read_csv("prices.csv", index_col=0, parse_dates=True)
df = pd.read_parquet(r"D:\Documentos\Vida_profesional\Coding\Projects\Quant_Risk_Engine\src\data\raw\AAPL.parquet")

#print(df.head())
#print(df.columns)
prices = pd.DataFrame({
    "AAPL": df["adj_close"]
})
log_returns = compute_log_returns(prices)


#plot_summary_statistics(summary_statistics(log_returns))
plot_rolling_volatility(log_returns, dates=df['index'])
#plot_return_vs_volatility(log_returns)
plot_return_distributions(log_returns)
plt.show()