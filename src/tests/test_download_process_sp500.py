import ast
import sys
from pathlib import Path

import pandas as pd

# test file: project_root/src/tests/this_file.py
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.data.statistics import compute_log_returns


if __name__ == "__main__":
    data_dir = project_root / "src" / "data"
    input_file = data_dir / "sp500_top100.txt"

    with input_file.open(encoding="utf-8") as file:
        sp500_top100 = ast.literal_eval(file.read())

    prices = pd.DataFrame()

    for stock in sp500_top100:
        stock_file = data_dir / "raw" / f"{stock}.parquet"
        stock_price = pd.read_parquet(stock_file)

        # Use the actual Date column as the index
        stock_price = stock_price.set_index("Date")

        prices[stock] = stock_price["adj_close"]


    # ------------------------------------------------------------
    # Save prices in long format
    # ------------------------------------------------------------

    prices_long = (
        prices
        .rename_axis("date")
        .reset_index()
        .melt(
            id_vars="date",
            var_name="ticker",
            value_name="adj_close"
        )
    )

    prices_long = prices_long.sort_values(["ticker", "date"])


    prices_file = data_dir / "processed" / "prices.csv"
    prices_long.to_csv(prices_file, index=False)


    # ------------------------------------------------------------
    # Calculate log returns
    # ------------------------------------------------------------

    log_returns = compute_log_returns(prices)


    # Convert to long format
    log_returns_long = (
        log_returns
        .rename_axis("date")
        .reset_index()
        .melt(
            id_vars="date",
            var_name="ticker",
            value_name="log_return",
        )
        .dropna(subset=["log_return"])
        .sort_values(["ticker", "date"])
    )

    log_returns_long = log_returns_long.sort_values(["ticker", "date"])


    returns_file = data_dir / "processed" / "log_returns.csv"
    log_returns_long.to_csv(returns_file, index=False)
    print(log_returns_long.head())