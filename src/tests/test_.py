import sys
import os
from pathlib import Path
import ast
import pandas as pd
project_root = Path.cwd().parents[1]
from src.data.statistics import compute_log_returns

if __name__ == "__main__":
    file_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "data",
        "sp500_top100.txt",
    )

    with open(file_path, encoding="utf-8") as file:
        sp500_top100 = ast.literal_eval(file.read())
    prices=pd.DataFrame()
    for stock in sp500_top100:
        stock_price=pd.read_parquet(f"..\data\raw\{stock}.parquet")
        prices[stock]=stock_price["adj_close"]
    log_returns=compute_log_returns(prices)
    log_returns.to_csv(r"..\data\processed\log_returns.csv",index=False)
