# risk_engine/data/yahoo.py

import yfinance as yf
import pandas as pd
from pathlib import Path


class YahooDownloader:

    def __init__(self, data_dir="data/raw"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def download_prices(
        self,
        ticker,
        start="2020-01-01",
        end=None,
        interval="1d"
    ):
        """
        Download historical price data from Yahoo Finance.
        """

        df = yf.download(
            ticker,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=False,
            progress=False
        )

        if df.empty:
            raise ValueError(f"No data found for {ticker}")

        # Make column names lowercase
        df.columns = [
    col[0].lower().replace(" ", "_")
    if isinstance(col, tuple)
    else col.lower().replace(" ", "_")
    for col in df.columns
]
        # Reset index so date becomes a column
        df = df.reset_index()

        # Rename adjusted close column
        if "adj close" in df.columns:
            df = df.rename(columns={"adj close": "adj_close"})

        return df

    def save_parquet(self, df, ticker):
        """
        Save dataframe as parquet file.
        """

        file_path = self.data_dir / f"{ticker}.parquet"

        df.to_parquet(file_path, index=False)

        print(f"Saved: {file_path}")

    def download_and_save(
        self,
        ticker,
        start="2020-01-01",
        end=None
    ):
        """
        Download and save data in one step.
        """

        df = self.download_prices(
            ticker=ticker,
            start=start,
            end=end
        )

        self.save_parquet(df, ticker)

        return df