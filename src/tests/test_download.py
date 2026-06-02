# test_download.py
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from risk_engine.data.yahoo import YahooDownloader
from risk_engine.features.returns import compute_returns

downloader = YahooDownloader()

df = downloader.download_prices("AAPL")

df = compute_returns(df)

print(df.head(10))