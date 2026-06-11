# test_download.py
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.data.yahoo import YahooDownloader
from src.features.returns import compute_returns

downloader = YahooDownloader()

df = downloader.download_and_save("JPM")

df = compute_returns(df)

print(df.head(10))