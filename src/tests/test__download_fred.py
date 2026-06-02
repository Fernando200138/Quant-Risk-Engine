import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from risk_engine.data.fred_data import FredDownloader


API_KEY = "557f634c2a4631c992cf0529d0da60e3"

fred = FredDownloader(API_KEY)

df = fred.download_series("DGS10")

print(df.head())