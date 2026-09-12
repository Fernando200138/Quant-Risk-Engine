# test_download.py
import sys
import os
import ast

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.data.yahoo import YahooDownloader
from src.features.returns import compute_returns

downloader = YahooDownloader()
if __name__ == "__main__":
    file_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "data",
        "sp500_top100.txt",
    )

    with open(file_path, encoding="utf-8") as file:
        sp500_top100 = ast.literal_eval(file.read())

    print(sp500_top100)
    for stock in sp500_top100:
        df=downloader.download_and_save(stock)

