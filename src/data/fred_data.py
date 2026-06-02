from fredapi import Fred
import pandas as pd


class FredDownloader:

    def __init__(self, api_key):
        self.fred = Fred(api_key=api_key)

    def download_series(
        self,
        series_id,
        start="2000-01-01"
    ):
        """
        Download a FRED time series.
        """

        data = self.fred.get_series(
            series_id,
            observation_start=start
        )

        df = pd.DataFrame({
            "date": data.index,
            "value": data.values
        })

        return df