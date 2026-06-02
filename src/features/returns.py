import numpy as np


def compute_returns(df):
    """
    Compute daily percentage returns.
    """

    df = df.copy()

    df["returns"] = df["adj_close"].pct_change()

    return df


def compute_log_returns(df):
    """
    Compute logarithmic returns.
    """

    df = df.copy()

    df["log_returns"] = np.log(
        df["adj_close"] / df["adj_close"].shift(1)
    )

    return df