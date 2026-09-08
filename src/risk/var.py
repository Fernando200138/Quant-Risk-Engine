import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.simulations.GBM import simulate_correlated_GBM
import pandas as pd
import numpy as np
import numpy as np

def loss_distribution(paths: np.ndarray) -> np.ndarray:
    """
    Computes losses = V0 - VT.
    It's not necessary a loss, it's the difference between the initial price and final price
    Accepts:
        (time, assets, paths)  -> Monte Carlo simulations
        (time, assets)         -> Historical prices

    Returns:
        (assets, paths) for simulations
        (assets,) for historical data
    """

    if paths.ndim not in (2, 3):
        raise ValueError(
            f"Expected array with 2 or 3 dimensions, got shape {paths.shape}"
        )

    print("NaNs at t0:", np.isnan(paths[0]).sum())
    print("NaNs at tT:", np.isnan(paths[-1]).sum())

    V0 = paths[0]
    VT = paths[-1]

    losses = V0 - VT

    return losses

