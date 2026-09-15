import ast
import sys
from pathlib import Path

import pandas as pd
import numpy as np
# test file: project_root/src/tests/this_file.py
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))
# ============================================================
# 1. Load data
# ============================================================
data_dir = project_root / "src" / "data"/"processed"
df = pd.read_csv(data_dir / "log_returns.csv")

# Convert date column to datetime
df["date"] = pd.to_datetime(df["date"])

# Sort by stock and date
df = df.sort_values(["ticker", "date"])


# ============================================================
# 2. Return features
# ============================================================

# Cumulative log returns over different horizons
df["return_5d"] = (
    df.groupby("ticker")["log_return"]
    .transform(lambda x: x.rolling(5).sum())
)

df["return_10d"] = (
    df.groupby("ticker")["log_return"]
    .transform(lambda x: x.rolling(10).sum())
)

df["return_20d"] = (
    df.groupby("ticker")["log_return"]
    .transform(lambda x: x.rolling(20).sum())
)


# ============================================================
# 3. Volatility features
# ============================================================

df["volatility_5d"] = (
    df.groupby("ticker")["log_return"]
    .transform(lambda x: x.rolling(5).std())
)

df["volatility_10d"] = (
    df.groupby("ticker")["log_return"]
    .transform(lambda x: x.rolling(10).std())
)

df["volatility_20d"] = (
    df.groupby("ticker")["log_return"]
    .transform(lambda x: x.rolling(20).std())
)

df["volatility_60d"] = (
    df.groupby("ticker")["log_return"]
    .transform(lambda x: x.rolling(60).std())
)


# ============================================================
# 4. Save feature dataset
# ============================================================

df.to_csv(data_dir / "features.csv", index=False)

print(df.head(20))
print("\nFeatures saved to features.csv")