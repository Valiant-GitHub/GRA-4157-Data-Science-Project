"""Feature engineering: time and lag features for spreads."""
import pandas as pd
import numpy as np

def create_time_features(df, ts_col="ts_oslo"):
    """Add hour, day-of-week, month and cyclical encodings."""
    d = df.copy()
    d["hour"] = d[ts_col].dt.hour
    d["dayofweek"] = d[ts_col].dt.dayofweek
    d["month"] = d[ts_col].dt.month

    d["hour_sin"] = np.sin(2 * np.pi * d["hour"] / 24)
    d["hour_cos"] = np.cos(2 * np.pi * d["hour"] / 24)
    d["dow_sin"] = np.sin(2 * np.pi * d["dayofweek"] / 7)
    d["dow_cos"] = np.cos(2 * np.pi * d["dayofweek"] / 7)
    d["month_sin"] = np.sin(2 * np.pi * (d["month"] - 1) / 12)
    d["month_cos"] = np.cos(2 * np.pi * (d["month"] - 1) / 12)

    return d


def create_lag_features(df, column, lags=(1, 2, 3, 4)):
    """Add specified lagged versions of a column."""
    d = df.copy()
    for k in lags:
        d[f"{column}_lag_{k}h"] = d[column].shift(k)
    return d
