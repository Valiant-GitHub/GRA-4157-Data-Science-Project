"""Hydrology processing helpers: seasonal normals, HPI, latest tables."""
import pandas as pd
import numpy as np
from .data_loader import OFFICIAL_CAP_TWH

def compute_seasonal_and_hpi(df):
    """
    Adds:
        capacity_twh
        storage_twh
        reservoir_pct_normal
        pct_vs_normal
        HPI
    """
    df = df.copy()
    df["capacity_twh"] = df["area"].map(OFFICIAL_CAP_TWH)
    df["storage_twh"] = df["capacity_twh"] * (df["reservoir_pct"] / 100)

    # Seasonal normal
    normals = df.groupby(["area","iso_week"], as_index=False)["reservoir_pct"].mean()
    normals = normals.rename(columns={"reservoir_pct":"reservoir_pct_normal"})
    df = df.merge(normals, on=["area","iso_week"], how="left")

    df["pct_vs_normal"] = df["reservoir_pct"] / df["reservoir_pct_normal"] * 100

    # HPI (z-score)
    stats = df.groupby("area")["pct_vs_normal"].agg(["mean","std"])
    df = df.join(stats, on="area")
    df["HPI"] = (df["pct_vs_normal"] - df["mean"]) / df["std"].replace(0, np.nan)

    df = df.drop(columns=["mean","std"])
    return df


def latest_week_table(df):
    latest = df["week_start"].max()
    out = df[df["week_start"] == latest].copy()

    return out[[
        "area","reservoir_pct","pct_vs_normal","capacity_twh","storage_twh","HPI"
    ]].sort_values("area")
