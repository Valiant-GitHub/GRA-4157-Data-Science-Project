"""Basic diagnostic plots for prices and spreads."""
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from modeling.palette import (
    COLOR_PURPLE,
    COLOR_BLUE,
    COLOR_CRIMSON,
    COLOR_ORANGE,
)


# ======================================================================
# PRICE SERIES PLOT
# ======================================================================

def plot_price_series(df, outpath: Path, area: str | None = None):
    """
    Plots DA and mFRR prices over time.
    """
    # Downsample and smooth for an operational timeline: daily means + 7-day rolling
    df_local = df.copy()
    df_local["ts"] = df_local["ts_oslo_tz"].dt.tz_localize(None)
    daily = (
        df_local.set_index("ts")[["price_da", "price_mfrr"]]
        .resample("D")
        .mean(numeric_only=True)
    )
    daily["price_da_roll7"] = daily["price_da"].rolling(7, min_periods=3).mean()
    daily["price_mfrr_roll7"] = daily["price_mfrr"].rolling(7, min_periods=3).mean()

    plt.figure(figsize=(12, 5))
    plt.plot(daily.index, daily["price_da"], label="DA (daily mean)", alpha=0.5, color=COLOR_ORANGE)
    plt.plot(daily.index, daily["price_mfrr"], label="mFRR (daily mean)", alpha=0.35, color=COLOR_PURPLE)
    plt.plot(daily.index, daily["price_da_roll7"], label="DA (7d roll)", alpha=1, color=COLOR_ORANGE, linewidth=1.5)
    plt.plot(daily.index, daily["price_mfrr_roll7"], label="mFRR (7d roll)", alpha=0.7, color=COLOR_PURPLE, linewidth=1.5)

    title_area = f"{area}: " if area else ""
    plt.title(f"{title_area}Day-Ahead vs mFRR Prices (Daily Mean + 7d Roll)")
    plt.xlabel("Time (Oslo)")
    plt.ylabel("EUR/MWh")
    plt.legend()
    plt.tight_layout()

    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=200)
    plt.close()


# ======================================================================
# SPREAD HISTOGRAM
# ======================================================================

def plot_spread_hist(df, outpath: Path):
    """
    Histogram of (mFRR - DA) spreads.
    """
    plt.figure(figsize=(10, 4))
    plt.hist(df["spread"].dropna(), bins=50, alpha=0.8, color=COLOR_CRIMSON)

    plt.title("Distribution of Spread (mFRR - DA)")
    plt.xlabel("Spread [EUR/MWh]")
    plt.ylabel("Frequency")
    plt.tight_layout()

    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=200)
    plt.close()


# ======================================================================
# SIMPLE MODEL OVERLAY (optional)
# ======================================================================

def plot_basic_actual_vs_pred(ts, y_true, y_pred, label, outpath: Path):
    """
    Lightweight overlay plot for quick inspection.
    """
    plt.figure(figsize=(12, 5))
    plt.plot(ts.dt.tz_localize(None), y_true, label="Actual", alpha=0.7, color=COLOR_BLUE)
    plt.plot(ts.dt.tz_localize(None), y_pred, label=label, alpha=0.7, color=COLOR_ORANGE)

    plt.title(f"Actual vs Predicted — {label}")
    plt.xlabel("Time (Oslo)")
    plt.ylabel("Spread [EUR/MWh]")
    plt.legend()
    plt.tight_layout()

    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=200)
    plt.close()
