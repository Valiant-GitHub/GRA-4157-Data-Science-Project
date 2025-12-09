"""Visualization helpers for hydrology plots (trend, HPI, map, weeks below normal)."""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import geopandas as gpd
from pathlib import Path
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import Normalize

# Cohesive colour system for hydrology visuals
COLOR_PURPLE = "#b000fb"   # primary accent
COLOR_BLUE = "#a1eafe"     # secondary accent
COLOR_CRIMSON = "#dc143c"  # tertiary accent for emphasis


# ============================================================
# Helper: Save figure to disk
# ============================================================
def savefig(path):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(p, dpi=150, bbox_inches="tight")


# ============================================================
# Calendar-based 24-week window selector 
# ============================================================
def last_24_week_window(df):
    """
    Returns calendar-accurate 24-week window.
    Includes ALL rows with week_start >= cutoff.
    """
    last_date = df["week_start"].max()
    cutoff = last_date - pd.Timedelta(weeks=24)
    return df[df["week_start"] >= cutoff].copy()


# ============================================================
# TREND PLOT 
# ============================================================
def plot_trend(df, area, outdir="outputs/figures"):
    sub = df[df["area"] == area].copy()
    sub = last_24_week_window(sub)

    plt.figure(figsize=(6, 3))
    plt.plot(sub["week_start"], sub["reservoir_pct"],
             label="Reservoir fill (%)", color=COLOR_BLUE)
    plt.plot(sub["week_start"], sub["pct_vs_normal"],
             label="% of seasonal normal", color=COLOR_PURPLE)

    plt.title(f"{area}: Reservoir & vs. normal (last 24 weeks)")
    plt.xlabel("Week")
    plt.ylabel("Percent")
    plt.legend()
    plt.grid(alpha=0.25)

    savefig(f"{outdir}/fig_timeseries_{area}.png")
    plt.close()


# ============================================================
# HPI TIME SERIES 
# ============================================================
def plot_hpi_timeseries(df, area, outdir="outputs/figures"):
    sub = df[df["area"] == area].copy()
    sub = last_24_week_window(sub)

    hl_color = COLOR_BLUE
    ref_color = COLOR_PURPLE
    roll_color = COLOR_CRIMSON

    plt.figure(figsize=(6, 3))
    # Global z
    if "HPI_z" in sub.columns:
        plt.plot(sub["week_start"], sub["HPI_z"],
                 label="HPI (global z)", color=hl_color)

    # Rolling z (optional)
    if "HPI_z_roll" in sub.columns and sub["HPI_z_roll"].notna().any():
        plt.plot(sub["week_start"], sub["HPI_z_roll"],
                 label="HPI (rolling z)", color=roll_color, linestyle="--")

    plt.axhline(0,  color=ref_color, linestyle="--", linewidth=1)
    plt.axhline(1,  color=ref_color, linestyle=":",  linewidth=1)
    plt.axhline(-1, color=ref_color, linestyle=":",  linewidth=1)

    plt.title(f"{area}: HPI (last 24 weeks)")
    plt.xlabel("Week")
    plt.ylabel("HPI")
    plt.legend()
    plt.grid(alpha=0.25)

    savefig(f"{outdir}/fig_hpi_ts_{area}.png")
    plt.close()


# ============================================================
# WEEKS BELOW NORMAL
# ============================================================
def plot_weeks_below_normal(df, outdir="outputs/figures"):
    wk = df.assign(below=(df["pct_vs_normal"] < 100).astype(int))
    summary = wk.groupby("area", as_index=False)["below"].sum()

    plt.figure(figsize=(6, 3.5))
    plt.barh(summary["area"], summary["below"], color=COLOR_CRIMSON)

    plt.title("Weeks below seasonal normal (since start of sample)")
    plt.xlabel("Weeks")
    plt.grid(axis="x", alpha=0.25)

    savefig(f"{outdir}/fig_weeks_below_normal.png")
    plt.close()


# ============================================================
# HPI BAR CHART
# ============================================================
def plot_hpi_bar(df, outdir="outputs/figures"):
    latest = df["week_start"].max()
    sub = df[df["week_start"] == latest].copy()

    plt.figure(figsize=(6, 3.5))
    # Prefer global HPI_z and include rolling if available
    have_global = "HPI_z" in sub.columns
    have_roll = "HPI_z_roll" in sub.columns and sub["HPI_z_roll"].notna().any()

    if not have_global and "HPI" in sub.columns:
        sub["HPI_z"] = sub["HPI"]  # fallback
        have_global = True

    if not have_global and not have_roll:
        print("⚠ No HPI column found; skipping HPI bar chart.")
        return

    # Global bar
    if have_global:
        plt.barh(sub["area"], sub["HPI_z"], color=COLOR_PURPLE, label="HPI (global z)")

    # Rolling bar
    if have_roll:
        plt.barh(sub["area"], sub["HPI_z_roll"], color=COLOR_CRIMSON, alpha=0.7, label="HPI (rolling z)")

    plt.title(f"Hydro Pressure Index (z-score)\nLatest week: {latest.date()}")
    plt.xlabel("HPI (z-score)")
    plt.grid(axis="x", alpha=0.25)
    plt.legend()

    savefig(f"{outdir}/fig_hpi_bar.png")
    plt.close()


# ============================================================
# CHOROPLETH MAP 
# ============================================================
def plot_choropleth(df, outdir="outputs/figures"):
    latest_week = df["week_start"].max()
    latest = df[df["week_start"] == latest_week][["area", "pct_vs_normal"]]

    url = (
        "https://nve.geodataonline.no/arcgis/rest/services/Mapservices/Elspot/"
        "MapServer/0/query?where=1%3D1&outFields=*&f=geojson&outSR=4326"
    )
    zones = gpd.read_file(url).to_crs(4326)

    col = "ElSpotOmr" if "ElSpotOmr" in zones.columns else \
        next(c for c in zones.columns if "ElSpot" in c)

    zones["area"] = zones[col].str.replace(" ", "", regex=False).str.upper()
    g = zones.merge(latest, on="area", how="left").to_crs(3857)

    # Notebook-like colour range: muted "cool" map
    norm = Normalize(vmin=90, vmax=110)

    fig, ax = plt.subplots(figsize=(6, 8))
    m = g.plot(
        column="pct_vs_normal",
        cmap="cool",
        norm=norm,
        edgecolor="white",
        linewidth=0.7,
        legend=True,
        ax=ax
    )

    cbar = m.get_figure().get_axes()[-1]
    cbar.set_ylabel("% of Seasonal Normal", fontsize=11)
    cbar.tick_params(labelsize=10)

    ax.set_aspect("auto")
    ax.set_axis_off()

    # Label zones
    g["pt"] = g.geometry.representative_point()
    for _, row in g.iterrows():
        if pd.isna(row["pct_vs_normal"]):
            label = row["area"]
        else:
            label = f"{row['area']}\n{row['pct_vs_normal']:.0f}%"

        ax.text(
            row["pt"].x, row["pt"].y,
            label, ha="center", va="center",
            fontsize=11, weight="bold"
        )

    plt.title(
        f"% of Seasonal Normal – Latest Week: {latest_week.date()}",
        fontsize=14, pad=15
    )

    savefig(f"{outdir}/map_pct_vs_normal.png")
    plt.close()
