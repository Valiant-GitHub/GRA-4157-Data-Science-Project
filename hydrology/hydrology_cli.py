"""Hydrology CLI: download, clean, and plot hydrology data."""
from pathlib import Path
import pandas as pd

from hydrology.hydrology_download import download_reservoir_data
from hydrology.hydrology_clean import clean_reservoir_data
from hydrology.visualization import (
    plot_trend,
    plot_hpi_timeseries,
    plot_weeks_below_normal,
    plot_hpi_bar,
    plot_choropleth,
)


def _load_clean_weekly(path: Path | str = Path("data/clean/weekly_hydro.parquet")) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Clean hydrology parquet not found: {path}")
    return pd.read_parquet(path)


def run_hydrology_cli(args):
    """
    Handle hydrology CLI flags: download/clean and plots.
    """
    if getattr(args, "download", False):
        raw_path = download_reservoir_data()
        clean_reservoir_data(raw_path)

    if args.plots:
        df = _load_clean_weekly()
        # Area-wise plots
        for area in ["NO1", "NO2", "NO3", "NO4", "NO5"]:
            plot_trend(df, area, outdir="outputs/hydro")
            plot_hpi_timeseries(df, area, outdir="outputs/hydro")

        # Global plots
        plot_weeks_below_normal(df, outdir="outputs/hydro")
        plot_hpi_bar(df, outdir="outputs/hydro")
        plot_choropleth(df, outdir="outputs/hydro")

        print("✔ Hydrology plots saved under outputs/hydro")

    if not getattr(args, "download", False) and not args.plots:
        print("Nothing to do. Use --download to fetch/clean, --plots to generate visuals.")
