"""Clean raw hydrology CSV to weekly parquet with derived fields."""
import pandas as pd
import numpy as np
from pathlib import Path
from hydrology.data_loader import OFFICIAL_CAP_TWH


def _detect_week_column(df: pd.DataFrame) -> str:
    """
    Try to find the date/week column in the raw CSV.
    """
    for key in ("uke", "week", "dato", "date"):
        for col in df.columns:
            if key in str(col).lower():
                return col
    return df.columns[0]


def clean_reservoir_data(
    raw_csv_path: Path | str = Path("data/raw/hydro/reservoir.csv"),
    weekly_out_path: Path | str = Path("data/clean/weekly_hydro.parquet"),
    hpi_out_path: Path | str = Path("data/clean/hpi.parquet"),
    latest_table_path: Path | str = Path("outputs/tables/tbl_hpi_latest.csv"),
) -> pd.DataFrame:
    """
    Clean hydrology data, derive seasonal normals/storage, and write weekly parquet(s).
    """
    raw_csv_path = Path(raw_csv_path)
    weekly_out_path = Path(weekly_out_path)
    hpi_out_path = Path(hpi_out_path)
    latest_table_path = Path(latest_table_path)

    if not raw_csv_path.exists():
        raise FileNotFoundError(f"Raw hydrology CSV not found: {raw_csv_path}")

    df = pd.read_csv(raw_csv_path, sep=None, engine="python")

    # Case 1: already tidy long form (week_start, area, reservoir_pct)
    if {"week_start", "area", "reservoir_pct"}.issubset(df.columns):
        tidy = df.copy()
        tidy["week_start"] = pd.to_datetime(tidy["week_start"], errors="coerce")
        tidy["area"] = tidy["area"].astype(str).str.strip().str.upper()
        tidy = tidy.dropna(subset=["week_start", "reservoir_pct", "area"])
    else:
        # Case 2: wide CSV, detect week/date col and melt
        week_col = _detect_week_column(df)
        df = df.rename(columns={week_col: "week_start"})
        df["week_start"] = pd.to_datetime(df["week_start"], errors="coerce", dayfirst=True)
        df = df[df["week_start"].notna()].copy()

        value_cols = [c for c in df.columns if c != "week_start"]
        if not value_cols:
            raise ValueError("No area columns detected in hydrology CSV.")

        tidy = df.melt(
            id_vars="week_start",
            value_vars=value_cols,
            var_name="area",
            value_name="reservoir_pct"
        )
        tidy["area"] = tidy["area"].astype(str).str.strip().str.upper().str.replace(" ", "")
        tidy["reservoir_pct"] = pd.to_numeric(tidy["reservoir_pct"], errors="coerce")
        tidy = tidy[tidy["area"].str.startswith("NO")]
        tidy = tidy.dropna(subset=["reservoir_pct"])

    weekly = (
        tidy.groupby(["week_start", "area"], as_index=False)["reservoir_pct"]
        .mean()
        .sort_values(["area", "week_start"])
        .reset_index(drop=True)
    )

    # ISO fields
    iso = weekly["week_start"].dt.isocalendar()
    weekly["iso_year"] = iso.year.astype(int)
    weekly["iso_week"] = iso.week.astype(int)

    # Capacity and storage (TWh)
    weekly["capacity_twh"] = weekly["area"].map(OFFICIAL_CAP_TWH)
    weekly["storage_twh"] = weekly["capacity_twh"] * (weekly["reservoir_pct"] / 100.0)

    # Seasonal normal and % vs normal
    normals = (
        weekly.groupby(["area", "iso_week"], as_index=False)["reservoir_pct"]
        .mean()
        .rename(columns={"reservoir_pct": "reservoir_pct_normal"})
    )
    weekly = weekly.merge(normals, on=["area", "iso_week"], how="left")
    weekly["pct_vs_normal"] = weekly["reservoir_pct"] / weekly["reservoir_pct_normal"] * 100

    # Compute HPI (global z-score of pct_vs_normal) per area
    stats = weekly.groupby("area")["pct_vs_normal"].agg(["mean", "std"])
    weekly = weekly.join(stats, on="area")
    weekly["HPI_z"] = (weekly["pct_vs_normal"] - weekly["mean"]) / weekly["std"].replace(0, np.nan)

    # Rolling HPI (104-week window, min 26 weeks) per area
    # This captures recent pressure relative to the last ~2 years
    weekly = weekly.sort_values(["area", "week_start"])
    weekly["HPI_z_roll"] = (
        weekly.groupby("area")["pct_vs_normal"]
        .transform(lambda s: (s - s.rolling(104, min_periods=26).mean()) /
                             s.rolling(104, min_periods=26).std())
    )

    # Drop rows with missing critical derived fields and clip percentages
    weekly = weekly.dropna(subset=["reservoir_pct", "reservoir_pct_normal", "pct_vs_normal"])
    weekly["reservoir_pct"] = weekly["reservoir_pct"].clip(0, 100)
    weekly["pct_vs_normal"] = weekly["pct_vs_normal"].clip(lower=0)

    # Clean up temp columns
    weekly = weekly.drop(columns=["mean", "std"])

    weekly_out_path.parent.mkdir(parents=True, exist_ok=True)
    weekly.to_parquet(weekly_out_path, index=False)
    print(f"✔ Saved weekly hydrology parquet → {weekly_out_path}")

    hpi_out_path.parent.mkdir(parents=True, exist_ok=True)
    weekly[["week_start", "area", "HPI_z", "HPI_z_roll"]].to_parquet(hpi_out_path, index=False)
    print(f"✔ Saved HPI parquet (global and rolling) → {hpi_out_path}")

    # Latest-week HPI table (global and rolling)
    latest_week = weekly["week_start"].max()
    hpi_latest = (
        weekly[weekly["week_start"] == latest_week]
        .loc[:, ["area", "reservoir_pct", "pct_vs_normal", "HPI_z", "HPI_z_roll"]]
        .sort_values("HPI_z")
    )
    latest_table_path.parent.mkdir(parents=True, exist_ok=True)
    hpi_latest.to_csv(latest_table_path, index=False)
    print(f"✔ Saved latest HPI table → {latest_table_path}")

    return weekly
