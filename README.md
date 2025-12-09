# Script Guide (Hydrology + Modeling)

## Prerequisites
- Python 3.11 
- Key packages (from my code usage): `pandas`, `numpy`, `requests`, `matplotlib`, `seaborn`, `geopandas`, `scikit-learn`, `xgboost`
- Ensure your environment can read GeoJSON over HTTPS (for choropleth in hydrology viz).
- Working dirs assume running commands from project root.

## CLI with argparse
Utilize the "--help" function for guidance within the CLI!
python main.py --help
python main.py hydro --help
python main.py forecast --help


## Hydrology pipeline
- Download + clean hydrology (writes raw CSV and clean parquets):
  ```bash
  python main.py hydro --download
  ```
- Generate hydrology plots (reads clean parquet, writes to `outputs/hydro/`):
  ```bash
  python main.py hydro --plots
  ```
- Or both in one go:
  ```bash
  python main.py hydro --download --plots
  ```
- Outputs:
  - `data/raw/hydro/reservoir.csv`
  - `data/clean/weekly_hydro.parquet`
  - `data/clean/hpi.parquet`
  - `outputs/hydro/*.png`
  - `outputs/tables/tbl_hpi_latest.csv`

## Hydrology features (how HPI and reservoir_pct are built)
- `reservoir_pct`: Weekly reservoir fill (%) per NO1–NO5, averaged if duplicates; clipped to 0–100. Derived from NVE weekly data (CSV or API fallback) in `hydrology_clean.py`.
- `reservoir_pct_normal`: Seasonal normal per area/week, computed as the mean reservoir_pct by ISO week across the sample; used as the baseline.
- `pct_vs_normal`: Reservoir_pct / reservoir_pct_normal * 100 (percent of seasonal normal).
- `HPI_z`: Hydro Pressure Index as a z-score of `pct_vs_normal` per area (centered and scaled within each area).
- `HPI_z_roll`: Rolling HPI z-score per area using a 104-week window (min 26) to emphasize the last ~2 years.
- Outputs saved by the hydrology pipeline:
  - `data/clean/weekly_hydro.parquet` (includes reservoir_pct, pct_vs_normal, HPI_z, HPI_z_roll)
  - `data/clean/hpi.parquet` (week_start, area, HPI_z, HPI_z_roll)
  - `outputs/tables/tbl_hpi_latest.csv` (latest week per area with reservoir_pct, pct_vs_normal, HPI_z, HPI_z_roll)


## Forecasting pipeline (modeling)
- Full run with hydrology and plots:
  ```bash
  python main.py forecast --area NO1 --start 2022-01-01 --end 2024-01-01 --hydro yes --allow-lag1 --plots full
  ```
- Flags:
  - `--hydro yes|no` (merge `weekly_hydro.parquet` if available)
  - `--allow-lag1` (train lag1 comparison models)
  - `--plots full|minimal` or `--plot-list a,b,c`
  - `--raw` / `--mfrr` to point to custom DA/mFRR data paths

### Models and horizons
- **Algorithms:**  
  - `Ridge` (linear regression with L2 regularization)  
  - `RandomForest` (ensemble of decision trees)  
  - `XGBoost` (gradient-boosted trees)
- **Horizon variants:**  
  - `primary` — 1-hour ahead using lags 2–4 (no lag1 leak).  
  - `lag1` — 1-hour ahead, comparison model that also uses lag1.  
  - `2h` — 2-hour ahead forecast (target shifted −2h), which is what I mainly used in the paper
  - `3h` — 3-hour ahead forecast (target shifted −3h).  
- Plots overlay the full history and highlight the test window; predictions are shown on the test slice.

## Data expectations
- DA raw: `data/raw/AuctionPriceIndex_*DayAhead*.csv`
- mFRR raw: `data/raw/BalanceMarket_*NO*.csv`
- Hydrology clean: `data/clean/weekly_hydro.parquet` (created by hydrology pipeline)

## Notes
- If hydrology files are missing, the forecast pipeline continues but skips hydrology features (prints a warning).
- Plot directories are created automatically; horizon-specific (i.e. 2h, 3h) plots are stored under `outputs/figures/<horizon>/...`.

## Folder hierarchy (key paths)
- `main.py` — root CLI (forecast, hydro)
- `modeling/` — forecasting pipeline, features, training, plotting and a palette file for colours
- `hydrology/` — hydrology ETL/CLI/viz helpers
  - `hydrology_download.py` — fetch raw hydrology CSV (API fallback)
  - `hydrology_clean.py` — clean + write `data/clean/weekly_hydro.parquet`, `hpi.parquet`
  - `hydrology_cli.py` — hydrology subcommand handler
  - `data_loader.py`, `processing.py`, `visualization.py` — EDA helpers
- `data/raw/` — DA/mFRR raw CSVs; `data/raw/hydro/reservoir.csv` from hydrology download
- `data/clean/` — cleaned datasets (e.g., `weekly_hydro.parquet`, price/hydro joins)
- `outputs/figures/` — forecast plots; subfolders per horizon (primary, lag1, 2h, 3h)
- `outputs/hydro/` — hydrology plots
- `outputs/tables/` — tables (latest-week, etc.)


## Used directly in the report
Run these commands from the project root to regenerate the referenced assets (replace dates as needed):

- `NO1_timeline_2h.png` and `NO1_zoom_2h.png`  
  ```bash
  python main.py forecast --area NO1 --start 2020-01-01 --end 2025-12-08 --hydro yes --plots full
  ```
  Outputs: `outputs/figures/2h/NO1_timeline_2h.png`, `outputs/figures/2h/NO1_zoom_2h.png`

- `NO1_2h_metrics.csv` and `NO1_3h_metrics.csv`  
  ```bash
  python main.py forecast --area NO1 --start 2020-01-01 --end 2025-12-08 --hydro yes --plots minimal
  ```
  Outputs: `outputs/metrics/NO1_2h_metrics.csv`, `outputs/metrics/NO1_3h_metrics.csv`

- `tbl_hpi_latest.csv` (latest hydrology summary)  
  ```bash
  python main.py hydro --download
  ```
  Output: `outputs/tables/tbl_hpi_latest.csv`

