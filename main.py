"""Root CLI entrypoint for forecasting and hydrology tasks."""
import argparse
import sys
from pathlib import Path

# ----------------------------------------------------------
# Ensure modeling/ is on Python path
# ----------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
MODELING_DIR = PROJECT_ROOT / "modeling"
HYDROLOGY_DIR = PROJECT_ROOT / "hydrology"

for p in (MODELING_DIR, HYDROLOGY_DIR):
    if str(p) not in sys.path:
        sys.path.append(str(p))

# Full forecasting pipeline
from run_pipeline_cli import run_pipeline_cli


# ==========================================================
# Root CLI — dispatch commands
# ==========================================================
def main():
    """Dispatch CLI subcommands for forecasting and hydrology ETL/plots."""

    parser = argparse.ArgumentParser(
        prog="GRA4157 Forecasting System",
        description="Full mFRR–DA forecasting pipeline with ML models, diagnostics, and hydrology.",
    )

    subparsers = parser.add_subparsers(dest="command")

    # ------------------------------------------------------
    # forecast — MAIN ENTRYPOINT FOR THE WHOLE PIPELINE
    # ------------------------------------------------------
    forecast = subparsers.add_parser(
        "forecast",
        help="Run the FULL mFRR–DA forecasting pipeline (all models, all diagnostics)."
    )

    forecast.add_argument("--area", type=str, default="NO1",
        help="Price area (NO1–NO5). Default = NO1.")

    forecast.add_argument("--start", type=str, required=True,
        help="Start date (YYYY-MM-DD).")

    forecast.add_argument("--end", type=str, required=True,
        help="End date (YYYY-MM-DD).")

    forecast.add_argument("--hydro", type=str, choices=["yes", "no"], default="no",
        help="Include hydrology dataset (reservoir_pct, HPI). Default=no.")

    forecast.add_argument("--allow-lag1", action="store_true",
        help="Include lag-1 model for comparison. Default = False.")

    forecast.add_argument("--plots", type=str, choices=["full", "minimal"], default="full",
        help="Set plotting mode. 'full' = all plots. 'minimal' = key plots only.")

    forecast.add_argument("--plot-list", type=str, default=None,
        help="Comma-separated list of specific plots to generate. Overrides --plots.")

    # Optional: custom paths — mostly for testing
    forecast.add_argument("--raw", type=str, default="data/raw",
        help="Path to raw Day-Ahead folder or file.")

    forecast.add_argument("--mfrr", type=str, default="data/raw",
        help="Path to raw mFRR folder or file.")

    # ------------------------------------------------------
    # hydrology — update/plot hydrology datasets
    # ------------------------------------------------------
    hydro = subparsers.add_parser(
        "hydro",
        help="Hydrology ETL + plots (download, clean, plot)."
    )
    hydro.add_argument("--download", action="store_true",
        help="Download and clean hydrology data.")
    hydro.add_argument("--plots", action="store_true",
        help="Generate hydrology-only plots into outputs/hydro.")

    args = parser.parse_args()

    # ------------------------------------------------------
    # Dispatch to the pipeline
    # ------------------------------------------------------
    if args.command == "forecast":
        run_pipeline_cli(args)
    elif args.command == "hydro":
        from hydrology_cli import run_hydrology_cli
        run_hydrology_cli(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
