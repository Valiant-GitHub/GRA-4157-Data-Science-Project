"""Forecasting pipeline: data prep, feature creation, model training, plotting."""
import pandas as pd
import numpy as np
from pathlib import Path
import os

from data_preparation import prepare_mspread
from features import create_time_features, create_lag_features
from train_models import (
    train_primary_models, 
    train_lag1_models, 
    train_two_hour_models,
    train_three_hour_models
)

from diagnostics import (
    plot_price_series,
    plot_spread_hist
)

from diagnostics_advanced import (
    plot_full_timeline,
    plot_zoom_week,
    plot_residual_diagnostics,
    plot_model_scatter,
    plot_feature_importance_plots,
    plot_combined_feature_importance,
    plot_heatmap_month_hour
)

from metrics_utils import (
    save_metrics_csv,
    save_feature_importances
)



# ======================================================================
# PLOTTING DISPATCHER — FULL / MINIMAL / CUSTOM
# ======================================================================

def generate_plots(args, results):
    """
    Routes plotting requests:
        --plots full        → ALL plots
        --plots minimal     → Only key plots
        --plot-list a,b,c   → User-specified subset
    """

    area = results["area"]
    mspread = results["mspread"]

    primary = results["primary"]
    lag1    = results["lag1"]
    h2      = results["h2"]
    h3      = results.get("h3")

    feature_cols_primary = results["feature_cols_primary"]
    feature_cols_lag1    = results["feature_cols_lag1"]
    feature_cols_h2      = results["feature_cols_h2"]
    feature_cols_h3      = results.get("feature_cols_h3")

    # Output directories
    figdir = Path("outputs/figures")
    figdir.mkdir(parents=True, exist_ok=True)

    def horizon_dir(label: str) -> Path:
        """
        Place horizon-specific plots into subfolders:
        outputs/figures/<label>/...
        """
        return figdir / label

    # --------------------------------------------------------------
    # DEFINE PLOT GROUPS (mapping names → functions)
    # --------------------------------------------------------------
    def run_basic_plots():
        plot_price_series(mspread, figdir / "DA_vs_mFRR_Dmean_7dRoll.png", area=area)
        plot_spread_hist(mspread, figdir / f"{area}_spread_hist.png")

    def run_full_models_plots(label, block, feature_cols):
        """
        block = {
            'ts': ...,
            'y_true': ...,
            'ridge': ...,
            'rf': ...,
            'xgb': ...
        }
        """
        subdir = horizon_dir(label)
        plot_full_timeline(block, area, subdir, label)
        plot_zoom_week(block, area, subdir, label)
        plot_residual_diagnostics(block, area, subdir, label)
        plot_model_scatter(block, area, subdir, label)
        plot_feature_importance_plots(
            area, subdir, feature_cols, label
        )
        plot_combined_feature_importance(area, subdir, label)

    def run_heatmap():
        plot_heatmap_month_hour(mspread, figdir / f"{area}_heatmap_month_hour.png")

    # --------------------------------------------------------------
    # Custom plot-list parsing
    # --------------------------------------------------------------
    if args.plot_list is not None:
        plot_items = [p.strip().lower() for p in args.plot_list.split(",")]
    else:
        plot_items = None

    # --------------------------------------------------------------
    # PLOTTING LOGIC
    # --------------------------------------------------------------

    print("\n🎨 Generating plots ...")

    if plot_items is not None:
        print(f"📌 Custom plot selection: {plot_items}")

        if "basic" in plot_items:
            run_basic_plots()

        if "primary" in plot_items:
            run_full_models_plots("primary", primary, feature_cols_primary)

        if "lag1" in plot_items and lag1 is not None:
            run_full_models_plots("lag1", lag1, feature_cols_lag1)

        if "2h" in plot_items:
            run_full_models_plots("2h", h2, feature_cols_h2)

        if "3h" in plot_items and h3 is not None:
            run_full_models_plots("3h", h3, feature_cols_h3)

        if "heatmap" in plot_items:
            run_heatmap()

        print("✔ Custom plotting complete.")
        return

    # --------------------------------------------------------------
    # MINIMAL MODE
    # --------------------------------------------------------------
    if args.plots == "minimal":
        print("📉 Minimal plotting mode: basic plots only.")
        run_basic_plots()
        return

    # --------------------------------------------------------------
    # FULL MODE
    # --------------------------------------------------------------
    print("🎨 FULL plotting mode: generating ALL figures ...")

    run_basic_plots()
    run_full_models_plots("primary", primary, feature_cols_primary)

    if lag1 is not None:
        run_full_models_plots("lag1", lag1, feature_cols_lag1)

    run_full_models_plots("2h", h2, feature_cols_h2)

    if h3 is not None:
        run_full_models_plots("3h", h3, feature_cols_h3)

    run_heatmap()

    print("✔ All plots saved.")


# ======================================================================
# PIPELINE FINISHER: CALL PLOTTER AFTER TRAINING
# ======================================================================

def run_pipeline_cli(args):
    """Top-level pipeline wrapper that runs training then plots."""
    results = run_pipeline_cli_inner(args)  # inner function executes full pipeline
    generate_plots(args, results)


# ======================================================================
# REFACTOR — shift main logic into inner function
# ======================================================================

def run_pipeline_cli_inner(args):
    """Full forecasting engine: load, prep, feature-engineer, train, save, return results."""
    # ======================================================================
    # LOADERS — DA + mFRR (whole folder ingestion)
    # ======================================================================

    def load_day_ahead_all(raw_dir):
        """
        Load all DA CSVs (AuctionPriceIndex_*DayAhead*) and convert to standardized format.
        """
        raw_dir = Path(raw_dir)
        files = sorted(raw_dir.glob("AuctionPriceIndex_*DayAhead*.csv"))
        if not files:
            raise FileNotFoundError(f"No DA files found in {raw_dir}")

        frames = []
        for f in files:
            df = pd.read_csv(f, sep=";", engine="python")
            df.columns = [c.strip() for c in df.columns]

            c_start = [c for c in df.columns if "Delivery Start" in c][0]
            price_cols = [c for c in df.columns if "Price" in c and "NO" in c]

            long = df.melt(
                id_vars=[c_start],
                value_vars=price_cols,
                var_name="area_col",
                value_name="price",
            )
            long["area"] = long["area_col"].str.extract(r"(NO[1-5])")
            long["ts_utc"] = pd.to_datetime(
                long[c_start],
                utc=True,
                errors="coerce",
                dayfirst=True
            )
            long["ts_oslo"] = long["ts_utc"].dt.tz_convert("Europe/Oslo")
            long["price"] = pd.to_numeric(long["price"], errors="coerce")

            long = long.dropna(subset=["ts_oslo", "price", "area"])
            frames.append(long[["ts_oslo", "area", "price"]])

        raw = pd.concat(frames, ignore_index=True)
        raw = raw.sort_values(["area", "ts_oslo"]).reset_index(drop=True)
        return raw


    def load_mfrr_all(raw_dir):
        """
        Load all BalanceMarket CSVs for mFRR.
        """
        raw_dir = Path(raw_dir)
        files = sorted(raw_dir.glob("BalanceMarket_*NO*.csv"))
        if not files:
            raise FileNotFoundError(f"No mFRR files found in {raw_dir}")

        parts = []
        for f in files:
            df = pd.read_csv(f, sep=";", engine="python")
            c_start = next(c for c in df.columns if "Delivery Start" in c)
            val_cols = [c for c in df.columns if c != c_start]

            long = df.melt(
                id_vars=[c_start],
                value_vars=val_cols,
                var_name="area_metric",
                value_name="value",
            )
            long[["area", "metric"]] = long["area_metric"].str.extract(r"(NO[1-5])\s+(.*)")
            long["value"] = pd.to_numeric(long["value"], errors="coerce")
            long["ts_start_oslo"] = pd.to_datetime(
                long[c_start],
                format="%d.%m.%Y %H:%M:%S",
                errors="coerce"
            )
            long = long.dropna(subset=["area", "metric", "ts_start_oslo", "value"])
            parts.append(long[["area", "metric", "value", "ts_start_oslo"]])

        long_all = pd.concat(parts, ignore_index=True)

        # Remove duplicates before aggregation (notebook logic)
        long_all = (
            long_all.groupby(["area", "ts_start_oslo", "metric"], as_index=False)
            .mean(numeric_only=True)
        )

        # Keep only Imbalance Price rows and tidy directly (avoid fragile pivot/melt)
        price_rows = long_all[long_all["metric"].str.contains("Imbalance Price", na=False)]
        if price_rows.empty:
            raise RuntimeError("No 'Imbalance Price' columns detected in mFRR data.")

        price_rows = price_rows.rename(columns={"value": "price_mfrr", "ts_start_oslo": "ts_oslo"})
        price_rows["ts_oslo"] = pd.to_datetime(price_rows["ts_oslo"], errors="coerce")
        price_rows = price_rows.dropna(subset=["price_mfrr", "ts_oslo", "area"])
        price_rows = price_rows.sort_values("ts_oslo").reset_index(drop=True)

        return price_rows[["ts_oslo", "area", "price_mfrr"]]


    # ======================================================================
    # FULL PIPELINE FUNCTION
    # ======================================================================

    def run_pipeline_cli(args):
        """
        MAIN FORECASTING ENGINE — called by main.py
        Always runs:
        • primary (lag2) model set
        • lag1 comparison model set
        • 2h-ahead model set
        • saves ALL metrics + feature importances
        • generates ALL plots (if requested)
        """

        area = args.area
        start_date = pd.to_datetime(args.start)
        end_date   = pd.to_datetime(args.end)

        print(f"\n=== RUNNING FULL PIPELINE FOR {area} ===")
        print(f"Date range: {start_date.date()} → {end_date.date()}")
        print(f"Hydrology: {args.hydro.upper()}")
        print(f"Lag1 comparison enabled: {args.allow_lag1}")
        print(f"Plotting mode: {args.plots}")

        # ----------------------------------------------------------
        # LOAD RAW DATA
        # ----------------------------------------------------------
        print("\n📡 Loading Day-Ahead (DA) data ...")
        raw = load_day_ahead_all(args.raw)

        print("📡 Loading mFRR data ...")
        mfrr = load_mfrr_all(args.mfrr)

        # ----------------------------------------------------------
        # FIX TIMEZONE + DATE FILTERING
        # ----------------------------------------------------------
        raw["ts_oslo"]  = pd.to_datetime(raw["ts_oslo"]).dt.tz_localize(None)
        mfrr["ts_oslo"] = pd.to_datetime(mfrr["ts_oslo"]).dt.tz_localize(None)

        raw  = raw[(raw["ts_oslo"] >= start_date) & (raw["ts_oslo"] <= end_date)]
        mfrr = mfrr[(mfrr["ts_oslo"] >= start_date) & (mfrr["ts_oslo"] <= end_date)]

        print(f"✔ DA rows kept: {len(raw)}")
        print(f"✔ mFRR rows kept: {len(mfrr)}")

        # ----------------------------------------------------------
        # BUILD MSPREAD
        # ----------------------------------------------------------
        print("\n🔧 Building mspread ...")
        mspread = prepare_mspread(raw, mfrr, area=area)

        # ----------------------------------------------------------
        # FEATURE ENGINEERING (time + lags)
        # ----------------------------------------------------------
        print("🔧 Creating time features ...")
        mspread = create_time_features(mspread)

        # Primary lags (lag2–4)
        primary_lags = (2, 3, 4)
        mspread_primary = create_lag_features(mspread.copy(), "spread", lags=primary_lags)
        mspread_primary = mspread_primary.dropna(subset=[f"spread_lag_{k}h" for k in primary_lags])

        # Lag1 comparison
        if args.allow_lag1:
            print("🔧 Creating lag1 comparison dataset ...")
            lag1_lags = (1, 2, 3, 4)
            mspread_lag1 = create_lag_features(mspread.copy(), "spread", lags=lag1_lags)
            mspread_lag1 = mspread_lag1.dropna(subset=[f"spread_lag_{k}h" for k in lag1_lags])
        else:
            mspread_lag1 = None

        # ----------------------------------------------------------
        # OPTIONAL HYDROLOGY MERGE
        # ----------------------------------------------------------
        if args.hydro == "yes":
            hydro_file = Path("data/clean/weekly_hydro.parquet")

            if hydro_file.exists():
                print("💧 Merging hydrology (reservoir_pct, HPI) ...")

                hydro = pd.read_parquet(hydro_file)
                hydro["week_start"] = pd.to_datetime(hydro["week_start"])
                hydro["week_period"] = hydro["week_start"].dt.to_period("W-MON")

                # Ensure legacy HPI name is available; prefer HPI_z if present
                if "HPI" not in hydro.columns:
                    if "HPI_z" in hydro.columns:
                        hydro = hydro.copy()
                        hydro["HPI"] = hydro["HPI_z"]
                    else:
                        # No HPI-style column; merge only reservoir_pct
                        print("⚠ No HPI column found in hydro data — merging reservoir_pct only.")

                # Attach weekly hydrology to mspread
                def merge_hydro(df):
                    temp = df.copy()
                    temp["week_period"] = temp["ts_oslo"].dt.to_period("W-MON")
                    merge_cols = ["area", "week_period", "reservoir_pct"]
                    if "HPI" in hydro.columns:
                        merge_cols.append("HPI")
                    if "HPI_z_roll" in hydro.columns:
                        merge_cols.append("HPI_z_roll")

                    merged = temp.merge(
                        hydro[merge_cols],
                        on=["area", "week_period"],
                        how="left",
                    )
                    return merged.drop(columns=["week_period"])

                mspread_primary = merge_hydro(mspread_primary)
                if mspread_lag1 is not None:
                    mspread_lag1 = merge_hydro(mspread_lag1)
            else:
                print("⚠ Hydrology file NOT AVAILABLE — continuing without hydrology")

        # ==================================================================
        # TRAIN MODELS — PRIMARY (lag2–4)
        # ==================================================================
        print("\n🤖 Training PRIMARY models (lags 2–4) ...")

        primary_results = train_primary_models(
            mspread_primary,
            hydro_enabled=(args.hydro == "yes")
        )

        (
            ridge_primary,
            rf_primary,
            xgb_primary,
            naive_primary,
            metrics_primary,
            X_test_primary,
            y_test_primary,
            y_pred_ridge_primary,
            y_pred_rf_primary,
            y_pred_xgb_primary,
            ts_test_primary,
            feature_cols_primary,
            df_primary_model,
            split_idx_primary
        ) = primary_results

        print("\n✅ PRIMARY MODEL METRICS:")
        for k, v in metrics_primary.items():
            print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")


        # ==================================================================
        # TRAIN MODELS — LAG1 COMPARISON (if allowed)
        # ==================================================================
        if mspread_lag1 is not None:
            print("\n🤖 Training LAG1 comparison models (lags 1–4) ...")

            lag1_results = train_lag1_models(
                mspread_lag1,
                hydro_enabled=(args.hydro == "yes")
            )

            (
                ridge_lag1,
                rf_lag1,
                xgb_lag1,
                naive_lag1,
                metrics_lag1,
                X_test_lag1,
                y_test_lag1,
                y_pred_ridge_lag1,
                y_pred_rf_lag1,
                y_pred_xgb_lag1,
                ts_test_lag1,
                feature_cols_lag1,
                df_lag1_model,
                split_idx_lag1
            ) = lag1_results

            print("\n📊 LAG1 MODEL METRICS:")
            for k, v in metrics_lag1.items():
                print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
        else:
            lag1_results = None
            metrics_lag1 = None
            feature_cols_lag1 = None


        # ==================================================================
        # TRAIN MODELS — 2-HOUR AHEAD MODEL
        # ==================================================================
        print("\n⏩ Training 2-hour ahead forecasting models ...")

        h2_results = train_two_hour_models(
            mspread_primary,
            hydro_enabled=(args.hydro == "yes")
        )

        (
            ridge_h2,
            rf_h2,
            xgb_h2,
            naive_h2,
            metrics_h2,
            X_test_h2,
            y_test_h2,
            y_pred_ridge_h2,
            y_pred_rf_h2,
            y_pred_xgb_h2,
            ts_test_h2,
            feature_cols_h2,
            df_h2_model,
            split_idx_h2
        ) = h2_results

        print("\n📊 2H-AHEAD MODEL METRICS:")
        for k, v in metrics_h2.items():
            print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

        # ==================================================================
        # TRAIN MODELS — 3-HOUR AHEAD MODEL
        # ==================================================================
        print("\n⏩ Training 3-hour ahead forecasting models ...")

        h3_results = train_three_hour_models(
            mspread_primary,
            hydro_enabled=(args.hydro == "yes")
        )

        (
            ridge_h3,
            rf_h3,
            xgb_h3,
            naive_h3,
            metrics_h3,
            X_test_h3,
            y_test_h3,
            y_pred_ridge_h3,
            y_pred_rf_h3,
            y_pred_xgb_h3,
            ts_test_h3,
            feature_cols_h3,
            df_h3_model,
            split_idx_h3
        ) = h3_results

        print("\n📊 3H-AHEAD MODEL METRICS:")
        for k, v in metrics_h3.items():
            print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")


        # ==================================================================
        # SAVE METRICS CSVs
        # ==================================================================
        out_metrics = Path("outputs/metrics")
        out_metrics.mkdir(parents=True, exist_ok=True)

        print("\n💾 Saving metrics CSVs ...")
        save_metrics_csv(area, metrics_primary, kind="primary")
        if metrics_lag1:
            save_metrics_csv(area, metrics_lag1, kind="lag1")
        save_metrics_csv(area, metrics_h2, kind="2h")
        save_metrics_csv(area, metrics_h3, kind="3h")

        print("✔ Metrics saved.")


        # ==================================================================
        # SAVE FEATURE IMPORTANCES
        # ==================================================================
        out_import = Path("outputs/feature_importances")
        out_import.mkdir(parents=True, exist_ok=True)

        print("\n💾 Saving feature importances ...")

        save_feature_importances(
            area=area,
            ridge_model=ridge_primary,
            rf_model=rf_primary,
            xgb_model=xgb_primary,
            feature_cols=feature_cols_primary,
            kind="primary"
        )

        if lag1_results is not None:
            save_feature_importances(
                area=area,
                ridge_model=ridge_lag1,
                rf_model=rf_lag1,
                xgb_model=xgb_lag1,
                feature_cols=feature_cols_lag1,
                kind="lag1"
            )

        save_feature_importances(
            area=area,
            ridge_model=ridge_h2,
            rf_model=rf_h2,
            xgb_model=xgb_h2,
            feature_cols=feature_cols_h2,
            kind="2h"
        )

        save_feature_importances(
            area=area,
            ridge_model=ridge_h3,
            rf_model=rf_h3,
            xgb_model=xgb_h3,
            feature_cols=feature_cols_h3,
            kind="3h"
        )

        print("✔ Feature importances saved.")


        # ==================================================================
        # RETURN DATA FOR PLOTTING
        # ==================================================================
        results = {
            "area": area,
            "primary": {
                "ts": ts_test_primary,
                "y_true": y_test_primary,
                "ridge": y_pred_ridge_primary,
                "rf": y_pred_rf_primary,
                "xgb": y_pred_xgb_primary,
                "ts_full": df_primary_model["ts_oslo_tz"],
                "y_full": df_primary_model["spread"],
                "split_idx": split_idx_primary
            },
            "lag1": None if lag1_results is None else {
                "ts": ts_test_lag1,
                "y_true": y_test_lag1,
                "ridge": y_pred_ridge_lag1,
                "rf": y_pred_rf_lag1,
                "xgb": y_pred_xgb_lag1,
                "ts_full": df_lag1_model["ts_oslo_tz"],
                "y_full": df_lag1_model["spread"],
                "split_idx": split_idx_lag1
            },
            "h2": {
                "ts": ts_test_h2,
                "y_true": y_test_h2,
                "ridge": y_pred_ridge_h2,
                "rf": y_pred_rf_h2,
                "xgb": y_pred_xgb_h2,
                "ts_full": df_h2_model["ts_oslo_tz"],
                "y_full": df_h2_model["spread_h2"],
                "split_idx": split_idx_h2
            },
            "h3": {
                "ts": ts_test_h3,
                "y_true": y_test_h3,
                "ridge": y_pred_ridge_h3,
                "rf": y_pred_rf_h3,
                "xgb": y_pred_xgb_h3,
                "ts_full": df_h3_model["ts_oslo_tz"],
                "y_full": df_h3_model["spread_h3"],
                "split_idx": split_idx_h3
            },
            "mspread": mspread_primary,
            "feature_cols_primary": feature_cols_primary,
            "feature_cols_lag1": feature_cols_lag1,
            "feature_cols_h2": feature_cols_h2,
            "feature_cols_h3": feature_cols_h3
        }

        return results

    # Execute the nested pipeline function and return its results.
    return run_pipeline_cli(args)
