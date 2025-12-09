"""Utilities for saving model metrics and feature importances."""
import pandas as pd
from pathlib import Path


# ==========================================================
# SAVE METRICS FOR EACH MODEL SET (primary, lag1, 2h)
# ==========================================================

def save_metrics_csv(area, metrics_dict, kind):
    """
    Saves model performance metrics to:
        outputs/metrics/<area>_<kind>_metrics.csv
    """
    outdir = Path("outputs/metrics")
    outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    for model_name, metric_values in metrics_dict.items():
        row = {"model": model_name}
        if isinstance(metric_values, dict):
            row.update(metric_values)
        else:
            # Support flat dicts where values are scalar (e.g., "ridge_mae": 1.23)
            row["value"] = metric_values
        rows.append(row)

    df = pd.DataFrame(rows)
    outfile = outdir / f"{area}_{kind}_metrics.csv"
    df.to_csv(outfile, index=False)
    print(f"✔ Saved metrics → {outfile}")


# ==========================================================
# SAVE FEATURE IMPORTANCES FOR RIDGE + RF + XGB
# ==========================================================

def save_feature_importances(area, ridge_model, rf_model, xgb_model, feature_cols, kind):
    """
    Saves feature importance CSVs for:
      - Ridge coefficients
      - RandomForest feature_importances_
      - XGBoost feature_importances_
    
    Files saved into:
        outputs/feature_importances/<area>_ridge_coef_<kind>.csv
        outputs/feature_importances/<area>_rf_importance_<kind>.csv
        outputs/feature_importances/<area>_xgb_importance_<kind>.csv
    """
    outdir = Path("outputs/feature_importances")
    outdir.mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # Ridge Coefficients
    # -----------------------------
    try:
        ridge_step = ridge_model.named_steps["ridge"]
        coefs = ridge_step.coef_
        df_ridge = pd.DataFrame({
            "feature": feature_cols,
            "coef": coefs
        })
        df_ridge.to_csv(outdir / f"{area}_ridge_coef_{kind}.csv", index=False)
    except Exception as e:
        print(f"⚠ Ridge feature save skipped: {e}")

    # -----------------------------
    # Random Forest Importances
    # -----------------------------
    try:
        rf_imp = rf_model.feature_importances_
        df_rf = pd.DataFrame({
            "feature": feature_cols,
            "importance": rf_imp
        })
        df_rf.to_csv(outdir / f"{area}_rf_importance_{kind}.csv", index=False)
    except Exception as e:
        print(f"⚠ RF feature save skipped: {e}")

    # -----------------------------
    # XGBoost Importances
    # -----------------------------
    try:
        xgb_imp = xgb_model.feature_importances_
        df_xgb = pd.DataFrame({
            "feature": feature_cols,
            "importance": xgb_imp
        })
        df_xgb.to_csv(outdir / f"{area}_xgb_importance_{kind}.csv", index=False)
    except Exception as e:
        print(f"⚠ XGB feature save skipped: {e}")

    print(f"✔ Saved feature importances ({kind})")
