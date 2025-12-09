"""Advanced diagnostic plots for model timelines, residuals, scatter, importances."""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from modeling.palette import (
    COLOR_PURPLE,
    COLOR_BLUE,
    COLOR_CRIMSON,
    COLOR_ORANGE,
    COLOR_SHADE,
)


# ======================================================================
# HELPER: Save figure
# ======================================================================
def _savefig(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=200)
    plt.close()


# ======================================================================
# FULL TIMELINE PLOT (primary, lag1, 2h)
# ======================================================================
def plot_full_timeline(model_block, area, figdir, label):
    """
    model_block = {
        'ts': ...,
        'y_true': ...,
        'ridge': ...,
        'rf': ...,
        'xgb': ...
    }
    label: 'primary' or 'lag1' or '2h'
    """

    ts = model_block["ts"].dt.tz_localize(None)
    y = model_block["y_true"]

    ts_full = model_block.get("ts_full")
    y_full = model_block.get("y_full")
    split_idx = model_block.get("split_idx")
    if ts_full is not None and y_full is not None:
        ts_full = ts_full.dt.tz_localize(None)

    plt.figure(figsize=(14, 5))

    if ts_full is not None and y_full is not None:
        plt.plot(ts_full, y_full, label="Actual (full)", alpha=0.35, color=COLOR_BLUE)

    # Highlight test window
    if ts_full is not None and split_idx is not None and split_idx < len(ts_full):
        start_test = ts_full.iloc[split_idx]
        end_test = ts_full.iloc[-1]
        plt.axvspan(start_test, end_test, color=COLOR_SHADE, alpha=0.15, label="Test window")

    # Test slice actuals (y) and predictions
    plt.plot(ts, y, label="Actual (test)", alpha=0.9, color=COLOR_BLUE)

    plt.plot(ts, model_block["ridge"], label="Ridge", alpha=0.8, color=COLOR_PURPLE)
    plt.plot(ts, model_block["rf"], label="RandomForest", alpha=0.75, color=COLOR_ORANGE)
    plt.plot(ts, model_block["xgb"], label="XGBoost", alpha=0.5, color=COLOR_CRIMSON)

    plt.title(f"Full Timeline — {label.upper()} Models ({area})")
    plt.xlabel("Time (Oslo)")
    plt.ylabel("Spread [EUR/MWh]")
    plt.legend()
    plt.tight_layout()

    _savefig(figdir / f"{area}_timeline_{label}.png")


# ======================================================================
# ZOOM WEEK (last 7 days of test set)
# ======================================================================
def plot_zoom_week(model_block, area, figdir, label, days=7):
    ts = model_block["ts"]
    y = model_block["y_true"]

    n = days * 24
    ts_zoom = ts.iloc[-n:].dt.tz_localize(None)
    y_zoom = y[-n:]

    plt.figure(figsize=(14, 4))
    plt.plot(ts_zoom, y_zoom, label="Actual", alpha=0.85, color=COLOR_BLUE)
    plt.plot(ts_zoom, model_block["ridge"][-n:], label="Ridge", alpha=0.85, color=COLOR_PURPLE)
    plt.plot(ts_zoom, model_block["rf"][-n:], label="RandomForest", alpha=0.8, color=COLOR_ORANGE)
    plt.plot(ts_zoom, model_block["xgb"][-n:], label="XGBoost", alpha=0.8, color=COLOR_CRIMSON)

    plt.title(f"{days}-Day Zoom — {label.upper()} Models ({area})")
    plt.xlabel("Time (Oslo)")
    plt.ylabel("Spread [EUR/MWh]")
    plt.legend()
    plt.tight_layout()

    _savefig(figdir / f"{area}_zoom_{label}.png")

# ======================================================================
# RESIDUAL DIAGNOSTICS (timeline, histogram, QQ)
# ======================================================================

def plot_residual_diagnostics(model_block, area, figdir, label):
    """
    model_block contains:
        ts, y_true, ridge, rf, xgb
    We use Ridge residuals as default for diagnostics.
    """

    ts = model_block["ts"].dt.tz_localize(None)
    y = model_block["y_true"]
    y_pred = model_block["ridge"]     # main model for residuals

    residuals = y - y_pred

    # ----------------------------------------------------------
    # Residual timeline
    # ----------------------------------------------------------
    plt.figure(figsize=(14, 4))
    plt.plot(ts, residuals, label="Residuals", alpha=0.8, color=COLOR_CRIMSON)
    plt.axhline(0, color=COLOR_PURPLE, linestyle="--", alpha=0.8)
    plt.title(f"Residuals Over Time — {label.upper()} ({area})")
    plt.xlabel("Time (Oslo)")
    plt.ylabel("Residual")
    plt.tight_layout()

    _savefig(figdir / f"{area}_residuals_timeline_{label}.png")

    # ----------------------------------------------------------
    # Residual histogram
    # ----------------------------------------------------------
    plt.figure(figsize=(8, 4))
    plt.hist(residuals, bins=40, alpha=0.85, color=COLOR_BLUE)
    plt.title(f"Residual Distribution — {label.upper()} ({area})")
    plt.xlabel("Residual")
    plt.tight_layout()

    _savefig(figdir / f"{area}_residuals_hist_{label}.png")

    # ----------------------------------------------------------
    # Q–Q plot
    # ----------------------------------------------------------
    import scipy.stats as stats
    plt.figure(figsize=(6, 4))
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title(f"Q-Q Plot — {label.upper()} ({area})")
    plt.tight_layout()

    _savefig(figdir / f"{area}_residuals_qq_{label}.png")

    # ----------------------------------------------------------
    # ACF / PACF of residuals
    # ----------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    plot_acf(residuals, lags=40, ax=axes[0], color=COLOR_CRIMSON)
    plot_pacf(residuals, lags=40, ax=axes[1], method="ywm", color=COLOR_PURPLE)
    axes[0].set_title("ACF of residuals")
    axes[1].set_title("PACF of residuals")
    plt.tight_layout()
    _savefig(figdir / f"{area}_residuals_acf_pacf_{label}.png")


# ======================================================================
# SCATTER: ACTUAL VS PREDICTED
# ======================================================================

def plot_model_scatter(model_block, area, figdir, label):
    ts = model_block["ts"]
    y = model_block["y_true"]

    # Ridge, RF, and XGB models
    preds = {
        "Ridge": (model_block["ridge"], COLOR_PURPLE),
        "RF": (model_block["rf"], COLOR_ORANGE),
        "XGB": (model_block["xgb"], COLOR_CRIMSON),
    }

    for name, (pred, color) in preds.items():
        plt.figure(figsize=(5, 5))
        plt.scatter(y, pred, alpha=0.35, s=12, color=color)
        plt.plot([y.min(), y.max()], [y.min(), y.max()], "--", color=COLOR_BLUE, alpha=0.7)
        plt.xlabel("Actual Spread")
        plt.ylabel(f"Predicted Spread ({name})")
        plt.title(f"Actual vs Predicted — {name} ({label.upper()})")
        plt.tight_layout()

        _savefig(figdir / f"{area}_scatter_{label}_{name.lower()}.png")


# ======================================================================
# FEATURE IMPORTANCE PLOTS (Ridge, RF, XGB)
# ======================================================================

def plot_feature_importance_plots(area, figdir, feature_cols, label,
                                  ridge=None, rf=None, xgb=None):
    """
    We pull coefficients/feature importances from models that the caller
    already saved to disk — to avoid recomputing.
    """

    # NOTE: The actual models are not passed directly. This plotting function
    # assumes feature importances already saved separately by metrics_utils.
    # This is a VISUAL ONLY wrapper that reads the saved importance CSVs.

    importances_dir = Path("outputs/feature_importances")

    # ----------------------------------------------------------
    # Ridge coefficients
    # ----------------------------------------------------------
    ridge_path = importances_dir / f"{area}_ridge_coef_{label}.csv"
    if ridge_path.exists():
        df = pd.read_csv(ridge_path)

        plt.figure(figsize=(8, 4))
        plt.bar(df["feature"], df["coef"], color=COLOR_PURPLE)
        plt.xticks(rotation=45, ha="right")
        plt.title(f"Ridge Coefficients — {label.upper()} ({area})")
        plt.tight_layout()

        _savefig(figdir / f"{area}_ridge_importance_{label}.png")

    # ----------------------------------------------------------
    # RF importances
    # ----------------------------------------------------------
    rf_path = importances_dir / f"{area}_rf_importance_{label}.csv"
    if rf_path.exists():
        df = pd.read_csv(rf_path).sort_values("importance", ascending=False)

        plt.figure(figsize=(8, 4))
        plt.bar(df["feature"], df["importance"], color=COLOR_ORANGE)
        plt.xticks(rotation=45, ha="right")
        plt.title(f"RandomForest Feature Importance — {label.upper()} ({area})")
        plt.tight_layout()

        _savefig(figdir / f"{area}_rf_importance_{label}.png")

    # ----------------------------------------------------------
    # XGB importances
    # ----------------------------------------------------------
    xgb_path = importances_dir / f"{area}_xgb_importance_{label}.csv"
    if xgb_path.exists():
        df = pd.read_csv(xgb_path).sort_values("importance", ascending=False)

        plt.figure(figsize=(8, 4))
        plt.bar(df["feature"], df["importance"], color=COLOR_CRIMSON)
        plt.xticks(rotation=45, ha="right")
        plt.title(f"XGBoost Feature Importance — {label.upper()} ({area})")
        plt.tight_layout()

        _savefig(figdir / f"{area}_xgb_importance_{label}.png")


# ======================================================================
# COMBINED FEATURE IMPORTANCE (Ridge, RF, XGB) — single figure
# ======================================================================

def plot_combined_feature_importance(area, figdir, label, top_n=20):
    """
    Combine Ridge, RF, and XGB importances into one cohesive figure.
    Reads the CSVs saved under outputs/feature_importances and plots
    three horizontal bar subplots.
    """
    importances_dir = Path("outputs/feature_importances")
    ridge_path = importances_dir / f"{area}_ridge_coef_{label}.csv"
    rf_path = importances_dir / f"{area}_rf_importance_{label}.csv"
    xgb_path = importances_dir / f"{area}_xgb_importance_{label}.csv"

    if not (ridge_path.exists() or rf_path.exists() or xgb_path.exists()):
        print(f"⚠ No importance files found for {area} / {label}; skipping combined plot.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 6))

    # Ridge (signed coefficients)
    if ridge_path.exists():
        df_ridge = pd.read_csv(ridge_path)
        df_ridge = df_ridge.assign(abs_coef=df_ridge["coef"].abs()) \
                           .sort_values("abs_coef", ascending=False) \
                           .head(top_n)
        axes[0].barh(df_ridge["feature"], df_ridge["coef"], color=COLOR_PURPLE)
        axes[0].invert_yaxis()
        axes[0].set_title("Ridge coef")
    else:
        axes[0].axis("off")

    # RF
    if rf_path.exists():
        df_rf = pd.read_csv(rf_path).sort_values("importance", ascending=False).head(top_n)
        axes[1].barh(df_rf["feature"], df_rf["importance"], color=COLOR_ORANGE)
        axes[1].invert_yaxis()
        axes[1].set_title("RF importance")
    else:
        axes[1].axis("off")

    # XGB
    if xgb_path.exists():
        df_xgb = pd.read_csv(xgb_path).sort_values("importance", ascending=False).head(top_n)
        axes[2].barh(df_xgb["feature"], df_xgb["importance"], color=COLOR_CRIMSON)
        axes[2].invert_yaxis()
        axes[2].set_title("XGB importance")
    else:
        axes[2].axis("off")

    for ax in axes:
        if ax.get_ylabel():
            ax.set_ylabel("")
        ax.tick_params(axis="y", labelsize=9)
        ax.tick_params(axis="x", labelsize=9)

    fig.suptitle(f"{area} — {label.upper()} feature importance (top {top_n})", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    _savefig(figdir / f"{area}_feature_importance_combined_{label}.png")


# ======================================================================
# HEATMAP (MONTH × HOUR)
# ======================================================================

def plot_heatmap_month_hour(mspread, outpath):
    """
    Recreates the heatmap:
        avg spread by month × hour
    """

    df = mspread.copy()
    df["month"] = df["ts_oslo"].dt.month
    df["hour"]  = df["ts_oslo"].dt.hour

    pivot = df.pivot_table(
        index="month",
        columns="hour",
        values="spread",
        aggfunc="mean"
    )

    # Clip extremes for a less saturated palette
    vals = pivot.values
    vmin = np.nanpercentile(vals, 5)
    vmax = np.nanpercentile(vals, 95)
    # Ensure symmetric bounds around zero when possible
    bound = max(abs(vmin), abs(vmax))

    plt.figure(figsize=(10, 4))
    sns.heatmap(
        pivot,
        cmap="coolwarm",
        annot=False,
        vmin=-bound,
        vmax=bound,
        center=0,
    )
    plt.title("Average Spread by Month × Hour")
    plt.tight_layout()

    _savefig(outpath)

# ======================================================================
# LAG1 VS PRIMARY COMPARISON
# ======================================================================

def plot_lag_comparison(primary_block, lag1_block, area, figdir):
    if lag1_block is None:
        return

    ts = primary_block["ts"].dt.tz_localize(None)
    y_true = primary_block["y_true"]

    plt.figure(figsize=(14, 5))
    plt.plot(ts, y_true, label="Actual", alpha=0.8, color=COLOR_BLUE)

    plt.plot(ts, primary_block["ridge"], label="Primary Ridge (lags 2–4)", alpha=0.85, color=COLOR_PURPLE)
    plt.plot(ts, lag1_block["ridge"], label="Lag1 Ridge (lags 1–4)", alpha=0.7, color=COLOR_ORANGE)

    plt.title(f"Lag1 vs Primary Comparison — Ridge ({area})")
    plt.xlabel("Time (Oslo)")
    plt.ylabel("Spread [EUR/MWh]")
    plt.legend()
    plt.tight_layout()

    _savefig(figdir / f"{area}_comparison_lag1_vs_primary_ridge.png")


# ======================================================================
# 2-HOUR AHEAD MODEL DIAGNOSTICS
# ======================================================================

def plot_two_hour_comparison(h2_block, area, figdir):
    """
    Equivalent of notebook section:
    - 2h actual vs predicted
    - Zoom week
    """

    ts = h2_block["ts"].dt.tz_localize(None)
    y = h2_block["y_true"]

    plt.figure(figsize=(14, 5))
    plt.plot(ts, y, label="Actual (t+2h)", alpha=0.8, color=COLOR_BLUE)
    plt.plot(ts, h2_block["ridge"], label="Ridge (2h)", alpha=0.8, color=COLOR_PURPLE)
    plt.plot(ts, h2_block["rf"], label="RF (2h)", alpha=0.75, color=COLOR_ORANGE)
    plt.plot(ts, h2_block["xgb"], label="XGBoost (2h)", alpha=0.5, color=COLOR_CRIMSON)

    plt.title(f"2h Ahead Predictions — Ridge/RF/XGB ({area})")
    plt.xlabel("Time (Oslo)")
    plt.ylabel("Spread [EUR/MWh]")
    plt.legend()
    plt.tight_layout()

    _savefig(figdir / f"{area}_2h_full.png")

    # 7-day zoom
    n = 7 * 24
    plt.figure(figsize=(14, 4))
    plt.plot(ts[-n:], y[-n:], label="Actual", alpha=0.85, color=COLOR_BLUE)
    plt.plot(ts[-n:], h2_block["ridge"][-n:], label="Ridge", alpha=0.85, color=COLOR_PURPLE)
    plt.plot(ts[-n:], h2_block["rf"][-n:], label="RF", alpha=0.8, color=COLOR_ORANGE)
    plt.plot(ts[-n:], h2_block["xgb"][-n:], label="XGB", alpha=0.8, color=COLOR_CRIMSON)

    plt.title(f"2h Ahead — Last 7 Days ({area})")
    plt.xlabel("Time (Oslo)")
    plt.ylabel("Spread [EUR/MWh]")
    plt.legend()
    plt.tight_layout()

    _savefig(figdir / f"{area}_2h_zoom.png")


# ======================================================================
# HYDROLOGY INTERACTIONS (reservoir × hour / HPI × hour)
# ======================================================================

def plot_hydrology_interactions(df, area, figdir):
    """
    Recreate notebook interaction plots:
        reservoir_pct * hour_sin
        HPI * hour_sin
    """

    if "reservoir_pct" not in df.columns:
        print("⚠ No hydrology found — skipping hydro interaction plots.")
        return

    if "hour_sin" not in df.columns:
        return

    df = df.copy()
    df["reservoir_x_hour"] = df["reservoir_pct"] * df["hour_sin"]

    if "HPI" in df.columns:
        df["HPI_x_hour"] = df["HPI"] * df["hour_sin"]
    else:
        df["HPI_x_hour"] = np.nan

    # ----------------------------------------------------------
    # Distribution plots
    # ----------------------------------------------------------
    plt.figure(figsize=(10, 4))
    sns.histplot(df["reservoir_x_hour"].dropna(), bins=40, kde=True, color=COLOR_PURPLE)
    plt.title(f"Interaction: Reservoir_pct × hour_sin ({area})")
    plt.tight_layout()
    _savefig(figdir / f"{area}_hydro_interaction_reservoir.png")

    if df["HPI_x_hour"].notna().sum() > 0:
        plt.figure(figsize=(10, 4))
        sns.histplot(df["HPI_x_hour"].dropna(), bins=40, kde=True, color=COLOR_CRIMSON)
        plt.title(f"Interaction: HPI × hour_sin ({area})")
        plt.tight_layout()
        _savefig(figdir / f"{area}_hydro_interaction_hpi.png")


# ======================================================================
# OPTIONAL: PARTIAL DEPENDENCE PLOTS (RF / XGB)
# ======================================================================

def plot_pdp(model, X, feature_cols, area, figdir, label):
    """
    Enables PDPs for hydrology or selected features.
    Only runs if sklearn ≥ 1.2.
    """

    try:
        from sklearn.inspection import PartialDependenceDisplay

        # Create PDP for each feature
        for feat in feature_cols:
            try:
                idx = feature_cols.index(feat)
                plt.figure(figsize=(8, 4))
                PartialDependenceDisplay.from_estimator(
                    model,
                    X,
                    [idx],
                    feature_names=feature_cols,
                )
                plt.title(f"PDP — {feat} ({label.upper()} / {area})")
                plt.tight_layout()
                _savefig(figdir / f"{area}_pdp_{label}_{feat}.png")
            except Exception:
                pass

    except Exception:
        print("⚠ PDP not available in your sklearn version — skipping.")


# ======================================================================
# AUTOCORRELATION PLOT (spread or residuals)
# ======================================================================

def plot_autocorrelation(block, area, figdir, label):
    """
    Uses pandas autocorrelation_plot to reproduce notebook cell.
    """

    y = block["y_true"]

    plt.figure(figsize=(6, 4))
    pd.plotting.autocorrelation_plot(y)
    plt.title(f"Autocorrelation — {label.upper()} ({area})")
    plt.tight_layout()

    _savefig(figdir / f"{area}_autocorr_{label}.png")

