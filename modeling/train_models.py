"""Model training utilities for primary, lag1, 2h, and 3h horizons."""
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


# =====================================================================
# HELPERS
# =====================================================================

def eval_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-6))) * 100
    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "MAPE": mape
    }


def time_split(df, feature_cols, target_col):
    """
    Chronological 80/20 split for time series evaluation.
    """
    X = df[feature_cols].values
    y = df[target_col].values

    split = int(len(df) * 0.8)
    return (
        X[:split],
        X[split:],
        y[:split],
        y[split:],
        df.iloc[split:]["ts_oslo_tz"],  # ts for plotting
        split
    )


# =====================================================================
# TRAINING: PRIMARY MODELS (lags 2–4)
# =====================================================================

def train_primary_models(df, hydro_enabled):
    """
    df: dataset with spread_lag_2h, spread_lag_3h, spread_lag_4h + time features.
    """
    # FEATURE LIST
    lag_cols = ["spread_lag_2h", "spread_lag_3h", "spread_lag_4h"]
    cyclical = ["hour_sin","hour_cos","dow_sin","dow_cos","month_sin","month_cos"]

    feature_cols = lag_cols + cyclical
    if hydro_enabled:
        for col in ["reservoir_pct", "HPI"]:
            if col in df.columns and df[col].notna().any():
                feature_cols.append(col)

    # 1-hour ahead target
    target_col = "spread"

    # Drop rows with missing features/target
    df = df.dropna(subset=feature_cols + [target_col])
    if df.empty:
        raise ValueError("No rows available for primary training after filtering.")

    # SPLIT
    X_train, X_test, y_train, y_test, ts_test, split_idx = time_split(df, feature_cols, target_col)

    # ------------------------------------------------------
    # Ridge
    # ------------------------------------------------------
    ridge = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", RidgeCV(alphas=[0.1, 1, 10, 100, 1000]))
    ])
    ridge.fit(X_train, y_train)
    y_pred_ridge = ridge.predict(X_test)

    # ------------------------------------------------------
    # Random Forest
    # ------------------------------------------------------
    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)

    # ------------------------------------------------------
    # XGBoost
    # ------------------------------------------------------
    xgb = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        n_jobs=-1,
        random_state=42
    )
    xgb.fit(X_train, y_train)
    y_pred_xgb = xgb.predict(X_test)

    # ------------------------------------------------------
    # Naive Baseline (lag-2)
    # ------------------------------------------------------
    idx = df.columns.get_loc("spread_lag_2h")
    y_pred_naive = df.iloc[len(y_train):]["spread_lag_2h"].values

    # ------------------------------------------------------
    # METRICS
    # ------------------------------------------------------
    metrics = {
        "ridge_mae": eval_metrics(y_test, y_pred_ridge)["MAE"],
        "ridge_rmse": eval_metrics(y_test, y_pred_ridge)["RMSE"],
        "ridge_r2": eval_metrics(y_test, y_pred_ridge)["R2"],

        "rf_mae": eval_metrics(y_test, y_pred_rf)["MAE"],
        "rf_rmse": eval_metrics(y_test, y_pred_rf)["RMSE"],
        "rf_r2": eval_metrics(y_test, y_pred_rf)["R2"],

        "xgb_mae": eval_metrics(y_test, y_pred_xgb)["MAE"],
        "xgb_rmse": eval_metrics(y_test, y_pred_xgb)["RMSE"],
        "xgb_r2": eval_metrics(y_test, y_pred_xgb)["R2"],

        "naive_mae": eval_metrics(y_test, y_pred_naive)["MAE"],
        "naive_rmse": eval_metrics(y_test, y_pred_naive)["RMSE"]
    }

    return (
        ridge,
        rf,
        xgb,
        y_pred_naive,
        metrics,
        X_test,
        y_test,
        y_pred_ridge,
        y_pred_rf,
        y_pred_xgb,
        ts_test,
        feature_cols,
        df,
        split_idx
    )


# =====================================================================
# TRAINING: LAG1 COMPARISON (lags 1–4)
# =====================================================================

def train_lag1_models(df, hydro_enabled):
    """
    Same as primary, but includes lag1.
    """
    lag_cols = ["spread_lag_1h","spread_lag_2h","spread_lag_3h","spread_lag_4h"]
    cyclical = ["hour_sin","hour_cos","dow_sin","dow_cos","month_sin","month_cos"]

    feature_cols = lag_cols + cyclical
    if hydro_enabled:
        for col in ["reservoir_pct", "HPI"]:
            if col in df.columns and df[col].notna().any():
                feature_cols.append(col)

    target_col = "spread"

    df = df.dropna(subset=feature_cols + [target_col])
    if df.empty:
        raise ValueError("No rows available for lag1 training after filtering.")

    X_train, X_test, y_train, y_test, ts_test, split_idx = time_split(df, feature_cols, target_col)

    # ------------------------------------------------------
    # Ridge
    # ------------------------------------------------------
    ridge = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", RidgeCV(alphas=[0.1,1,10,100,1000]))
    ])
    ridge.fit(X_train, y_train)
    y_pred_ridge = ridge.predict(X_test)

    # ------------------------------------------------------
    # RF
    # ------------------------------------------------------
    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)

    # ------------------------------------------------------
    # XGB
    # ------------------------------------------------------
    xgb = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        random_state=42,
        n_jobs=-1
    )
    xgb.fit(X_train, y_train)
    y_pred_xgb = xgb.predict(X_test)

    # ------------------------------------------------------
    # Naive baseline (lag-1)
    # ------------------------------------------------------
    y_pred_naive = df.iloc[len(y_train):]["spread_lag_1h"].values

    # ------------------------------------------------------
    # METRICS
    # ------------------------------------------------------
    metrics = {
        "ridge_mae": eval_metrics(y_test, y_pred_ridge)["MAE"],
        "ridge_rmse": eval_metrics(y_test, y_pred_ridge)["RMSE"],
        "ridge_r2": eval_metrics(y_test, y_pred_ridge)["R2"],

        "rf_mae": eval_metrics(y_test, y_pred_rf)["MAE"],
        "rf_rmse": eval_metrics(y_test, y_pred_rf)["RMSE"],
        "rf_r2": eval_metrics(y_test, y_pred_rf)["R2"],

        "xgb_mae": eval_metrics(y_test, y_pred_xgb)["MAE"],
        "xgb_rmse": eval_metrics(y_test, y_pred_xgb)["RMSE"],
        "xgb_r2": eval_metrics(y_test, y_pred_xgb)["R2"],

        "naive_mae": eval_metrics(y_test, y_pred_naive)["MAE"],
        "naive_rmse": eval_metrics(y_test, y_pred_naive)["RMSE"]
    }

    return (
        ridge, rf, xgb,
        y_pred_naive, metrics,
        X_test, y_test,
        y_pred_ridge, y_pred_rf, y_pred_xgb,
        ts_test,
        feature_cols,
        df,
        split_idx
    )


# =====================================================================
# TRAINING: 2-HOUR AHEAD MODELS (spread_h2)
# =====================================================================

def train_two_hour_models(df, hydro_enabled):
    """
    df must contain spread_h2 = spread.shift(-2)
    Already created in run_pipeline.
    """

    df = df.copy()
    df["spread_h2"] = df["spread"].shift(-2)
    df = df.dropna(subset=["spread_h2"])

    lag_cols = ["spread_lag_2h", "spread_lag_3h", "spread_lag_4h"]
    cyclical = ["hour_sin","hour_cos","dow_sin","dow_cos","month_sin","month_cos"]

    feature_cols = lag_cols + cyclical
    if hydro_enabled:
        for col in ["reservoir_pct", "HPI"]:
            if col in df.columns and df[col].notna().any():
                feature_cols.append(col)

    target_col = "spread_h2"

    df = df.dropna(subset=feature_cols + [target_col])
    if df.empty:
        raise ValueError("No rows available for 2h training after filtering.")

    X_train, X_test, y_train, y_test, ts_test, split_idx = time_split(df, feature_cols, target_col)

    # Ridge
    ridge = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", RidgeCV(alphas=[0.1,1,10,100,1000]))
    ])
    ridge.fit(X_train, y_train)
    y_pred_ridge = ridge.predict(X_test)

    # RF
    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)

    # XGBoost
    xgb = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        n_jobs=-1,
        random_state=42
    )
    xgb.fit(X_train, y_train)
    y_pred_xgb = xgb.predict(X_test)

    # Naive (lag-2)
    y_pred_naive = df.iloc[len(y_train):]["spread_lag_2h"].values

    metrics = {
        "ridge_mae": eval_metrics(y_test, y_pred_ridge)["MAE"],
        "ridge_rmse": eval_metrics(y_test, y_pred_ridge)["RMSE"],
        "ridge_r2": eval_metrics(y_test, y_pred_ridge)["R2"],

        "rf_mae": eval_metrics(y_test, y_pred_rf)["MAE"],
        "rf_rmse": eval_metrics(y_test, y_pred_rf)["RMSE"],
        "rf_r2": eval_metrics(y_test, y_pred_rf)["R2"],

        "xgb_mae": eval_metrics(y_test, y_pred_xgb)["MAE"],
        "xgb_rmse": eval_metrics(y_test, y_pred_xgb)["RMSE"],
        "xgb_r2": eval_metrics(y_test, y_pred_xgb)["R2"],

        "naive_mae": eval_metrics(y_test, y_pred_naive)["MAE"],
        "naive_rmse": eval_metrics(y_test, y_pred_naive)["RMSE"],
    }

    return (
        ridge, rf, xgb,
        y_pred_naive, metrics,
        X_test, y_test,
        y_pred_ridge, y_pred_rf, y_pred_xgb,
        ts_test,
        feature_cols,
        df,
        split_idx
    )


# =====================================================================
# TRAINING: 3-HOUR AHEAD MODELS (spread_h3)
# =====================================================================

def train_three_hour_models(df, hydro_enabled):
    """
    df must contain spread_h3 = spread.shift(-3)
    Already created in run_pipeline.
    """

    df = df.copy()
    df["spread_h3"] = df["spread"].shift(-3)
    df = df.dropna(subset=["spread_h3"])

    lag_cols = ["spread_lag_2h", "spread_lag_3h", "spread_lag_4h"]
    cyclical = ["hour_sin","hour_cos","dow_sin","dow_cos","month_sin","month_cos"]

    feature_cols = lag_cols + cyclical
    if hydro_enabled:
        for col in ["reservoir_pct", "HPI"]:
            if col in df.columns and df[col].notna().any():
                feature_cols.append(col)

    target_col = "spread_h3"

    df = df.dropna(subset=feature_cols + [target_col])
    if df.empty:
        raise ValueError("No rows available for 3h training after filtering.")

    X_train, X_test, y_train, y_test, ts_test, split_idx = time_split(df, feature_cols, target_col)

    # Ridge
    ridge = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", RidgeCV(alphas=[0.1,1,10,100,1000]))
    ])
    ridge.fit(X_train, y_train)
    y_pred_ridge = ridge.predict(X_test)

    # RF
    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)

    # XGBoost
    xgb = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        n_jobs=-1,
        random_state=42
    )
    xgb.fit(X_train, y_train)
    y_pred_xgb = xgb.predict(X_test)

    # Naive (lag-2)
    y_pred_naive = df.iloc[len(y_train):]["spread_lag_2h"].values

    metrics = {
        "ridge_mae": eval_metrics(y_test, y_pred_ridge)["MAE"],
        "ridge_rmse": eval_metrics(y_test, y_pred_ridge)["RMSE"],
        "ridge_r2": eval_metrics(y_test, y_pred_ridge)["R2"],

        "rf_mae": eval_metrics(y_test, y_pred_rf)["MAE"],
        "rf_rmse": eval_metrics(y_test, y_pred_rf)["RMSE"],
        "rf_r2": eval_metrics(y_test, y_pred_rf)["R2"],

        "xgb_mae": eval_metrics(y_test, y_pred_xgb)["MAE"],
        "xgb_rmse": eval_metrics(y_test, y_pred_xgb)["RMSE"],
        "xgb_r2": eval_metrics(y_test, y_pred_xgb)["R2"],

        "naive_mae": eval_metrics(y_test, y_pred_naive)["MAE"],
        "naive_rmse": eval_metrics(y_test, y_pred_naive)["RMSE"],
    }

    return (
        ridge, rf, xgb,
        y_pred_naive, metrics,
        X_test, y_test,
        y_pred_ridge, y_pred_rf, y_pred_xgb,
        ts_test,
        feature_cols,
        df,
        split_idx
    )

