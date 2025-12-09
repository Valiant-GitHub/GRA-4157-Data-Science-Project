"""Builds mFRR–DA spread dataset (mspread) for modeling."""
import pandas as pd


def prepare_mspread(raw, mfrr, area="NO1"):
    """
    Build the operational mFRR–DA spread dataset (mspread).

    Steps (CLI already ensured tz-naive Oslo):
    1) Parse timestamps as tz-naive.
    2) Filter DA and mFRR for the selected area.
    3) Merge on (ts_oslo, area).
    4) Create tz-aware ts_oslo_tz for reporting only.
    5) Compute spread.
    """

    def _ensure_mfrr_long(df):
        """
        Accepts mfrr either in long form (area, ts_oslo, price_mfrr)
        or the wide pivoted form from load_mfrr_all, and returns long.
        """
        # Already long-form
        if {"area", "price_mfrr"}.issubset(df.columns):
            out = df.copy()
            if "ts_start_oslo" in out.columns and "ts_oslo" not in out.columns:
                out = out.rename(columns={"ts_start_oslo": "ts_oslo"})
            return out[["ts_oslo", "area", "price_mfrr"]]

        # Wide pivoted form with MultiIndex columns (area, metric)
        if "ts_start_oslo" in df.columns and isinstance(df.columns, pd.MultiIndex):
            wide = df.rename(columns={"ts_start_oslo": "ts_oslo"}).copy()
            stacked = (
                wide.set_index("ts_oslo")
                .stack(level=[0, 1])
                .reset_index()
                .rename(columns={"level_1": "area", "level_2": "metric", 0: "value"})
            )
            stacked = stacked.dropna(subset=["value"])
            stacked = stacked[stacked["metric"].str.contains("Imbalance Price", na=False)]
            stacked = stacked.rename(columns={"value": "price_mfrr"})
            stacked["ts_oslo"] = pd.to_datetime(stacked["ts_oslo"], errors="coerce")
            return stacked[["ts_oslo", "area", "price_mfrr"]]

        # Fallback: return as-is (will likely fail downstream but keeps visibility)
        return df

    # Normalize mfrr to long format
    mfrr = _ensure_mfrr_long(mfrr)

    # --- DA PREP ---
    _da = raw.copy()
    _da["ts_oslo"] = pd.to_datetime(_da["ts_oslo"], errors="coerce")   # <-- FIXED
    _da = (
        _da[_da["area"] == area][["ts_oslo", "area", "price"]]
        .rename(columns={"price": "price_da"})
    )

    # --- mFRR PREP ---
    mfrr_area = mfrr[mfrr["area"] == area].copy()
    mfrr_area["ts_oslo"] = pd.to_datetime(mfrr_area["ts_oslo"], errors="coerce")  # <-- FIXED

    # --- MERGE ---
    mspread = mfrr_area.merge(_da, on=["ts_oslo", "area"], how="left")

    # --- RESTORE TZ-AWARE TS (for plotting/reporting only) ---
    # underlying merge stays tz-naive
    mspread["ts_oslo_tz"] = (
        pd.to_datetime(mspread["ts_oslo"], errors="coerce")
        .dt.tz_localize("Europe/Oslo", nonexistent="shift_forward", ambiguous="NaT")
    )

    # --- SPREAD ---
    mspread["spread"] = mspread["price_mfrr"] - mspread["price_da"]

    # --- CLEANUP ---
    mspread = (
        mspread.dropna(subset=["price_mfrr", "price_da"])
        .sort_values("ts_oslo")
        .reset_index(drop=True)
    )

    return mspread
