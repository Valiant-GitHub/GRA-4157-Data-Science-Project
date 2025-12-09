"""API fetch for weekly hydropower reservoir data (Magasinstatistikk)."""
import pandas as pd
import numpy as np
import requests
import time

OFFICIAL_CAP_TWH = {
    "NO1": 5.976, "NO2": 33.899, "NO3": 9.147,
    "NO4": 20.851, "NO5": 17.331
}

def fetch_reservoir_weekly(start, end):
    """
    Fetch weekly hydropower reservoir data from NVE.
    Returns cleaned DataFrame with:
        week_start, area, reservoir_pct, iso_year, iso_week
    """
    url = "https://biapi.nve.no/magasinstatistikk/api/Magasinstatistikk/HentOffentligData"

    # Retry a few times to mitigate transient slow responses/timeouts
    last_err = None
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            break
        except requests.exceptions.ReadTimeout as exc:
            last_err = exc
            if attempt == 2:
                raise
            # small backoff before retry
            time.sleep(2 * (attempt + 1))
        except Exception as exc:
            last_err = exc
            if attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))
    else:
        # Should not reach here, but keep for clarity
        if last_err:
            raise last_err

    df = pd.json_normalize(r.json())

    df = df.rename(columns={
        "dato_Id": "week_start",
        "omrnr": "area_num",
        "fyllingsgrad": "fill_frac"
    })

    df["week_start"] = pd.to_datetime(df["week_start"])
    df["area"] = "NO" + df["area_num"].astype(str)

    df["reservoir_pct"] = df["fill_frac"] * 100
    df = df[df["area"].isin(["NO1","NO2","NO3","NO4","NO5"])]

    mask = (df["week_start"] >= pd.to_datetime(start)) & \
           (df["week_start"] <= pd.to_datetime(end))
    df = df.loc[mask]

    iso = df["week_start"].dt.isocalendar()
    df["iso_year"] = iso.year.astype(int)
    df["iso_week"] = iso.week.astype(int)

    # Deduplicate to one row per area/week (keep latest)
    df = (
        df.sort_values(["area", "week_start"])
          .drop_duplicates(["area", "iso_year", "iso_week"], keep="last")
    )

    df = df[["week_start","area","reservoir_pct","iso_year","iso_week"]]
    df = df.sort_values(["area","week_start"]).reset_index(drop=True)
    return df
