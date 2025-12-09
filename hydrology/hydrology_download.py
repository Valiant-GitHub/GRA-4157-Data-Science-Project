"""Download weekly hydrology data (CSV or API fallback)."""
import requests
import pandas as pd
from pathlib import Path
from datetime import date
from hydrology.data_loader import fetch_reservoir_weekly

#not the actual link, just the base case condition, actual link is in data_loader.py
DEFAULT_URL = "https://www.nve.no/energy/hydropower/weekly-reservoir-filling/download-csv/" 


def _try_direct_csv(url: str) -> bytes | None:
    try:
        resp = requests.get(url, timeout=60)
        if resp.status_code == 200:
            return resp.content
    except Exception:
        return None
    return None


def download_reservoir_data(
    output_path: Path | str = Path("data/raw/hydro/reservoir.csv"),
    url: str = DEFAULT_URL,
    start: str | None = None,
    end: str | None = None,
) -> Path:
    """
    Download weekly reservoir data and save CSV to disk.
    Primary attempt: direct CSV download.
    Fallback: API fetch (Magasinstatistikk) via data_loader.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 1) Try direct CSV
    content = _try_direct_csv(url)

    if content is not None:
        output_path.write_bytes(content)
        print(f"✔ Saved raw hydrology CSV → {output_path}")
        return output_path

    # 2) Fallback: use API to build a tidy CSV
    print("⚠ Direct CSV failed, falling back to API (Magasinstatistikk)...")
    if start is None:
        start = "1990-01-01"
    if end is None:
        end = date.today().isoformat()

    df = fetch_reservoir_weekly(start=start, end=end)
    df.to_csv(output_path, index=False)
    print(f"✔ Saved raw hydrology CSV (from API) → {output_path}")
    return output_path
