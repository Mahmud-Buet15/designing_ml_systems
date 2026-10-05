"""Download NYC TLC yellow-taxi Parquet files and the taxi zone lookup (Milestone 2).

Source: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
"""

from __future__ import annotations

import shutil
import urllib.request
from pathlib import Path

from taxieta import config


def trip_url(year: int, month: int, color: str = "yellow") -> str:
    return f"{config.TLC_BASE_URL}/trip-data/{color}_tripdata_{year}-{month:02d}.parquet"


def zone_lookup_url() -> str:
    return f"{config.TLC_BASE_URL}/misc/taxi_zone_lookup.csv"


def _fetch(url: str, dest: Path, force: bool = False) -> Path:
    if dest.exists() and not force:
        print(f"skip  {dest.name} (exists)")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"get   {url}")
    with urllib.request.urlopen(url) as resp, open(tmp, "wb") as out:  # noqa: S310 (trusted URL)
        shutil.copyfileobj(resp, out, length=1 << 20)
    tmp.rename(dest)
    return dest


def download_months(months: list[tuple[int, int]], force: bool = False) -> list[Path]:
    """months: list of (year, month). Returns local paths."""
    config.ensure_dirs()
    paths = [_fetch(zone_lookup_url(), config.ZONE_LOOKUP, force)]
    for year, month in months:
        dest = config.RAW_DIR / f"yellow_tripdata_{year}-{month:02d}.parquet"
        paths.append(_fetch(trip_url(year, month), dest, force))
    return paths


def parse_months(spec: str) -> list[tuple[int, int]]:
    """'2019-01:2019-04,2020-03' -> [(2019,1),(2019,2),(2019,3),(2019,4),(2020,3)]"""
    out: list[tuple[int, int]] = []
    for part in spec.split(","):
        part = part.strip()
        if ":" in part:
            start, end = part.split(":")
            y, m = map(int, start.split("-"))
            ey, em = map(int, end.split("-"))
            while (y, m) <= (ey, em):
                out.append((y, m))
                y, m = (y + 1, 1) if m == 12 else (y, m + 1)
        else:
            y, m = map(int, part.split("-"))
            out.append((y, m))
    return out
