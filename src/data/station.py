"""Real weather-station observations via Meteostat (free, no key).

This is the data the markets actually resolve on — e.g. NWS Central Park for NYC.
We use it for the historical/climatology branch and to compute the grid->station
bias for the forecast branch, so the model is anchored to the exact resolution
source rather than a model grid cell near the city center.

Returns archives in the same shape as src.data.openmeteo so they drop straight
into fit_climatology / fit_gev / regime_offset.
"""
from __future__ import annotations

import logging
import warnings
from datetime import date, datetime

import numpy as np
from meteostat import Daily, Stations

# Meteostat warns for every missing yearly bulk file; quiet it down.
logging.getLogger("meteostat").setLevel(logging.ERROR)

_CACHE: dict = {}


def nearest_station(lat: float, lon: float) -> dict:
    df = Stations().nearby(lat, lon).fetch(1)
    if df.empty:
        raise ValueError(f"No Meteostat station near ({lat:.3f}, {lon:.3f})")
    row = df.iloc[0]
    return {
        "id": df.index[0],
        "name": str(row["name"]),
        "lat": float(row["latitude"]),
        "lon": float(row["longitude"]),
        "distance_km": round(float(row["distance"]) / 1000.0, 2),
    }


def station_archive(lat: float, lon: float, start: date, end: date,
                    units: str = "celsius") -> tuple[dict, dict]:
    """Daily Tmax for the nearest station, as an openmeteo-style archive + meta."""
    key = (round(lat, 3), round(lon, 3), start, end, units)
    if key in _CACHE:
        return _CACHE[key]
    info = nearest_station(lat, lon)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = Daily(info["id"], datetime(start.year, start.month, start.day),
                   datetime(end.year, end.month, end.day)).fetch()
    s = df["tmax"].dropna() if "tmax" in df.columns else df.get("tmax")
    if s is None or s.empty:
        raise ValueError(f"No Tmax history for station {info['name']!r}")
    vals = s.to_numpy(dtype=float)
    if str(units).lower() in ("f", "fahrenheit"):
        vals = vals * 9.0 / 5.0 + 32.0
    archive = {"daily": {
        "time": [t.strftime("%Y-%m-%d") for t in s.index],
        "temperature_2m_max": [float(v) for v in vals],
    }}
    info["n_days"] = int(vals.size)
    out = (archive, info)
    _CACHE[key] = out
    return out


def tail_archive(archive: dict, n: int) -> dict:
    """Last n daily entries of an archive (for the regime/persistence window)."""
    t = archive["daily"]["time"][-n:]
    v = archive["daily"]["temperature_2m_max"][-n:]
    return {"daily": {"time": t, "temperature_2m_max": v}}


def grid_to_station_offset(grid_archive: dict, station_archive_: dict) -> float:
    """median(station - grid) over overlapping dates: maps grid forecasts to the
    station the market resolves on."""
    g = dict(zip(grid_archive["daily"]["time"],
                 grid_archive["daily"]["temperature_2m_max"]))
    diffs = [s - g[t] for t, s in zip(station_archive_["daily"]["time"],
                                      station_archive_["daily"]["temperature_2m_max"])
             if t in g and s is not None and g[t] is not None]
    return float(np.median(diffs)) if diffs else 0.0
