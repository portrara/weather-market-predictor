"""Open-Meteo data access: geocoding, historical archive, and ensemble forecasts.

All endpoints are free and need no API key. Responses are cached to .cache/ so
repeated runs (and offline experimentation) don't re-hit the network.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests

CACHE_DIR = Path(__file__).resolve().parents[2] / ".cache"
CACHE_DIR.mkdir(exist_ok=True)

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ENSEMBLE_URL = "https://ensemble-api.open-meteo.com/v1/ensemble"

# Ensemble systems combined into one "super-ensemble" cloud.
ENSEMBLE_MODELS = "gfs025,ecmwf_ifs025,icon_seamless,gem_global"


@dataclass
class Location:
    name: str
    latitude: float
    longitude: float
    elevation: Optional[float] = None
    timezone: str = "auto"


def _cache_key(url: str, params: dict) -> Path:
    raw = url + "?" + json.dumps(params, sort_keys=True)
    h = hashlib.sha1(raw.encode()).hexdigest()[:16]
    return CACHE_DIR / f"{h}.json"


def _get(url: str, params: dict, ttl_hours: float = 6.0, use_cache: bool = True,
         retries: int = 3, timeout: int = 90) -> dict:
    """GET with on-disk caching and retry/backoff.
    ttl_hours<=0 means cache forever (good for immutable history)."""
    cache_file = _cache_key(url, params)
    if use_cache and cache_file.exists():
        age_h = (time.time() - cache_file.stat().st_mtime) / 3600.0
        if ttl_hours <= 0 or age_h < ttl_hours:
            return json.loads(cache_file.read_text(encoding="utf-8"))
    last_err = None
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            cache_file.write_text(json.dumps(data), encoding="utf-8")
            return data
        except requests.RequestException as e:
            last_err = e
            if attempt < retries - 1:
                time.sleep(2 ** attempt)  # 1s, 2s, 4s backoff
    raise RuntimeError(f"Open-Meteo request failed after {retries} tries: {last_err}")


def geocode(name: str) -> Location:
    """Resolve a place name to coordinates. Raises if not found.

    The Open-Meteo geocoder wants a bare place name, so if "City, Region"
    yields nothing we retry with just the first comma-separated token.
    """
    queries = [name]
    if "," in name:
        queries.append(name.split(",")[0].strip())
    results = None
    for q in queries:
        data = _get(GEOCODE_URL, {"name": q, "count": 1, "language": "en"}, ttl_hours=0)
        results = data.get("results")
        if results:
            break
    if not results:
        raise ValueError(f"Could not geocode location: {name!r}")
    r = results[0]
    label = ", ".join(x for x in [r.get("name"), r.get("admin1"), r.get("country")] if x)
    return Location(
        name=label,
        latitude=r["latitude"],
        longitude=r["longitude"],
        elevation=r.get("elevation"),
        timezone="auto",
    )


def fetch_archive(loc: Location, start: str, end: str,
                  temperature_unit: str = "celsius") -> dict:
    """Historical daily Tmax/Tmin (ERA5 reanalysis). Cached permanently."""
    params = {
        "latitude": loc.latitude,
        "longitude": loc.longitude,
        "start_date": start,
        "end_date": end,
        "daily": "temperature_2m_max,temperature_2m_min",
        "temperature_unit": temperature_unit,
        "timezone": loc.timezone,
    }
    return _get(ARCHIVE_URL, params, ttl_hours=0)


def fetch_recent(loc: Location, past_days: int = 35,
                 temperature_unit: str = "celsius") -> dict:
    """Recent observed daily Tmax (for regime/persistence). Short cache."""
    params = {
        "latitude": loc.latitude,
        "longitude": loc.longitude,
        "daily": "temperature_2m_max",
        "temperature_unit": temperature_unit,
        "past_days": past_days,
        "forecast_days": 1,
        "timezone": loc.timezone,
    }
    return _get(FORECAST_URL, params, ttl_hours=6)


def observed_high_today(loc: Location, target, temperature_unit: str = "celsius"):
    """If `target` is the location's current local day, return the max temperature
    observed SO FAR today (a hard floor on the daily high). Else None.

    On the day of resolution this is a real edge: the daily max can only be >= what
    has already happened. Returns (high_so_far, hours_elapsed) or None.
    """
    from datetime import datetime, timedelta, timezone
    params = {
        "latitude": loc.latitude, "longitude": loc.longitude,
        "hourly": "temperature_2m", "temperature_unit": temperature_unit,
        "past_days": 1, "forecast_days": 1, "timezone": "auto",
    }
    data = _get(FORECAST_URL, params, ttl_hours=1)
    off = data.get("utc_offset_seconds", 0)
    # Open-Meteo's hourly timestamps are naive local time, so compute a matching
    # naive local "now" (UTC now -> drop tzinfo -> add the location's offset).
    now_utc = datetime.now(timezone.utc).replace(tzinfo=None)
    now_local = now_utc + timedelta(seconds=off)
    today_local = now_local.date()
    if target != today_local:
        return None
    times = data["hourly"]["time"]
    temps = data["hourly"]["temperature_2m"]
    so_far = [t for ti, t in zip(times, temps)
              if t is not None and datetime.fromisoformat(ti) <= now_local
              and datetime.fromisoformat(ti).date() == today_local]
    if not so_far:
        return None
    return float(max(so_far)), now_local.hour


def fetch_ensemble(loc: Location, forecast_days: int = 16,
                   temperature_unit: str = "celsius") -> dict:
    """Multi-model ensemble daily Tmax. Each member is its own column."""
    params = {
        "latitude": loc.latitude,
        "longitude": loc.longitude,
        "daily": "temperature_2m_max",
        "temperature_unit": temperature_unit,
        "models": ENSEMBLE_MODELS,
        "forecast_days": forecast_days,
        "timezone": loc.timezone,
    }
    return _get(ENSEMBLE_URL, params, ttl_hours=3)
