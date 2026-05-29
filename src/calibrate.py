"""Per-station auto-calibration of the spread-inflation factor, cached to disk.

Running the walk-forward backtest is slow, so the first time we calibrate a
station we compute the spread_inflation that yields ~80% interval coverage and
cache it. Subsequent predictions for that station reuse it, so the model's bucket
probabilities are well-calibrated by default instead of overconfident.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from .data.openmeteo import Location

CACHE_DIR = Path(__file__).resolve().parents[1] / ".cache"
CACHE_DIR.mkdir(exist_ok=True)
TTL_DAYS = 30


def _cache_file(lat: float, lon: float, units: str) -> Path:
    return CACHE_DIR / f"calib_{round(lat,3)}_{round(lon,3)}_{units[:1].lower()}.json"


def get_spread_inflation(lat: float, lon: float, units: str = "celsius",
                         name: str = "station", refresh: bool = False) -> dict:
    """Return {'spread_inflation', 'crps_skill', 'cached'} for a location."""
    cf = _cache_file(lat, lon, units)
    if cf.exists() and not refresh:
        age_d = (time.time() - cf.stat().st_mtime) / 86400.0
        if age_d < TTL_DAYS:
            data = json.loads(cf.read_text())
            data["cached"] = True
            return data

    from .backtest import walk_forward  # lazy import (heavy)
    r = walk_forward(location=Location(name=name, latitude=lat, longitude=lon),
                     units=units, eval_days=365, step=10)
    data = {
        "spread_inflation": r.suggested_spread_inflation,
        "crps_skill": round(r.crps_skill, 3),
        "coverage_at_1x": round(r.coverage_80, 3),
        "cached": False,
    }
    cf.write_text(json.dumps(data))
    return data
