"""End-to-end pipeline: location + date + buckets -> probabilities + edge.

Grid mode (default):
    history (ERA5 grid) ─► climatology (harmonic + trend)
    recent obs ─────────► regime offset ─┐
    ensemble NWP ───────► member cloud ──┼─► blend ─► buckets
    config ─────────────► station offset ┘        └► EVT context

Station mode (station={"lat","lon","name"}):
    history = REAL station record (Meteostat) -> climatology/regime/EVT
    ensemble = Open-Meteo grid + (grid->station offset)   # bridge to the station
    => anchored to the exact source the market resolves on (e.g. NWS Central Park)

Same-day floor: if the target is the location's current local day, the max
observed SO FAR is a hard lower bound applied to the distribution.
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np

from .blend import Bucket, auto_buckets, build_distribution, summarize
from .data.openmeteo import (
    Location,
    fetch_archive,
    fetch_ensemble,
    fetch_recent,
    geocode,
    observed_high_today,
)
from .models.bias_correct import regime_offset
from .models.climatology import fit_climatology
from .models.ensemble import lead_days, sample as ensemble_sample
from .models.extremes import fit_gev


def predict(
    *,
    city: str | None = None,
    location: Location | None = None,
    target: date,
    buckets: list[Bucket] | None = None,
    history_years: int = 20,
    station_offset: float = 0.0,
    spread_inflation: float = 1.0,
    units: str = "celsius",
    station: dict | None = None,      # {"lat","lon","name"} -> use real station data
    same_day_floor: bool = True,
    today: date | None = None,
    seed: int = 7,
    use_llm: bool = True,
    llm_key: str | None = None,       # per-request key (e.g. pasted in the web UI)
    llm_provider: str | None = None,
) -> dict:
    today = today or date.today()
    rng = np.random.default_rng(seed)
    tunit = "fahrenheit" if str(units).lower() in ("f", "fahrenheit") else "celsius"
    sym = "°F" if tunit == "fahrenheit" else "°C"

    start_d = today - timedelta(days=int(history_years * 365.25))
    grid_to_station = 0.0
    station_meta = None

    # --- history & climatology (always long grid history for a stable seasonal
    #     shape); in station mode we additionally anchor the mean to the real
    #     resolution station via a grid->station offset. ---
    loc = location or (Location(name=station.get("name", "station"),
                                latitude=float(station["lat"]), longitude=float(station["lon"]))
                       if station and station.get("lat") is not None else geocode(city))
    archive = fetch_archive(loc, start_d.isoformat(),
                            (today - timedelta(days=5)).isoformat(),
                            temperature_unit=tunit)
    clim = fit_climatology(archive)
    gev = fit_gev(archive)
    regime = regime_offset(fetch_recent(loc, temperature_unit=tunit), clim)

    station_warning = None
    if station and station.get("lat") is not None:
        # Station anchoring is best-effort: if Meteostat is unreachable / changed /
        # has no usable record, fall back to grid mode rather than killing the run.
        try:
            from .data.station import grid_to_station_offset, station_archive
            st_archive, station_meta = station_archive(
                float(station["lat"]), float(station["lon"]),
                today - timedelta(days=6 * 365), today, units=tunit)
            grid_to_station = grid_to_station_offset(archive, st_archive)
        except Exception as e:  # any Meteostat/network/parse failure
            station_meta = None
            grid_to_station = 0.0
            station_warning = f"station anchoring unavailable, using grid ({e})"

    effective_offset = station_offset + grid_to_station

    # --- ensemble forecast (grid) ---
    lead = lead_days(target, today)
    ens_samples = None
    if -1 <= lead <= 35:
        days = min(max(lead, 0) + 3, 35)
        ens = fetch_ensemble(loc, forecast_days=days, temperature_unit=tunit)
        ens_samples = ensemble_sample(ens, target, n=20_000, rng=rng)

    # --- blend (whole grid-based distribution shifted to the station) ---
    samples, w_ens = build_distribution(
        clim, target, lead, ens_samples, regime, effective_offset, rng,
        spread_inflation=spread_inflation,
    )

    # --- same-day floor: daily high can't be below what already happened today ---
    floor_info = None
    if same_day_floor:
        try:
            hi = observed_high_today(loc, target, temperature_unit=tunit)
        except Exception:
            hi = None
        if hi is not None:
            floor_val, hour = hi
            samples = np.maximum(samples, floor_val + effective_offset)
            floor_info = {"high_so_far": round(floor_val + effective_offset, 1),
                          "local_hour": hour}

    if not buckets:
        buckets = auto_buckets(float(np.mean(samples)))
    summary = summarize(samples, buckets)

    result = {
        "location": loc.name,
        "coordinates": [round(loc.latitude, 4), round(loc.longitude, 4)],
        "target_date": target.isoformat(),
        "lead_days": lead,
        "unit": sym,
        "history_years": history_years,
        "warming_trend_per_decade": round(clim.trend_per_year * 10, 3),
        "regime_offset": round(regime, 2),
        "station_offset": station_offset,
        "grid_to_station_offset": round(grid_to_station, 2),
        "ensemble_weight": round(w_ens, 3),
        "spread_inflation": spread_inflation,
        **summary,
    }
    if station_meta:
        result["station"] = {
            "name": station_meta["name"],
            "distance_km": station_meta["distance_km"],
            "n_history_days": station_meta["n_days"],
        }
    if station_warning:
        result["station_warning"] = station_warning
    if floor_info:
        result["same_day"] = floor_info
    if gev is not None:
        result["climate_context"] = {
            "record_tmax": round(gev.record, 1),
            "return_level_10yr": round(gev.return_level(10), 1),
            "return_level_50yr": round(gev.return_level(50), 1),
        }

    # LLM is presentation-only: it READS the finished result and writes ONLY to
    # result["commentary"]. It must never feed back into any number above — the
    # math is already complete and frozen at this point.
    if use_llm:
        from .llm.adapter import explain
        result["commentary"] = explain(result, api_key=llm_key, provider=llm_provider)

    return result
