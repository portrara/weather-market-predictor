"""Validation harness — proves the statistical core actually has skill.

Two checks, both fully automated against free Open-Meteo data:

1. walk_forward(): leak-free walk-forward backtest of the climatology+regime
   model. For each past eval date we fit climatology ONLY on data strictly
   before it, add the regime offset from the preceding 30 days, draw the
   predictive distribution, and score it against the reanalysis truth. Scored
   with MAE/RMSE/bias (point), CRPS (full distribution), and multi-bucket Brier,
   each compared to two baselines: climatology-only and persistence.

2. forecast_bias(): compares archived operational forecasts to reanalysis truth.
   The mean difference is the model's systematic grid bias — a sensible starting
   value for `station_offset` before you calibrate to the market's exact station.

Note: the free API doesn't expose true multi-day-ahead past forecasts, so the
ensemble branch can't be backtested at long lead here. The walk-forward validates
the climatology+regime branch, which is what carries weight at medium/long lead
(where markets are most mispriced); forecast_bias validates short-lead calibration.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

from .blend import build_distribution
from .data.openmeteo import Location, fetch_archive, geocode
from .models.climatology import fit_climatology


def _archive_df(archive: dict) -> pd.DataFrame:
    d = archive["daily"]
    return pd.DataFrame({
        "time": pd.to_datetime(d["time"]),
        "tmax": d["temperature_2m_max"],
    }).dropna().reset_index(drop=True)


def _df_to_archive(df: pd.DataFrame) -> dict:
    return {"daily": {
        "time": df["time"].dt.strftime("%Y-%m-%d").tolist(),
        "temperature_2m_max": df["tmax"].tolist(),
    }}


def _crps(samples: np.ndarray, y: float, rng: np.random.Generator, m: int = 1500) -> float:
    """Empirical CRPS = E|X-y| - 0.5 E|X-X'|  (lower is better)."""
    s = samples if samples.size <= m else rng.choice(samples, m, replace=False)
    t1 = np.mean(np.abs(s - y))
    t2 = np.mean(np.abs(s[:, None] - s[None, :]))
    return float(t1 - 0.5 * t2)


def _brier_multi(samples: np.ndarray, y: float, center: float) -> float:
    """Multi-category Brier over 1C buckets centered near the prediction."""
    lo = np.floor(center) - 3
    edges = [lo + i for i in range(7)]  # 6 interior + 2 open ends => 8 buckets
    bounds = [(-np.inf, edges[0])] + \
             [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)] + \
             [(edges[-1], np.inf)]
    score = 0.0
    for low, high in bounds:
        p = float(np.mean((samples >= low) & (samples < high)))
        o = 1.0 if (low <= y < high) else 0.0
        score += (p - o) ** 2
    return score


@dataclass
class BacktestResult:
    location: str
    n: int
    eval_days: int
    step: int
    mae: float
    rmse: float
    bias: float
    crps: float
    brier: float
    mae_climatology: float
    mae_persistence: float
    crps_climatology: float
    crps_skill: float          # 1 - crps/crps_climatology  (>0 = beats baseline)
    coverage_80: float         # fraction of truths inside the 80% interval
    spread_inflation: float    # inflation used for this run
    suggested_spread_inflation: float  # value that would yield ~80% coverage
    unit: str = "°C"           # display unit for the scores


def _coverage_at(means, p10s, p90s, obs, kappa):
    lo = means + (p10s - means) * kappa
    hi = means + (p90s - means) * kappa
    return float(np.mean((obs >= lo) & (obs <= hi)))


def _solve_inflation(means, p10s, p90s, obs, target=0.80):
    """Bisect for the spread scale that hits target 80% coverage."""
    lo, hi = 0.5, 4.0
    for _ in range(40):
        mid = (lo + hi) / 2
        if _coverage_at(means, p10s, p90s, obs, mid) < target:
            lo = mid
        else:
            hi = mid
    return round((lo + hi) / 2, 3)


def walk_forward(*, city: str | None = None, location: Location | None = None,
                 history_years: int = 18, eval_days: int = 365, step: int = 7,
                 min_train_years: int = 8, seed: int = 11,
                 spread_inflation: float = 1.0, units: str = "celsius") -> BacktestResult:
    loc = location or geocode(city)
    rng = np.random.default_rng(seed)
    tunit = "fahrenheit" if str(units).lower() in ("f", "fahrenheit") else "celsius"

    today = date.today()
    total_back = int((history_years + eval_days / 365.25 + 1) * 365.25)
    start = (today - timedelta(days=total_back)).isoformat()
    end = (today - timedelta(days=6)).isoformat()  # reanalysis lag
    df = _archive_df(fetch_archive(loc, start, end, temperature_unit=tunit))
    df_idx = df.set_index("time")["tmax"]

    last = df["time"].max()
    eval_dates = pd.date_range(end=last, periods=eval_days // step, freq=f"{step}D")

    err, ae_clim, ae_pers, crps_m, crps_c, brier, inside = [], [], [], [], [], [], []
    means, p10s, p90s, obss = [], [], [], []
    for ts in eval_dates:
        d = ts.date()
        train = df[df["time"] < ts]
        if train.empty or (train["time"].max() - train["time"].min()).days < min_train_years * 365:
            continue
        if ts not in df_idx.index:
            continue
        obs = float(df_idx.loc[ts])

        clim = fit_climatology(_df_to_archive(train))

        # regime: median(obs - clim) over the preceding 30 days
        win = df[(df["time"] >= ts - pd.Timedelta(days=31)) & (df["time"] < ts)]
        regime = float(np.median([o - clim.mean(t.date())
                                  for t, o in zip(win["time"], win["tmax"])])) if not win.empty else 0.0

        samples, _ = build_distribution(clim, d, lead=1, ensemble_samples=None,
                                        regime=regime, station_offset=0.0, rng=rng,
                                        n=8000, spread_inflation=spread_inflation)
        base_samples = clim.sample(d, 8000, rng)  # climatology-only baseline

        pred = float(np.mean(samples))
        err.append(pred - obs)
        ae_clim.append(abs(clim.mean(d) - obs))
        # persistence baseline = yesterday's observed
        prev = df[df["time"] < ts]["tmax"]
        ae_pers.append(abs(float(prev.iloc[-1]) - obs))
        crps_m.append(_crps(samples, obs, rng))
        crps_c.append(_crps(base_samples, obs, rng))
        brier.append(_brier_multi(samples, obs, pred))
        lo, hi = np.percentile(samples, [10, 90])
        inside.append(lo <= obs <= hi)
        means.append(pred); p10s.append(lo); p90s.append(hi); obss.append(obs)

    err = np.array(err)
    means, p10s, p90s, obss = map(np.array, (means, p10s, p90s, obss))
    suggested = _solve_inflation(means, p10s, p90s, obss) * spread_inflation
    crps_model = float(np.mean(crps_m))
    crps_clim = float(np.mean(crps_c))
    return BacktestResult(
        location=loc.name, n=len(err), eval_days=eval_days, step=step,
        mae=float(np.mean(np.abs(err))),
        rmse=float(np.sqrt(np.mean(err ** 2))),
        bias=float(np.mean(err)),
        crps=crps_model,
        brier=float(np.mean(brier)),
        mae_climatology=float(np.mean(ae_clim)),
        mae_persistence=float(np.mean(ae_pers)),
        crps_climatology=crps_clim,
        crps_skill=float(1 - crps_model / crps_clim) if crps_clim else 0.0,
        coverage_80=float(np.mean(inside)),
        spread_inflation=spread_inflation,
        suggested_spread_inflation=float(suggested),
        unit="°F" if tunit == "fahrenheit" else "°C",
    )


def forecast_bias(*, city: str | None = None, location: Location | None = None,
                  days: int = 120, units: str = "celsius") -> dict:
    """Mean (archived forecast - reanalysis truth) over recent days.
    Suggests a starting `station_offset` to remove the model's grid bias."""
    from .data.openmeteo import ARCHIVE_URL, _get

    tunit = "fahrenheit" if str(units).lower() in ("f", "fahrenheit") else "celsius"
    loc = location or geocode(city)
    end = (date.today() - timedelta(days=6))
    start = end - timedelta(days=days)
    common = {"latitude": loc.latitude, "longitude": loc.longitude,
              "start_date": start.isoformat(), "end_date": end.isoformat(),
              "daily": "temperature_2m_max", "temperature_unit": tunit,
              "timezone": loc.timezone}
    truth = _get(ARCHIVE_URL, common, ttl_hours=0)
    fc = _get("https://historical-forecast-api.open-meteo.com/v1/forecast", common, ttl_hours=24)

    tt = dict(zip(truth["daily"]["time"], truth["daily"]["temperature_2m_max"]))
    ff = dict(zip(fc["daily"]["time"], fc["daily"]["temperature_2m_max"]))
    diffs = [ff[k] - tt[k] for k in tt
             if k in ff and tt[k] is not None and ff[k] is not None]
    diffs = np.array(diffs, dtype=float)
    return {
        "location": loc.name,
        "unit": "°F" if tunit == "fahrenheit" else "°C",
        "n_days": int(diffs.size),
        "mean_forecast_bias": round(float(np.mean(diffs)), 2) if diffs.size else None,
        "mae": round(float(np.mean(np.abs(diffs))), 2) if diffs.size else None,
        "suggested_station_offset": round(-float(np.mean(diffs)), 2) if diffs.size else 0.0,
    }
