"""Unit tests for the math core (no network). Run:  py -m pytest -q"""
import numpy as np
import pandas as pd
import pytest

from src.blend import Bucket, build_distribution, ensemble_weight, kelly_fraction, summarize
from src.models.climatology import fit_climatology
from src.backtest import _crps, _solve_inflation, _coverage_at


def _synthetic_archive(years=20, amp=8.0, mean=25.0, trend=0.02, noise=1.5, seed=0):
    """Daily Tmax = mean + trend*years + seasonal sine + noise."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2004-01-01", periods=years * 365, freq="D")
    doy = dates.dayofyear.to_numpy()
    yr = (dates.year - dates.year[0]).to_numpy()
    season = amp * np.sin(2 * np.pi * (doy - 110) / 365.25)
    tmax = mean + trend * yr + season + rng.normal(0, noise, len(dates))
    return {"daily": {"time": dates.strftime("%Y-%m-%d").tolist(),
                      "temperature_2m_max": tmax.tolist()}}


def test_climatology_recovers_trend_and_amplitude():
    clim = fit_climatology(_synthetic_archive(trend=0.03, amp=8.0))
    # trend coefficient is per-year; should be close to the injected 0.03
    assert abs(clim.trend_per_year - 0.03) < 0.02
    # seasonal swing: hottest minus coldest day of a year ~ 2*amp
    from datetime import date
    temps = [clim.mean(date(2015, m, 15)) for m in range(1, 13)]
    swing = max(temps) - min(temps)
    assert 12 < swing < 20  # ~2*8


def test_kelly_fraction():
    assert kelly_fraction(0.5, 0.5) == 0.0          # no edge
    assert kelly_fraction(0.6, 0.5) > 0.0           # positive edge -> positive stake
    assert kelly_fraction(0.4, 0.5) == 0.0          # negative edge -> no bet
    assert kelly_fraction(0.99, 0.5) <= 0.25        # capped


def test_ensemble_weight_decay():
    assert ensemble_weight(0) > ensemble_weight(9) > ensemble_weight(30)
    assert 0.05 <= ensemble_weight(100) <= 0.95
    assert ensemble_weight(-5) <= 0.95


def test_summarize_probs_and_edge():
    rng = np.random.default_rng(1)
    samples = rng.normal(33, 1.5, 50000)
    buckets = [Bucket("<=32", float("-inf"), 32, price=0.2),
               Bucket("32-34", 32, 34, price=0.5),
               Bucket(">=34", 34, float("inf"), price=0.3)]
    out = summarize(samples, buckets)
    total = sum(b["model_p"] for b in out["buckets"])
    assert abs(total - 1.0) < 1e-9
    for b in out["buckets"]:
        assert "edge" in b and "kelly" in b
        assert abs(b["edge"] - (b["model_p"] - b["market_p"])) < 1e-9


def test_spread_inflation_widens():
    clim = fit_climatology(_synthetic_archive())
    from datetime import date
    rng = np.random.default_rng(2)
    d = date(2026, 6, 1)
    narrow, _ = build_distribution(clim, d, 1, None, 0.0, 0.0, rng, n=40000, spread_inflation=1.0)
    wide, _ = build_distribution(clim, d, 1, None, 0.0, 0.0, rng, n=40000, spread_inflation=2.0)
    assert np.std(wide) > 1.8 * np.std(narrow)
    assert abs(np.mean(wide) - np.mean(narrow)) < 0.2  # mean preserved


def test_crps_perfect_vs_noisy():
    rng = np.random.default_rng(3)
    y = 30.0
    perfect = np.full(2000, y)
    noisy = rng.normal(y, 3.0, 2000)
    assert _crps(perfect, y, rng) < _crps(noisy, y, rng)
    assert _crps(perfect, y, rng) < 0.01


def test_kalshi_parse_event_ticker():
    from datetime import date
    from src.data.kalshi import parse_event_ticker
    assert parse_event_ticker("KXHIGHNY-26MAY28") == ("KXHIGHNY", date(2026, 5, 28))
    assert parse_event_ticker("KXHIGHLAX-26JAN03") == ("KXHIGHLAX", date(2026, 1, 3))


def test_kalshi_bucket_bounds():
    from src.data.kalshi import _market_bucket
    between = _market_bucket({"strike_type": "between", "floor_strike": 78,
                              "cap_strike": 79, "subtitle": "78° to 79°", "ticker": "x"})
    assert (between.low, between.high) == (77.5, 79.5)
    below = _market_bucket({"strike_type": "less", "cap_strike": 76,
                            "subtitle": "75° or below", "ticker": "x"})
    assert below.low == float("-inf") and below.high == 75.5
    above = _market_bucket({"strike_type": "greater", "floor_strike": 83,
                            "subtitle": "84° or above", "ticker": "x"})
    assert above.low == 83.5 and above.high == float("inf")


def test_devig_and_liquidity_gating():
    rng = np.random.default_rng(9)
    samples = rng.normal(33, 1.5, 50000)
    # raw mid prices sum to 1.2 (vig); tight book on bucket 0, wide on bucket 2
    buckets = [
        Bucket("<=32", float("-inf"), 32, price=0.30, bid=0.28, ask=0.32),
        Bucket("32-34", 32, 34, price=0.60, bid=0.58, ask=0.62),
        Bucket(">=34", 34, float("inf"), price=0.30, bid=0.05, ask=0.55),  # illiquid
    ]
    out = summarize(samples, buckets)
    assert out["vig"] == pytest.approx(1.2, abs=1e-6)
    fair_sum = sum(b["market_p_fair"] for b in out["buckets"])
    assert fair_sum == pytest.approx(1.0, abs=1e-6)        # de-vigged sums to 1
    assert out["buckets"][2]["liquid"] is False            # wide spread -> not liquid
    assert out["buckets"][2]["value"] is False             # never a signal if illiquid


def test_grid_to_station_offset():
    from src.data.station import grid_to_station_offset
    grid = {"daily": {"time": ["2024-01-01", "2024-01-02", "2024-01-03"],
                      "temperature_2m_max": [10.0, 12.0, 11.0]}}
    stn = {"daily": {"time": ["2024-01-01", "2024-01-02", "2024-01-03"],
                     "temperature_2m_max": [11.0, 13.5, 12.0]}}
    # diffs: +1.0, +1.5, +1.0 -> median 1.0
    assert grid_to_station_offset(grid, stn) == pytest.approx(1.0)


def test_kalshi_implied_prob():
    from src.data.kalshi import _implied_prob
    assert _implied_prob({"yes_bid_dollars": 0.5, "yes_ask_dollars": 0.6}) == 0.55
    assert _implied_prob({"yes_bid_dollars": None, "yes_ask_dollars": None,
                          "last_price_dollars": 0.42}) == 0.42
    assert _implied_prob({"yes_bid_dollars": None, "yes_ask_dollars": None,
                          "last_price_dollars": None}) is None


def test_solve_inflation_hits_target():
    rng = np.random.default_rng(4)
    n = 2000
    means = np.full(n, 30.0)
    obs = rng.normal(30.0, 2.0, n)
    # intervals too narrow (sd 1 instead of 2): solver should suggest ~2x
    p10 = means - 1.2816 * 1.0
    p90 = means + 1.2816 * 1.0
    k = _solve_inflation(means, p10, p90, obs, target=0.80)
    assert _coverage_at(means, p10, p90, obs, k) == pytest.approx(0.80, abs=0.05)
    assert 1.6 < k < 2.4
