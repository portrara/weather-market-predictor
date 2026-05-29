"""Kalshi live market data (read-only, no auth needed).

Kalshi runs a market per US city per day, e.g. event KXHIGHNY-26MAY28 =
"Highest temperature in NYC on May 28". We pull its buckets + live prices and
hand them to the model. Kalshi resolves against official NWS station data, so the
resolution source is explicit (great for calibration).

Temperatures are in Fahrenheit; the pipeline runs in F for these markets.
Trading needs a verified Kalshi account, but market DATA is public.
"""
from __future__ import annotations

import time
from datetime import date

import requests

from ..blend import Bucket

KALSHI = "https://api.elections.kalshi.com/trade-api/v2"

# Kalshi high-temperature series -> resolution station (verified against each
# market's rules_primary). (geocode label, station name, station lat, station lon)
CITY_SERIES = {
    "KXHIGHNY":   ("New York",      "Central Park, NYC (KNYC)",        40.7789,  -73.9692),
    "KXHIGHLAX":  ("Los Angeles",   "Los Angeles Intl (KLAX)",        33.9381, -118.3889),
    "KXHIGHCHI":  ("Chicago",       "Chicago Midway (KMDW)",          41.7860,  -87.7524),
    "KXHIGHMIA":  ("Miami",         "Miami Intl (KMIA)",              25.7906,  -80.2906),
    "KXHIGHAUS":  ("Austin",        "Austin-Bergstrom (KAUS)",        30.1975,  -97.6664),
    "KXHIGHDEN":  ("Denver",        "Denver Intl (KDEN)",             39.8467, -104.6562),
    "KXHIGHPHIL": ("Philadelphia",  "Philadelphia Intl (KPHL)",       39.8721,  -75.2411),
}

_MONTHS = {m: i for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"], 1)}


def _get(path: str, params: dict, retries: int = 3, timeout: int = 30) -> dict:
    last = None
    for attempt in range(retries):
        try:
            r = requests.get(f"{KALSHI}{path}", params=params, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            last = e
            if attempt < retries - 1:
                time.sleep(2 ** attempt)  # 1s, 2s backoff
    raise RuntimeError(f"Kalshi request failed after {retries} tries: {last}")


def parse_event_ticker(event_ticker: str) -> tuple[str, date]:
    """'KXHIGHNY-26MAY28' -> ('KXHIGHNY', date(2026,5,28))."""
    try:
        series, datestr = event_ticker.split("-", 1)
        yy, mon, dd = int(datestr[:2]), datestr[2:5].upper(), int(datestr[5:])
        return series, date(2000 + yy, _MONTHS[mon], dd)
    except (ValueError, KeyError, IndexError):
        raise ValueError(
            f"Malformed Kalshi event ticker: {event_ticker!r} "
            f"(expected like KXHIGHNY-26MAY28)")


def _implied_prob(m: dict) -> float | None:
    """Mid of yes bid/ask (in dollars = probability), else last trade, else None."""
    bid, ask = m.get("yes_bid_dollars"), m.get("yes_ask_dollars")
    last = m.get("last_price_dollars")
    if bid is not None and ask is not None and float(ask) > 0:
        return round((float(bid) + float(ask)) / 2, 4)
    if last is not None and float(last) > 0:
        return round(float(last), 4)
    return None


def _market_bucket(m: dict) -> Bucket:
    st = m.get("strike_type")
    floor, cap = m.get("floor_strike"), m.get("cap_strike")
    inf = float("inf")
    if st == "between":
        low, high = float(floor) - 0.5, float(cap) + 0.5
    elif st in ("less", "less_or_equal"):
        low, high = -inf, float(cap) - (0.5 if st == "less" else -0.5)
    elif st in ("greater", "greater_or_equal"):
        low, high = float(floor) + (0.5 if st == "greater" else -0.5), inf
    else:  # fallback: use whatever bounds exist
        low = float(floor) - 0.5 if floor is not None else -inf
        high = float(cap) + 0.5 if cap is not None else inf
    label = m.get("subtitle") or m.get("yes_sub_title") or m["ticker"].split("-")[-1]
    bid, ask = m.get("yes_bid_dollars"), m.get("yes_ask_dollars")
    return Bucket(label=label, low=low, high=high, price=_implied_prob(m),
                  bid=float(bid) if bid is not None else None,
                  ask=float(ask) if ask is not None else None)


def fetch_event_markets(event_ticker: str) -> list[dict]:
    data = _get("/markets", {"event_ticker": event_ticker, "limit": 100})
    return data.get("markets", [])


def open_events(series_ticker: str, limit: int = 200) -> list[str]:
    """Upcoming event tickers (today onward) for a series, soonest first.
    Robust to Kalshi status-string changes: we filter by the date in the ticker."""
    events = {}
    for params in ({"series_ticker": series_ticker, "status": "open", "limit": limit},
                   {"series_ticker": series_ticker, "limit": limit}):
        try:
            data = _get("/markets", params)
        except RuntimeError:
            continue
        for m in data.get("markets", []):
            et = m.get("event_ticker")
            if not et or et in events:
                continue
            try:
                events[et] = parse_event_ticker(et)[1]
            except Exception:
                continue
        if events:  # first (status=open) query was enough
            break
    today = date.today()
    upcoming = [(d, et) for et, d in events.items() if d >= today]
    return [et for _, et in sorted(upcoming)]


def load_event(event_ticker: str) -> dict:
    """Everything the pipeline needs for one Kalshi market."""
    series, target = parse_event_ticker(event_ticker)
    markets = fetch_event_markets(event_ticker)
    if not markets:
        raise ValueError(f"No Kalshi markets found for event {event_ticker!r}")
    meta = CITY_SERIES.get(series)
    if meta:
        city, station, slat, slon = meta
    else:
        city, station, slat, slon = series, "unknown station", None, None
    buckets = sorted((_market_bucket(m) for m in markets),
                     key=lambda b: (b.low, b.high))
    has_prices = any(b.price is not None for b in buckets)
    # Event-level title. markets[0]["title"] is only ONE sub-market's question
    # (e.g. ">85°"), which contradicts the full multi-bucket table, so build the
    # event title ourselves from the resolved city + date.
    title = f"Highest temperature in {city} on {target.strftime('%b %d, %Y')}"
    return {
        "event_ticker": event_ticker,
        "series": series,
        "city": city,
        "station_note": station,
        "station_latlon": (slat, slon) if slat is not None else None,
        "target": target,
        "units": "fahrenheit",
        "buckets": buckets,
        "has_prices": has_prices,
        "title": title,
        "rules": (markets[0].get("rules_primary") or "")[:400],
    }
