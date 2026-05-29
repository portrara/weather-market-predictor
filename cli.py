"""Command-line interface for the weather prediction-market model.

Examples:
    py cli.py --market singapore_2026_05_28
    py cli.py --city "Singapore" --date 2026-05-28
    py cli.py --city "Singapore" --date 2026-05-28 --no-llm --json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime
from pathlib import Path

import yaml
from dotenv import load_dotenv

from src.blend import Bucket
from src.predict import predict

load_dotenv()
CONFIG = Path(__file__).parent / "config" / "markets.yaml"


def _bucket(d: dict) -> Bucket:
    return Bucket(
        label=d["label"],
        low=float(d["low"]) if d.get("low") is not None else float("-inf"),
        high=float(d["high"]) if d.get("high") is not None else float("inf"),
        price=float(d["price"]) if d.get("price") is not None else None,
    )


def load_market(name: str):
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    if name not in cfg:
        sys.exit(f"Market {name!r} not found in {CONFIG}. Available: {', '.join(cfg)}")
    m = cfg[name]
    return {
        "city": m["city"],
        "target": datetime.fromisoformat(str(m["date"])).date(),
        "history_years": int(m.get("history_years", 20)),
        "station_offset": float(m.get("station_offset", 0.0)),
        "spread_inflation": float(m.get("spread_inflation", 1.0)),
        "buckets": [_bucket(b) for b in m["buckets"]],
    }


def render(r: dict) -> str:
    L = []
    L.append("")
    L.append(f"  {r['location']}  ({r['coordinates'][0]:.3f}, {r['coordinates'][1]:.3f})")
    L.append(f"  Target: {r['target_date']}   lead: {r['lead_days']}d   "
             f"ensemble weight: {r['ensemble_weight']:.0%}")
    u = r.get("unit", "°C")
    if r.get("station"):
        s = r["station"]
        L.append(f"  Station: {s['name']} ({s['distance_km']} km, {s['n_history_days']} days) "
                 f"| grid→station {r['grid_to_station_offset']:+.1f}{u}")
    L.append(f"  Expected Tmax: {r['mean']:.1f}{u}   median {r['median']:.1f}{u}   "
             f"80% range [{r['p10']:.1f}, {r['p90']:.1f}]{u}")
    L.append(f"  Regime offset: {r['regime_offset']:+.1f}{u}   "
             f"station offset: {r['station_offset']:+.1f}{u}   "
             f"warming: {r['warming_trend_per_decade']:+.2f}{u}/decade")
    if r.get("station_warning"):
        L.append(f"  Note: {r['station_warning']}")
    if r.get("same_day"):
        sd = r["same_day"]
        L.append(f"  Same-day floor: high so far {sd['high_so_far']}{u} "
                 f"(by {sd['local_hour']}:00 local)")
    if "climate_context" in r:
        c = r["climate_context"]
        L.append(f"  Climate: record {c['record_tmax']}{u}, "
                 f"10yr {c['return_level_10yr']}{u}, 50yr {c['return_level_50yr']}{u}")
    L.append("")
    has_mkt = any("market_p" in b for b in r["buckets"])
    if has_mkt:
        L.append(f"  {'Bucket':<14}{'Model':>7}{'Fair':>7}{'Edge':>7}{'Kelly':>7}")
        L.append("  " + "-" * 44)
        for b in r["buckets"]:
            fair = f"{b.get('market_p_fair', b.get('market_p', 0)):.0%}" if "market_p" in b else "-"
            ed = f"{b['edge']:+.0%}" if "edge" in b else "-"
            ke = f"{b['kelly']:.1%}" if "kelly" in b else "-"
            star = "  BET" if b.get("value") else ""
            L.append(f"  {b['label']:<14}{b['model_p']:>6.0%}{fair:>7}{ed:>7}{ke:>7}{star}")
    else:
        L.append(f"  {'Bucket':<12}{'Model prob':>12}")
        L.append("  " + "-" * 24)
        for b in r["buckets"]:
            bar = "#" * int(round(b["model_p"] * 30))
            L.append(f"  {b['label']:<12}{b['model_p']:>11.0%}  {bar}")
    if r.get("commentary"):
        L.append("")
        L.append("  Analyst read:")
        for line in r["commentary"].splitlines():
            L.append(f"    {line}")
    L.append("")
    return "\n".join(L)


def render_backtest(r) -> str:
    skill = "BEATS" if r.crps_skill > 0 else "loses to"
    u = getattr(r, "unit", "°C")
    return "\n".join([
        "",
        f"  Walk-forward backtest — {r.location}",
        f"  {r.n} eval days over the last {r.eval_days} (every {r.step}d)",
        "  " + "-" * 50,
        f"  Point error   MAE {r.mae:.2f}{u}   RMSE {r.rmse:.2f}{u}   bias {r.bias:+.2f}{u}",
        f"  Distribution  CRPS {r.crps:.3f}   multi-bucket Brier {r.brier:.3f}",
        f"  80% interval coverage: {r.coverage_80:.0%}  (target ~80%, "
        f"spread_inflation={r.spread_inflation})",
        "",
        f"  vs baselines  MAE: model {r.mae:.2f} | climatology {r.mae_climatology:.2f} "
        f"| persistence {r.mae_persistence:.2f}",
        f"                CRPS: model {r.crps:.3f} | climatology {r.crps_climatology:.3f}",
        f"  => model {skill} climatology baseline (CRPS skill {r.crps_skill:+.1%})",
        "",
        f"  Suggested spread_inflation for ~80% coverage: {r.suggested_spread_inflation}",
        "  (set this in config/markets.yaml so bucket probabilities are well-calibrated)",
        "",
    ])


def render_calibrate(b: dict) -> str:
    u = b.get("unit", "°C")
    return "\n".join([
        "",
        f"  Calibration — {b['location']}",
        f"  Compared {b['n_days']} days of archived forecast vs reanalysis truth.",
        f"  Forecast bias: {b['mean_forecast_bias']:+}{u}   (MAE {b['mae']}{u})",
        "",
        f"  Suggested starting station_offset: {b['suggested_station_offset']:+}{u}",
        "  (This removes model grid bias. Still calibrate further to the market's",
        "   exact resolution station once you have a few days of its reported highs.)",
        "",
    ])


def _parse_date(s: str) -> date:
    try:
        return datetime.fromisoformat(s).date()
    except ValueError:
        raise ValueError(f"--date must be YYYY-MM-DD (got {s!r})")


def _dispatch(args, ap):
    # Live Kalshi market: auto-pull buckets + prices, run prediction in F
    if args.kalshi_event:
        from src.data.kalshi import load_event
        ev = load_event(args.kalshi_event)
        title = ev["title"].replace("**", "")
        print(f"\n  Kalshi: {title}")
        print(f"  Event {ev['event_ticker']} | resolves at {ev['station_note']}"
              f" | live prices: {'yes' if ev['has_prices'] else 'none yet'}")
        station = None
        if ev.get("station_latlon"):
            station = {"lat": ev["station_latlon"][0], "lon": ev["station_latlon"][1],
                       "name": ev["station_note"]}
        # auto-calibrate spread inflation for this station (cached) unless overridden
        spread = args.spread_inflation
        if spread == 1.0 and station and not args.no_calibrate:
            from src.calibrate import get_spread_inflation
            try:
                cal = get_spread_inflation(station["lat"], station["lon"], ev["units"],
                                           name=station["name"])
                spread = cal["spread_inflation"]
                print(f"  Auto-calibrated spread_inflation = {spread} "
                      f"({'cached' if cal['cached'] else 'computed'})")
            except Exception as e:
                print(f"  Auto-calibration unavailable ({e}); using spread_inflation = 1.0")
        result = predict(target=ev["target"], buckets=ev["buckets"], units=ev["units"],
                         station=station, station_offset=args.station_offset,
                         spread_inflation=spread, use_llm=not args.no_llm)
        print(json.dumps(result, indent=2, default=str) if args.json else render(result))
        return

    # Resolve a city for backtest/calibrate from --city or a named market
    aux_city = args.city
    if not aux_city and args.market:
        aux_city = load_market(args.market)["city"]

    if args.backtest:
        from src.backtest import walk_forward
        if not aux_city:
            ap.error("--backtest needs --city or --market")
        r = walk_forward(city=aux_city, eval_days=args.eval_days, step=args.step,
                         spread_inflation=args.spread_inflation, units=args.units)
        print(render_backtest(r) if not args.json else json.dumps(r.__dict__, indent=2))
        return

    if args.calibrate:
        from src.backtest import forecast_bias
        if not aux_city:
            ap.error("--calibrate needs --city or --market")
        b = forecast_bias(city=aux_city, units=args.units)
        print(json.dumps(b, indent=2) if args.json else render_calibrate(b))
        return

    if args.market:
        m = load_market(args.market)
        kwargs = dict(city=m["city"], target=m["target"], buckets=m["buckets"],
                      history_years=m["history_years"], station_offset=m["station_offset"],
                      spread_inflation=m["spread_inflation"])
    elif args.city and args.date:
        kwargs = dict(city=args.city, target=_parse_date(args.date), buckets=None,
                      history_years=args.history_years, station_offset=args.station_offset,
                      spread_inflation=args.spread_inflation, units=args.units)
    else:
        ap.error("Provide --market NAME, or both --city and --date.")

    result = predict(use_llm=not args.no_llm, **kwargs)
    print(json.dumps(result, indent=2, default=str) if args.json else render(result))


def main():
    # Windows consoles default to cp1252 and mangle the degree symbol; force UTF-8.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="Predict daily max temperature for prediction markets.")
    ap.add_argument("--market", help="Named market from config/markets.yaml")
    ap.add_argument("--kalshi-event", help="Live Kalshi event, e.g. KXHIGHNY-26MAY28")
    ap.add_argument("--city", help="Location name (ad-hoc run)")
    ap.add_argument("--date", help="Target date YYYY-MM-DD (ad-hoc run)")
    ap.add_argument("--history-years", type=int, default=20)
    ap.add_argument("--units", choices=["celsius", "fahrenheit"], default="celsius",
                    help="Units for --city/--date, --backtest and --calibrate "
                         "(Kalshi events are always Fahrenheit)")
    ap.add_argument("--station-offset", type=float, default=0.0)
    ap.add_argument("--spread-inflation", type=float, default=1.0,
                    help="Widen distribution for calibration (see --backtest)")
    ap.add_argument("--no-llm", action="store_true", help="Skip LLM commentary")
    ap.add_argument("--json", action="store_true", help="Raw JSON output")
    ap.add_argument("--backtest", action="store_true",
                    help="Walk-forward skill test (needs --city or --market)")
    ap.add_argument("--calibrate", action="store_true",
                    help="Suggest a station_offset from recent forecast bias")
    ap.add_argument("--no-calibrate", action="store_true",
                    help="Skip auto spread-inflation calibration for Kalshi events")
    ap.add_argument("--eval-days", type=int, default=365, help="Backtest window")
    ap.add_argument("--step", type=int, default=7, help="Backtest eval spacing (days)")
    args = ap.parse_args()

    try:
        _dispatch(args, ap)
    except (ValueError, RuntimeError, KeyError) as e:
        sys.exit(f"Error: {e}")
    except KeyboardInterrupt:
        sys.exit("\nInterrupted.")


if __name__ == "__main__":
    main()
