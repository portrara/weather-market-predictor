"""Streamlit dashboard for the Weather Market Predictor.

Run locally:   streamlit run app.py
Deploy free:   push to GitHub -> share.streamlit.io -> point at app.py

Pick a live Kalshi market (or any city/date), and see the model's probability per
temperature bucket vs the live market price, with edge and Kelly stake.
"""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from src.data.kalshi import CITY_SERIES, load_event, open_events
from src.predict import predict

load_dotenv()
st.set_page_config(page_title="Weather Market Predictor", page_icon="🌡️", layout="wide")


@st.cache_data(ttl=300, show_spinner=False)
def _open_events(series): return open_events(series)


@st.cache_data(ttl=300, show_spinner=False)
def _load_event(ticker): return load_event(ticker)


st.title("🌡️ Weather Market Predictor")
st.caption("Predict daily max temperature → bucket probabilities → edge vs live market prices. "
           "Forecast = climatology (harmonic + trend) + recent regime + multi-model ensemble, "
           "lead-time-weighted.")

# ---------------- sidebar ----------------
with st.sidebar:
    st.header("Market")
    mode = st.radio("Source", ["Kalshi (live)", "Manual city + date"])
    st.divider()
    st.header("Model knobs")
    station_offset = st.number_input("Station offset", value=0.0, step=0.1,
        help="Calibrate model grid to the market's resolution station (deg).")
    spread_inflation = st.number_input("Spread inflation", value=1.0, min_value=0.5, step=0.05,
        help="Widen distribution for calibration. Run the backtest tab to find the right value.")
    history_years = st.slider("History years", 10, 30, 20)
    use_llm = st.checkbox("LLM commentary (needs API key in .env)", value=False)

# ---------------- inputs ----------------
kwargs, header, station_note = None, "", ""
if mode == "Kalshi (live)":
    labels = {f"{c} — {note}": k for k, (c, note, *_rest) in CITY_SERIES.items()}
    pick = st.selectbox("City", list(labels))
    series = labels[pick]
    auto_cal = st.checkbox("Auto-calibrate spread (slow first time)", value=True)
    try:
        events = _open_events(series)
    except Exception as e:
        events = []
        st.error(f"Couldn't reach Kalshi: {e}")
    if events:
        event = st.selectbox("Market (event)", events)
        if st.button("Predict", type="primary"):
            ev = _load_event(event)
            header = ev["title"].replace("**", "")
            station_note = f"Resolves at {ev['station_note']} · live prices: " \
                           f"{'yes' if ev['has_prices'] else 'none yet'}"
            station = None
            spread = spread_inflation
            if ev.get("station_latlon"):
                station = {"lat": ev["station_latlon"][0], "lon": ev["station_latlon"][1],
                          "name": ev["station_note"]}
                if auto_cal and spread_inflation == 1.0:
                    from src.calibrate import get_spread_inflation
                    try:
                        with st.spinner("Calibrating spread (first time per station)…"):
                            spread = get_spread_inflation(station["lat"], station["lon"],
                                                          ev["units"], name=station["name"])["spread_inflation"]
                    except Exception as e:
                        st.warning(f"Auto-calibration unavailable ({e}); using spread = 1.0")
            kwargs = dict(target=ev["target"], buckets=ev["buckets"], units=ev["units"],
                          station=station, station_offset=station_offset,
                          spread_inflation=spread, use_llm=use_llm)
    else:
        st.info("No open events for this city right now.")
else:
    c1, c2, c3 = st.columns(3)
    city = c1.text_input("City", "Singapore")
    target = c2.date_input("Date", date.today() + timedelta(days=1))
    units = c3.selectbox("Units", ["celsius", "fahrenheit"])
    if st.button("Predict", type="primary"):
        header = f"{city} — {target}"
        kwargs = dict(city=city, target=target, buckets=None, units=units,
                      station_offset=station_offset, spread_inflation=spread_inflation,
                      use_llm=use_llm)

# ---------------- run + display ----------------
if kwargs:
    with st.spinner("Fetching data and running the model…"):
        try:
            r = predict(**kwargs)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

    if header:
        st.subheader(header)
    if station_note:
        st.caption(station_note)

    u = r["unit"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Expected high", f"{r['mean']:.1f}{u}")
    m2.metric("80% range", f"{r['p10']:.1f}–{r['p90']:.1f}{u}")
    m3.metric("Lead", f"{r['lead_days']} d")
    m4.metric("Ensemble weight", f"{r['ensemble_weight']:.0%}")

    if r.get("station"):
        s = r["station"]
        st.caption(f"📍 Real station: **{s['name']}** ({s['distance_km']} km, "
                   f"{s['n_history_days']} days) · grid→station "
                   f"{r['grid_to_station_offset']:+.1f}{u}")
    if r.get("station_warning"):
        st.warning(f"⚠️ {r['station_warning']}")
    if r.get("same_day"):
        sd = r["same_day"]
        st.caption(f"🌡️ Same-day floor: high so far **{sd['high_so_far']}{u}** "
                   f"(by {sd['local_hour']}:00 local) — daily max can't be below this.")

    buckets = r["buckets"]
    has_mkt = any("market_p" in b for b in buckets)

    # chart: model (+ de-vigged market) probability per bucket
    chart_df = pd.DataFrame({"Model": [b["model_p"] for b in buckets]},
                            index=[b["label"] for b in buckets])
    if has_mkt:
        chart_df["Market (fair)"] = [b.get("market_p_fair", float("nan")) for b in buckets]
    st.bar_chart(chart_df)

    # table with de-vigged edge + kelly
    rows = []
    for b in buckets:
        row = {"Bucket": b["label"], "Model": f"{b['model_p']:.0%}"}
        if "market_p" in b:
            row["Market"] = f"{b['market_p']:.0%}"
            row["Fair"] = f"{b['market_p_fair']:.0%}"
            row["Edge"] = f"{b['edge']:+.0%}"
            row["Kelly"] = f"{b['kelly']:.1%}"
            row["Bet?"] = "✅" if b.get("value") else ""
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    if has_mkt:
        st.caption(f"Market vig ≈ {(r.get('vig',1)-1)*100:.1f}% (removed for 'Fair'). "
                   "✅ = liquid book and edge clears the spread.")

    if "climate_context" in r:
        c = r["climate_context"]
        st.caption(f"Climate context — record {c['record_tmax']}{u}, "
                   f"10-yr return {c['return_level_10yr']}{u}, "
                   f"50-yr return {c['return_level_50yr']}{u} · "
                   f"warming {r['warming_trend_per_decade']:+.2f}{u}/decade · "
                   f"regime {r['regime_offset']:+.1f}{u}")

    if r.get("commentary"):
        st.info(r["commentary"])

    with st.expander("⚠️ Calibration reminder"):
        st.markdown(
            "- **Station offset**: each market resolves on one specific station; "
            "set it so the model matches that source.\n"
            "- **Spread inflation**: run the backtest (CLI: `--backtest`) for this city "
            "to get the value that yields ~80% interval coverage, so edge isn't inflated.")
else:
    st.info("Set up a market on the left and hit **Predict**.")
