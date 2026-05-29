# Weather Market Predictor

Predicts the **daily maximum temperature** for a city/airport and turns it into a
**probability for each prediction-market bucket** (e.g. Polymarket / Kalshi
"Highest temperature in Singapore on May 28?"). When you supply the market's
prices, it also reports your **edge** and a suggested **Kelly stake**.

The core predictor is statistical and physics-driven — no API key, runs free. An
**optional, pluggable LLM layer** (Groq by default) writes a plain-English read of
the situation. You can swap the LLM provider with two lines in `.env`.

---

## How it works

```
 history (ERA5, ~20 yrs) ─► climatology  (harmonic regression + warming trend)
 recent observations ─────► regime offset (current hot/cool spell, decays w/ lead)
 ensemble forecast (NWP) ─► member cloud  (GFS + ECMWF + ICON + GEM, ~100 members)
 config ──────────────────► station offset (calibrate to the resolution source)
                                   │
                                   ▼
                 lead-time-weighted Monte-Carlo blend  (60k samples)
                                   │
                ┌──────────────────┼───────────────────┐
                ▼                  ▼                   ▼
        bucket probabilities   summary stats      EVT context
        + edge + Kelly         (mean, p10/p90)   (record, return levels)
```

**Why this is accurate:** near-term, a multi-model ensemble already encodes the
real atmospheric state — its spread *is* the probability distribution. As lead
time grows, ensemble skill decays, so the blend smoothly hands weight to
climatology (`ensemble_weight` in the output shows the mix). Extreme Value Theory
contextualizes the hot-tail buckets that mean-regression alone underestimates.

### The math
- **Climatology:** `T(d,y) = a0 + trend·y + Σ aₖ sin(2πk d/365.25) + bₖ cos(...)`
  fit by least squares; spread comes from the *empirical* residuals in a ±15-day
  window (so it's not forced Gaussian).
- **Blend weight:** logistic decay `w = 1/(1+e^((lead−9)/2))`, clipped to [5%, 95%].
- **Kelly:** `f* = (p(b+1) − 1)/b`, `b = 1/price − 1`, clipped to [0, 25%].
- **Extremes:** GEV fit to annual maxima for record / N-year return levels.

---

## The #1 accuracy lever: station calibration

A market resolves against **one specific station** (a named weather service /
airport), and a model grid point can differ from it by 1–3°.

- **Kalshi markets: handled automatically** — the tool reads the real station's
  record (Meteostat) and applies a computed grid→station offset.
- **Manual markets (`config/markets.yaml`): set `station_offset` yourself** —
  compare a few days of the official source's reported highs to this tool's `mean`
  and enter the average difference. Or run `--calibrate` for a starting value.

> Example: geocoding "Singapore" returns the city center grid (~30.9 °C), but a
> market may resolve on **Changi Airport** (warmer). Calibrate accordingly.

---

## Validation (proven skill, not vibes)

A leak-free **walk-forward backtest** is built in. On Singapore (52 weekly eval
days over the last year, climatology+regime branch):

| Metric | Model | Climatology | Persistence |
|---|---|---|---|
| MAE | **1.16 °C** | 1.36 °C | 1.03 °C |
| CRPS | **0.83** | 1.05 | — |

- **CRPS skill +23%** over the climatology baseline — the regime term adds real value.
- The backtest also caught that raw intervals were **overconfident** (62% coverage
  for a nominal 80%), and **solves for the `spread_inflation`** that fixes it
  (1.54 → 79% coverage). Calibrated probabilities matter: overconfidence invents
  fake betting edge.
- Persistence wins at lead-1 (tropics barely move day-to-day); the model's edge is
  at the medium/long lead the markets actually price. The ensemble branch (not in
  this backtest) sharpens the short lead.

**Skill varies by climate — always backtest the specific city.** In stable climates
the regime term shines (Singapore CRPS skill **+21%**); in volatile mid-latitude
maritime weather it ties climatology (London **−0.2%**) because fast-moving systems
aren't captured by a 30-day anomaly — there the ensemble does the work at short
lead and climatology is the long-lead fallback. The backtest tells you which case
you're in before you risk money.

Run it yourself: `py cli.py --city "Singapore" --backtest`

## Live market data (Kalshi) — station-accurate

Kalshi runs a **daily high-temperature market per US city** (NY, LA, Chicago,
Miami, Austin, Denver, Philadelphia) via a **free, no-auth data API**. The tool
pulls the exact buckets + live prices automatically. (Polymarket has no comparable
standing temp markets / open API, so live data uses Kalshi; manual YAML config
still works for anything else.)

```bash
.venv\Scripts\python.exe cli.py --kalshi-event KXHIGHNY-26MAY29
```

For each market the tool automatically:
- **Anchors to the real resolution station.** Each market's coords are verified
  against its `rules_primary` (e.g. NY → NWS Central Park, Austin → Bergstrom). It
  pulls that **station's actual record via Meteostat** and computes a
  **grid→station offset** so the forecast matches the source the market settles on.
- **Applies a same-day floor.** On the day of resolution, the high observed *so
  far* is a hard lower bound on the daily max — a real edge late in the day.
- **De-vigs the market.** Bucket mid-prices are normalized to remove the vig, and
  a bet is only flagged when the book is **liquid** (tight spread) *and* the edge
  clears the spread. Kelly is sized against the ask you'd actually pay.
- **Auto-calibrates the spread** per station (cached) so probabilities aren't
  overconfident. (Calibrated on the climatology branch — most rigorous at
  medium/long lead; short-lead ensemble calibration is approximate, limited by the
  free APIs' lack of archived per-lead forecasts.)

> Trading on Kalshi needs a verified US account, but market **data** is public —
> fine for the model and research wherever you are.

## Web UI (Streamlit)

```bash
.venv\Scripts\python.exe -m streamlit run app.py
```

Pick a live Kalshi market (or any city/date), and see model probabilities vs live
prices, edge, Kelly, a bucket chart, and climate context. Deploy free by pushing
to GitHub and pointing share.streamlit.io at `app.py`.

## Setup

Requires **Python 3.10+**.

```bash
# Windows
py -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pytest -q          # 12 unit tests on the math core

# macOS / Linux
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pytest -q
```

> For a byte-for-byte reproducible environment, use the pinned set:
> `pip install -r requirements.lock`. The commands below use the Windows venv path
> (`.venv\Scripts\python.exe`); on macOS/Linux use `.venv/bin/python`.

## Usage

```bash
# Run a named market from config/markets.yaml
.venv\Scripts\python.exe cli.py --market singapore_2026_05_28

# Ad-hoc, any city + date (auto 1-degree buckets)
.venv\Scripts\python.exe cli.py --city "Singapore" --date 2026-05-28

# Raw JSON (for piping into other tools / your own AI)
.venv\Scripts\python.exe cli.py --market phoenix_example --json

# Skip the LLM commentary
.venv\Scripts\python.exe cli.py --market phoenix_example --no-llm

# Validate skill (walk-forward backtest, suggests spread_inflation)
.venv\Scripts\python.exe cli.py --city "Singapore" --backtest

# Suggest a starting station_offset from recent forecast bias
.venv\Scripts\python.exe cli.py --city "Singapore" --calibrate
```

### Recommended workflow for a new market
1. `--calibrate` to get a starting `station_offset` (removes model grid bias).
2. `--backtest` to get the `spread_inflation` for well-calibrated probabilities.
3. Put both (plus buckets + market prices) in `config/markets.yaml`.
4. Fine-tune `station_offset` against the market's exact resolution station once
   you have a few days of its reported highs — the last and most important step.

Define markets and (optionally) market prices in `config/markets.yaml`. With
prices set, the table shows Model %, Market %, Edge, and Kelly; `<<` flags buckets
where your model sees > 5% edge.

## Optional LLM layer

The LLM only writes a plain-English read of the finished forecast — it never feeds
back into the math. Three ways to give it a key:

- **Web app, easiest:** just paste a key into the sidebar field. It's used only for
  that browser session — never saved, logged, or shared.
- **Locally (CLI or app):** `copy .env.example .env`, then set `LLM_API_KEY`.
- **Public deploy with a server key:** set both `LLM_API_KEY` and `ENABLE_LLM=1` in
  the host's secrets. ⚠️ Leave `ENABLE_LLM` **unset** on a public deploy unless you
  want anonymous visitors spending your key — they can always bring their own.

Set `LLM_PROVIDER` to `groq` (free tier — get a key at
https://console.groq.com/keys), `openai`, `ollama` (local, no key), or
`anthropic`. Change `LLM_MODEL` to use any model that provider offers. Without a
key the layer is silently skipped — the numbers are unaffected.

---

## Project layout

```
app.py                   Streamlit web dashboard
config/markets.yaml      market definitions, buckets, station_offset, prices
src/data/openmeteo.py    geocoding + historical + ensemble + same-day hourly (cached)
src/data/kalshi.py       live Kalshi markets: buckets + prices + station coords
src/data/station.py      real station observations via Meteostat + grid→station offset
src/calibrate.py         per-station spread-inflation auto-calibration (cached)
src/models/climatology.py  harmonic regression + warming trend
src/models/ensemble.py     ensemble members -> samples
src/models/bias_correct.py regime + station offsets
src/models/extremes.py     GEV / return levels
src/blend.py             lead-time-weighted blend -> buckets, edge, Kelly
src/backtest.py          walk-forward skill test + spread calibration + forecast bias
src/llm/adapter.py       pluggable optional LLM commentary
src/predict.py           pipeline orchestration
cli.py                   command line interface
tests/test_core.py       unit tests for the math (py -m pytest)
```

## Limits & disclaimer
- Daily-max forecast skill is strong to ~10–14 days; beyond that the blend leans
  on climatology and ranges widen — treat long-lead buckets as soft.
- Always calibrate `station_offset` to the market's resolution source.
- For research/educational use. Markets involve risk; size positions responsibly.
