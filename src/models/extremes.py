"""Extreme Value Theory for context on the hot-tail buckets.

We fit a GEV to the block (annual) maxima of Tmax. This answers questions a
mean-regression model can't: "what's the all-time record vibe?" and "how rare is
a 35C day here?" Used for reporting/sanity-checking the extreme buckets, not to
override the ensemble+climatology blend.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class GEVFit:
    shape: float   # xi (genextreme uses c = -xi convention)
    loc: float
    scale: float
    record: float

    def return_level(self, years: float) -> float:
        """Tmax expected to be exceeded once every `years` years."""
        p = 1.0 - 1.0 / years
        return float(stats.genextreme.ppf(p, self.shape, loc=self.loc, scale=self.scale))

    def exceedance_prob_annual(self, temp: float) -> float:
        """P(annual max > temp)."""
        cdf = stats.genextreme.cdf(temp, self.shape, loc=self.loc, scale=self.scale)
        return float(1.0 - cdf)


def fit_gev(archive: dict) -> GEVFit | None:
    daily = archive["daily"]
    df = pd.DataFrame({"time": pd.to_datetime(daily["time"]),
                       "tmax": daily["temperature_2m_max"]}).dropna()
    if df.empty:
        return None
    annual_max = df.groupby(df["time"].dt.year)["tmax"].max().to_numpy()
    if annual_max.size < 5:
        return None
    c, loc, scale = stats.genextreme.fit(annual_max)
    return GEVFit(shape=float(c), loc=float(loc), scale=float(scale),
                  record=float(annual_max.max()))
