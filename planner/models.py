"""Feature building and the three forecasting methods (fit offline in scripts/03_fit.py)."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

WEATHER = ["temp", "rain", "cloud", "wind"]
CAL = ["bank_holiday", "day_before_bh", "easter_sunday", "school_holiday", "black_friday", "xmas_shopping",
       "xmas_eve", "xmas_day", "boxing_day", "xmas_to_ny", "days_to_xmas"]


def build_features(frame: pd.DataFrame, days: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """frame: site, ts (+ footfall optional). Adds calendar + weather features.
    Weather beyond the observed record is filled with the month x hour climatology."""
    f = frame.copy()
    f["date"] = f.ts.dt.normalize()
    f = f.merge(days.drop(columns=["event"]), on="date", how="left")
    w = weather.reindex(f.ts.values)
    clim = weather.groupby([weather.index.month, weather.index.hour]).mean()
    key = pd.MultiIndex.from_arrays([f.ts.dt.month.values, f.ts.dt.hour.values])
    fill = clim.reindex(key)
    for c in WEATHER:
        v = w[c].values
        f[c] = np.where(np.isnan(v), fill[c].values, v)
    f["weather_observed"] = ~np.isnan(w["temp"].values)
    f["hour"] = f.ts.dt.hour
    f["dow"] = f.ts.dt.dayofweek
    f["month"] = f.ts.dt.month
    f["doy"] = f.ts.dt.dayofyear
    f["wet"] = (f.rain > 0.2).astype(int)
    return f


# ---------- lag features (horizon-safe) ----------
LAGS = ["naive", "sw_mean", "lag364"]


def add_lags(f: pd.DataFrame, series: pd.Series) -> pd.DataFrame:
    """f needs site, ts, h_week (1 or 2). series: footfall indexed by (site, ts).
    naive  = same hour/weekday h_week weeks earlier (the latest week known at the origin)
    sw_mean = mean of the same hour/weekday 2-5 weeks earlier
    lag364 = same hour/weekday 52 weeks earlier.
    For a row h_week weeks after the origin every lag is strictly before the origin."""
    s = series.copy()
    s.index = pd.MultiIndex.from_arrays([s.index.get_level_values(0).astype(str), s.index.get_level_values(1)])
    site = f.site.astype(str).values

    def look(delta_days):
        return s.reindex(pd.MultiIndex.from_arrays([site, (f.ts - pd.to_timedelta(delta_days, unit="D")).values])).values.astype(float)

    f = f.copy()
    k = f.h_week.values.astype(int)
    f["naive"] = np.where(k == 1, look(7), look(14))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        f["sw_mean"] = np.nanmean(np.vstack([look(d) for d in (14, 21, 28, 35)]), axis=0)
    f["lag364"] = look(364)
    return f


def training_rows(feat: pd.DataFrame, series: pd.Series) -> pd.DataFrame:
    """Each historical hour appears twice: as a 1-week-ahead and a 2-week-ahead example."""
    a = add_lags(feat.assign(h_week=1), series)
    b = add_lags(feat.assign(h_week=2), series)
    out = pd.concat([a, b], ignore_index=True)
    return out[out.naive.notna()].reset_index(drop=True)


def horizon_week(ts: pd.Series, origin: pd.Timestamp) -> np.ndarray:
    return (((ts - origin) // pd.Timedelta(days=7)).astype(int) + 1).values


# ---------- 1. seasonal naive ----------
def seasonal_naive(hist: pd.DataFrame, target: pd.DataFrame, origin: pd.Timestamp) -> np.ndarray:
    """Same hour, same weekday from the last observed week before origin."""
    lookup = hist.set_index(["site", "ts"])["footfall"]
    weeks_ahead = ((target.ts - origin) // pd.Timedelta(days=7)).astype(int) + 1
    src = target.ts - pd.to_timedelta(7 * weeks_ahead, unit="D")
    idx = pd.MultiIndex.from_arrays([target.site.astype(str).values, src.values])
    lookup.index = pd.MultiIndex.from_arrays([lookup.index.get_level_values(0).astype(str), lookup.index.get_level_values(1)])
    return lookup.reindex(idx).values.astype(float)


# ---------- 2. regression (per site, log scale) ----------
def _design(f: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=f.index)
    hd = f.dow.astype(str) + "_" + f.hour.astype(str)
    X = pd.get_dummies(hd, prefix="hd", dtype=float)
    X = X.join(pd.get_dummies(f.month, prefix="m", dtype=float))
    band = pd.cut(f.hour, [-1, 6, 9, 17, 20, 23], labels=["night", "morning", "day", "evening", "late"]).astype(str)
    for c in ["bank_holiday", "easter_sunday", "school_holiday", "black_friday", "xmas_shopping", "xmas_eve",
              "xmas_day", "boxing_day", "xmas_to_ny", "day_before_bh"]:
        for b in ["night", "morning", "day", "evening", "late"]:
            X[f"{c}_{b}"] = f[c].values * (band.values == b)
    day = f.hour.between(9, 19).values
    X["temp"] = f.temp.values
    X["temp2"] = (f.temp.values - 12) ** 2
    X["rain_day"] = np.log1p(f.rain.values) * day
    X["wet_day"] = f.wet.values * day
    X["wind_day"] = f.wind.values * day
    X["cloud_day"] = f.cloud.values * day
    nv = np.log1p(f.naive.values)
    sw = np.log1p(np.where(np.isnan(f.sw_mean.values), f.naive.values, f.sw_mean.values))
    ly = np.log1p(np.where(np.isnan(f.lag364.values), f.sw_mean.fillna(f.naive).values, f.lag364.values))
    X["l_naive"], X["l_sw"], X["l_364"] = nv, sw, ly
    X["l_naive_w2"] = nv * (f.h_week.values == 2)
    X["l_naive_special"] = nv * (f.day_type.values != "normal")
    return X


class RidgeHourly:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.models = {}

    def fit(self, f: pd.DataFrame):
        from sklearn.linear_model import Ridge
        for s, g in f.groupby("site", observed=True):
            X = _design(g)
            self.models[s] = (Ridge(alpha=self.alpha).fit(X.values, np.log1p(g.footfall.values)), X.columns)
        return self

    def predict(self, f: pd.DataFrame) -> np.ndarray:
        out = np.full(len(f), np.nan)
        for s, g in f.groupby("site", observed=True):
            m, cols = self.models[s]
            X = _design(g).reindex(columns=cols, fill_value=0.0)
            out[f.index.get_indexer(g.index)] = np.expm1(m.predict(X.values)).clip(0)
        return out


# ---------- 3. LightGBM (global, Poisson) ----------
LGB_FEATS = ["site_code", "hour", "dow", "month", "doy", "h_week"] + LAGS + CAL + WEATHER


class LGBHourly:
    def __init__(self, use_weather=True, rounds=600, seed=7):
        self.use_weather, self.rounds, self.seed = use_weather, rounds, seed

    def _X(self, f):
        X = f.assign(site_code=f.site.cat.codes if hasattr(f.site, "cat") else f.site.astype("category").cat.codes)
        cols = [c for c in LGB_FEATS if self.use_weather or c not in WEATHER]
        return X[cols]

    def fit(self, f: pd.DataFrame):
        import lightgbm as lgb
        params = dict(objective="poisson", learning_rate=0.05, num_leaves=63, min_data_in_leaf=40,
                      feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=1, seed=self.seed, verbose=-1,
                      num_threads=4)
        X = self._X(f)
        ds = lgb.Dataset(X, f.footfall.values, categorical_feature=["site_code"], free_raw_data=False)
        self.model = lgb.train(params, ds, num_boost_round=self.rounds)
        return self

    def predict(self, f: pd.DataFrame) -> np.ndarray:
        return self.model.predict(self._X(f)).clip(0)


# ---------- intervals ----------
def interval_group(f: pd.DataFrame, origin: pd.Timestamp) -> pd.Series:
    hw = np.where(f.ts < origin + pd.Timedelta(days=7), "w1", "w2")
    band = np.where(f.hour.between(9, 17), "trading", "edge")
    return pd.Series(hw, index=f.index) + "|" + f.day_type.astype(str).values + "|" + band


def ratio_quantiles(pool: pd.DataFrame, lo_q=0.10, hi_q=0.90, min_n=150) -> dict:
    """pool: rows with group, actual, pred. Multiplicative 80% band per group, with fallbacks."""
    p = pool[(pool.pred > 5)].copy()
    p["r"] = p.actual / p.pred
    out = {}
    overall = (p.r.quantile(lo_q), p.r.quantile(hi_q))
    by_hw_band = {}
    for k, g in p.groupby(p.group.str.split("|").str[0] + "|" + p.group.str.split("|").str[2]):
        by_hw_band[k] = (g.r.quantile(lo_q), g.r.quantile(hi_q))
    for k, g in p.groupby("group"):
        if len(g) >= min_n:
            out[k] = (float(g.r.quantile(lo_q)), float(g.r.quantile(hi_q)), int(len(g)))
    out["_fallback"] = {k: (float(a), float(b)) for k, (a, b) in by_hw_band.items()}
    out["_overall"] = (float(overall[0]), float(overall[1]))
    return out


DAY_TYPE_ORDER = ["normal", "holiday period", "special"]


def _ratios(g: str, q: dict):
    if g in q:
        return q[g][0], q[g][1]
    hw, _, band = g.split("|")
    return q["_fallback"].get(f"{hw}|{band}", q["_overall"])


def band_for(g: str, q: dict):
    """Multiplicative band for a group. A holiday-period or special day never gets a narrower band
    than a less unusual day type with the same horizon and hour band (sensible prior when such days are rare)."""
    hw, dt, band = g.split("|")
    a, b = _ratios(g, q)
    for lower in DAY_TYPE_ORDER[:DAY_TYPE_ORDER.index(dt)] if dt in DAY_TYPE_ORDER else []:
        a2, b2 = _ratios(f"{hw}|{lower}|{band}", q)
        a, b = min(a, a2), max(b, b2)
    return min(a, 1.0), max(b, 1.0)


def apply_intervals(pred: np.ndarray, groups: pd.Series, q: dict):
    cache = {g: band_for(g, q) for g in pd.unique(groups.values)}
    a = np.array([cache[g][0] for g in groups.values])
    b = np.array([cache[g][1] for g in groups.values])
    return pred * a, pred * b
