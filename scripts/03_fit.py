"""Rolling-origin backtest of three methods, residual-based 80% intervals, and the forward forecast.
Writes data/forecast_hourly.parquet, data/backtest.csv, data/intervals.csv, data/metrics.json.
Usage: python scripts/03_fit.py [--quick]
"""
import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from planner.models import (LGBHourly, RidgeHourly, add_lags, apply_intervals, build_features,  # noqa: E402
                            horizon_week, interval_group, ratio_quantiles, seasonal_naive, training_rows)

DATA = ROOT / "data"
BACKTEST_WEEKS = 8      # scored origins
POOL_WEEKS = 52         # residual pool for intervals
REFIT_EVERY = 4         # weeks between model refits (lags are refreshed at every origin)
HORIZON = 14            # days
METHODS = {"seasonal_naive": "Seasonal naive", "regression": "Regression (hour x weekday, holidays, weather)",
           "lightgbm": "LightGBM (calendar + weather)"}


def target_frame(sites, origin, days=HORIZON):
    ts = pd.date_range(origin, origin + pd.Timedelta(days=days) - pd.Timedelta(hours=1), freq="h")
    t = pd.MultiIndex.from_product([sites, ts], names=["site", "ts"]).to_frame(index=False)
    t["site"] = pd.Categorical(t.site, categories=sites)
    return t


def fit_models(train, names=("regression", "lightgbm"), extra=False):
    fitted = {}
    if "regression" in names:
        fitted["regression"] = RidgeHourly().fit(train)
    if "lightgbm" in names:
        fitted["lightgbm"] = LGBHourly().fit(train)
    if extra:
        fitted["lightgbm_no_weather"] = LGBHourly(use_weather=False).fit(train)
    return fitted


def forecast_all(fitted, hist, target_f, origin):
    out = {"seasonal_naive": seasonal_naive(hist, target_f, origin)}
    for k, m in fitted.items():
        out[k] = m.predict(target_f)
    return out


def main(quick=False):
    t0 = time.time()
    ff = pd.read_parquet(DATA / "footfall_hourly.parquet")
    days = pd.read_parquet(DATA / "calendar_days.parquet")
    weather = pd.read_parquet(DATA / "weather_hourly.parquet")
    sites = list(ff.site.cat.categories)
    ff["site"] = pd.Categorical(ff.site.astype(str), categories=sites)
    feat = build_features(ff, days, weather)
    end = ff.ts.max() + pd.Timedelta(hours=1)            # first hour with no data (a Monday 00:00)
    assert end.dayofweek == 0 and end.hour == 0
    n_orig = BACKTEST_WEEKS + (8 if quick else POOL_WEEKS)
    origins = [end - pd.Timedelta(weeks=k) for k in range(n_orig, 0, -1)]
    actual = ff.set_index(["site", "ts"]).footfall

    rows = []
    fitted, fitted_at = None, None
    first_scored = end - pd.Timedelta(weeks=BACKTEST_WEEKS)
    for i, o in enumerate(origins):
        hist = ff[ff.ts < o]
        if fitted is None or (o - fitted_at) >= pd.Timedelta(weeks=REFIT_EVERY) or (o >= first_scored and fitted_at < first_scored):
            train = training_rows(feat[feat.ts < o].reset_index(drop=True), hist.set_index(["site", "ts"]).footfall)
            fitted, fitted_at = fit_models(train, extra=True), o
            train_end = train.ts.max()
        tf = build_features(target_frame(sites, o), days, weather)
        tf = tf[tf.ts < end].reset_index(drop=True)
        tf["h_week"] = horizon_week(tf.ts, o)
        tf = add_lags(tf, hist.set_index(["site", "ts"]).footfall)
        preds = forecast_all(fitted, hist, tf, o)
        r = tf[["site", "ts", "day_type", "hour"]].copy()
        r["origin"] = o
        r["group"] = interval_group(tf, o).values
        r["actual"] = actual.reindex(pd.MultiIndex.from_arrays([r.site.astype(str), r.ts])).values
        for k, v in preds.items():
            r[k] = v
        r["train_end"] = train_end
        rows.append(r)
        print(f"origin {o.date()} ({i+1}/{len(origins)}) {time.time()-t0:.0f}s", flush=True)
    bt_all = pd.concat(rows, ignore_index=True)
    assert (bt_all.train_end < bt_all.origin).all()

    # ---- score the last 8 origins, week-1 horizon ----
    sc = bt_all[(bt_all.origin >= end - pd.Timedelta(weeks=BACKTEST_WEEKS)) &
                (bt_all.ts < bt_all.origin + pd.Timedelta(days=7))].copy()
    def wape(a, p): return float(np.abs(a - p).sum() / a.sum())
    trading = sc.hour.between(9, 17)
    metrics_models = {}
    for m in list(METHODS) + ["lightgbm_no_weather"]:
        metrics_models[m] = dict(mae=float(np.abs(sc.actual - sc[m]).mean()), wape=wape(sc.actual, sc[m]),
                                 wape_trading=wape(sc.actual[trading], sc[m][trading]))
    best = min(METHODS, key=lambda m: metrics_models[m]["wape"])

    # ---- intervals for the scored origins: pool only residuals observed before that origin ----
    sc_parts = []
    for o in sorted(sc.origin.unique()):
        pool = bt_all[(bt_all.ts < o) & (bt_all.origin >= o - pd.Timedelta(weeks=POOL_WEEKS))]
        q = ratio_quantiles(pool.rename(columns={best: "pred"})[["group", "actual", "pred"]])
        part = sc[sc.origin == o].copy()
        part["lo"], part["hi"] = apply_intervals(part[best].values, part.group, q)
        sc_parts.append(part)
    sc = pd.concat(sc_parts)
    cover = float(((sc.actual >= sc.lo) & (sc.actual <= sc.hi)).mean())
    cover_trading = float(((sc.actual >= sc.lo) & (sc.actual <= sc.hi))[sc.hour.between(9, 17)].mean())

    by_hour = (sc.assign(**{f"ae_{m}": (sc.actual - sc[m]).abs() for m in METHODS})
                 .groupby("hour")[[f"ae_{m}" for m in METHODS] + ["actual"]].sum())
    wape_by_hour = {m: (by_hour[f"ae_{m}"] / by_hour.actual).round(4).tolist() for m in METHODS}

    # ---- forward forecast from the end of the data ----
    tf = build_features(target_frame(sites, end), days, weather)
    tf["h_week"] = horizon_week(tf.ts, end)
    tf = add_lags(tf, ff.set_index(["site", "ts"]).footfall)
    if best == "seasonal_naive":
        preds = {best: seasonal_naive(ff, tf, end)}
    else:
        final = fit_models(training_rows(feat, ff.set_index(["site", "ts"]).footfall), names=(best,))
        preds = forecast_all(final, ff, tf, end)
    pool = bt_all[bt_all.origin >= end - pd.Timedelta(weeks=POOL_WEEKS)]
    q_final = ratio_quantiles(pool.rename(columns={best: "pred"})[["group", "actual", "pred"]])
    fwd = tf[["site", "ts", "day_type", "weather_observed"]].copy()
    fwd["pred"] = preds[best]
    fwd["group"] = interval_group(tf, end).values
    fwd["lo"], fwd["hi"] = apply_intervals(fwd.pred.values, fwd.group, q_final)
    fwd["actual"] = np.nan
    fwd["kind"] = "forecast"
    fwd["origin"] = end

    bwd = sc[["site", "ts", "day_type", "group", "lo", "hi", "actual", "origin"]].copy()
    bwd["pred"] = sc[best].values
    bwd["kind"] = "backtest"
    bwd["weather_observed"] = True
    out = pd.concat([fwd, bwd], ignore_index=True)
    out["week_start"] = out.ts.dt.normalize() - pd.to_timedelta(out.ts.dt.dayofweek, unit="D")
    out["site"] = out.site.astype(str)
    for c in ["pred", "lo", "hi", "actual"]:
        out[c] = out[c].astype("float32")
    out.drop(columns=["group"]).to_parquet(DATA / "forecast_hourly.parquet", index=False)

    bt_out = sc[["origin", "site", "ts", "actual"] + list(METHODS) + ["lightgbm_no_weather", "lo", "hi"]].copy()
    bt_out[list(METHODS) + ["lightgbm_no_weather", "lo", "hi"]] = bt_out[list(METHODS) + ["lightgbm_no_weather", "lo", "hi"]].round(1)
    bt_out.to_csv(DATA / "backtest.csv", index=False)

    iv = [dict(group=k, lo_ratio=v[0], hi_ratio=v[1], n=v[2]) for k, v in q_final.items() if not k.startswith("_")]
    pd.DataFrame(iv).sort_values("group").to_csv(DATA / "intervals.csv", index=False)

    # interval width by day type (forward + backtest), relative to prediction, trading hours
    w = out[out.ts.dt.hour.between(9, 17) & (out.pred > 0)]
    width = (((w.hi - w.lo) / w.pred).groupby(w.day_type).mean()).round(3).to_dict()

    metrics = dict(
        data_start=str(ff.ts.min()), data_end=str(ff.ts.max()), sites=sites, n_rows=int(len(ff)),
        forecast_start=str(end), forecast_end=str(end + pd.Timedelta(days=HORIZON) - pd.Timedelta(hours=1)),
        backtest_origins=[str(o.date()) for o in sorted(sc.origin.unique())],
        methods=METHODS, best_method=best, models=metrics_models, wape_by_hour=wape_by_hour,
        interval_coverage_80=cover, interval_coverage_80_trading=cover_trading,
        mean_rel_interval_width_trading_by_day_type=width,
        weekly_forecast_footfall={s: float(fwd[(fwd.site == s) & (fwd.ts < end + pd.Timedelta(days=7))].pred.sum())
                                  for s in sites},
        fit_seconds=round(time.time() - t0),
    )
    (DATA / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k: metrics[k] for k in ["best_method", "models", "interval_coverage_80",
                                               "mean_rel_interval_width_trading_by_day_type"]}, indent=1))


if __name__ == "__main__":
    main(quick="--quick" in sys.argv)
