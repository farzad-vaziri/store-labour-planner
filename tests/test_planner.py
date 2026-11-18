import json
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from planner.calendar import black_friday, day_table  # noqa: E402
from planner.models import add_lags, horizon_week, seasonal_naive  # noqa: E402
from planner.staffing import (StoreAssumptions, customers_per_staff_hour, is_open, staff_needed,  # noqa: E402
                              staff_to_cope, staffing_plan, summarise)

DATA = ROOT / "data"


# ---------------- data integrity ----------------
@pytest.fixture(scope="module")
def ff():
    return pd.read_parquet(DATA / "footfall_hourly.parquet")


def test_no_duplicate_timestamps_per_site(ff):
    assert not ff.duplicated(["site", "ts"]).any()


def test_series_is_complete_and_hourly(ff):
    for _, g in ff.groupby("site", observed=True):
        full = pd.date_range(g.ts.min(), g.ts.max(), freq="h")
        assert len(g) == len(full)
    assert (ff.footfall >= 0).all()
    assert (ff.footfall == ff.count_in + ff.count_out).all()


def test_forecast_file_no_duplicates_and_ordered_intervals():
    fc = pd.read_parquet(DATA / "forecast_hourly.parquet")
    assert not fc.duplicated(["site", "ts", "kind", "origin"]).any()
    assert (fc.lo <= fc.pred + 1e-3).all() and (fc.pred <= fc.hi + 1e-3).all()


# ---------------- staffing calculator: hand examples ----------------
def test_staff_needed_hand_example():
    a = StoreAssumptions(capture_rate=0.02, conversion=0.30, service_min=6, utilisation=0.75, min_staff=2)
    # 4000 passers-by -> 80 visitors -> 24 buyers -> 144 staff-minutes / (60*0.75=45) = 3.2 -> 4 staff
    assert staff_needed([4000], a)[0] == 4
    # 1000 -> 20 visitors -> 6 buyers -> 36 min / 45 = 0.8 -> 1, floored to min cover 2
    assert staff_needed([1000], a)[0] == 2
    # exact boundary: 3750 -> 75 -> 22.5 buyers -> 135 min / 45 = 3.0 -> exactly 3, not 4
    assert staff_needed([3750], a)[0] == 3


def test_staff_to_cope_hand_example():
    a = StoreAssumptions(capture_rate=0.02, conversion=0.30, service_min=6, min_staff=1)
    # 5000 -> 100 visitors -> 30 buyers -> 180 min / 60 = 3 staff at 100% busy
    assert staff_to_cope([5000], a)[0] == 3


def test_customers_per_staff_hour():
    assert customers_per_staff_hour(StoreAssumptions(service_min=6, utilisation=0.75)) == pytest.approx(7.5)


def test_plan_cost_and_risk():
    a = StoreAssumptions(min_staff=2, hourly_cost=10.0, open_hour=9, close_hour=11)
    ts = pd.date_range("2026-11-02 08:00", periods=4, freq="h")  # Monday 08-11
    fc = pd.DataFrame({"ts": ts, "pred": [4000] * 4, "lo": [3000] * 4, "hi": [4000, 4000, 8000, 8000]})
    plan = staffing_plan(fc, a)
    assert list(plan.ts.dt.hour) == [9, 10]                 # closed at 08 and 11
    assert list(plan.staff) == [4, 4]
    assert list(plan.at_risk) == [False, True]              # 8000 -> 48 buyers -> 288 min -> 5 staff > 4
    s = summarise(plan)
    assert s["staff_hours"] == 8 and s["cost"] == 80.0 and s["risk_hours"] == 1


def test_plan_to_upper_uses_upper_bound():
    a = StoreAssumptions(plan_to="upper")
    fc = pd.DataFrame({"ts": [pd.Timestamp("2026-11-02 12:00")], "pred": [1000.0], "lo": [800.0], "hi": [6000.0]})
    assert staffing_plan(fc, a).staff.iloc[0] == staff_needed([6000], a)[0]


def test_sunday_hours_and_empty_plan():
    a = StoreAssumptions(sun_open=11, sun_close=17)
    sun = pd.Series(pd.date_range("2026-11-01 00:00", periods=24, freq="h"))
    assert is_open(sun, a).sum() == 6
    closed = StoreAssumptions(open_hour=9, close_hour=9, sun_open=9, sun_close=9)
    fc = pd.DataFrame({"ts": sun, "pred": 100.0, "lo": 50.0, "hi": 150.0})
    plan = staffing_plan(fc, closed)
    assert plan.empty and summarise(plan)["staff_hours"] == 0


# ---------------- backtest: no leakage ----------------
def test_lags_never_reach_past_origin():
    ts = pd.date_range("2026-01-05", periods=24 * 7 * 8, freq="h")
    s = pd.Series(np.arange(len(ts), dtype=float), index=pd.MultiIndex.from_arrays([["A"] * len(ts), ts]))
    origin = pd.Timestamp("2026-02-16")
    target = pd.DataFrame({"site": "A", "ts": pd.date_range(origin, periods=24 * 14, freq="h")})
    target["h_week"] = horizon_week(target.ts, origin)
    known = s[s.index.get_level_values(1) < origin]
    full = add_lags(target, s)
    cut = add_lags(target, known)        # only data before the origin
    pd.testing.assert_frame_equal(full, cut)
    hist = pd.DataFrame({"site": "A", "ts": known.index.get_level_values(1), "footfall": known.values})
    sn = seasonal_naive(hist, target, origin)
    assert not np.isnan(sn).any()


def test_backtest_windows_do_not_leak():
    bt = pd.read_csv(DATA / "backtest.csv", parse_dates=["origin", "ts"])
    assert (bt.ts >= bt.origin).all()
    assert (bt.ts < bt.origin + pd.Timedelta(days=7)).all()
    fc = pd.read_parquet(DATA / "forecast_hourly.parquet")
    ff = pd.read_parquet(DATA / "footfall_hourly.parquet")
    fwd = fc[fc.kind == "forecast"]
    assert fwd.ts.min() > ff.ts.max()


# ---------------- calendar and intervals ----------------
def test_calendar_flags():
    d = day_table("2026-11-01", "2026-12-31").set_index("date")
    assert black_friday(2026) == pd.Timestamp("2026-11-27")
    assert d.loc["2026-12-25", "event"] == "Christmas Day"
    assert d.loc["2026-12-28", "bank_holiday"] == 1          # substitute Boxing Day
    assert d.loc["2026-11-10", "day_type"] == "normal"


def test_intervals_wider_on_special_days():
    iv = pd.read_csv(DATA / "intervals.csv")
    from planner.models import band_for
    q = {r.group: (r.lo_ratio, r.hi_ratio, r.n) for r in iv.itertuples()}
    q["_fallback"], q["_overall"] = {}, (0.8, 1.2)
    for hw in ["w1", "w2"]:
        for band in ["trading", "edge"]:
            n = band_for(f"{hw}|normal|{band}", q)
            sp = band_for(f"{hw}|special|{band}", q)
            assert sp[1] - sp[0] >= n[1] - n[0]


def test_metrics_match_forecast_file():
    m = json.loads((DATA / "metrics.json").read_text())
    fc = pd.read_parquet(DATA / "forecast_hourly.parquet")
    start = pd.Timestamp(m["forecast_start"])
    for site, v in m["weekly_forecast_footfall"].items():
        got = fc[(fc.site == site) & (fc.kind == "forecast") & (fc.ts < start + pd.Timedelta(days=7))].pred.sum()
        assert got == pytest.approx(v, rel=1e-4)
    assert m["best_method"] == min(m["methods"], key=lambda k: m["models"][k]["wape"])
