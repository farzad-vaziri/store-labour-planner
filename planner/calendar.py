"""Calendar features: UK bank holidays, Leeds school holidays, retail special days."""
from __future__ import annotations

import json
import pathlib

import numpy as np
from dateutil.easter import easter
import pandas as pd

# Leeds City Council (local-authority) school holidays, inclusive dates.
# 2025-26 and 2026-27: leeds.gov.uk "School holidays and term dates" and Leeds LA school
# calendars. 2023-24 and 2024-25: reconstructed from the Leeds LA calendar pattern
# (Easter = first two full weeks of April); individual schools may differ by a day.
SCHOOL_HOLIDAYS = [
    ("2023-10-30", "2023-11-03"), ("2023-12-25", "2024-01-05"), ("2024-02-12", "2024-02-16"),
    ("2024-04-01", "2024-04-12"), ("2024-05-27", "2024-05-31"), ("2024-07-23", "2024-08-30"),
    ("2024-10-28", "2024-11-01"), ("2024-12-23", "2025-01-03"), ("2025-02-17", "2025-02-21"),
    ("2025-04-07", "2025-04-21"), ("2025-05-26", "2025-05-30"), ("2025-07-23", "2025-08-29"),
    ("2025-10-27", "2025-10-31"), ("2025-12-22", "2026-01-02"), ("2026-02-16", "2026-02-20"),
    ("2026-04-03", "2026-04-17"), ("2026-05-25", "2026-05-29"), ("2026-07-22", "2026-08-28"),
    ("2026-10-26", "2026-10-30"), ("2026-12-21", "2027-01-01"), ("2027-02-15", "2027-02-19"),
    ("2027-04-05", "2027-04-16"), ("2027-05-31", "2027-06-04"), ("2027-07-23", "2027-09-03"),
]

# Fallback list (England & Wales) used only if the gov.uk JSON is not supplied.
_BH_FALLBACK = [
    "2023-12-25", "2023-12-26", "2024-01-01", "2024-03-29", "2024-04-01", "2024-05-06",
    "2024-05-27", "2024-08-26", "2024-12-25", "2024-12-26", "2025-01-01", "2025-04-18",
    "2025-04-21", "2025-05-05", "2025-05-26", "2025-08-25", "2025-12-25", "2025-12-26",
    "2026-01-01", "2026-04-03", "2026-04-06", "2026-05-04", "2026-05-25", "2026-08-31",
    "2026-12-25", "2026-12-28", "2027-01-01",
]


def bank_holidays(path: str | pathlib.Path | None = None) -> pd.DatetimeIndex:
    if path and pathlib.Path(path).exists():
        ev = json.loads(pathlib.Path(path).read_text())["england-and-wales"]["events"]
        return pd.DatetimeIndex(sorted({e["date"] for e in ev}))
    return pd.DatetimeIndex(_BH_FALLBACK)


def school_holiday_days() -> pd.DatetimeIndex:
    days = [pd.date_range(a, b, freq="D") for a, b in SCHOOL_HOLIDAYS]
    return pd.DatetimeIndex(np.concatenate([d.values for d in days]))


def black_friday(year: int) -> pd.Timestamp:
    nov30 = pd.Timestamp(year=year, month=11, day=30)
    # last Friday of November
    return nov30 - pd.Timedelta(days=(nov30.weekday() - 4) % 7)


def day_table(start, end, bh_path=None) -> pd.DataFrame:
    """One row per date with calendar flags and a day_type label used for intervals."""
    d = pd.DataFrame({"date": pd.date_range(start, end, freq="D")})
    bh = bank_holidays(bh_path)
    sh = school_holiday_days()
    years = range(d.date.dt.year.min(), d.date.dt.year.max() + 1)
    bf = pd.DatetimeIndex([black_friday(y) for y in years])
    es = pd.DatetimeIndex([pd.Timestamp(easter(y)) for y in years])
    md = d.date.dt.month * 100 + d.date.dt.day
    d["bank_holiday"] = d.date.isin(bh).astype(int)
    d["day_before_bh"] = (d.date + pd.Timedelta(days=1)).isin(bh).astype(int)
    d["easter_sunday"] = d.date.isin(es).astype(int)
    d["school_holiday"] = d.date.isin(sh).astype(int)
    d["black_friday"] = d.date.isin(bf).astype(int)
    d["xmas_shopping"] = ((md >= 1201) & (md <= 1224)).astype(int)
    d["xmas_eve"] = (md == 1224).astype(int)
    d["xmas_day"] = (md == 1225).astype(int)
    d["boxing_day"] = (md == 1226).astype(int)
    d["xmas_to_ny"] = ((md >= 1227) | (md <= 101)).astype(int)
    d["days_to_xmas"] = np.where(md <= 1225, (pd.to_datetime(d.date.dt.year.astype(str) + "-12-25") - d.date).dt.days, 400).clip(0, 60)
    special = (d.bank_holiday | d.easter_sunday | d.black_friday | d.xmas_eve | d.xmas_day | d.boxing_day | d.xmas_to_ny).astype(bool)
    d["day_type"] = np.select([special, d.school_holiday.astype(bool) | d.xmas_shopping.astype(bool)],
                              ["special", "holiday period"], "normal")
    def label(r):
        if r.xmas_day: return "Christmas Day"
        if r.boxing_day: return "Boxing Day"
        if r.xmas_eve: return "Christmas Eve"
        if r.black_friday: return "Black Friday"
        if r.easter_sunday: return "Easter Sunday"
        if r.bank_holiday: return "Bank holiday"
        return ""
    d["event"] = d.apply(label, axis=1)
    return d
