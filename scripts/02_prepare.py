"""Build data/footfall_hourly.parquet and the model feature table from the raw downloads.
Usage: python scripts/02_prepare.py <raw_dir>
"""
import glob
import json
import pathlib
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from planner.calendar import day_table  # noqa: E402

RAW = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "raw")
DATA = ROOT / "data"


def load_footfall() -> pd.DataFrame:
    files = sorted(glob.glob(str(RAW / "footfall" / "*.csv")))
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    d["ts"] = pd.to_datetime(d["Date"], format="%d-%b-%y") + pd.to_timedelta(d["Hour"].str[:2].astype(int), unit="h")
    d = d.rename(columns={"LocationName": "site", "InCount": "count_in", "OutCount": "count_out",
                          "ReportCount": "footfall", "BRCYear": "brc_year", "BRCWeekNum": "brc_week"})
    d = d[["site", "ts", "footfall", "count_in", "count_out", "brc_year", "brc_week"]]
    # Weekly files can overlap after re-publication: keep the last copy of any site-hour.
    d = d.drop_duplicates(["site", "ts"], keep="last").sort_values(["site", "ts"]).reset_index(drop=True)
    for c in ["footfall", "count_in", "count_out"]:
        d[c] = d[c].astype("int32")
    d["site"] = d["site"].astype("category")
    return d


def load_weather() -> pd.DataFrame:
    w = json.loads((RAW / "weather_hourly.json").read_text())["hourly"]
    w = pd.DataFrame(w).rename(columns={"time": "ts", "temperature_2m": "temp", "cloud_cover": "cloud",
                                       "wind_speed_10m": "wind", "precipitation": "rain"})
    w["ts"] = pd.to_datetime(w["ts"])
    return w.drop_duplicates("ts").set_index("ts").sort_index()


def main():
    DATA.mkdir(exist_ok=True)
    ff = load_footfall()
    ff.to_parquet(DATA / "footfall_hourly.parquet", index=False)
    w = load_weather()
    w.to_parquet(DATA / "weather_hourly.parquet")
    days = day_table(ff.ts.min().normalize(), ff.ts.max().normalize() + pd.Timedelta(days=60), RAW / "bank_holidays.json")
    days.to_parquet(DATA / "calendar_days.parquet", index=False)
    print(ff.shape, ff.ts.min(), ff.ts.max(), ff.site.nunique(), "sites")


if __name__ == "__main__":
    main()
