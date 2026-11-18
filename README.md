# Store Labour Planner

**Live app:** https://store-labour-planner.streamlit.app *(planned address; confirmed at launch)*

How many staff should a high-street shop have on the floor each hour? This one-page Streamlit app forecasts real
hourly city-centre footfall in Leeds, turns it into an hour-by-hour staffing plan, and shows what covering the peaks costs.

## What it does

1. **Footfall patterns.** A weekday × hour heatmap and a three-year daily trend for each of Leeds City Council's
   eight footfall cameras, with bank holidays, Easter Sunday, Black Friday and Christmas marked.
2. **Forecast.** A two-week hourly forecast from the end of the published data, with an 80% interval. Three methods
   are compared in a rolling-origin backtest over the last 8 weeks: seasonal naive, a ridge regression
   (hour × weekday, holidays, weather, lagged levels) and LightGBM (calendar, weather, lagged levels).
   The lowest-error method, LightGBM, drives the plan.
3. **Staffing plan.** Store visitors = footfall × capture rate. Staff needed per hour =
   `ceil(visitors × conversion × minutes per sale / (60 × target busy %))`, floored at the minimum cover, for
   the store's opening hours. An hour is at **service risk** when the top of the 80% interval would need more
   staff than planned even with everyone 100% busy. The plan downloads as CSV.
4. **What if.** Capture rate and conversion sliders, with the change in staff-hours and cost, and a cost grid.

Store figures (capture 2%, conversion 30%, 6 staff-minutes per sale, 85% busy target, minimum 2 staff,
£14.50/hour including on-costs, open 09–18 and Sunday 11–17) are **illustrative assumptions**, labelled as such
on the page and editable in the sidebar.

Weeks to plan: the two forecast weeks plus the eight backtest weeks, where the forecast can be compared with what
the cameras counted. A link such as `?site=Park%20Row&week=2026-08-31` opens a given site and week.

## Results (from `data/metrics.json`)

| Method | WAPE, last 8 weeks, 1 week ahead | WAPE, 09:00–18:00 |
|---|---|---|
| Seasonal naive | 13.4% | 11.4% |
| Regression | 16.3% | 14.7% |
| **LightGBM** | **12.8%** | **10.8%** |
| LightGBM without weather | 14.4% | 12.0% |

80% intervals contained 80.6% of actual hours in the backtest. Interval width is set from how far past forecasts
missed for the same horizon week, day type (normal / school-holiday or Christmas-shopping period / special day) and
hour band; special days get bands roughly twice as wide as normal days.

**Caveat:** the weather inputs are observed weather, so the backtest assumes a perfect weather forecast. The
"without weather" row shows the lower bound on what weather adds.

## Data sources and licences

- **Footfall:** Leeds City Council, *Leeds city centre footfall data*, Data Mill North
  (https://datamillnorth.org/dataset/leeds-city-centre-footfall-data-2rlld). Open Government Licence v3.0.
  *Contains public sector information licensed under the Open Government Licence v3.0.*
  Only the **Revised Weekly Data** files are used (30 Oct 2023 – 25 Oct 2026, 8 cameras, hourly, no gaps).
  Leeds republished these after finding that the ageing camera system and some camera relocations had
  under-counted. Footfall = people passing in both directions (`ReportCount` = in + out).
- **Camera changes:** between April and July 2023 the cameras were renamed to show their current locations, some
  after relocation: Albion Street South → Albion Street at Bond Street; Briggate → Briggate M&S; Briggate at
  McDonalds → Briggate at Swan Street; Dortmund Square → Headrow at Broadgate; Headrow → Headrow at Lands Lane.
  All of this predates the revised series, so there are no relocation breaks inside the modelled data. The trend
  chart marks where the revised series starts.
- **Weather:** Open-Meteo historical weather API, hourly, central Leeds (https://open-meteo.com). CC BY 4.0.
  *Weather data by Open-Meteo.com.*
- **Bank holidays:** gov.uk `bank-holidays.json`, Open Government Licence v3.0.
- **School holidays:** Leeds City Council term dates (leeds.gov.uk) for 2025–27; 2023–25 reconstructed from the
  Leeds local-authority pattern (see `planner/calendar.py`). BRC retail week numbers come with the footfall files.

## Run locally

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
streamlit run app.py
```

The app only reads the precomputed files in `data/`. No keys or paid APIs.

## Refit with new data

```bash
pip install -r requirements-fit.txt
python scripts/01_download.py raw      # revised weekly CSVs, bank holidays, weather -> raw/ (git-ignored)
python scripts/02_prepare.py raw       # -> data/footfall_hourly.parquet, weather, calendar
python scripts/03_fit.py               # 60 weekly origins, about 10 minutes -> forecast, backtest, intervals, metrics
pytest -q
```

Add new school-holiday dates to `planner/calendar.py` each year. Data Mill North publishes a new weekly file roughly
every week; the forecast always starts at the Monday after the latest data.

## Files

| Path | Contents |
|---|---|
| `app.py` | The Streamlit page |
| `planner/` | Calendar features, forecasting models, staffing calculator |
| `scripts/` | Download, prepare, fit (all offline) |
| `data/footfall_hourly.parquet` | Site × hour footfall (revised series) |
| `data/forecast_hourly.parquet` | Forward forecast and backtest weeks, with 80% intervals |
| `data/backtest.csv` | Last 8 origins, all methods, 1-week horizon |
| `data/intervals.csv`, `data/metrics.json` | Interval ratios, headline metrics |
| `tests/` | Data integrity, calculator hand examples, leakage checks, every filter combination |
| `SPEC.md` | Original build brief |

Portfolio demo, not operational advice.
