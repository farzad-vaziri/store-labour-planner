# store-labour-planner – Build brief

Written by the ideas chat, 18 Nov 2026. Follow the project instructions' build command. This file is your spec.

**This chat will be deleted once the dashboard is live.** Everything anyone needs later must end up in the repo or in `handover/store-labour-planner-live.md`, never only in conversation.

## 1. Purpose
- **Story:** shoppers don't arrive evenly. Using real hourly city-centre footfall, forecast next week's footfall for a high-street store and turn it into how many staff to have on the floor each hour, at what cost and service level.
- **Audience:** recruiters and interviewers for data analyst, data scientist and business analyst roles (retail operations, workforce management). Skills to show: hourly time-series forecasting with calendar effects and weather, demand-to-staffing conversion, and scenario costing.
- **Visual standard:** polished and striking, not a default Streamlit page.
- **Not covered here:** till queue simulation (checkout-queue-simulator), rota construction (shift-rota-optimiser).

## 2. Data
- **Leeds City Council city-centre footfall data** (Data Mill North; also on data.gov.uk) — hourly counts from camera sites (Briggate, Headrow, Albion Street, Commercial Street, Park Row, etc.). **Licence: Open Government Licence** (checked 18 Nov 2026). Note: Leeds published **revised weekly data back to 2020** after camera-age and relocation problems; use the revised series and document camera changes. Period: 2021 to latest (skip lockdown periods or flag them).
- **Weather (optional, improves forecasts):** Met Office or Open-Meteo historical hourly weather for Leeds (Open-Meteo data is CC BY 4.0; check). Use only if the download works cleanly.
- **Calendar:** UK bank holidays (gov.uk JSON, OGL), school holidays (Leeds City Council), BRC retail calendar weeks.
- **Store assumptions (editable, labelled):** a store on Briggate captures X% of passing footfall (default 2%), conversion rate, average transaction time, customers per staff member per hour at the target service level, staff hourly cost, minimum staff on floor.
- **Data shape (`data/`):** `footfall_hourly.parquet` (site × hour), `forecast_hourly.parquet`, `backtest.csv`, `metrics.json`. Scripts in `scripts/`.
- **If the data fails:** if the Leeds data can't be downloaded, look for another UK council's open hourly footfall data (OGL) and tell Farzad. Don't fabricate real-looking footfall; if nothing usable exists, stop and recommend parking.

## 3. Models and methods (one forecast model plus a staffing calculator)
1. **Hourly footfall forecast** per site, next 2 weeks: compare seasonal naive, a regression with hour × weekday effects + holidays + weather, and LightGBM on calendar/weather features. Rolling-origin backtest (last 8 weeks), MAE and WAPE by hour; 80% intervals from quantile models or residual bootstrap.
2. **Staffing calculator:** expected store visitors = footfall × capture rate; staff needed per hour = ceil(visitors × conversion × service time / (60 × target utilisation)), with minimum cover; cost per hour; service risk where forecast upper bound exceeds staffing.
- **Unit tests:** hourly series has no duplicate timestamps per site; calculator matches hand examples; backtest windows don't leak.

## 4. The page (one page, top to bottom)
- **Header:** "Store Labour Planner", one-line story, data credit.
- **Controls:** camera site (store location), week to plan, store assumptions, service target.
- **KPI strip:** forecast footfall next week, peak hour, staff-hours needed, weekly labour cost, hours at service risk.
- **Panel 1 — Footfall patterns.** Heatmap weekday × hour; trend with events (holidays, Christmas) marked.
- **Panel 2 — Forecast.** Next two weeks hourly with intervals; backtest comparison of methods.
- **Panel 3 — Staffing plan.** Staff needed by hour (bars) for each day of the planned week, with risk shading; downloadable table.
- **Panel 4 — What if.** Capture rate and conversion sliders → change in staff-hours and cost.
- **"How to read this" expander** and **footer** (Leeds City Council OGL credit, revised-data note, assumptions, "Portfolio demo").

## 5. Look and feel
- Custom Plotly template and `.streamlit/config.toml` theme; fixed colours per category used consistently on every chart.
- Generous spacing, card-style KPIs, no default Streamlit grey. Works at phone width (panels stack; wide tables collapse to key columns). Readable in light and dark.

## 6. Performance
- Cold start under about 10 seconds, repo under about 50 MB, `st.cache_data` throughout. All heavy downloads and model fitting happen offline in `scripts/`; raw files stay out of git.

## 7. Checks before stopping
- Unit tests pass; every filter combination works, including empty ones (friendly message, no errors).
- The app runs from a clean install of `requirements.txt`.
- Desktop and phone checked visually, light and dark.
- Headline numbers on the page match the exported `metrics.json` (or equivalent) and, where stated, the published official figures.
- Specific: camera relocations and the revised series are documented and visible as markers where relevant
- Specific: store assumptions are labelled as assumptions on the page
- Specific: forecast intervals widen sensibly on holidays and special days

## 8. Files
- `app.py`, `scripts/`, `data/`, `tests/`, `requirements.txt` (pinned, Python 3.12+), `.streamlit/config.toml`, `README.md`, `SPEC.md`, `.gitignore`.
- The README must carry the live app URL, what the dashboard does, the data sources and licences (with required credits), and how to run and refit it. Assume the reader has no other context.
- After launch: `preview.png` (about 1300×690 @2x) and `screenshots/` — **committed to the repo**, not only saved locally.

## 9. Launch details (used only after Farzad says go)
- Repo: `farzad-vaziri/store-labour-planner` (private), branch `main`.
- App URL (public): `store-labour-planner.streamlit.app`. No `farzad-` prefix and no fallback name: if the name is taken, stop and ask Farzad.
- Suggested home-page card text (put it in the live note; the home-page chat reads it from there): "How many staff should a high-street shop have on the floor each hour? Forecast real city-centre footfall, turn it into an hour-by-hour staffing plan, and see the cost of covering the peaks."

## Choices made by the ideas chat (defaults)
- Real hourly Leeds footfall (OGL), revised series from 2020; weather optional.
- Three forecasting methods compared by backtest.
- Store capture and service assumptions editable and labelled.
