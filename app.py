"""Store Labour Planner - forecast city-centre footfall and turn it into an hourly staffing plan."""
import json
import pathlib

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from planner.staffing import StoreAssumptions, customers_per_staff_hour, staffing_plan, summarise, staff_needed

ROOT = pathlib.Path(__file__).parent
DATA = ROOT / "data"
DEFAULT_SITE = "Briggate M&S"

st.set_page_config(page_title="Store Labour Planner", page_icon="🛍️", layout="wide",
                   initial_sidebar_state="collapsed")

# ---------------------------------------------------------------- theme
DARK = getattr(getattr(st.context, "theme", None), "type", "light") == "dark"
C = dict(
    surface="#1a1a19" if DARK else "#fcfcfb",
    ink="#f0efec" if DARK else "#0b0b0b",
    ink2="#c3c2b7" if DARK else "#52514e",
    muted="#898781",
    grid="#2c2c2a" if DARK else "#e1e0d9",
    axis="#383835" if DARK else "#c3c2b7",
    forecast="#3987e5" if DARK else "#2a78d6",     # footfall forecast (blue) - every chart
    actual="#f0efec" if DARK else "#0b0b0b",       # observed footfall (ink)
    staff="#d95926" if DARK else "#eb6834",        # staff on floor (orange) - every chart
    regression="#9085e9" if DARK else "#6250d6",   # method: regression (violet)
    naive="#898781",                               # method: seasonal naive (grey)
    risk="#d03b3b",                                # status: service risk (critical)
    event="#c98500" if DARK else "#b07800",        # calendar events
)
METHOD_COLOURS = {"seasonal_naive": C["naive"], "regression": C["regression"], "lightgbm": C["forecast"],
                  "lightgbm_no_weather": C["forecast"]}
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQ_DARK = ["#1f2a38", "#184f95", "#1c5cab", "#2a78d6", "#5598e7", "#86b6ef", "#cde2fb"]


def rgba(hex_, a):
    h = hex_.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{a})"


def style(fig, height=360, legend=True):
    fig.update_layout(
        height=height, margin=dict(l=8, r=8, t=36, b=8), paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif",
                                                 color=C["ink2"], size=13),
        hoverlabel=dict(bgcolor=C["surface"], bordercolor=C["axis"], font=dict(color=C["ink"])),
        showlegend=legend, legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0,
                                       font=dict(color=C["ink2"]), bgcolor="rgba(0,0,0,0)"),
        hovermode="x unified" if legend else "closest", bargap=0.18,
    )
    fig.update_xaxes(gridcolor=C["grid"], linecolor=C["axis"], zeroline=False, tickfont=dict(color=C["muted"]),
                     showgrid=False)
    fig.update_yaxes(gridcolor=C["grid"], linecolor=C["axis"], zeroline=False, tickfont=dict(color=C["muted"]),
                     title_font=dict(color=C["muted"], size=12))
    return fig


st.markdown(f"""
<style>
.block-container {{padding-top: 3.2rem; padding-bottom: 3rem; max-width: 1240px;}}
h1, h2, h3 {{letter-spacing: -0.01em;}}
.hero-kicker {{font-size: .78rem; font-weight: 600; letter-spacing: .12em; text-transform: uppercase;
  color: {C['forecast']}; margin-bottom: .2rem;}}
.hero-title {{font-size: 2.5rem; font-weight: 750; line-height: 1.1; margin: 0 0 .5rem 0; color: {C['ink']};}}
.hero-sub {{font-size: 1.08rem; color: {C['ink2']}; max-width: 760px; margin-bottom: .3rem;}}
.credit {{font-size: .8rem; color: {C['muted']};}}
.kpis {{display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 14px; margin: 8px 0 6px 0;}}
.kpi {{background: {C['surface']}; border: 1px solid {rgba(C['ink'], .10)}; border-radius: 14px;
  padding: 16px 18px 14px 18px;}}
.kpi .lab {{font-size: .78rem; color: {C['ink2']}; font-weight: 600; margin-bottom: 6px;}}
.kpi .val {{font-size: 1.75rem; font-weight: 700; color: {C['ink']}; line-height: 1.15; font-variant-numeric: tabular-nums;}}
.kpi .sub {{font-size: .78rem; color: {C['muted']}; margin-top: 4px;}}
.kpi .bar {{height: 3px; width: 28px; border-radius: 2px; margin-bottom: 10px;}}
.panel-h {{font-size: 1.35rem; font-weight: 700; margin: .1rem 0 .1rem 0; color: {C['ink']};}}
.panel-n {{margin-top: 2.6rem; font-size: .74rem; font-weight: 700; letter-spacing: .12em; color: {C['muted']}; text-transform: uppercase;}}
.panel-s {{font-size: .95rem; color: {C['ink2']}; margin-bottom: .6rem; max-width: 820px;}}
.assume {{display: inline-block; white-space: nowrap; font-size: .7rem; font-weight: 700; letter-spacing: .06em; text-transform: uppercase;
  color: {C['event']}; border: 1px solid {rgba(C['event'], .5)}; border-radius: 999px; padding: 1px 8px; margin-left: 6px;}}
.chip {{display:block; line-height:1.9; font-size:.8rem; color:{C['ink2']}; margin-right: 14px;}}
.dot {{width:10px; height:10px; border-radius:3px; display:inline-block;}}
.foot {{font-size: .8rem; color: {C['muted']}; border-top: 1px solid {rgba(C['ink'], .10)}; margin-top: 2.5rem; padding-top: 1rem;}}
div[data-testid="stExpander"] details {{border-radius: 12px;}}
@media (max-width: 900px) {{
  .kpis {{grid-template-columns: repeat(2, minmax(0, 1fr));}}
  .hero-title {{font-size: 1.9rem;}}
  .kpi .val {{font-size: 1.4rem;}}
}}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------- data
@st.cache_data
def load():
    ff = pd.read_parquet(DATA / "footfall_hourly.parquet")
    ff["site"] = ff.site.astype(str)
    fc = pd.read_parquet(DATA / "forecast_hourly.parquet")
    bt = pd.read_csv(DATA / "backtest.csv", parse_dates=["origin", "ts"])
    days = pd.read_parquet(DATA / "calendar_days.parquet")
    metrics = json.loads((DATA / "metrics.json").read_text())
    return ff, fc, bt, days, metrics


@st.cache_data
def daily(ff):
    d = ff.assign(date=ff.ts.dt.normalize()).groupby(["site", "date"], as_index=False).footfall.sum()
    d["roll7"] = d.groupby("site").footfall.transform(lambda s: s.rolling(7, min_periods=4).mean())
    return d


@st.cache_data
def heat(ff, site):
    last = ff.ts.max() - pd.Timedelta(weeks=52)
    g = ff[(ff.site == site) & (ff.ts > last)]
    return g.groupby([g.ts.dt.dayofweek, g.ts.dt.hour]).footfall.mean().unstack()


ff, fc, bt, days, M = load()
SITES = M["sites"]
BEST = M["best_method"]
fc_start = pd.Timestamp(M["forecast_start"])

weeks = (fc[["week_start", "kind"]].drop_duplicates().sort_values("week_start", ascending=False))
week_labels = {}
for _, r in weeks.iterrows():
    ws = r.week_start
    tag = "Forecast" if r.kind == "forecast" else "Backtest (actual known)"
    week_labels[f"{ws:%a %d %b %Y} · {tag}"] = (ws, r.kind)
fwd_weeks = [k for k, v in week_labels.items() if v[1] == "forecast"]
fwd_weeks = sorted(fwd_weeks, key=lambda k: week_labels[k][0])
week_options = fwd_weeks + [k for k in week_labels if k not in fwd_weeks]

# ---------------------------------------------------------------- header
st.markdown(f"""
<div class="hero-kicker">Retail operations · workforce planning</div>
<div class="hero-title">Store Labour Planner</div>
<div class="hero-sub">Shoppers don't arrive evenly. This page forecasts real hourly footfall on Leeds city-centre
streets and turns it into how many staff a high-street store needs on the floor each hour, what that costs,
and where the service is at risk.</div>
<div class="credit">Footfall: Leeds City Council city-centre cameras, Revised Weekly Data via Data Mill North
(Open Government Licence v3.0), {pd.Timestamp(M['data_start']):%d %b %Y} – {pd.Timestamp(M['data_end']):%d %b %Y}.
Weather: Open-Meteo (CC BY 4.0).</div>
""", unsafe_allow_html=True)
st.write("")

# ---------------------------------------------------------------- controls
qp = st.query_params
site0 = qp.get("site", DEFAULT_SITE)
site_idx = SITES.index(site0) if site0 in SITES else (SITES.index(DEFAULT_SITE) if DEFAULT_SITE in SITES else 0)
week_idx = next((i for i, k in enumerate(week_options) if f"{week_labels[k][0]:%Y-%m-%d}" == qp.get("week")), 0)
c1, c2, c3, c4 = st.columns([1.3, 1.6, 1.0, 1.0])
site = c1.selectbox("Store location (camera site)", SITES, index=site_idx)
week_key = c2.selectbox("Week to plan", week_options, index=week_idx,
                        help="Forecast weeks start where the published data ends. Backtest weeks were forecast "
                             "one week ahead in a rolling test, so you can compare the plan with what happened.")
util = c3.slider("Service target: staff busy %", 50, 95, 85, 5,
                 help="Target share of staff time spent serving customers. Lower = more slack, shorter waits.")
plan_to = c4.radio("Plan staff to", ["Median forecast", "80% upper bound"], index=0,
                   help="Planning to the upper bound covers busy surprises at a higher cost.")

with st.sidebar:
    st.markdown(f"### Store assumptions <span class='assume'>assumption</span>", unsafe_allow_html=True)
    st.caption("Illustrative values for a mid-size fashion store. Edit them to match your own store.")
    capture = st.slider("Capture rate: % of passers-by who enter", 0.5, 6.0, 2.0, 0.1) / 100
    conversion = st.slider("Conversion: % of visitors who buy", 5, 80, 30, 1) / 100
    service = st.slider("Staff minutes per buying customer", 1.0, 15.0, 6.0, 0.5,
                        help="Till, help on the floor and fitting rooms, per sale.")
    min_staff = st.number_input("Minimum staff on floor while open", 1, 10, 2)
    cost = st.number_input("Staff cost per hour (£, incl. on-costs)", 8.0, 40.0, 14.50, 0.25)
    oh = st.slider("Opening hours Mon–Sat", 6, 23, (9, 18))
    sh = st.slider("Opening hours Sunday", 6, 23, (11, 17))

A = StoreAssumptions(capture_rate=capture, conversion=conversion, service_min=service, utilisation=util / 100,
                     min_staff=int(min_staff), hourly_cost=float(cost), open_hour=oh[0], close_hour=oh[1],
                     sun_open=sh[0], sun_close=sh[1], plan_to="upper" if plan_to.startswith("80") else "median")

st.markdown(f"<span class='chip'><span class='assume'>assumptions</span>&nbsp; capture {capture:.1%} · conversion "
            f"{conversion:.0%} · {service:g} min per sale · min {min_staff} staff · £{cost:.2f}/h · open "
            f"{oh[0]:02d}–{oh[1]:02d} (Sun {sh[0]:02d}–{sh[1]:02d}) · one person serves "
            f"{customers_per_staff_hour(A):.1f} buyers/h at {util}% busy. Edit in the sidebar (› top left).</span>",
            unsafe_allow_html=True)

ws, kind = week_labels[week_key]
wk = fc[(fc.site == site) & (fc.week_start == ws) & (fc.kind == kind)].sort_values("ts")
plan = staffing_plan(wk, A)
S = summarise(plan)

# ---------------------------------------------------------------- KPIs
def kpi(label, value, sub, colour):
    return (f"<div class='kpi'><div class='bar' style='background:{colour}'></div><div class='lab'>{label}</div>"
            f"<div class='val'>{value}</div><div class='sub'>{sub}</div></div>")

if wk.empty:
    st.info("No forecast for this site and week. Pick another week.")
    st.stop()
peak = wk.loc[wk.pred.idxmax()]
wk_total = float(wk.pred.sum())
k = [
    kpi("Forecast footfall, week", f"{wk_total/1000:,.0f}k", f"people passing {site}", C["forecast"]),
    kpi("Peak hour", f"{peak.ts:%a %H}:00", f"{peak.pred:,.0f} passers-by expected", C["forecast"]),
    kpi("Staff-hours needed", f"{S['staff_hours']:,}", f"over {len(plan)} open hours", C["staff"]),
    kpi("Weekly labour cost", f"£{S['cost']:,.0f}", f"at £{cost:.2f} per staff-hour", C["staff"]),
    kpi("Hours at service risk", f"⚠ {S['risk_hours']}" if S["risk_hours"] else "✓ 0",
        "busy-case demand beyond the planned team", C["risk"]),
]
st.markdown(f"<div class='kpis'>{''.join(k)}</div>", unsafe_allow_html=True)
if plan.empty:
    st.warning("The store is closed all week with these opening hours, so there is nothing to staff. "
               "Widen the opening hours in the sidebar.")

# ---------------------------------------------------------------- panel 1
st.markdown("<div class='panel-n'>Panel 1</div><div class='panel-h'>Footfall patterns</div>"
            "<div class='panel-s'>When people walk past. Average footfall by weekday and hour over the last "
            "52 weeks, and the daily trend with holidays and big shopping days marked.</div>", unsafe_allow_html=True)
p1a, p1b = st.columns([1, 1.35], gap="large")
with p1a:
    h = heat(ff, site)
    dn = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    fig = go.Figure(go.Heatmap(z=h.values, x=[f"{x:02d}" for x in h.columns], y=dn, xgap=2, ygap=2,
                               colorscale=[[i / (len(SEQ) - 1), c] for i, c in enumerate(SEQ_DARK if DARK else SEQ)],
                               colorbar=dict(title=dict(text="per hour", font=dict(color=C["muted"], size=11)),
                                             thickness=10, tickfont=dict(color=C["muted"])),
                               hovertemplate="%{y} %{x}:00<br>%{z:,.0f} people per hour<extra></extra>"))
    style(fig, 330, legend=False)
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title=None, dtick=3)
    fig.update_layout(title=dict(text="Average people per hour", font=dict(size=14, color=C["ink"]), x=0))
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
with p1b:
    d = daily(ff)
    d = d[d.site == site]
    ev = days[(days.event != "") & days.date.between(d.date.min(), d.date.max())].merge(d, on="date", how="left")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d.date, y=d.footfall, mode="lines", line=dict(color=rgba(C["forecast"], .28), width=1),
                             name="Daily footfall", hovertemplate="%{x|%a %d %b %Y}: %{y:,.0f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=d.date, y=d.roll7, mode="lines", line=dict(color=C["forecast"], width=2),
                             name="7-day average", hovertemplate="7-day avg %{y:,.0f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=ev.date, y=ev.footfall, mode="markers", name="Holiday / event",
                             marker=dict(size=9, color=C["event"], symbol="diamond", line=dict(color=C["surface"], width=2)),
                             customdata=ev.event, hovertemplate="%{customdata}: %{y:,.0f}<extra></extra>"))
    fig.add_vline(x=pd.Timestamp(M["data_start"]), line=dict(color=C["muted"], dash="dot", width=1))
    fig.add_annotation(x=pd.Timestamp(M["data_start"]), y=1, yref="paper", text="Revised series starts", showarrow=False,
                       xanchor="left", yanchor="top", font=dict(size=11, color=C["muted"]), xshift=4)
    style(fig, 330)
    fig.update_layout(hovermode="closest", title=dict(text="People per day", font=dict(size=14, color=C["ink"]), x=0, y=.98),
                      legend=dict(y=1.0, x=1, xanchor="right"))
    fig.update_yaxes(rangemode="tozero")
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

# ---------------------------------------------------------------- panel 2
is_fwd = kind == "forecast"
st.markdown("<div class='panel-n'>Panel 2</div><div class='panel-h'>Forecast</div>"
            f"<div class='panel-s'>{'Hourly forecast for the two weeks after the latest published data' if is_fwd else 'The one-week-ahead forecast made for this backtest week, against what the cameras counted'}"
            f", with an 80% interval. Bands are wider on bank holidays, Christmas and other unusual days because "
            f"forecasts have missed by more on those days in the past.</div>", unsafe_allow_html=True)
if is_fwd:
    view = fc[(fc.site == site) & (fc.kind == "forecast")].sort_values("ts")
else:
    view = wk
fig = go.Figure()
fig.add_trace(go.Scatter(x=view.ts, y=view.hi, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
fig.add_trace(go.Scatter(x=view.ts, y=view.lo, mode="lines", line=dict(width=0), fill="tonexty",
                         fillcolor=rgba(C["forecast"], .30 if DARK else .18), name="80% interval",
                         customdata=np.c_[view.lo, view.hi],
                         hovertemplate="80%%: %{customdata[0]:,.0f} – %{customdata[1]:,.0f}<extra></extra>"))
fig.add_trace(go.Scatter(x=view.ts, y=view.pred, mode="lines", line=dict(color=C["forecast"], width=2),
                         name="Forecast", hovertemplate="Forecast %{y:,.0f}<extra></extra>"))
if not is_fwd:
    fig.add_trace(go.Scatter(x=view.ts, y=view.actual, mode="lines", line=dict(color=C["actual"], width=1.5, dash="dot"),
                             name="Actual", hovertemplate="Actual %{y:,.0f}<extra></extra>"))
if is_fwd:
    fig.add_vrect(x0=ws, x1=ws + pd.Timedelta(days=7), fillcolor=rgba(C["forecast"], .06), line_width=0,
                  annotation_text="planned week", annotation_position="top left",
                  annotation_font=dict(size=11, color=C["muted"]))
spec = days[(days.day_type != "normal") & days.date.between(view.ts.min().normalize(), view.ts.max())]
for _, r in spec.iterrows():
    label = r.event or ("School holiday" if r.school_holiday else "Christmas shopping")
    fig.add_vrect(x0=r.date, x1=r.date + pd.Timedelta(days=1), fillcolor=rgba(C["event"], .08), line_width=0)
if len(spec):
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", marker=dict(size=10, color=rgba(C["event"], .35), symbol="square"),
                             name=", ".join(sorted({(r.event or ('School holiday' if r.school_holiday else 'Christmas shopping'))
                                                    for _, r in spec.iterrows()}))))
style(fig, 360)
fig.update_yaxes(title="people per hour", rangemode="tozero")
fig.update_xaxes(tickformat="%a %d %b")
st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
if is_fwd and not bool(view.weather_observed.all()):
    st.caption("Weather beyond the observed record is filled with the Leeds hourly average for that month.")

# backtest comparison
b = bt[bt.site == site]
names = {"seasonal_naive": "Seasonal naive", "regression": "Regression", "lightgbm": "LightGBM",
         "lightgbm_no_weather": "LightGBM, no weather"}
wapes = {m: float((b.actual - b[m]).abs().sum() / b.actual.sum()) for m in names}
p2a, p2b = st.columns([1, 1.35], gap="large")
with p2a:
    order = sorted(names, key=lambda m: wapes[m], reverse=True)
    fig = go.Figure(go.Bar(y=[names[m] for m in order], x=[wapes[m] * 100 for m in order], orientation="h",
                           marker=dict(color=[METHOD_COLOURS[m] for m in order], cornerradius=4,
                                       pattern=dict(shape=["/" if m == "lightgbm_no_weather" else "" for m in order],
                                                    fgcolor=C["surface"])),
                           text=[f"{wapes[m]*100:.1f}%" for m in order], textposition="outside",
                           textfont=dict(color=C["ink2"]),
                           hovertemplate="%{y}: WAPE %{x:.1f}%<extra></extra>"))
    style(fig, 280, legend=False)
    fig.update_xaxes(title="WAPE, % (lower is better)", range=[0, max(wapes.values()) * 135], showgrid=True)
    fig.update_layout(title=dict(text=f"Backtest error at {site}", font=dict(size=14, color=C["ink"]), x=0))
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
with p2b:
    gh = b.groupby(b.ts.dt.hour)
    fig = go.Figure()
    for m in ["seasonal_naive", "regression", "lightgbm"]:
        e = gh.apply(lambda g: (g.actual - g[m]).abs().sum() / max(g.actual.sum(), 1) * 100)
        fig.add_trace(go.Scatter(x=e.index, y=e.values, mode="lines+markers", name=names[m],
                                 line=dict(color=METHOD_COLOURS[m], width=2), marker=dict(size=6),
                                 hovertemplate=names[m] + " %{y:.1f}%<extra></extra>"))
    style(fig, 280)
    fig.update_yaxes(title="WAPE %", rangemode="tozero", range=[0, 60])
    fig.update_xaxes(title="hour of day", dtick=3)
    fig.update_layout(title=dict(text="Error by hour (night hours are tiny counts)", font=dict(size=14, color=C["ink"]), x=0, y=.98),
                      legend=dict(y=1.0, x=1, xanchor="right"))
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
cov = float(((b.actual >= b.lo) & (b.actual <= b.hi)).mean())
st.caption(f"Rolling-origin backtest: {len(M['backtest_origins'])} weekly origins "
           f"({pd.Timestamp(M['backtest_origins'][0]):%d %b} – {pd.Timestamp(M['backtest_origins'][-1]):%d %b %Y}), each forecasting "
           f"the next 7 days from data before it. WAPE = total absolute error ÷ total footfall. The plan uses "
           f"**{names[BEST]}**, best across all sites (WAPE {M['models'][BEST]['wape']*100:.1f}%). "
           f"80% intervals contained {cov:.0%} of actual hours here. The weather inputs are observed weather "
           f"(a perfect weather forecast); the hatched bar shows the same model without weather.")

# ---------------------------------------------------------------- panel 3
st.markdown("<div class='panel-n'>Panel 3</div><div class='panel-h'>Staffing plan</div>"
            "<div class='panel-s'>Staff needed on the floor for each open hour of the planned week. "
            "Red hours are at service risk: if footfall comes in at the top of the 80% interval, the planned "
            "team would be 100% busy and still fall behind. The tick shows the staff needed just to keep up in that case.</div>", unsafe_allow_html=True)
if plan.empty:
    st.info("No open hours this week with the current opening hours.")
else:
    plan = plan.copy()
    plan["day"] = plan.ts.dt.strftime("%a %d")
    dlist = list(dict.fromkeys(plan.day))
    tabs = st.tabs(["Whole week"] + dlist)
    with tabs[0]:
        x = [plan.ts.dt.strftime("%a").values, plan.ts.dt.strftime("%H").values]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=x, y=plan.staff, name="Staff planned",
                             marker=dict(color=np.where(plan.at_risk, C["risk"], C["staff"]), cornerradius=3),
                             customdata=np.c_[plan.pred, plan.visitors, plan.staff_if_high],
                             hovertemplate="%{x}:00<br>Staff planned %{y}<br>Footfall %{customdata[0]:,.0f}"
                                           "<br>Visitors %{customdata[1]:,.0f}<br>Busy case: %{customdata[2]} needed to cope<extra></extra>"))
        fig.add_trace(go.Scatter(x=x, y=plan.staff_if_high, mode="markers", name="Staff to cope in busy case",
                                 marker=dict(symbol="line-ew-open", size=12, color=C["ink2"], line=dict(width=2, color=C["ink2"])),
                                 hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name="⚠ Service risk",
                                 marker=dict(size=10, color=C["risk"], symbol="square")))
        style(fig, 340)
        fig.update_layout(hovermode="closest", bargap=0.15)
        fig.update_yaxes(title="staff on floor", dtick=1, rangemode="tozero", showgrid=True)
        fig.update_xaxes(tickfont=dict(size=10))
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    for t, dname in zip(tabs[1:], dlist):
        with t:
            g = plan[plan.day == dname]
            fig = go.Figure(go.Bar(x=g.ts.dt.strftime("%H:00"), y=g.staff, name="Staff planned",
                                   marker=dict(color=np.where(g.at_risk, C["risk"], C["staff"]), cornerradius=4),
                                   text=g.staff, textposition="outside", textfont=dict(color=C["ink2"]),
                                   customdata=np.c_[g.pred, g.lo, g.hi],
                                   hovertemplate="%{x}<br>Staff %{y}<br>Footfall %{customdata[0]:,.0f} "
                                                 "(%{customdata[1]:,.0f}–%{customdata[2]:,.0f})<extra></extra>"))
            style(fig, 300, legend=False)
            fig.update_yaxes(title="staff on floor", dtick=1, range=[0, max(g.staff_if_high.max(), g.staff.max()) + 1.5])
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
            st.caption(f"{dname}: {int(g.staff.sum())} staff-hours, £{g.cost.sum():,.0f}, "
                       f"{int(g.at_risk.sum())} hour(s) at risk.")
    table = pd.DataFrame({
        "Day": plan.ts.dt.strftime("%a %d %b"), "Hour": plan.ts.dt.strftime("%H:00"),
        "Footfall (forecast)": plan.pred.round(0).astype(int), "Footfall low": plan.lo.round(0).astype(int),
        "Footfall high": plan.hi.round(0).astype(int), "Store visitors": plan.visitors.round(1),
        "Buyers": plan.buyers.round(1), "Staff": plan.staff, "Staff if busy": plan.staff_if_high,
        "Service risk": np.where(plan.at_risk, "yes", ""), "Cost £": plan.cost.round(2)})
    if not is_fwd:
        table.insert(3, "Footfall (actual)", plan.actual.round(0).astype("Int64"))
    with st.expander("Hour-by-hour table"):
        st.dataframe(table[["Day", "Hour", "Footfall (forecast)", "Staff", "Service risk", "Cost £"]],
                     hide_index=True, width="stretch", height=300)
        st.caption("Key columns shown; the download has every column.")
    st.download_button("⬇ Download staffing plan (CSV)", table.to_csv(index=False).encode(),
                       file_name=f"staffing-plan-{site.lower().replace(' ', '-').replace('&', 'and')}-{ws:%Y-%m-%d}.csv",
                       mime="text/csv")

# ---------------------------------------------------------------- panel 4
st.markdown("<div class='panel-n'>Panel 4</div><div class='panel-h'>What if</div>"
            "<div class='panel-s'>How sensitive is the plan to the store assumptions? Move capture and conversion "
            "to see the change in staff-hours and cost against the current plan.</div>", unsafe_allow_html=True)
open_wk = wk.loc[wk.ts.isin(plan.ts)] if not plan.empty else wk.iloc[:0]
if open_wk.empty:
    st.info("No open hours to test. Widen the opening hours in the sidebar.")
else:
    w1, w2 = st.columns([1, 1.35], gap="large")
    with w1:
        cap2 = st.slider("What-if capture rate (%)", 0.5, 6.0, round(capture * 100, 1), 0.1, key="wi_cap") / 100
        conv2 = st.slider("What-if conversion (%)", 5, 80, int(round(conversion * 100)), 1, key="wi_conv") / 100
        A2 = StoreAssumptions(**{**A.dict(), "capture_rate": cap2, "conversion": conv2})
        S2 = summarise(staffing_plan(wk, A2))
        dh, dc = S2["staff_hours"] - S["staff_hours"], S2["cost"] - S["cost"]
        st.markdown("<div class='kpis' style='grid-template-columns:repeat(2,minmax(0,1fr))'>" +
                    kpi("Staff-hours", f"{S2['staff_hours']:,}", f"{dh:+,} vs current plan", C["staff"]) +
                    kpi("Weekly cost", f"£{S2['cost']:,.0f}", f"{'+' if dc >= 0 else '−'}£{abs(dc):,.0f} vs current plan", C["staff"]) +
                    "</div>", unsafe_allow_html=True)
        st.caption(f"Hours at service risk: {S2['risk_hours']} (current plan {S['risk_hours']}).")
    with w2:
        caps = np.round(np.linspace(0.5, 6.0, 12), 1)
        convs = np.arange(10, 81, 7.5)
        basis = (open_wk.hi if A.plan_to == "upper" else open_wk.pred).values
        z = np.array([[staff_needed(basis, StoreAssumptions(**{**A.dict(), "capture_rate": c / 100, "conversion": v / 100})).sum()
                       * A.hourly_cost for c in caps] for v in convs])
        fig = go.Figure(go.Heatmap(z=z, x=caps, y=convs, xgap=2, ygap=2,
                                   colorscale=[[i / (len(SEQ) - 1), c] for i, c in enumerate(["#fde9df", "#f7c3aa", "#f19a74", "#eb6834", "#c4501f", "#9a3d16", "#6e2a0e"] if not DARK else ["#2a1a12", "#6e2a0e", "#9a3d16", "#c4501f", "#eb6834", "#f19a74", "#fde9df"])],
                                   colorbar=dict(title=dict(text="£ / week", font=dict(color=C["muted"], size=11)),
                                                 thickness=10, tickfont=dict(color=C["muted"])),
                                   hovertemplate="capture %{x}%<br>conversion %{y}%<br>£%{z:,.0f} per week<extra></extra>"))
        fig.add_trace(go.Scatter(x=[cap2 * 100], y=[conv2 * 100], mode="markers", name="what-if",
                                 marker=dict(size=14, symbol="circle-open", line=dict(width=3, color=C["ink"])),
                                 hovertemplate="what-if<extra></extra>"))
        fig.add_trace(go.Scatter(x=[capture * 100], y=[conversion * 100], mode="markers", name="current",
                                 marker=dict(size=9, color=C["ink"], line=dict(width=2, color=C["surface"])),
                                 hovertemplate="current plan<extra></extra>"))
        style(fig, 330)
        fig.update_layout(hovermode="closest", title=dict(text="Weekly labour cost", font=dict(size=14, color=C["ink"]), x=0, y=.98),
                          legend=dict(y=1.0, x=1, xanchor="right"))
        fig.update_xaxes(title="capture rate %")
        fig.update_yaxes(title="conversion %")
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

# ---------------------------------------------------------------- explainer + footer
st.write("")
with st.expander("How to read this"):
    st.markdown(f"""
**The question.** A shop on a busy street knows roughly how many people walk past. How many staff should be on
the floor each hour next week, and what will that cost?

**Footfall.** Eight Leeds City Council cameras count people walking past every hour (both directions added).
Panel 1 shows the rhythm: quiet mornings, a lunchtime and Saturday-afternoon peak, Sundays shorter,
and big spikes before Christmas.

**Forecast.** Three methods are tested on the last {len(M['backtest_origins'])} weeks, each time using only data
available before that week: *seasonal naive* (same hour last week), a *regression* on hour × weekday,
holidays and weather, and *LightGBM* on calendar, weather and recent-level features. The lowest-error method
drives the plan. The 80% band comes from how wrong past forecasts were for the same kind of day and hour;
bank holidays, Christmas and similar days get wider bands.

**From footfall to staff.** Visitors = footfall × capture rate. Staff needed =
⌈visitors × conversion × minutes per sale ÷ (60 × target busy %)⌉, never below the minimum cover.
A higher busy % runs leaner; planning to the 80% upper bound buys protection against busy surprises.

**Service risk.** An hour is at risk when, in the busy case (top of the 80% band), the planned team would be
fully busy and still not keep up, so queues would build. Lower the busy % or plan to the upper bound to remove it.

**Assumptions.** Capture, conversion, minutes per sale, cost and opening hours are illustrative assumptions,
not data. They are labelled as such and editable in the sidebar.
""")

with st.expander("About the data: revised series and camera changes"):
    st.markdown("""
- **Source:** Leeds City Council, *Leeds city centre footfall data*, Data Mill North. Open Government Licence v3.0.
- **Revised series.** Leeds found that some cameras had reported low counts because of the age of the camera
  system and the relocation of some cameras. It republished corrected data as *Revised Weekly Data*. This page
  uses only the revised files, which run continuously from 30 Oct 2023 (marked on the trend chart). Older,
  unrevised files are not used.
- **Camera names and moves.** Between April and July 2023 the cameras were renamed to show where they now sit,
  some after relocation: Albion Street South → *Albion Street at Bond Street*; Briggate → *Briggate M&S*;
  Briggate at McDonalds → *Briggate at Swan Street*; Dortmund Square → *Headrow at Broadgate*;
  Headrow → *Headrow at Lands Lane*. Albion Street at McDonalds, Commercial Street at Sharps and Park Row kept their names.
  All moves predate the revised series, so there are no relocation breaks inside the data shown.
- **Calendar:** UK bank holidays from gov.uk (OGL); Leeds local-authority school holidays; Easter Sunday,
  Black Friday and the Christmas period added as retail days.
- **Weather:** Open-Meteo historical hourly weather for central Leeds (CC BY 4.0).
""")

st.markdown(f"""<div class='foot'>Contains public sector information licensed under the Open Government Licence v3.0:
footfall data © Leeds City Council (Data Mill North), revised weekly series. Weather data by Open-Meteo.com (CC BY 4.0).
Store figures are assumptions, not data. Portfolio demo, not operational advice. Model fitted offline;
best method {names[BEST]}.</div>""", unsafe_allow_html=True)
