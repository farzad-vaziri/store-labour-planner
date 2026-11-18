"""Demand-to-staffing conversion. Pure functions, no Streamlit."""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd


@dataclass
class StoreAssumptions:
    capture_rate: float = 0.02        # share of passing footfall that enters the store
    conversion: float = 0.30          # share of visitors who buy
    service_min: float = 6.0          # staff minutes per buying customer (till, help, fitting room)
    utilisation: float = 0.75         # target share of staff time spent serving (service target)
    min_staff: int = 2                # minimum staff on the floor while open
    hourly_cost: float = 14.50        # £ per staff-hour incl. employer on-costs
    open_hour: int = 9                # Mon-Sat opening (first trading hour starts here)
    close_hour: int = 18              # Mon-Sat closing (last trading hour ends here)
    sun_open: int = 11
    sun_close: int = 17
    plan_to: str = "median"           # "median" or "upper" (80% upper bound)

    def dict(self):
        return asdict(self)


def customers_per_staff_hour(a: StoreAssumptions) -> float:
    """Buying customers one person can serve in an hour at the target utilisation."""
    return 60.0 * a.utilisation / a.service_min


def staff_needed(footfall, a: StoreAssumptions) -> np.ndarray:
    """ceil(visitors x conversion x service minutes / (60 x utilisation)), floored at min cover."""
    f = np.clip(np.asarray(footfall, dtype=float), 0, None)
    workload = f * a.capture_rate * a.conversion * a.service_min
    raw = np.ceil(np.round(workload / (60.0 * a.utilisation), 9))
    return np.maximum(raw, a.min_staff).astype(int)


def staff_to_cope(footfall, a: StoreAssumptions) -> np.ndarray:
    """Staff needed to keep up at all (100% busy). Fewer than this and queues build without limit."""
    f = np.clip(np.asarray(footfall, dtype=float), 0, None)
    workload = f * a.capture_rate * a.conversion * a.service_min
    return np.maximum(np.ceil(np.round(workload / 60.0, 9)), a.min_staff).astype(int)


def is_open(ts: pd.Series, a: StoreAssumptions) -> np.ndarray:
    ts = pd.to_datetime(ts)
    h, sun = ts.dt.hour.values, (ts.dt.dayofweek == 6).values
    return np.where(sun, (h >= a.sun_open) & (h < a.sun_close), (h >= a.open_hour) & (h < a.close_hour))


def staffing_plan(fc: pd.DataFrame, a: StoreAssumptions) -> pd.DataFrame:
    """fc needs columns ts, pred, lo, hi (footfall). Returns one row per open hour."""
    df = fc.loc[is_open(fc.ts, a)].copy()
    basis = df["hi"] if a.plan_to == "upper" else df["pred"]
    df["visitors"] = df["pred"] * a.capture_rate
    df["buyers"] = df["visitors"] * a.conversion
    df["staff"] = staff_needed(basis, a)
    df["staff_if_high"] = staff_to_cope(df["hi"], a)      # busy case, every staff member 100% busy
    df["at_risk"] = df["staff_if_high"] > df["staff"]     # planned team could not keep up in the busy case
    df["cost"] = df["staff"] * a.hourly_cost
    return df


def summarise(plan: pd.DataFrame) -> dict:
    if plan.empty:
        return dict(staff_hours=0, cost=0.0, risk_hours=0, footfall=0.0)
    return dict(staff_hours=int(plan.staff.sum()), cost=float(plan.cost.sum()),
                risk_hours=int(plan.at_risk.sum()), footfall=float(plan.pred.sum()))
