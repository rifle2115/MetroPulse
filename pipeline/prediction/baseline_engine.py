"""
Expected crowd baseline from time-of-day and calendar — independent of
whatever a camera currently sees, meant to sit alongside the live reactive
`zone/state` reading so an operator can compare predicted vs. actual.

Backed by real Bangalore Namma Metro (BMRCL) data, not a placeholder:
  - data/station-hourly.parquet — real per-station hourly entries, Aug-Sep
    2025, 83 stations (github.com/Vonter/bmrcl-ridership-hourly, RTI-sourced,
    ODbL 1.0). Aggregated into a (day-of-week, hour) -> average system-wide
    ridership lookup, then bucketed into low/med/high by tercile — this is
    what "peak hours" means here: empirically the busiest third of hourly
    slots in real data, not a guessed rush-hour window.
  - data/significant_dates.csv — real Bangalore events/holidays 2024-2026
    (github.com/thecont1/namma-metro-ridership-tracker, BSD-0). Any date in
    this file overrides the hour/weekday estimate with "high".

The (day-of-week, hour) pattern generalizes past the Aug-Sep 2025 collection
window (it's a weekly cycle, not tied to specific dates) — the significant-
dates list is what's actually date-bound, and only covers 2024-2026.
"""

from datetime import datetime
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent / "data"


def _load_hourly_pattern() -> tuple[dict, float, float]:
    df = pd.read_parquet(DATA_DIR / "station-hourly.parquet")
    df["Date"] = pd.to_datetime(df["Date"])
    by_date_hour = df.groupby(["Date", "Hour"])["Ridership"].sum().reset_index()
    by_date_hour["dow"] = by_date_hour["Date"].dt.dayofweek
    avg = by_date_hour.groupby(["dow", "Hour"])["Ridership"].mean()
    pattern = {(int(dow), int(hour)): float(v) for (dow, hour), v in avg.items()}
    return pattern, float(avg.quantile(0.33)), float(avg.quantile(0.67))


def _load_significant_dates() -> dict:
    df = pd.read_csv(DATA_DIR / "significant_dates.csv")
    events: dict[str, list] = {}
    for date_str, event in zip(df["Date"], df["Event"]):
        events.setdefault(date_str, []).append(event)
    return {date_str: ", ".join(names) for date_str, names in events.items()}


_HOURLY_PATTERN, _LOW_CUTOFF, _HIGH_CUTOFF = _load_hourly_pattern()
_SIGNIFICANT_DATES = _load_significant_dates()


def expected_level(dt: datetime) -> tuple[str, str]:
    date_str = dt.strftime("%Y-%m-%d")
    if date_str in _SIGNIFICANT_DATES:
        return "high", f"{_SIGNIFICANT_DATES[date_str]} (Bangalore Metro significant date)"

    key = (dt.weekday(), dt.hour)
    avg = _HOURLY_PATTERN.get(key)
    if avg is None:  # shouldn't happen — all 7*24 combos are present in the data
        return "med", "no BMRCL data for this hour/weekday"

    day_name = dt.strftime("%a")
    if avg <= _LOW_CUTOFF:
        return "low", f"{day_name} {dt.hour:02d}:00 avg ~{avg:,.0f} riders/hr (BMRCL Aug-Sep 2025 data)"
    if avg <= _HIGH_CUTOFF:
        return "med", f"{day_name} {dt.hour:02d}:00 avg ~{avg:,.0f} riders/hr (BMRCL Aug-Sep 2025 data)"
    return "high", f"{day_name} {dt.hour:02d}:00 avg ~{avg:,.0f} riders/hr - peak hour (BMRCL Aug-Sep 2025 data)"
