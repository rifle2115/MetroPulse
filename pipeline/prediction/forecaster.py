"""
Passenger crowd forecaster — predicts hourly station entries ahead of time.

Used only by the passenger app (user_app/serve.py serves it at /api/forecast).
Nothing here publishes to MQTT, so the Control Room and the Twin never see it.

Model
-----
Gradient-boosted trees (scikit-learn HistGradientBoostingRegressor) trained on
real BMRCL hourly entries (data/station-hourly.parquet, 83 stations, Aug-Sep
2025) plus the Bangalore significant-dates list. Features:

    station, hour, day of week, weekend flag, event/holiday flag,
    day-before-event flag, day-after-event flag

A Poisson model gives the expected count. The likely range shown in the app
comes from how far real crowds landed from the model's predictions on days it
had not seen (10th-90th percentile of actual/predicted, per station).

The model learns the weekly cycle and how events shift it. It cannot learn a
long-term trend: 48 days of history is not enough to see one, so forecasts for
2026 dates assume 2025 ridership levels.

Usage
-----
    python forecaster.py train      # fit, evaluate, save models/forecaster.joblib
    python forecaster.py demo       # print a sample forecast
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
MODEL_DIR = HERE / "models"
MODEL_PATH = MODEL_DIR / "forecaster.joblib"
METRICS_PATH = MODEL_DIR / "forecaster_metrics.json"

# The Twin's stations are fictional; each borrows the ridership pattern of a
# real Purple Line station with a similar role on the line.
TWIN_STATION_PROFILE = {
    "Central City": "Nadaprabhu Kempegowda Station, Majestic",
    "University Road": "Jnanabharathi",
    "Lakeside": "Halasuru",
    "North Gate": "Dr. B. R. Ambedkar Station, Vidhana Soudha",
    "Airport Terminal": "Whitefield (Kadugodi)",
}
DEFAULT_STATION = "Central City"

FEATURES = ["station", "hour", "dow", "weekend", "event", "pre_event", "post_event"]
TEST_DAYS = 14


# ----------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------
def load_events() -> dict[str, str]:
    df = pd.read_csv(DATA_DIR / "significant_dates.csv")
    out: dict[str, list[str]] = {}
    for d, e in zip(df["Date"], df["Event"]):
        out.setdefault(str(d), []).append(str(e))
    return {d: ", ".join(v) for d, v in out.items()}


def _calendar(dates: pd.Series, events: dict[str, str]) -> pd.DataFrame:
    ev = set(events)
    day = dates.dt.normalize()
    key = day.dt.strftime("%Y-%m-%d")
    before = (day + pd.Timedelta(days=1)).dt.strftime("%Y-%m-%d")
    after = (day - pd.Timedelta(days=1)).dt.strftime("%Y-%m-%d")
    return pd.DataFrame({
        "dow": day.dt.day_of_week.astype(int),
        "weekend": (day.dt.day_of_week >= 5).astype(int),
        "event": key.isin(ev).astype(int),
        "pre_event": before.isin(ev).astype(int),
        "post_event": after.isin(ev).astype(int),
    }, index=dates.index)


def load_training_frame(events: dict[str, str]) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_parquet(DATA_DIR / "station-hourly.parquet")
    df["Date"] = pd.to_datetime(df["Date"])
    stations = sorted(df["Station"].unique())
    code = {s: i for i, s in enumerate(stations)}
    cal = _calendar(df["Date"], events)
    X = pd.concat([
        pd.DataFrame({"station": df["Station"].map(code).astype(int),
                      "hour": df["Hour"].astype(int)}),
        cal,
    ], axis=1)[FEATURES]
    X["date"] = df["Date"]
    X["y"] = df["Ridership"].astype(float)
    return X, stations


# ----------------------------------------------------------------------------
# training
# ----------------------------------------------------------------------------
def _fit(X: pd.DataFrame, y: pd.Series):
    from sklearn.ensemble import HistGradientBoostingRegressor

    common = dict(categorical_features=[0], max_iter=400, learning_rate=0.08,
                  max_leaf_nodes=63, min_samples_leaf=20, random_state=0)
    return HistGradientBoostingRegressor(loss="poisson", **common).fit(X, y)


def _ratio_bands(df: pd.DataFrame, pred: np.ndarray, stations: list[str]) -> dict:
    """Per-station 10th/90th percentile of actual / predicted, busy hours only."""
    d = df.assign(p=pred)
    d = d[(d["hour"] >= 6) & (d["p"] >= 20)]
    d = d.assign(r=d["y"] / d["p"])
    q = d.groupby("station")["r"].quantile([0.10, 0.90]).unstack()
    allq = d["r"].quantile([0.10, 0.90]).to_numpy()
    return {stations[c]: (float(q.loc[c, 0.10]), float(q.loc[c, 0.90])) if c in q.index
            else (float(allq[0]), float(allq[1])) for c in range(len(stations))}


def _wape(y, p) -> float:
    return float(np.abs(y - p).sum() / max(np.abs(y).sum(), 1e-9))


def train() -> dict:
    import joblib

    events = load_events()
    frame, stations = load_training_frame(events)
    days = sorted(frame["date"].unique())
    cutoff = days[-TEST_DAYS]
    tr, te = frame[frame["date"] < cutoff], frame[frame["date"] >= cutoff]

    # --- held-out evaluation: train on the first ~5 weeks, score the last 2
    model = _fit(tr[FEATURES], tr["y"])
    p = np.clip(model.predict(te[FEATURES]), 0, None)

    # range check without peeking: learn the ranges on the first test week,
    # measure how often the second week's real crowds land inside them
    te_days = sorted(te["date"].unique())
    first = (te["date"] < te_days[len(te_days) // 2]).to_numpy()
    rb = _ratio_bands(te[first], p[first], stations)
    second = te[~first]
    lo_r = second["station"].map(lambda c: rb[stations[c]][0]).to_numpy()
    hi_r = second["station"].map(lambda c: rb[stations[c]][1]).to_numpy()
    y2, p2 = second["y"].to_numpy(), p[~first]
    busy2 = (second["hour"] >= 6).to_numpy() & (p2 >= 20)
    coverage = float(np.mean((y2[busy2] >= p2[busy2] * lo_r[busy2]) & (y2[busy2] <= p2[busy2] * hi_r[busy2])))

    # baseline the model has to beat: average for that station/weekday/hour
    base_tbl = tr.groupby(["station", "dow", "hour"])["y"].mean()
    base = te.set_index(["station", "dow", "hour"]).index.map(base_tbl).to_numpy(dtype=float)
    base = np.nan_to_num(base)

    y = te["y"].to_numpy()
    open_hours = (te["hour"] >= 6).to_numpy()
    metrics = {
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "history": {"from": str(pd.Timestamp(days[0]).date()), "to": str(pd.Timestamp(days[-1]).date()),
                    "days": len(days), "stations": len(stations), "rows": int(len(frame))},
        "test_window": {"from": str(pd.Timestamp(cutoff).date()), "days": TEST_DAYS},
        "model": {
            "mae": float(np.mean(np.abs(y - p))),
            "wape": _wape(y[open_hours], p[open_hours]),
            "range_coverage": coverage,
        },
        "baseline_average": {
            "mae": float(np.mean(np.abs(y - base))),
            "wape": _wape(y[open_hours], base[open_hours]),
        },
    }
    ev_mask = te["event"].to_numpy() == 1
    if ev_mask.any():
        metrics["event_days"] = {
            "model_wape": _wape(y[ev_mask], p[ev_mask]),
            "baseline_wape": _wape(y[ev_mask], base[ev_mask]),
        }
    metrics["improvement_vs_baseline"] = 1 - metrics["model"]["wape"] / metrics["baseline_average"]["wape"]

    # --- final model on all history
    ranges = _ratio_bands(te, p, stations)   # all held-out residuals
    mean = _fit(frame[FEATURES], frame["y"])

    # per-station crowd bands, from that station's own busy hours (06:00+)
    awake = frame[frame["hour"] >= 6]
    bands = {}
    for code, g in awake.groupby("station"):
        hourly = g.groupby(["dow", "hour"])["y"].mean()
        bands[stations[code]] = (float(hourly.quantile(0.40)), float(hourly.quantile(0.75)))

    # per-station whole-day bands, from that station's real daily totals
    daily = frame.groupby(["station", "date"])["y"].sum().reset_index()
    day_bands = {stations[c]: (float(g["y"].quantile(0.33)), float(g["y"].quantile(0.75)))
                 for c, g in daily.groupby("station")}

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump({"mean": mean, "ranges": ranges, "stations": stations, "bands": bands,
                 "day_bands": day_bands}, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    return metrics


# ----------------------------------------------------------------------------
# inference
# ----------------------------------------------------------------------------
class Forecaster:
    def __init__(self):
        import joblib

        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"{MODEL_PATH} missing — run: python forecaster.py train")
        b = joblib.load(MODEL_PATH)
        self.mean, self.ranges = b["mean"], b["ranges"]
        self.stations, self.bands = b["stations"], b["bands"]
        self.day_bands = b["day_bands"]
        self.code = {s: i for i, s in enumerate(self.stations)}
        self.events = load_events()
        self.metrics = json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else {}

    def _level(self, real_station: str, value: float) -> str:
        lo, hi = self.bands[real_station]
        return "low" if value < lo else "high" if value >= hi else "med"

    def _predict(self, real_station: str, times: list[datetime]) -> pd.DataFrame:
        ts = pd.Series(pd.to_datetime(times))
        X = pd.concat([
            pd.DataFrame({"station": self.code[real_station], "hour": ts.dt.hour.astype(int)}),
            _calendar(ts, self.events),
        ], axis=1)[FEATURES]
        mean = np.clip(self.mean.predict(X), 0, None)
        rlo, rhi = self.ranges[real_station]
        lo, hi = mean * min(rlo, 1.0), mean * max(rhi, 1.0)
        return pd.DataFrame({"time": ts, "expected": mean, "low": lo, "high": hi})

    def forecast(self, twin_station: str | None, at: datetime, hours: int = 18) -> dict:
        twin_station = twin_station if twin_station in TWIN_STATION_PROFILE else DEFAULT_STATION
        real = TWIN_STATION_PROFILE[twin_station]
        start = at.replace(minute=0, second=0, microsecond=0)

        times = [start + timedelta(hours=h) for h in range(hours)]
        df = self._predict(real, times)
        rows = [{
            "time": t.strftime("%Y-%m-%dT%H:00"),
            "hour": t.hour,
            "expected": round(float(e)),
            "low": round(float(l)),
            "high": round(float(h)),
            "level": self._level(real, float(e)),
        } for t, e, l, h in zip(times, df["expected"], df["low"], df["high"])]

        now = rows[0]
        # quietest service hour in the next six (metro runs ~05:00-23:00)
        window = [r for r in rows[:6] if 5 <= r["hour"] <= 22] or rows[:1]
        quiet = min(window, key=lambda r: r["expected"])
        busy_today = [r for r in rows if r["time"][:10] == now["time"][:10] and 5 <= r["hour"] <= 22]
        peak = max(busy_today, key=lambda r: r["expected"]) if busy_today else None

        return {
            "station": twin_station,
            "profile_station": real,
            "at": start.strftime("%Y-%m-%dT%H:00"),
            "now": now,
            "hours": rows,
            "quietest_soon": quiet,
            "peak_today": peak,
            "event_today": self.events.get(start.strftime("%Y-%m-%d")),
            "event_tomorrow": self.events.get((start + timedelta(days=1)).strftime("%Y-%m-%d")),
            "week": self._week(real, start),
            "model": {
                "name": "Gradient-boosted trees (scikit-learn)",
                "history": self.metrics.get("history"),
                "accuracy_wape": self.metrics.get("model", {}).get("wape"),
                "improvement_vs_baseline": self.metrics.get("improvement_vs_baseline"),
                "range_coverage": self.metrics.get("model", {}).get("range_coverage"),
            },
        }

    # ------------------------------------------------------------------
    # Planner: any real station, any date and hour (passenger app only)
    # ------------------------------------------------------------------
    SERVICE_HOURS = range(5, 24)   # first and last hours with trains

    def station_list(self) -> list[str]:
        return list(self.stations)

    def plan(self, real_station: str, at: datetime) -> dict:
        """Predicted crowd at a real station for one date and hour, plus that
        whole day's profile so the app can show where the hour sits."""
        if real_station not in self.code:
            raise KeyError(f"unknown station: {real_station}")
        day0 = at.replace(hour=0, minute=0, second=0, microsecond=0)
        times = [day0 + timedelta(hours=h) for h in range(24)]
        df = self._predict(real_station, times)
        day = [{
            "hour": t.hour,
            "expected": round(float(e)),
            "low": round(float(l)),
            "high": round(float(h)),
            "level": self._level(real_station, float(e)),
            "service": t.hour in self.SERVICE_HOURS,
        } for t, e, l, h in zip(times, df["expected"], df["low"], df["high"])]

        sel = day[at.hour]
        service = [r for r in day if r["service"]]
        avg = sum(r["expected"] for r in service) / max(len(service), 1)
        nearby = [r for r in service if abs(r["hour"] - at.hour) <= 3]
        quiet = min(nearby, key=lambda r: r["expected"]) if nearby else None
        peak = max(service, key=lambda r: r["expected"]) if service else None
        key = day0.strftime("%Y-%m-%d")
        return {
            "station": real_station,
            "date": key,
            "hour": at.hour,
            "selected": sel,
            "in_service": sel["service"],
            "vs_day_average": (sel["expected"] - avg) / avg if avg else 0.0,
            "quietest_nearby": quiet,
            "peak": peak,
            "day": day,
            "event": self.events.get(key),
            "event_tomorrow": self.events.get((day0 + timedelta(days=1)).strftime("%Y-%m-%d")),
            "event_yesterday": self.events.get((day0 - timedelta(days=1)).strftime("%Y-%m-%d")),
        }

    def _week(self, real: str, start: datetime) -> list[dict]:
        """Next 7 days: daily total and busiest hour, for the outlook strip."""
        day0 = start.replace(hour=0)
        out = []
        for d in range(7):
            day = day0 + timedelta(days=d)
            times = [day + timedelta(hours=h) for h in range(24)]
            df = self._predict(real, times)
            i = int(df["expected"].idxmax())
            peak = float(df["expected"].iloc[i])
            total = float(df["expected"].sum())
            lo, hi = self.day_bands[real]
            key = day.strftime("%Y-%m-%d")
            out.append({
                "date": key,
                "total": round(total),
                "peak_hour": times[i].hour,
                "peak": round(peak),
                "level": "low" if total < lo else "high" if total >= hi else "med",
                "event": self.events.get(key),
            })
        return out


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "train"
    if cmd == "train":
        print(json.dumps(train(), indent=2))
    elif cmd == "demo":
        f = Forecaster()
        print(json.dumps(f.forecast("Central City", datetime(2026, 11, 8, 14)), indent=2)[:3000])
    else:
        sys.exit(__doc__)
