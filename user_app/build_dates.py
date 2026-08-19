"""Turn the vendored significant-dates CSV into JSON the passenger app can fetch.

The app has to answer "is today a big day in Bangalore, and why" with no
backend of its own — it is a static page a passenger opens — so the list ships
alongside it. Generated rather than hand-copied so it stays traceable to the
one source of truth the rest of the pipeline already uses:

    pipeline/prediction/data/significant_dates.csv
    (github.com/thecont1/namma-metro-ridership-tracker, BSD-0)

which is the same file baseline_engine.py reads for the Control Room's
prediction/baseline. Both ends therefore agree on which dates are significant
and on the event names they carry.

Re-run after updating the CSV:
    python user_app/build_dates.py
"""

import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "pipeline" / "prediction" / "data" / "significant_dates.csv"
OUT = HERE / "data" / "significant_dates.json"


def main() -> None:
    if not SRC.exists():
        raise SystemExit(f"source CSV not found: {SRC}")

    events: dict[str, list[str]] = {}
    with SRC.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            date = (row.get("Date") or "").strip()
            name = (row.get("Event") or "").strip()
            if date and name:
                events.setdefault(date, []).append(name)

    # One date can carry more than one event (the CSV already joins some with
    # "+", but there are also genuinely repeated dates) — collapse to one label
    # per date, which is what the app displays.
    merged = {date: ", ".join(names) for date, names in sorted(events.items())}

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "source": "pipeline/prediction/data/significant_dates.csv",
        "upstream": "github.com/thecont1/namma-metro-ridership-tracker (BSD-0)",
        "count": len(merged),
        "events": merged,
    }, indent=1, ensure_ascii=False), encoding="utf-8")

    first, last = min(merged), max(merged)
    print(f"wrote {OUT} — {len(merged)} dates, {first} to {last}")


if __name__ == "__main__":
    main()
