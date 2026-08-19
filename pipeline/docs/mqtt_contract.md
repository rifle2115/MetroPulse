# MetroPulse MQTT Contract v1

This defines the message contract between pipeline stages, so every stage
(perception, decision engine, actuators, dashboard) can be built and tested
independently against the same interface. This is the "exact same message
contracts" referenced in Idea.txt — nothing here should change when the
simulated components are swapped for real cameras/edge hardware in Phase-II.

Broker: Mosquitto, `localhost:1883`, no auth (local dev only — see
`pipeline/mqtt/mosquitto.conf`).

## Topic tree

```
metropulse/<station_id>/zone/<zone_id>/telemetry   — perception stack output (MCNN + zone split)
metropulse/<station_id>/zone/<zone_id>/state        — decision engine output (color state, density+flow blended)
metropulse/<station_id>/system/status               — pipeline heartbeat / LWT
metropulse/<station_id>/gate/<zone_id>/entry_exit   — directional entry/exit counts, per zone (Control Room)
metropulse/<station_id>/compartment/<zone_id>/occupancy — running real occupancy per compartment (the actual product)
metropulse/<station_id>/prediction/baseline         — expected crowd baseline (time/calendar heuristic)
metropulse/<station_id>/control/train               — Control Room -> Twin: announce/depart commands
metropulse/<station_id>/control/time                — Control Room -> Twin: set displayed clock to date/hour
metropulse/<station_id>/control/reset               — Control Room -> decision engine: zero compartment occupancy
metropulse/<station_id>/twin/state                  — Twin -> passenger app: current station + per-coach load
```

- `station_id` — e.g. `stn1` (matches the 5-station line already modeled in
  `MetroPulse 0/index.html` — `STATIONS[i].id`).
- `zone_id` — `A` | `B` | `C` (matches `ZONES` in the simulator).

## Zone = compartment

A zone is one stretch of platform **and the compartment whose doors open onto
it** — the same thing from a passenger's point of view, and the MCNN's
left/middle/right thirds map onto them one-to-one. This matters because the
product is not "how dense is this bit of platform", it is "which compartment
should I board". Two numbers combine into that answer:

- `zone/<id>/telemetry` → `zone/<id>/state.count` — how many people are
  queued at that compartment's doors right now.
- `compartment/<id>/occupancy` — how many are already inside it, carried
  over from previous stations.

Their sum, against a coach's capacity, is the **projected load** — what that
compartment will hold once these doors close. That is what the platform LEDs
show (`compartment_status`), what the Twin's strips and floating boards
render, and what the "board via coach N" recommendation ranks on. Platform
density alone is still published and still inspectable, it just doesn't drive
the lights.

## `zone/<zone_id>/telemetry`

Published by the perception layer (today: `telemetry_publisher_test.py`
stand-in; later: preprocessing + MCNN + zone density calc). Not retained —
this is a live stream, a new subscriber shouldn't act on a stale count.

- QoS: 0
- Retained: false

```json
{
  "schema": "metropulse.zone_telemetry.v1",
  "station_id": "stn1",
  "zone_id": "A",
  "ts": "2026-08-10T12:34:56.123Z",
  "count": 14,
  "capacity": 40,
  "confidence": 0.97
}
```

- `count` — estimated head count in the zone (MCNN output, post zone-split).
- `capacity` — nominal zone capacity used for thresholding (matches
  `CFG.zoneCapacity` in the simulator, currently `40`).
- `confidence` — 0..1, stability-based confidence of the reading.

## `zone/<zone_id>/state`

Published by the decision engine (`decision_engine.py`) after applying
thresholds + trend detection to the telemetry stream. This is what actuators
(virtual ESP32 today, real LED strips in Phase-II) and the digital twin
dashboard subscribe to — never telemetry directly, so the actuation logic
lives in exactly one place.

The LED-driving `status` is not raw density alone: the decision engine also
tracks each zone's `gate/<zone_id>/entry_exit` flow (entered − exited, an
independent technique) and takes the max of the two as `count` before
classifying — see `blend_count()` in `decision_engine.py`. Max, not an
average: `flow_estimate` assumes the zone started empty, so it under-reads
real footage that already has people in frame from the start — averaging
that in would drag a correct high density reading down, which is the wrong
direction for a crowd-safety signal. The raw MCNN reading and the flow-only
estimate are both still included so the blend is inspectable, not a black
box.

- QoS: 1
- Retained: **true** — a subscriber that connects mid-run (e.g. the
  dashboard on page load) must immediately get the last known state instead
  of waiting for the next telemetry tick.

```json
{
  "schema": "metropulse.zone_state.v3",
  "station_id": "stn1",
  "zone_id": "A",
  "ts": "2026-08-10T12:34:57.500Z",
  "status": "med",
  "trend": "rising",
  "count": 14,
  "raw_count": 11,
  "flow_estimate": 17,
  "capacity": 40,
  "confidence": 0.97,
  "compartment_occupancy": 8,
  "projected_load": 22,
  "compartment_capacity": 30,
  "compartment_status": "med"
}
```

- `status` — `low` | `med` | `high` for **platform density**. Thresholds:
  `count <= 12` → `low`, `13..26` → `med`, `>= 27` → `high`, applied to the
  blended `count`, not `raw_count`. Retained for inspection; it is not what
  the LEDs show.
- `trend` — `rising` | `falling` | `steady`, derived from the last few
  telemetry readings for that zone (pre-blend).
- `count` — the blended estimate actually used for `status`:
  `max(raw_count, flow_estimate)` — see `blend_count()`.
- `raw_count` — the MCNN telemetry reading, unblended.
- `flow_estimate` — `max(0, entered_total - exited_total)` for this zone
  from the entry/exit gate — an independent estimate of current occupancy,
  assuming the zone started empty when the Control Room began processing.

### v3 — the compartment fields

v2 published only platform density, which left every consumer to invent its
own compartment rule (or, in the Twin's case, to render a simulation beside
the real answer instead of from it). v3 makes the decision once, here, so the
virtual ESP32, the Control Room and the Twin all light the same colour.

- `compartment_occupancy` — people already in this compartment, i.e. the
  latest value this engine published on `compartment/<zone_id>/occupancy`.
- `projected_load` — `compartment_occupancy + count`, rounded. How full the
  coach is heading for once these doors close.
- `compartment_capacity` — nominal per-coach capacity (30, matching
  `CFG.compCapacity` in the simulator).
- `compartment_status` — `low` | `med` | `high` from `projected_load /
  compartment_capacity`: `>= 0.90` → `high`, `>= 0.55` → `med`, else `low`.
  **This is the LED.**

v2 consumers keep working — every v2 field is unchanged and still present.
`virtual_esp32.py` falls back to `status` when `compartment_status` is
absent, so a v2 publisher and a v3 one can be mixed during a rollout.

## `system/status`

Heartbeat from the decision engine, published on connect and as an MQTT
Last Will and Testament so downstream consumers know if the pipeline drops.

- QoS: 1
- Retained: true

```json
{ "schema": "metropulse.system_status.v1", "station_id": "stn1", "online": true, "ts": "2026-08-10T12:30:00.000Z" }
```

`online: false` is the LWT payload the broker publishes automatically if the
decision engine disconnects uncleanly.

## `gate/<zone_id>/entry_exit`

Published by **`decision_engine.py`**, not the Control Room. It carries the
boarding and alighting that engine decided for a dwell, and it is decided by
measuring the platform rather than by following individuals.

Counting people across a virtual line needs a tracker that holds identities
frame to frame, and at the frame rate this pipeline can afford on CPU it
cannot: crossings came out at or near zero and compartment occupancy never
moved. Detecting people is cheap and reliable; following a named person is
neither. So the Control Room reads each zone several times across the dwell
(`DWELL_SAMPLES`, currently 6) and this engine splits the resulting
trajectory — see `decompose()`:

    every fall in the crowd  ->  boarding    (entered_delta)
    every rise in the crowd  ->  alighting   (exited_delta)

Two readings would only measure NET flow, and that is not a detail: on the
project's own clip the endpoints alone give 41 boarded and **zero** alighted,
because 20 people getting off is cancelled by 61 getting on. The same clip
sampled six times gives 61/20, and the net still agrees. Changes below
`FLOW_DEADBAND` are ignored, or the detector's own jitter would be reported as
a trickle of passengers in both directions.

Readings carry `frame_index` because the mid-dwell samples are taken after
playback has finished and therefore arrive out of order; the engine sorts on
it before decomposing.

Boarding is capped by the coach's free space; the remainder is left behind for
the next service. The assumption it rests on — that people who left the
platform boarded the train rather than walking away — holds for a platform
during a dwell and is stated in the Twin's own footer rather than hidden here.

Both deltas fire once, at `control/train` `depart`. Per-zone (not a single station-wide total) specifically so it's
directly comparable to that zone's density reading — the earlier v1 global
total didn't correspond to any single zone, which made it impossible to
meaningfully combine with `zone/telemetry`. Not retained — an event stream,
like telemetry.

- QoS: 0
- Retained: false

```json
{
  "schema": "metropulse.gate_entry_exit.v2",
  "station_id": "stn1",
  "zone_id": "A",
  "ts": "2026-08-11T09:00:00.000Z",
  "entered_total": 12,
  "exited_total": 9,
  "entered_delta": 1,
  "exited_delta": 0
}
```

- `entered_total` / `exited_total` — running totals for this zone since the
  Control Room started processing the current source.
- `entered_delta` / `exited_delta` — crossings since the previous publish
  (usually 0 or 1 per tick, occasionally more if several people cross close
  together).

## `compartment/<zone_id>/occupancy`

**This is the actual product.** Platform zone density (`zone/telemetry`,
`zone/state`) tells you how crowded the *platform* is — useful for LED
guidance, but not what a passenger needs to pick a less-crowded compartment
to board. This topic answers that: how many people are on the train, in
this specific compartment, right now.

Published by `decision_engine.py`, derived from `gate/<zone_id>/entry_exit`
— a crossing at a zone's door **is** a boarding or alighting event for the
compartment behind that door (`entered_delta` = boarded, `exited_delta` =
alighted), so `handle_gate()` accumulates those deltas into a running
per-zone tally rather than deriving anything from `zone/telemetry`. Unlike
`gate_state` (which mirrors each video's own from-zero totals), this
persists across however many videos the Control Room loads in a session —
it's meant to represent the compartment's real occupancy as the train moves
from station to station, not a single station's activity. `control/reset`
(below) is the reset, for a fresh demo journey without restarting the
process.

- QoS: 1
- Retained: true

```json
{ "schema": "metropulse.compartment_occupancy.v1", "station_id": "stn1", "zone_id": "A", "ts": "2026-08-11T09:00:00.000Z", "occupancy": 18 }
```

- `occupancy` — running count of people currently in this compartment,
  floored at 0. No upper clamp — a real compartment can exceed whatever
  nominal capacity a simulator assigns it.

## `control/train`

**One clip = one station's dwell.** The Control Room bookends every video it
processes on this topic, and the Twin's service follows the operator rather
than its own timer.

- `arrive` — fires when **playback** starts. The Twin starts the run if it is
  idle, brings the train in, and then **holds** at the platform with the doors
  open. Note this is playback, not load: the Control Room watches a clip in
  full before playing it (see below), and during that analysis phase no crowd
  data exists yet, so there is nothing for the train to arrive to.
- `announce` — fires when the clip ends: banner only.
- `depart` — fires 10s after `announce`. The Twin boards the waiting crowd,
  closes up, and runs to the next stop, where it docks and holds again for
  that station's clip.

"The clip ends" means either of two things, and they are treated identically:
the source ran out, or the operator pressed STOP during playback. Cutting a
clip short is the same decision as letting it finish — *this station is done,
send the train* — and the handoff uses whatever the pipeline has read so far,
which is already current in the Twin because `zone/state` and
`compartment/occupancy` publish continuously. Previously only a natural end
published anything, so pressing STOP left the train holding indefinitely.

Cancelling during the *analysis* phase is not that. Nothing has been played,
no `arrive` was sent, and the train has nothing to depart with — so it
publishes nothing and the Twin stays where it is.

The one stop that does **not** hand off is being superseded: loading a new
clip also halts the current one, but there the incoming clip owns this
station, and departing would pull the train out from under it.
`server.py` keeps a separate `superseded_event` for exactly this distinction,
and the post-`announce` delay waits on that event rather than on the stop
flag — after an operator stop the stop flag is already set, so waiting on it
would return instantly and skip the `depart` entirely.

While an MQTT session is live the Twin disables its dwell timer entirely
(`pipeline.active` in `MetroPulse 0/index.html`). Without `arrive` there was
nothing telling it a station's crowd was now being fed to it, so from the
second stop onward it fell back to that timer and pulled out roughly six
seconds after docking — long before the next clip had even been loaded.

`depart` is tolerant about ordering: a clip short enough to finish while the
train is still sliding in sets `pipeline.pendingDepart`, and the command
fires once the train docks instead of being dropped.

- QoS: 1
- Retained: false — one-shot commands, not state to sync to.

```json
{ "schema": "metropulse.control_train.v1", "station_id": "stn1", "ts": "2026-08-11T09:00:00.000Z", "action": "arrive", "source_name": "platform_a.mp4" }
{ "schema": "metropulse.control_train.v1", "station_id": "stn1", "ts": "2026-08-11T09:00:30.000Z", "action": "announce", "message": "STARTING TRAIN" }
{ "schema": "metropulse.control_train.v1", "station_id": "stn1", "ts": "2026-08-11T09:00:40.000Z", "action": "depart" }
```

- `action` — `arrive` | `announce` | `depart`, as above.
- `source_name` — present on `arrive`, the clip's filename, for the log line.
- `message` — present on `announce`, human-readable text for the banner.

## `control/time`

The simulated clock, and the one piece of state **either** console may set.
A demo shows "it's Republic Day, 9am" rather than whatever the wall clock
says, and both the Control Room (CHANGE TIME → `POST /api/change-time`) and
the Twin (SET CLOCK, in its operations console) publish here. Every consumer
stores the offset between the target and the moment it received the message,
then keeps ticking forward from it — a clock *jump*, not a freeze. Retained,
so a console that connects afterwards lands on the same simulated moment.

Subscribers:

- **Twin** — moves its CLOCK and stamp, and fills its date/hour inputs.
- **Control Room** (`server.py`) — mirrors the offset into `/api/stats` as
  `sim_offset_ms` so its own clock display follows a Twin-set time, and, when
  the message did **not** come from itself, answers with a fresh
  `prediction/baseline` for that date/hour. The Twin can't run the BMRCL
  model itself, so without that reply its clock and its PREDICTED readout
  would describe two different moments.
- **`baseline_publisher.py`** — holds the same offset and publishes the
  baseline for the *simulated* moment. It previously ran on wall-clock time
  against this same retained topic, which silently reverted any manually set
  baseline within 60 seconds of setting it.

- QoS: 1
- Retained: true

```json
{ "schema": "metropulse.control_time.v1", "station_id": "stn1", "ts": "2026-08-11T09:00:00.000Z", "target_iso": "2026-01-26T09:00:00", "origin": "control_room" }
```

- `target_iso` — the moment the clock should read as of `ts`. Naive, no
  timezone suffix, and it means **IST** — every consumer attaches
  `+05:30` explicitly rather than inheriting whatever timezone the host or
  browser happens to be set to.
- `origin` — `control_room` | `twin`, whoever set it. Only used to suppress
  echo handling: the Control Room skips republishing a baseline for its own
  message (it already sent one), and the Twin ignores its own (already
  applied locally before publishing). A message with no `origin` is treated
  as external by both, which is the safe default.

## `control/reset`

Published by the Control Room's RESET OCCUPANCY button (`POST
/api/reset-occupancy`). `compartment/<zone_id>/occupancy` has no other reset
mechanism by design (see that topic's docs) — this is it. `decision_engine.py`
subscribes, zeroes its in-memory `compartment_occupancy` dict for every zone,
and republishes `occupancy: 0` (retained) for each so every subscriber —
Twin, Control Room, anyone reconnecting later — converges on zero too, not
just the sender. Not retained itself — a one-shot command, not state.

- QoS: 1
- Retained: false

```json
{ "schema": "metropulse.control_reset.v1", "station_id": "stn1", "ts": "2026-08-11T09:00:00.000Z" }
```

## `prediction/baseline`

Published two ways: automatically by `prediction/baseline_publisher.py`
every 60s **for the simulated clock** (see `control/time` — not the wall
clock, or it would overwrite a manually set baseline a minute later), or
on-demand by the Control Room whenever `control/time` moves, whether that
came from its own CHANGE TIME action (`POST /api/change-time`) or from the
Twin. Both routes go through `publish_baseline()` in `server.py` so they
emit an identical message rather than two that could drift. An expected
crowd level independent
of what any camera currently sees, meant to sit alongside the live
`zone/state` reading so an operator can compare predicted vs. actual, not
replace it. Backed by real Bangalore Namma Metro (BMRCL) data, not a
placeholder — see `prediction/baseline_engine.py`'s
header for the data sources and how the low/med/high buckets are derived.
Since both publish to the same retained topic, whichever fires most recently
wins — expected when manually testing a specific date/hour.

- QoS: 1
- Retained: true

```json
{
  "schema": "metropulse.prediction_baseline.v2",
  "station_id": "stn1",
  "ts": "2026-08-11T09:00:00.000Z",
  "for_iso": "2026-01-26T09:00:00",
  "expected_level": "high",
  "reason": "Republic Day (Bangalore Metro significant date)"
}
```

- `expected_level` — `low` | `med` | `high`, same vocabulary as `zone/state`
  so the Twin can render both with the same color convention.
- `reason` — short human-readable explanation (which rule fired), for
  display and for debugging why a baseline is what it is.
- `for_iso` (v2) — **the simulated moment this forecast is for**, IST, naive.
  A forecast is only true of one moment, and this topic is retained, so
  without it a subscriber cannot tell a current baseline from one computed
  for a completely different day. The passenger app reported "Republic Day"
  at 14:00 on an ordinary Tuesday for exactly this reason. Consumers must
  ignore a baseline whose `for_iso` is not the moment they are asking about;
  a v1 message (no `for_iso`) should be treated as unusable rather than as a
  forecast for now.

## `twin/state`

Published by the Digital Twin (`MetroPulse 0/index.html`), consumed by the
passenger app (`user_app/`). The Twin's own view of the platform it is
currently rendering: which station the service is at, what each coach is
carrying, and which coach the LEDs are recommending.

This exists because the rest of the tree only carries that answer in **live**
mode. Offline — a sent batch, the sliders, the rush scenario — the Twin owns
`boarded` outright and the decision engine never sees any of it, so an app
subscribed to `compartment/occupancy` would sit at zero through a whole demo
run. This topic is the Twin's view in either mode, which is what a passenger
app is meant to mirror. It is strictly Twin -> app; the app never publishes,
so nothing a passenger does can perturb a run an operator is driving.

- QoS: 1
- Retained: **true** — an app opened mid-run is current immediately rather
  than blank until the next change.

```json
{
  "schema": "metropulse.twin_state.v1",
  "station_id": "stn1",
  "ts": "2026-08-19T08:30:00.000Z",
  "sim_iso": "2026-01-26T09:00:00",
  "run_state": "RUNNING",
  "train_state": "DOCKED",
  "live": true,
  "station": { "index": 1, "id": 2, "name": "University Road", "count": 5, "terminus": false },
  "next_station": "Lakeside",
  "line": ["Central City", "University Road", "Lakeside", "North Gate", "Airport Terminal"],
  "compartment_capacity": 30,
  "coaches": [
    { "id": "A", "name": "COACH 1", "onboard": 17, "queue": 0, "load": 17, "status": "med", "recommended": false },
    { "id": "B", "name": "COACH 2", "onboard": 15, "queue": 0, "load": 15, "status": "low", "recommended": false },
    { "id": "C", "name": "COACH 3", "onboard": 14, "queue": 0, "load": 14, "status": "low", "recommended": true }
  ],
  "best_coach": 2,
  "waiting_total": 0,
  "onboard_total": 46
}
```

- `sim_iso` — the Twin's simulated clock, IST. Rides along with the state so
  the app's crowd forecast tracks an operator jumping the date without a
  separate round trip through `control/time`.
- `coaches[].onboard` / `queue` / `load` — already aboard, queued at those
  doors on this platform, and their sum. `load` against
  `compartment_capacity` is the projected load the platform LED is showing.
- `coaches[].recommended` / `best_coach` — the emptiest coach, i.e. the
  redistribution advice itself. Duplicated per coach so a renderer doesn't
  have to cross-reference an index.
- `ts` — real wall-clock publish time, so a consumer can tell a live Twin
  from a retained message left behind by one that has since closed. Republished
  on change and on a 5s heartbeat.

## Versioning

Every payload carries a `schema` field (`<name>.v1`). Breaking changes bump
the version and get a new topic segment or field rather than silently
changing the meaning of an existing field in place.
