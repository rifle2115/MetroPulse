"""
Decision engine: subscribes to raw zone telemetry (MCNN density) AND the
entry/exit gate's per-zone flow, blends the two into the count that actually
drives the LED color, and publishes the retained state that actuators
(virtual ESP32, real LED strips in Phase-II) and the digital twin dashboard
consume. This is real pipeline logic, not a test stand-in — see
pipeline/docs/mqtt_contract.md for the topic/payload contract it implements.

Why blend two signals instead of trusting density alone: MCNN is a single
model that can misjudge scenes far from its training distribution (measured:
near-zero readings on a real platform video with a visibly present crowd —
see the diagnostic session that led to this file). The entry/exit gate is a
completely independent technique (motion tracking, not a trained model), so
combining them is a small hedge against either one being wrong alone — not a
guarantee of correctness, just not betting everything on one signal.

Thresholds match CFG in MetroPulse 0/index.html so the state this emits
lines up with what the simulator already renders once it's wired to MQTT.

This process also owns the actual product goal: real per-compartment
occupancy. zone_id A/B/C is the same door/compartment a passenger boards
through, so a gate crossing there IS a boarding or alighting event for that
compartment — `compartment_occupancy` accumulates those deltas into a
running tally that survives across stations (this process stays up across
however many videos the Control Room loads in a session; only restarting
this script resets it — a fresh journey for a fresh demo run).
"""

import json
from collections import defaultdict, deque
from datetime import datetime, timezone

import paho.mqtt.client as mqtt

BROKER_HOST = "localhost"
BROKER_PORT = 1883
# One decision engine instance per station (matches the real deployment: one
# edge device per platform), same as STATION_ID in telemetry_publisher_test.py.
STATION_ID = "stn1"
ZONE_IDS = ["A", "B", "C"]
TELEMETRY_FILTER = f"metropulse/{STATION_ID}/zone/+/telemetry"
GATE_TOPIC = f"metropulse/{STATION_ID}/gate/{{zone_id}}/entry_exit"
RESET_FILTER = f"metropulse/{STATION_ID}/control/reset"
TRAIN_FILTER = f"metropulse/{STATION_ID}/control/train"
STATUS_TOPIC = f"metropulse/{STATION_ID}/system/status"

THR_LOW = 12   # count <= THR_LOW        -> low
THR_MED = 26   # THR_LOW < count <= THR_MED -> med, above -> high
TREND_WINDOW = 3     # readings kept per zone for trend detection
TREND_DELTA = 1.5    # minimum average change to call it rising/falling

# What the LEDs actually show. Platform density answers "is this bit of
# platform busy"; a passenger deciding where to stand needs "how full will
# that compartment be once I'm in it" — who is already aboard plus who is
# queued at those doors, against the coach's capacity. Matches CFG's
# compCapacity/compMedFrac/compHighFrac in MetroPulse 0/index.html.
COMP_CAPACITY = 30
COMP_MED = 0.55    # >= 55% of a coach -> med
COMP_HIGH = 0.90   # >= 90% of a coach -> high

# per (station_id, zone_id) rolling history of recent counts, for trend
history = defaultdict(lambda: deque(maxlen=TREND_WINDOW))
# per (station_id, zone_id) latest gate totals, for the flow estimate
gate_state: dict[tuple, dict] = {}
# per (station_id, zone_id) running compartment occupancy — persists across
# video loads/stations, unlike gate_state above (which mirrors each video's
# own from-zero totals)
compartment_occupancy: dict[tuple, int] = {}
# Every platform reading taken during the current dwell, per (station, zone),
# as {frame_index: count}. Keyed by frame rather than appended, because the
# mid-dwell samples are taken after playback and arrive out of order.
dwell_readings: dict[tuple, dict] = {}
# Newest frame_index whose state has been published per (station, zone).
# zone/state is RETAINED, so it is the answer every late subscriber gets to
# "what does this platform look like right now" — and the mid-dwell samples
# arrive last while describing an earlier moment, so publishing them in
# arrival order left that retained answer sitting on a frame from the middle
# of a clip that had already finished. Measured: A:2 B:10 C:26 retained from
# frame 183 when the clip actually ended on A:1 B:14 C:25, which also moved
# zone B across a threshold and lit the wrong colour. Readings are still all
# collected into dwell_readings — the decomposition needs the whole
# trajectory — this only orders what gets published as current.
latest_published: dict[tuple, int] = {}

# A reading has to move by at least this much to count as somebody moving.
# The detector's own count jitters by a person or so between frames, and
# decomposing a trajectory turns every wobble into a boarding or an alighting
# if you let it.
FLOW_DEADBAND = 2


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def classify(count: float) -> str:
    if count <= THR_LOW:
        return "low"
    if count <= THR_MED:
        return "med"
    return "high"


def classify_compartment(load: float) -> str:
    frac = load / COMP_CAPACITY
    if frac >= COMP_HIGH:
        return "high"
    if frac >= COMP_MED:
        return "med"
    return "low"


def trend_for(key) -> str:
    hist = history[key]
    if len(hist) < 2:
        return "steady"
    delta = hist[-1] - hist[0]
    if delta >= TREND_DELTA:
        return "rising"
    if delta <= -TREND_DELTA:
        return "falling"
    return "steady"


# `flow_estimate` was a second, independent opinion on occupancy, back when
# gate totals came from the Control Room's own line-crossing counter. That
# counter is gone and this engine publishes the gate totals itself, so reading
# them back would be blending this process's output into its own input. The
# field stays in the payload at 0 for contract compatibility; a real gate in
# Phase-II can restore the second opinion by publishing on the topic again.
def on_connect(client, userdata, flags, reason_code, properties):
    print(f"[decision] connected ({reason_code}), subscribing to {TELEMETRY_FILTER}, {RESET_FILTER}, {TRAIN_FILTER}")
    client.subscribe(TELEMETRY_FILTER, qos=0)
    client.subscribe(RESET_FILTER, qos=1)
    client.subscribe(TRAIN_FILTER, qos=1)
    client.publish(
        STATUS_TOPIC,
        json.dumps({"schema": "metropulse.system_status.v1", "station_id": STATION_ID, "online": True, "ts": now_iso()}),
        qos=1, retain=True,
    )


def publish_occupancy(client, station_id: str, zone_id: str, occupancy: int) -> None:
    client.publish(
        f"metropulse/{station_id}/compartment/{zone_id}/occupancy",
        json.dumps({
            "schema": "metropulse.compartment_occupancy.v1",
            "station_id": station_id,
            "zone_id": zone_id,
            "ts": now_iso(),
            "occupancy": occupancy,
        }),
        qos=1, retain=True,
    )


def publish_gate(client, station_id: str, zone_id: str, boarded: int, alighted: int) -> None:
    """The boarding and alighting this engine decided on, published on the
    same topic real gate hardware would use in Phase-II. The Control Room no
    longer publishes here — it reports what it sees on the platform, and this
    process decides what that means for the train."""
    key = (station_id, zone_id)
    totals = gate_state.setdefault(key, {"entered_total": 0, "exited_total": 0})
    totals["entered_total"] += boarded
    totals["exited_total"] += alighted
    client.publish(
        GATE_TOPIC.format(zone_id=zone_id),
        json.dumps({
            "schema": "metropulse.gate_entry_exit.v2",
            "station_id": station_id,
            "zone_id": zone_id,
            "ts": now_iso(),
            "entered_total": totals["entered_total"],
            "exited_total": totals["exited_total"],
            "entered_delta": boarded,
            "exited_delta": alighted,
        }),
        qos=0,
    )


def decompose(counts: list) -> tuple[int, int]:
    """Split a platform trajectory into boarding and alighting.

    Two readings can only give net flow: a dwell where 41 board and 20 get off
    reads as a drop of 21, and there is no arithmetic that recovers the two
    numbers from that one difference. Sampled across the dwell instead, the
    shape carries the answer — the crowd falls while people are boarding and
    rises while people are stepping off — so falls sum to boarding and rises
    to alighting. Measured on the project's clip: 41/0 from the endpoints
    alone, 61/20 from six readings, and the net agrees.

    Changes smaller than FLOW_DEADBAND are dropped, or detector jitter would
    be reported as a steady trickle of passengers in both directions."""
    boarded = alighted = 0
    for a, b in zip(counts, counts[1:]):
        d = b - a
        if abs(d) < FLOW_DEADBAND:
            continue
        if d < 0:
            boarded += -d
        else:
            alighted += d
    return boarded, alighted


def handle_train(client, payload: dict) -> None:
    """Where boarding and alighting are decided — by measuring the platform,
    not by tracking people.

    Counting individual crossings needed a tracker holding identities across
    frames, and at the frame rate this pipeline can afford on CPU it could
    not: crossings came out at or near zero and compartment occupancy never
    moved. Detecting people is cheap and reliable; following a specific person
    from frame to frame is neither, so the design stops trying to.

    Instead the Control Room reads the platform several times across a dwell
    and this decomposes the resulting trajectory — see decompose(). It needs
    no tracking, no identities and no frame-rate floor. What it assumes is
    that people who left the platform boarded the train rather than walking
    away, which for a platform during a dwell is reasonable and is stated in
    the Twin's own footer rather than hidden here.

    Boarding is capped by what the coach can hold; the remainder is left
    behind for the next service.
    """
    action = payload.get("action")
    station_id = payload.get("station_id", STATION_ID)

    if action == "arrive":
        for zone_id in ZONE_IDS:
            dwell_readings.pop((station_id, zone_id), None)
            # frame numbering restarts with each clip, so the ordering guard in
            # handle_telemetry has to restart with it or the first readings of
            # the new dwell look like backfill from the old one and are dropped
            latest_published.pop((station_id, zone_id), None)
        print(f"[decision] {station_id}: service arriving — platform readings cleared")

    elif action == "depart":
        parts = []
        for zone_id in ZONE_IDS:
            key = (station_id, zone_id)
            readings = dwell_readings.get(key) or {}
            if len(readings) < 2:
                parts.append(f"{zone_id}:no readings")
                continue

            # chronological, not arrival order — the mid-dwell samples are
            # taken after playback and land last
            counts = [readings[i] for i in sorted(readings)]
            boarded, alighted = decompose(counts)

            aboard = compartment_occupancy.get(key, 0)
            alighted = min(alighted, aboard + boarded)
            free = max(0, COMP_CAPACITY - aboard)
            left = max(0, boarded - free)
            boarded = min(boarded, free)

            compartment_occupancy[key] = max(0, aboard + boarded - alighted)
            publish_occupancy(client, station_id, zone_id, compartment_occupancy[key])
            publish_gate(client, station_id, zone_id, boarded, alighted)
            dwell_readings.pop(key, None)
            latest_published.pop(key, None)
            history[key].clear()
            # the crowd this dwell measured is now aboard — see the docstring
            publish_empty_platform(client, station_id, zone_id, compartment_occupancy[key])

            parts.append(f"{zone_id}:[{','.join(str(c) for c in counts)}] "
                         f"+{boarded}/-{alighted}" + (f" ({left} left)" if left else ""))
        print(f"[decision] {station_id}: doors closed — " + "  ".join(parts))


def publish_zone_state(client, station_id: str, zone_id: str, state: dict) -> None:
    client.publish(f"metropulse/{station_id}/zone/{zone_id}/state",
                   json.dumps(state), qos=1, retain=True)


def publish_empty_platform(client, station_id: str, zone_id: str, aboard: int) -> None:
    """Retain a zone/state that says "nobody is queued here".

    zone/state is retained, so whatever was last published is what every future
    subscriber is told the platform looks like. After a departure the last
    reading is a crowd that has, by this engine's own reckoning, just got on
    the train — so leaving it retained makes both consoles add the same people
    twice: `projected_load` is occupancy + queue, and the queue is where the
    occupancy came from. The Control Room panel showed COACH 3 at 48/30 for
    this reason, a load above the hard capacity. handle_reset() already zeroes
    the platform for exactly this argument; a departure is the other moment it
    applies. Anyone genuinely still standing there is measured again by the
    next clip's own opening reading."""
    publish_zone_state(client, station_id, zone_id, {
        "schema": "metropulse.zone_state.v3", "station_id": station_id, "zone_id": zone_id,
        "ts": now_iso(), "status": "low", "trend": "steady", "count": 0, "raw_count": 0,
        "flow_estimate": 0, "capacity": None, "confidence": 1.0,
        "compartment_occupancy": aboard, "projected_load": aboard,
        "compartment_capacity": COMP_CAPACITY,
        "compartment_status": classify_compartment(aboard),
    })


def handle_reset(client, payload: dict) -> None:
    """Zeroes compartment_occupancy for every zone at this station — the
    only reset mechanism for that running tally (see compartment/occupancy
    in mqtt_contract.md). Publishes occupancy:0 for all three zones
    explicitly, not just ones that happen to be in compartment_occupancy
    already, so a zone that never reported activity still ends up retained
    at a known 0 rather than staying unset."""
    station_id = payload["station_id"]
    for zone_id in ZONE_IDS:
        compartment_occupancy[(station_id, zone_id)] = 0
        dwell_readings.pop((station_id, zone_id), None)
        gate_state[(station_id, zone_id)] = {"entered_total": 0, "exited_total": 0}
        occ_payload = {
            "schema": "metropulse.compartment_occupancy.v1",
            "station_id": station_id,
            "zone_id": zone_id,
            "ts": now_iso(),
            "occupancy": 0,
        }
        client.publish(f"metropulse/{station_id}/compartment/{zone_id}/occupancy", json.dumps(occ_payload), qos=1, retain=True)

        # zone/state is retained too, so leaving it alone meant a console that
        # connected after a reset resurrected the last clip's crowd — zeroed
        # coaches sat next to a platform full of people who had long gone
        publish_empty_platform(client, station_id, zone_id, 0)
        latest_published.pop((station_id, zone_id), None)
        history[(station_id, zone_id)].clear()
    print(f"[decision] {station_id}: occupancy AND platform state reset to 0 for {ZONE_IDS}")


def handle_telemetry(client, telemetry: dict) -> None:
    station_id = telemetry["station_id"]
    zone_id = telemetry["zone_id"]
    raw_count = telemetry["count"]
    key = (station_id, zone_id)
    frame_index = int(telemetry.get("frame_index", 0))

    # every reading of this dwell is kept, filed under the frame it came from
    dwell_readings.setdefault(key, {})[frame_index] = int(round(raw_count))

    # A backfilled mid-dwell sample describes an earlier moment than what is
    # already published, so it must not overwrite the retained "now" — nor
    # feed the trend, which is meant to read the direction the crowd is
    # actually moving, not the order frames happened to be analysed in.
    if frame_index < latest_published.get(key, -1):
        print(f"[decision] {station_id}/{zone_id}: backfill frame {frame_index} "
              f"(platform={raw_count}) recorded, not published", flush=True)
        return
    latest_published[key] = frame_index

    history[key].append(raw_count)
    trend = trend_for(key)

    blended = raw_count
    status = classify(blended)

    # the guidance signal: this platform's queue for that coach, on top of
    # what the coach is already carrying
    aboard = compartment_occupancy.get(key, 0)
    projected = aboard + blended
    compartment_status = classify_compartment(projected)

    state = {
        "schema": "metropulse.zone_state.v3",
        "station_id": station_id,
        "zone_id": zone_id,
        "ts": now_iso(),
        "status": status,
        "trend": trend,
        "count": round(blended),
        "raw_count": raw_count,
        "flow_estimate": 0,
        "capacity": telemetry.get("capacity"),
        "confidence": telemetry.get("confidence"),
        # v3 additions — see mqtt_contract.md
        "compartment_occupancy": aboard,
        "projected_load": round(projected),
        "compartment_capacity": COMP_CAPACITY,
        "compartment_status": compartment_status,
    }
    state_topic = f"metropulse/{station_id}/zone/{zone_id}/state"
    client.publish(state_topic, json.dumps(state), qos=1, retain=True)
    print(f"[decision] {station_id}/{zone_id}: platform={raw_count} ({status}, {trend}) | "
          f"aboard={aboard} -> coach {projected:.0f}/{COMP_CAPACITY} {compartment_status}")


def on_disconnect(client, userdata, disconnect_flags, reason_code, properties):
    print(f"[decision] disconnected ({reason_code}) — retrying")


def on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode())
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        print(f"[decision] dropped malformed message on {msg.topic}: {e}")
        return

    if msg.topic.endswith("/control/train"):
        handle_train(client, payload)
    elif msg.topic.endswith("/control/reset"):
        handle_reset(client, payload)
    else:
        handle_telemetry(client, payload)


def main() -> None:
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.will_set(
        STATUS_TOPIC,
        json.dumps({"schema": "metropulse.system_status.v1", "station_id": STATION_ID, "online": False, "ts": now_iso()}),
        qos=1, retain=True,
    )
    client.on_connect = on_connect
    client.on_message = on_message
    client.on_disconnect = on_disconnect
    # connect_async + loop_forever, not connect(): a blocking connect raises
    # ConnectionRefusedError and kills this process outright if mosquitto isn't
    # up yet, so starting the pipeline in the wrong order (or restarting the
    # broker) meant silently losing the engine — every subsequent zone reading
    # arriving at a broker with nothing listening. This way paho owns the retry
    # loop and the engine reconnects and re-subscribes on its own (on_connect
    # runs again on every reconnect, which is why the subscribes live there).
    client.connect_async(BROKER_HOST, BROKER_PORT)
    print(f"[decision] connecting to {BROKER_HOST}:{BROKER_PORT} (retries until it comes up)")

    try:
        client.loop_forever()
    except KeyboardInterrupt:
        pass
    finally:
        client.disconnect()


if __name__ == "__main__":
    main()
