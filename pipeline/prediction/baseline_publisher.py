"""
Publishes the current expected crowd baseline (baseline_engine.py) to MQTT
on an interval. Independent of the perception pipeline — this can run even
with no camera/video active, since it's purely time/calendar-driven.

It follows the simulated clock, not the wall clock. Both consoles can move
the demo's clock (control/time — see mqtt_contract.md), and this publisher
shares a retained topic with them, so ticking on real time meant a baseline
set to "Republic Day, 09:00" silently reverted to whatever the actual hour
was within a minute of being set. Instead it holds the same offset the Twin
and Control Room hold and reports the level for the moment they're showing.
"""

import argparse
import json
import time
from datetime import datetime, timedelta, timezone

import paho.mqtt.client as mqtt

from baseline_engine import expected_level

# the ridership data behind expected_level() is Bangalore's, so "now" means
# now in IST regardless of how this machine's clock is configured
IST = timezone(timedelta(hours=5, minutes=30))

# seconds between the simulated clock and the real one, moved by control/time
sim_offset_s = 0.0


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def sim_now() -> datetime:
    """The moment the demo currently believes it is, naive IST — the same
    shape expected_level() wants."""
    return (datetime.now(IST) + timedelta(seconds=sim_offset_s)).replace(tzinfo=None)


def on_connect(client, userdata, flags, reason_code, properties):
    client.subscribe(userdata["time_topic"], qos=1)


def on_message(client, userdata, msg):
    """control/time carries a naive timestamp that means IST."""
    global sim_offset_s
    try:
        payload = json.loads(msg.payload.decode())
        target = datetime.fromisoformat(str(payload["target_iso"]))
    except (json.JSONDecodeError, UnicodeDecodeError, KeyError, TypeError, ValueError):
        return
    if target.tzinfo is None:
        target = target.replace(tzinfo=IST)
    sim_offset_s = target.timestamp() - time.time()
    print(f"[prediction] clock moved to {target.isoformat()} (offset {sim_offset_s / 3600:+.1f}h)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--station-id", default="stn1")
    ap.add_argument("--interval", type=float, default=60.0, help="seconds between publishes")
    ap.add_argument("--broker-host", default="localhost")
    ap.add_argument("--broker-port", type=int, default=1883)
    args = ap.parse_args()

    topic = f"metropulse/{args.station_id}/prediction/baseline"
    time_topic = f"metropulse/{args.station_id}/control/time"

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, userdata={"time_topic": time_topic})
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(args.broker_host, args.broker_port)
    client.loop_start()
    print(f"connected to {args.broker_host}:{args.broker_port}, publishing every {args.interval}s", flush=True)

    last_level = None
    try:
        while True:
            dt = sim_now()
            level, reason = expected_level(dt)
            payload = {
                "schema": "metropulse.prediction_baseline.v1",
                "station_id": args.station_id,
                "ts": now_iso(),
                "expected_level": level,
                "reason": reason,
            }
            client.publish(topic, json.dumps(payload), qos=1, retain=True)
            # the reason line repeats verbatim for an hour at a time, so only
            # say something when the answer actually changes
            if level != last_level:
                last_level = level
                print(f"[prediction] {dt:%Y-%m-%d %H:%M} IST -> {level} - {reason}", flush=True)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass
    finally:
        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    main()
