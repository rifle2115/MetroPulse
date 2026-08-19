"""
Stand-in for the perception stack (preprocessing + MCNN + zone density calc).
Publishes fake per-zone telemetry on metropulse/<station>/zone/<zone>/telemetry
so the decision engine and downstream consumers can be built/tested before
the real crowd-counting pipeline exists. Same topic + payload shape a real
publisher would use, per pipeline/docs/mqtt_contract.md.
"""

import json
import random
import time
from datetime import datetime, timezone

import paho.mqtt.client as mqtt

BROKER_HOST = "localhost"
BROKER_PORT = 1883
STATION_ID = "stn1"
ZONE_IDS = ["A", "B", "C"]
ZONE_CAPACITY = 40
PUBLISH_INTERVAL_S = 2.0

# each zone drifts around its own target instead of jumping randomly every
# tick, so the trend the decision engine computes actually means something
zone_state = {z: random.uniform(4, 20) for z in ZONE_IDS}


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def step_zone(count: float) -> float:
    count += random.uniform(-3, 3.4)
    return max(0, min(ZONE_CAPACITY, count))


def main() -> None:
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.connect(BROKER_HOST, BROKER_PORT)
    client.loop_start()
    print(f"[telemetry] connected to {BROKER_HOST}:{BROKER_PORT}, publishing every {PUBLISH_INTERVAL_S}s")

    try:
        while True:
            for zone_id in ZONE_IDS:
                zone_state[zone_id] = step_zone(zone_state[zone_id])
                payload = {
                    "schema": "metropulse.zone_telemetry.v1",
                    "station_id": STATION_ID,
                    "zone_id": zone_id,
                    "ts": now_iso(),
                    "count": round(zone_state[zone_id]),
                    "capacity": ZONE_CAPACITY,
                    "confidence": round(random.uniform(0.90, 0.99), 2),
                }
                topic = f"metropulse/{STATION_ID}/zone/{zone_id}/telemetry"
                client.publish(topic, json.dumps(payload), qos=0)
                print(f"[telemetry] -> {topic} {payload}")
            time.sleep(PUBLISH_INTERVAL_S)
    except KeyboardInterrupt:
        pass
    finally:
        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    main()
