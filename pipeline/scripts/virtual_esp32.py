"""
Virtual ESP32 — the actuation-layer stand-in described in Idea.txt: subscribes
to the decision engine's zone/state topic and does exactly what a real
ESP32-driven LED strip controller would do with it (log/forward the color),
minus the GPIO. Same MQTT contract a physical device would use in Phase-II —
swapping this script for real hardware means no change anywhere upstream.
"""

import argparse
import json

import paho.mqtt.client as mqtt

STATE_FILTER = "metropulse/+/zone/+/state"
STATUS_FILTER = "metropulse/+/system/status"

# what a real LED strip would actually show for each status
LED = {"low": "GREEN", "med": "AMBER", "high": "RED (PULSING)"}


def on_connect(client, userdata, flags, reason_code, properties):
    print(f"[esp32] connected ({reason_code}), subscribing to {STATE_FILTER}", flush=True)
    client.subscribe(STATE_FILTER, qos=1)
    client.subscribe(STATUS_FILTER, qos=1)


def on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode())
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        print(f"[esp32] dropped malformed message on {msg.topic}: {e}", flush=True)
        return

    if msg.topic.endswith("/system/status"):
        state = "ONLINE" if payload.get("online") else "OFFLINE - HOLDING LAST COLOR"
        print(f"[esp32] decision engine {state}", flush=True)
        return

    station_id, zone_id = payload["station_id"], payload["zone_id"]
    # The strip is telling somebody which coach to walk to, so it follows the
    # projected compartment load, not the density of the platform they're
    # already standing on. Falls back to `status` for a v2 publisher.
    status = payload.get("compartment_status", payload["status"])
    led = LED.get(status, "UNKNOWN")
    load = payload.get("projected_load", payload["count"])
    cap = payload.get("compartment_capacity")
    print(f"[esp32] {station_id}/ZONE-{zone_id} LED -> {led}  "
          f"(coach {load}{'/' + str(cap) if cap else ''}, "
          f"platform={payload['count']}, trend={payload['trend']})", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--broker-host", default="localhost")
    ap.add_argument("--broker-port", type=int, default=1883)
    args = ap.parse_args()

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(args.broker_host, args.broker_port)

    try:
        client.loop_forever()
    except KeyboardInterrupt:
        pass
    finally:
        client.disconnect()


if __name__ == "__main__":
    main()
