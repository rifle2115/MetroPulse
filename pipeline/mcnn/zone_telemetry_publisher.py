"""
Real perception-stack publisher — replaces telemetry_publisher_test.py.
Runs the trained MCNN on a live camera / video / still image, splits the
predicted density map into the same three left-to-right platform zones
(A/B/C) the digital twin simulator already uses, and publishes per-zone
counts over MQTT using the exact same topic/payload contract, so
decision_engine.py and the dashboard need zero changes to consume it.

Source can be:
  - a webcam index, e.g. --source 0
  - a still image path (re-inferred every tick, simulating a static camera)
  - a video file path (loops back to the start at EOF)

Thin CLI wrapper around inference.py — control_room/server.py uses the same
underlying functions for the web-based version of this.
"""

import argparse
import json
import time

import torch
import paho.mqtt.client as mqtt

from inference import ZONE_CAPACITY, ZONE_IDS, frame_to_tensor, load_model, make_frame_source, now_iso, zone_counts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0", help="webcam index, image path, or video path")
    ap.add_argument("--checkpoint", default="checkpoints/best.pt")
    ap.add_argument("--station-id", default="stn1")
    ap.add_argument("--interval", type=float, default=2.0, help="seconds between inferences")
    ap.add_argument("--broker-host", default="localhost")
    ap.add_argument("--broker-port", type=int, default=1883)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)
    print(f"loaded {args.checkpoint} on {device}", flush=True)

    get_frame = make_frame_source(args.source)

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.connect(args.broker_host, args.broker_port)
    client.loop_start()
    print(f"connected to {args.broker_host}:{args.broker_port}, publishing every {args.interval}s", flush=True)

    try:
        while True:
            frame = get_frame()
            if frame is None:
                print("frame source exhausted", flush=True)
                break

            with torch.no_grad():
                img = frame_to_tensor(frame).to(device)
                density = model(img)[0, 0].cpu().numpy()

            counts = zone_counts(density)
            for zone_id, count in zip(ZONE_IDS, counts):
                payload = {
                    "schema": "metropulse.zone_telemetry.v1",
                    "station_id": args.station_id,
                    "zone_id": zone_id,
                    "ts": now_iso(),
                    "count": round(count),
                    "capacity": ZONE_CAPACITY,
                    # MCNN has no native uncertainty estimate — this is a
                    # fixed placeholder, not a modeled confidence value.
                    # A real one would need e.g. MC-dropout or an ensemble.
                    "confidence": 1.0,
                }
                topic = f"metropulse/{args.station_id}/zone/{zone_id}/telemetry"
                client.publish(topic, json.dumps(payload), qos=0)
            print(f"[mcnn] counts A={counts[0]:.1f} B={counts[1]:.1f} C={counts[2]:.1f}", flush=True)

            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass
    finally:
        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    main()
