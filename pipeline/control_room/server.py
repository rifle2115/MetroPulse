"""
Control Room backend — the operator app: load a video, it plays once (not
looped) at true source speed, while a person detector counts who is in each
zone and MCNN estimates crowd density over the same frames, both published to
MQTT (same contract zone_telemetry_publisher.py uses) so the Twin updates
live.

One clip = one station's dwell. This bookends every clip on
metropulse/<station>/control/train so the Twin's service follows the
operator rather than its own timer: "arrive" fires the moment processing
starts (the Twin brings the train in and HOLDS at the platform), then when
the clip ends "announce" fires immediately and "depart" 10s later. Without
the opening "arrive" the Twin had nothing telling it a station's crowd was
now being fed to it, so from the second stop on it just ran its own dwell
timer and pulled out before the clip had been loaded. See
pipeline/docs/mqtt_contract.md.

Neither model keeps up with 30fps on this CPU (tiled detection ~1s a pass,
MCNN ~206ms), so neither runs during playback at all.

A clip is read exactly twice — its first frame and its last — and played once
in between at true speed. Boarding is measured as the change in platform crowd
across the dwell (see handle_train in decision_engine.py), which needs those
two readings and nothing else; analysing the frames between them produced
numbers no part of the system consumed, while taking enough CPU to drag
playback below real time. See process_video.
"""

import base64
import json
import os
import re
import shutil
import sys
import tempfile
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
import paho.mqtt.client as mqtt
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "mcnn"))
from camera_calibration import CalibrationError, CalibrationStore, CameraCalibration  # noqa: E402
from person_counter import PersonCounter  # noqa: E402
from train_motion import TrainMotionDetector  # noqa: E402
from door_state import DoorCropper, DoorStateDetector, DoorStateModel  # noqa: E402
from inference import (  # noqa: E402
    MAX_FRAME_DIM, ZONE_CAPACITY, ZONE_IDS, frame_to_tensor, load_model, make_frame_source,
    now_iso, resize_max, video_fps, zone_counts,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "prediction"))
from baseline_engine import expected_level  # noqa: E402

STATIC_DIR = Path(__file__).parent / "static"
UPLOAD_DIR = Path(__file__).parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
# Per-camera platform calibration (see camera_calibration.py), marked once per
# camera on /calibrate. The reference frame each one was marked on is kept
# beside it so the marking can be reopened and checked later.
CAMERAS_FILE = Path(__file__).parent / "cameras.json"
CAMERA_FRAMES_DIR = Path(__file__).parent / "camera_frames"
CAMERA_FRAMES_DIR.mkdir(exist_ok=True)
cameras = CalibrationStore(CAMERAS_FILE)
# Every reading a clip gets — the raw frame, the annotated frame and the
# numbers — kept per run, so a count can be checked by eye afterwards instead
# of trusted. One folder per clip load: reading_logs/<when>_<clip>/.
READING_LOG_DIR = Path(__file__).parent / "reading_logs"
READING_LOG_DIR.mkdir(exist_ok=True)
_LOG_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]{1,120}$")

CHECKPOINT = Path(__file__).resolve().parent.parent / "mcnn" / "checkpoints" / "best.pt"
BROKER_HOST, BROKER_PORT = "localhost", 1883
STATION_ID = "stn1"
MCNN_EVERY_S = 2.0
# Readings taken across one dwell. Two (the ends) measure only NET flow, so a
# stop where 41 board and 20 get off reports 41 boarded and zero alighted —
# the two cancel and one of them vanishes. Sampling the middle as well lets
# the decision engine split the trajectory: every fall is boarding, every rise
# is alighting. Measured on the project's clip, 2 readings gave 41/0 and 6
# gave 61/20. The extra four are taken during the announce window, while the
# Twin is already showing STARTING TRAIN, so they cost no visible time.
DWELL_SAMPLES = 6
# Preview frame rate. Served as an MJPEG stream (see /api/stream.mjpg), so
# this is what the operator actually sees. Matched to the source rate so the
# preview period does not beat against the frame period and drop every other
# frame — at 25 against a 30fps source that measured 15fps in the browser.
PREVIEW_FPS = 30.0
# Detection sees the source at full resolution; everything else works from a
# downsized copy. 1920 is a cap, not an upscale — smaller sources pass through.
SOURCE_MAX_DIM = 1920
PREVIEW_MAX_DIM = 960  # what the browser panel actually needs
# MJPEG framing. Built from chr() rather than typed as escapes so the
# multipart separator survives being edited by tooling that rewrites
# backslashes.
CRLF = ("\r\n").encode()
MJPEG_BOUNDARY = ("--frame\r\nContent-Type: image/jpeg\r\n\r\n").encode()

PLAYBACK_SPEED = 1.0   # true source speed — 30fps footage plays at 30fps
TRAIN_DEPART_DELAY_S = 10.0
# When the departure is seen in the video (train_motion.py) the train is
# already moving, so the Twin's train follows within a couple of seconds
# rather than the end-of-clip announce window above.
DETECTED_DEPART_DELAY_S = 2.0
# Live counting while a clip plays: one reading every this many seconds of
# video, so the coach counts (and the LEDs downstream) follow the platform as
# it changes instead of jumping once at each end. Overridable per clip from the
# console; 0 goes back to reading only the ends plus the post-playback samples.
READ_EVERY_S = 3.0

# Everything time-related here is Bangalore local time, because that's what
# the ridership data behind expected_level() is. Pinning it explicitly (not
# via the host's timezone) keeps the Control Room, the Twin and the baseline
# publisher agreeing on what "09:00" means on a laptop set to anything else.
IST = timezone(timedelta(hours=5, minutes=30))

# same thresholds as decision_engine.py — used here only for the Control
# Room's own immediate display, independent of the MQTT round-trip
THR_LOW, THR_MED = 12, 26


def classify(count: float) -> str:
    if count <= THR_LOW:
        return "low"
    if count <= THR_MED:
        return "med"
    return "high"


app = FastAPI()

# Leave the capture loop some CPU. Torch defaults to 12 of this machine's 16
# logical cores and the detector will happily take all of them; with both
# models running flat out, decoding and pacing frames got starved and a 7.67s
# clip took 11.2s to play — the 30fps this loop is built around collapsed to
# 0.68x. Perception is allowed most of the machine, not all of it: the models
# lose a little throughput, and they're already sampling rather than seeing
# every frame, so the trade is heavily in playback's favour.
torch.set_num_threads(max(2, (os.cpu_count() or 4) // 2))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model(str(CHECKPOINT), device)
print(f"[control-room] loaded {CHECKPOINT} on {device}", flush=True)

# Warm both models here rather than letting the first clip pay for it. A cold
# first detection measured 4.6s against ~1s warm — all of it landing as dead
# air before the operator's first video started.
_warm = PersonCounter()
_warm.update(np.zeros((1080, 1920, 3), dtype=np.uint8))
with torch.no_grad():
    model(frame_to_tensor(np.zeros((432, 768, 3), dtype=np.uint8)).to(device))
print("[control-room] detector + MCNN warmed", flush=True)
# Door-state classifier for train arrival/departure, if one has been trained
# (pipeline/train_detect/train.py). Without it the Control Room falls back to
# the train-strip detector, or to departing at the end of each clip.
DOOR_MODEL_PATH = Path(__file__).resolve().parent.parent / "train_detect" / "models" / "door_state.pt"
door_model = None
if DOOR_MODEL_PATH.exists():
    try:
        door_model = DoorStateModel(DOOR_MODEL_PATH)
        print(f"[control-room] door-state model loaded (epoch {door_model.info['epoch']}, "
              f"val macro-F1 {door_model.info['val_macro_f1']})", flush=True)
    except Exception as exc:  # noqa: BLE001 — a bad checkpoint must not stop the console
        print(f"[control-room] door-state model not loaded: {exc}", flush=True)
else:
    print("[control-room] no door-state model yet — train detection uses the train strip",
          flush=True)

_active_cam = cameras.active()
print(f"[control-room] camera calibration: "
      f"{_active_cam.name + ' (' + _active_cam.id + ')' if _active_cam else 'none — zones are frame thirds'}",
      flush=True)

# The warmed detector is kept for /api/test-frame, so testing a frame never
# pays a model load and never touches the instance a running clip is using.
# One test at a time: the detector holds per-frame state between update() and
# draw_overlay().
_test_counter = _warm
_test_lock = threading.Lock()

state_lock = threading.Lock()
state = {
    "running": False,
    "source_name": None,
    "zone_counts": {z: 0 for z in ZONE_IDS},
    "zone_status": {z: "low" for z in ZONE_IDS},
    "entered_total": 0,
    "exited_total": 0,
    "entered_by_zone": {z: 0 for z in ZONE_IDS},
    "exited_by_zone": {z: 0 for z in ZONE_IDS},
    "last_frame_jpeg": None,
    # mirrors decision_engine.py's compartment_occupancy — that process owns
    # the real running tally, this is just a read-only display of it so the
    # operator doesn't need the Twin open to see the same numbers
    "compartment_occupancy": {z: 0 for z in ZONE_IDS},
    # idle | playing — kept so the console can label its own state without
    # inferring it from `running`
    "phase": "idle",
    # last load failure, surfaced to the console — a file OpenCV cannot decode
    # used to kill the worker thread with a traceback nobody watching the UI
    # would ever see, leaving the operator staring at an idle panel
    "error": None,
    # gap between simulated IST and real time, in ms. Mirrors control/time
    # whoever set it — this console or the Twin — so both show the same
    # clock instead of each keeping a private one.
    "sim_offset_ms": 0,
    # Whether we are actually talking to the broker. The console's MQTT
    # indicator used to be driven by whether its own stats stream was
    # delivering, which is a different connection entirely — so with mosquitto
    # down it cheerfully read LIVE while nothing this console published was
    # reaching the Twin. paho owns the truth here; the UI reads it.
    "mqtt_connected": False,
    # Highest frame_index published so far in the current dwell. The mid-dwell
    # samples are read after playback (see sample_dwell) and therefore arrive
    # AFTER the final frame, so without this the console's "platform now"
    # readout ends up showing a frame from the middle of the clip: measured
    # A:2 B:10 C:26 from frame 183 while the clip actually ended on
    # A:1 B:14 C:25. Readings still all go out on the wire — the decision
    # engine needs the whole trajectory — this only gates the live display.
    "display_frame_index": -1,
    # what train_motion.py thinks the train in the video is doing, for the
    # console; and an operator's TRAIN ARRIVED / TRAIN LEFT waiting to be
    # picked up by the playback loop
    "train_state": "not detected",
    "manual_train_event": None,
}


def _on_mqtt_connect(client, userdata, flags, reason_code, properties) -> None:
    with state_lock:
        state["mqtt_connected"] = reason_code == 0
    print(f"[control-room] mqtt connected ({reason_code})", flush=True)
    # re-subscribes automatically on reconnect too, not just the first connect
    client.subscribe(f"metropulse/{STATION_ID}/compartment/+/occupancy", qos=1)
    client.subscribe(f"metropulse/{STATION_ID}/control/time", qos=1)
    # boarding/alighting is decided by decision_engine.py now — this console
    # reports what it sees on the platform and displays what the engine
    # concluded, rather than guessing at crossings itself
    client.subscribe(f"metropulse/{STATION_ID}/gate/+/entry_exit", qos=0)


def _on_mqtt_disconnect(client, userdata, disconnect_flags, reason_code, properties) -> None:
    """paho reconnects on its own from here (loop_start keeps retrying); this
    exists so the console can say the broker is gone instead of showing a
    LIVE indicator over a pipeline that has stopped moving."""
    with state_lock:
        state["mqtt_connected"] = False
    print(f"[control-room] mqtt disconnected ({reason_code}) — retrying", flush=True)


def _parse_ist(target_iso: str) -> datetime | None:
    """control/time carries a naive local timestamp that means IST (see the
    topic docs). Attaching the zone explicitly avoids inheriting whatever
    timezone this machine happens to be configured for."""
    try:
        dt = datetime.fromisoformat(str(target_iso))
    except (TypeError, ValueError):
        return None
    return dt.replace(tzinfo=IST) if dt.tzinfo is None else dt


def _handle_time_message(payload: dict) -> None:
    """Someone moved the simulated clock. Mirror it either way; if it wasn't
    us, also answer with the crowd baseline for that date/hour, since the
    Twin has no way to run the BMRCL model itself and would otherwise be
    showing a prediction for a completely different moment than its clock."""
    dt = _parse_ist(payload.get("target_iso"))
    if dt is None:
        return
    with state_lock:
        state["sim_offset_ms"] = int(dt.timestamp() * 1000 - time.time() * 1000)

    if payload.get("origin") == "control_room":
        return   # our own echo — the baseline went out with it
    publish_baseline(dt)
    print(f"[control-room] clock set from twin -> {dt.isoformat()}", flush=True)


def _on_mqtt_message(client, userdata, msg) -> None:
    try:
        payload = json.loads(msg.payload.decode())
    except (json.JSONDecodeError, UnicodeDecodeError):
        return

    if msg.topic.endswith("/control/time"):
        _handle_time_message(payload)
        return

    if msg.topic.endswith("/entry_exit"):
        # A zone_id this console doesn't model (another station's layout, a
        # malformed publisher) raised KeyError straight inside paho's callback,
        # which takes the whole message loop down with it — every subsequent
        # occupancy and gate update silently stops arriving. Skip it instead.
        zone_id = payload.get("zone_id")
        if zone_id not in ZONE_IDS:
            return
        with state_lock:
            state["entered_by_zone"][zone_id] = payload.get("entered_total", 0)
            state["exited_by_zone"][zone_id] = payload.get("exited_total", 0)
            state["entered_total"] = sum(state["entered_by_zone"].values())
            state["exited_total"] = sum(state["exited_by_zone"].values())
        return

    zone_id = payload.get("zone_id")
    if zone_id in ZONE_IDS and isinstance(payload.get("occupancy"), (int, float)):
        with state_lock:
            state["compartment_occupancy"][zone_id] = int(payload["occupancy"])


mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
mqtt_client.on_connect = _on_mqtt_connect
mqtt_client.on_disconnect = _on_mqtt_disconnect
mqtt_client.on_message = _on_mqtt_message
# connect_async, not connect: a blocking connect() at import time raises
# ConnectionRefusedError when the broker isn't up yet, which means uvicorn
# never finishes importing the app and the whole Control Room fails to start
# — with a traceback about a socket rather than anything telling the operator
# to bring mosquitto up. Async hands the retry loop to paho, so the console
# boots either way, shows MQTT DOWN, and connects itself the moment the broker
# appears.
mqtt_client.connect_async(BROKER_HOST, BROKER_PORT)
mqtt_client.loop_start()
print(f"[control-room] mqtt connecting to {BROKER_HOST}:{BROKER_PORT}", flush=True)

worker_thread: threading.Thread | None = None
# Stopping a clip has two very different meanings and one event can't carry
# both. STOP is the operator saying "that's enough footage, send the train" —
# it must hand off to the Twin exactly like a clip running to its end, using
# whatever the pipeline has read so far. Loading a new clip also stops the
# current one, but there the run is being *superseded*: the incoming clip
# owns this station now, and handing off would depart the train out from
# under it. `superseded_event` is what tells the two apart; without it, an
# operator stop published nothing at all and the train held forever.
stop_event = threading.Event()
superseded_event = threading.Event()


def publish_zone_telemetry(detected: list, density: list | None = None,
                           frame_index: int = 0) -> None:
    """Two independent readings of the same zone, and which one is being used.

    `count` is the detection head count, because it's the one that can be
    checked: every person in it has a box you can look at. MCNN's density
    estimate ships alongside rather than instead — it degrades more gracefully
    once a crowd is dense enough that individual detection saturates, so it's
    the reading that matters at the top of the range, and keeping both visible
    means a disagreement is a fact on the wire instead of a mystery."""
    for i, (zone_id, count) in enumerate(zip(ZONE_IDS, detected)):
        payload = {
            "schema": "metropulse.zone_telemetry.v2",
            "station_id": STATION_ID,
            "zone_id": zone_id,
            "ts": now_iso(),
            "count": round(count),
            "detected_count": round(count),
            "density_count": round(density[i]) if density else None,
            "method": "detection",
            # which frame of the clip this reading came from. The mid-dwell
            # samples are taken after playback has finished, so they arrive
            # out of order and the decision engine sorts on this.
            "frame_index": frame_index,
            "capacity": ZONE_CAPACITY,
            "confidence": 1.0,  # see zone_telemetry_publisher.py — no native uncertainty estimate
        }
        mqtt_client.publish(f"metropulse/{STATION_ID}/zone/{zone_id}/telemetry", json.dumps(payload), qos=0)


def publish_control_train(action: str, **extra) -> None:
    payload = {"schema": "metropulse.control_train.v1", "station_id": STATION_ID, "ts": now_iso(),
               "action": action, **extra}
    mqtt_client.publish(f"metropulse/{STATION_ID}/control/train", json.dumps(payload), qos=1)


def publish_baseline(dt: datetime) -> tuple[str, str]:
    """Expected crowd level for a moment in IST, retained so a console that
    connects later still sees it. Shared by the CHANGE TIME endpoint and by
    a clock change arriving from the Twin, so both routes produce exactly the
    same message rather than two near-identical ones that could drift."""
    level, reason = expected_level(dt.replace(tzinfo=None))
    mqtt_client.publish(
        f"metropulse/{STATION_ID}/prediction/baseline",
        json.dumps({
            "schema": "metropulse.prediction_baseline.v2",
            "station_id": STATION_ID,
            "ts": now_iso(),
            # WHICH MOMENT this is the forecast for, in IST. Without it the
            # message is just a level and a sentence, and because it is
            # retained a subscriber has no way to tell a fresh forecast from
            # one computed for a completely different day — the passenger app
            # showed "Republic Day" at 14:00 on an ordinary Tuesday, because
            # that was the last baseline anyone had published. Consumers should
            # ignore a baseline whose for_iso is not the moment they are asking
            # about.
            "for_iso": dt.replace(tzinfo=None).isoformat(),
            "expected_level": level,
            "reason": reason,
        }),
        qos=1, retain=True,
    )
    return level, reason


def draw_zone_lines(frame: np.ndarray, calibration=None) -> np.ndarray:
    """The marked platform and coach seams when this camera is calibrated,
    the frame thirds when it isn't."""
    if calibration is not None:
        return calibration.draw(frame)
    w = frame.shape[1]
    for x in (w // 3, 2 * w // 3):
        cv2.line(frame, (x, 0), (x, frame.shape[0]), (255, 200, 0), 1)
    return frame


def density_zone_counts(density: np.ndarray, calibration=None) -> list:
    """MCNN's estimate per coach, split the same way the detector's people
    are: by the calibrated platform when there is one, by thirds otherwise."""
    if calibration is None:
        return zone_counts(density)
    masks = calibration.zone_masks(*density.shape[:2])
    return [max(0.0, float(density[masks[z]].sum())) for z in ZONE_IDS]


def label_frame(frame: np.ndarray, counts: dict, caption: str, calibration=None) -> np.ndarray:
    """Zone lines and the standing counts, without boxes. Used for the frames
    between the two readings: the boxes belong to a frame that is now seconds
    old and drawing them here would put them over empty floor."""
    out = draw_zone_lines(frame.copy(), calibration)
    total = sum(counts.values())
    cv2.putText(out, f"PEOPLE {total}", (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, "  ".join(f"{z}:{counts[z]}" for z in ZONE_IDS), (12, 54),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 1, cv2.LINE_AA)
    cv2.putText(out, caption, (12, out.shape[0] - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 160, 180), 1, cv2.LINE_AA)
    return out


def set_preview(image: np.ndarray) -> None:
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if ok:
        with state_lock:
            state["last_frame_jpeg"] = buf.tobytes()


def paste_pip(frame: np.ndarray) -> np.ndarray:
    """The last live reading's annotated frame, inset top-right over the
    playing video. Boxes can't go on the live frames themselves — they belong
    to a frame up to READ_EVERY_S old and would sit over empty floor — so the
    reading that produced the counts is shown beside them instead."""
    with state_lock:
        pip = state.get("reading_pip")
    if pip is None:
        return frame
    ph, pw = pip.shape[:2]
    h, w = frame.shape[:2]
    if pw + 20 > w or ph + 20 > h:
        return frame
    x0, y0 = w - pw - 10, 10
    frame[y0:y0 + ph, x0:x0 + pw] = pip
    cv2.rectangle(frame, (x0 - 1, y0 - 1), (x0 + pw, y0 + ph), (120, 255, 160), 1)
    return frame


def start_reading_log(source_name: str, calibration, fps: float, read_every_s: float) -> Path:
    """A fresh log folder for this clip, with what the readings were taken
    under — the camera layout matters as much as the frame when checking a
    count later, and the camera may have been re-marked since."""
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", Path(source_name or "clip").stem)[:40] or "clip"
    run_dir = READING_LOG_DIR / f"{datetime.now():%Y%m%d-%H%M%S}_{stem}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "meta.json").write_text(json.dumps({
        "clip": source_name,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "fps": fps,
        "read_every_s": read_every_s,
        "camera": calibration.to_dict() if calibration else None,
    }, indent=2), encoding="utf-8")
    return run_dir


def log_reading(log_dir: Path, frame: np.ndarray, shot: np.ndarray, record: dict) -> None:
    """Raw frame (full resolution — drop it into TEST FRAMES to score it), the
    annotated frame (what was counted), and one line of numbers. A failure
    here is reported, never raised: losing a log entry must not lose the
    reading itself."""
    try:
        base = f"{record['frame_index']:06d}_{record['kind']}"
        cv2.imwrite(str(log_dir / f"{base}_raw.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        cv2.imwrite(str(log_dir / f"{base}_annotated.jpg"), shot, [cv2.IMWRITE_JPEG_QUALITY, 88])
        record = dict(record, raw=f"{base}_raw.jpg", annotated=f"{base}_annotated.jpg")
        with (log_dir / "readings.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except OSError as exc:
        print(f"[control-room] could not write reading log: {exc}", flush=True)


def take_reading(counter: PersonCounter, frame: np.ndarray, label: str,
                 frame_index: int = 0, preview: bool = True, pip: bool = False,
                 log_dir: Path | None = None, kind: str = "reading",
                 video_s: float | None = None) -> dict:
    """One of the two readings a clip gets. Runs tiled detection and MCNN on
    this single frame, publishes the telemetry the decision engine turns into
    boarding, and leaves the annotated frame on the preview so the operator can
    see what was actually counted."""
    t0 = time.time()
    counter.update(frame)
    counts = dict(counter.zone_people)

    with torch.no_grad():
        small = resize_max(frame, MAX_FRAME_DIM)
        density = model(frame_to_tensor(small).to(device))[0, 0].cpu().numpy()
    density_counts = density_zone_counts(density, counter.calibration)

    detected = [float(counts[z]) for z in ZONE_IDS]
    publish_zone_telemetry(detected, density_counts, frame_index)
    # Every reading goes on the wire — the decision engine decomposes the whole
    # trajectory and needs all of them — but only the newest one is what the
    # platform looks like *now*. The mid-dwell samples are read after playback
    # has finished, so they arrive last while describing an earlier moment; let
    # them write here and the console ends up reporting a frame from the middle
    # of a clip that has already ended.
    with state_lock:
        if frame_index >= state["display_frame_index"]:
            state["display_frame_index"] = frame_index
            for zone_id, n in zip(ZONE_IDS, detected):
                state["zone_counts"][zone_id] = round(n)
                state["zone_status"][zone_id] = classify(n)

    # The mid-dwell samples come from the middle of a clip that has already
    # finished playing; showing them would leave the operator looking at an
    # arbitrary earlier moment as if it were the final state.
    if preview or pip or log_dir is not None:
        shot = draw_zone_lines(counter.draw_overlay(resize_max(frame, PREVIEW_MAX_DIM)),
                               counter.calibration)
        cv2.putText(shot, label, (12, shot.shape[0] - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 255, 160), 2, cv2.LINE_AA)
        if preview:
            set_preview(shot)
        if pip:
            small = resize_max(shot, PREVIEW_MAX_DIM // 3)
            with state_lock:
                state["reading_pip"] = small
        if log_dir is not None:
            log_reading(log_dir, frame, shot, {
                "frame_index": frame_index,
                "video_s": round(video_s, 2) if video_s is not None else None,
                "kind": kind,
                "label": label,
                "ts": now_iso(),
                "zones": counts,
                "on_platform": sum(counts.values()),
                "detected_total": counter.detected_total,
                "off_platform": counter.off_platform,
                "density_zones": {z: round(v, 1) for z, v in zip(ZONE_IDS, density_counts)},
                "camera_id": counter.calibration.id if counter.calibration else None,
            })

    off = f", {counter.off_platform} off platform" if counter.off_platform else ""
    print(f"[control-room] {label}: {counts} (total {sum(counts.values())}{off}) "
          f"in {time.time() - t0:.1f}s", flush=True)
    return counts


def process_video(video_path: str, source_name: str,
                  stop_ev: threading.Event, superseded_ev: threading.Event,
                  read_every_s: float = READ_EVERY_S) -> None:
    """Runs _process_video and makes sure a failure reaches the operator.

    Anything raised in here happens on a background thread, so without this the
    only trace is a traceback in a log file while the console sits looking
    idle — which is exactly what an unreadable video did."""
    try:
        _process_video(video_path, source_name, stop_ev, superseded_ev, read_every_s)
    except Exception as exc:                      # noqa: BLE001 — report anything
        print(f"[control-room] FAILED on {source_name}: {exc!r}", flush=True)
        with state_lock:
            state["running"] = False
            state["phase"] = "idle"
            state["error"] = f"{source_name}: {exc}"


def _process_video(video_path: str, source_name: str,
                   stop_ev: threading.Event, superseded_ev: threading.Event,
                   read_every_s: float = READ_EVERY_S) -> None:
    """Two detections per clip — the first frame and the last — with the whole
    video played at true speed in between.

    Everything else was tried first. Detecting continuously alongside playback
    starves the capture loop and drags 30fps down; detecting the whole clip up
    front costs ~6.5x the clip in dead waiting; detecting at a paced interval
    refreshes the preview about once a second, which is a slideshow, not a
    video. None of that work was buying anything, because boarding is measured
    as the change in platform crowd across the dwell (see handle_train in
    decision_engine.py) and that needs exactly two readings: one as the service
    arrives, one as it leaves. Every frame in between was being analysed to
    produce a number nothing consumed.

    So the clip is read twice and played once. Playback has the CPU entirely to
    itself and holds true speed, and the operator sees the annotated 'before'
    frame, then the video, then the annotated 'after' frame left on screen.

    Live counting (read_every_s > 0) adds a reading every that many seconds of
    video, taken on a side thread so playback never waits for the detector.
    At ~1s per reading against a 3s interval the detector is busy a third of
    the time, not continuously, which is what starved playback before. If a
    reading is still running when the next is due (a slow machine, a short
    interval) that slot is skipped rather than queued — a backlog would only
    publish counts for moments the video has long since left. Every live
    reading goes out as telemetry, so the coach counts and the LEDs follow the
    platform, and the decision engine gets the whole dwell trajectory instead
    of the four post-playback samples.
    """
    # events passed in rather than read off the globals: /api/load rebinds
    # those for the incoming run, and this thread must keep watching the pair
    # it was actually started with
    with state_lock:
        state["error"] = None
        state["source_name"] = source_name      # claimed up front, so a failure
                                                # can't leave the previous clip's
                                                # name on screen
        state["display_frame_index"] = -1       # new clip, new frame numbering
        state["reading_pip"] = None
    # The camera is fixed for the whole clip: switching the active camera
    # mid-dwell would split one before/after measurement across two layouts.
    calibration = cameras.active()
    counter = PersonCounter(calibration=calibration)
    print(f"[control-room] {source_name}: camera "
          f"{calibration.id if calibration else 'uncalibrated (frame thirds)'}", flush=True)
    get_frame = make_frame_source(video_path, loop=False, max_dim=SOURCE_MAX_DIM)
    fps = video_fps(video_path)
    frame_interval = (1.0 / fps) / PLAYBACK_SPEED
    preview_interval = 1.0 / PREVIEW_FPS

    with state_lock:
        state["running"] = True
        state["phase"] = "reading"
        state["source_name"] = source_name

    frame = get_frame()
    if frame is None:
        print(f"[control-room] {source_name} has no readable frames", flush=True)
        with state_lock:
            state["running"] = False
            state["phase"] = "idle"
        return

    log_dir = start_reading_log(source_name, calibration, fps, read_every_s)
    print(f"[control-room] {source_name} (~{fps:.1f} fps) — reading platform before the dwell "
          f"(log: {log_dir.name})", flush=True)
    # `arrive` first, then the reading. The decision engine clears its
    # before/after pair on arrive to mark a fresh service, so a reading
    # published ahead of it gets wiped and both ends of the measurement end up
    # being the same "after" value — boarding then comes out as zero for every
    # coach, which is exactly what happened when this ran the other way round.
    publish_control_train("arrive", source_name=source_name)

    with state_lock:
        state["phase"] = "reading"
    counts = take_reading(counter, frame, "BEFORE - platform as the service arrives", 0,
                          pip=read_every_s > 0, log_dir=log_dir, kind="before", video_s=0.0)
    with state_lock:
        state["phase"] = "playing"

    # in frames, so the interval is video time: 3s of a 3fps clip is every 9th
    # frame, of a 30fps clip every 90th, whatever the playback machine manages
    read_every = max(1, round(read_every_s * fps)) if read_every_s > 0 else 0
    if read_every:
        print(f"[control-room] live counting every {read_every_s:g}s "
              f"(every {read_every} frames)", flush=True)
    live = {"counts": counts}
    reader: threading.Thread | None = None
    next_read_index = read_every
    skipped = 0
    # one detector instance: live readings and train-event readings run on
    # their own threads and must take turns on it
    counter_lock = threading.Lock()

    def live_read(img: np.ndarray, idx: int) -> None:
        with counter_lock:
            live["counts"] = take_reading(counter, img, f"LIVE {idx / fps:.1f}s (frame {idx})", idx,
                                          preview=False, pip=True, log_dir=log_dir, kind="live",
                                          video_s=idx / fps)

    # Train arrival/departure read from the video itself (train_motion.py),
    # when this camera has a train strip marked. Without one the clip keeps
    # its old meaning: arrive at the first frame, depart at the last.
    # The trained door-state classifier when there is one and this camera has
    # its doors marked; else the strip detector; else none (the clip departs
    # at its end).
    detector = None
    # the classifier compares against this camera's empty track: its
    # "THIS FRAME = NO TRAIN" still on /calibrate
    empty_still = (cv2.imread(str(_train_ref_path(calibration.id, "empty")))
                   if calibration is not None else None)
    if (calibration is not None and calibration.door_crop and door_model is not None
            and empty_still is not None):
        detector = DoorStateDetector(calibration, fps, door_model,
                                     DoorCropper(calibration)(empty_still))
    elif calibration is not None and calibration.door_crop and door_model is not None:
        print(f"[control-room] camera {calibration.id} has a door crop but no NO TRAIN still — "
              f"press THIS FRAME = NO TRAIN on /calibrate to use the classifier", flush=True)
    if detector is None and calibration is not None and calibration.train_strip:
        ref_train = cv2.imread(str(_train_ref_path(calibration.id, "train")))
        ref_empty = cv2.imread(str(_train_ref_path(calibration.id, "empty")))
        detector = TrainMotionDetector(calibration, fps, ref_train=ref_train, ref_empty=ref_empty)
    motion_log = (log_dir / "motion.csv").open("w", encoding="utf-8") if detector else None
    if motion_log:
        # `response` is the presence score in presence mode (+1 train
        # reference, -1 empty) and the class confidence in classifier mode;
        # `label` is the classifier's class for that frame
        motion_log.write("frame_index,t,moving,slide,change,shift,response,speed,state,event,label\n")
    modes = {"classifier": " (door-state model on the door crop)",
             "presence": " (train / no-train reference stills)",
             "motion": " (no reference stills — needs continuous video)"}
    print(f"[control-room] train detection: "
          + (f"on — {detector.mode} mode{modes.get(detector.mode, '')}" if detector
             else "off — no door crop + model, or train strip, on this camera"),
          flush=True)
    train = {"present": True, "departed": False, "service": 1}
    last_still = {"frame": frame, "index": 0}
    with state_lock:
        state["train_state"] = detector.state if detector else "not detected"
        state["manual_train_event"] = None

    def on_departure(still_frame: np.ndarray, still_index: int, how: str) -> None:
        """The train is pulling out in the video: read the platform as it was
        the moment before it moved — the people left behind — then send the
        Twin's train within DETECTED_DEPART_DELAY_S, while the clip plays on."""
        started = time.time()
        with counter_lock:
            take_reading(counter, still_frame, f"DEPARTURE ({how}) at {still_index / fps:.1f}s",
                         still_index, preview=False, pip=True, log_dir=log_dir,
                         kind="departure", video_s=still_index / fps)
        if superseded_ev.is_set():
            return
        publish_control_train("announce", message="STARTING TRAIN")
        time.sleep(max(0.0, DETECTED_DEPART_DELAY_S - (time.time() - started)))
        if superseded_ev.is_set():
            return
        publish_control_train("depart")
        print(f"[control-room] train left in the video ({how}) -> depart sent", flush=True)
        with state_lock:
            # the measured crowd is aboard now; whoever is still there shows
            # up in the next live reading
            for zone_id in ZONE_IDS:
                state["zone_counts"][zone_id] = 0
                state["zone_status"][zone_id] = "low"

    def on_arrival(img: np.ndarray, idx: int, how: str) -> None:
        """A new train has stopped in the video: a fresh service, so a fresh
        `arrive` (the decision engine clears its readings on it) and a BEFORE
        reading of the queue it is about to take."""
        train["service"] += 1
        publish_control_train("arrive", source_name=f"{source_name} · train {train['service']}")
        with counter_lock:
            live["counts"] = take_reading(counter, img, f"ARRIVAL ({how}) at {idx / fps:.1f}s", idx,
                                          preview=False, pip=True, log_dir=log_dir,
                                          kind="arrival", video_s=idx / fps)
        print(f"[control-room] train {train['service']} arrived in the video ({how})", flush=True)

    def handle_train_event(ev: str, img: np.ndarray, idx: int, how: str) -> None:
        if ev == "departure" and train["present"]:
            train["present"], train["departed"] = False, True
            threading.Thread(target=on_departure, args=(last_still["frame"].copy(),
                                                        last_still["index"], how),
                             daemon=True).start()
        elif ev == "arrival" and not train["present"]:
            train["present"] = True
            threading.Thread(target=on_arrival, args=(img.copy(), idx, how), daemon=True).start()

    base_caption = (f"LIVE — counting every {read_every_s:g}s" if read_every
                    else "PLAYING — next reading when the doors close")

    finished_naturally = False
    last_frame = frame
    last_index = 0
    last_preview_at = 0.0
    # absolute schedule, not "sleep the leftover of each frame" — a per-frame
    # remainder never makes up time lost to a slow frame, so any hitch would
    # permanently slow the rest of the clip instead of being absorbed
    next_frame_at = time.time()
    while not stop_ev.is_set():
        now = time.time()
        if now - last_preview_at >= preview_interval:
            last_preview_at = now
            caption = base_caption + (f"  ·  TRAIN {detector.state.upper()}" if detector else "")
            shown = label_frame(resize_max(last_frame, PREVIEW_MAX_DIM), live["counts"],
                                caption, calibration)
            set_preview(paste_pip(shown) if read_every else shown)

        next_frame_at += frame_interval
        sleep_for = next_frame_at - time.time()
        if sleep_for > 0:
            time.sleep(sleep_for)
        elif sleep_for < -1.0:
            next_frame_at = time.time()   # far behind; resync rather than sprint

        nxt = get_frame()
        if nxt is None:
            finished_naturally = True
            break
        last_frame = nxt
        last_index += 1

        if detector:
            ev = detector.update(nxt, last_index)
            m = detector.last
            if m["anchor"]:
                last_still["frame"], last_still["index"] = nxt, last_index
            motion_log.write(f"{last_index},{m['t']},{int(m['moving'])},{int(m['slide'])},"
                             f"{m['change']:.3f},{m['shift']:.2f},{m['response']:.3f},"
                             f"{m['speed']:.3f},{m['state']},{ev or ''},{m.get('label', '')}\n")
            if ev:
                print(f"[control-room] detected {ev} at {m['t']:.1f}s (frame {last_index})",
                      flush=True)
                handle_train_event(ev, nxt, last_index, "detected")
            elif detector.state == "gone" and train["present"]:
                # the clip opened on an empty track — nothing left, nothing to
                # send; the next train in is an arrival
                train["present"] = False
                print(f"[control-room] no train at the platform at {m['t']:.1f}s — "
                      f"waiting for one", flush=True)
            with state_lock:
                state["train_state"] = detector.state
        with state_lock:
            manual = state["manual_train_event"]
            state["manual_train_event"] = None
        if manual:
            print(f"[control-room] operator: {manual} at {last_index / fps:.1f}s", flush=True)
            if manual == "departure":
                last_still["frame"], last_still["index"] = nxt, last_index
            handle_train_event(manual, nxt, last_index, "operator")
            if detector:
                # the operator outranks the detector; carry on from their word
                detector.state = "gone" if manual == "departure" else "standing"

        if read_every and last_index >= next_read_index:
            next_read_index += read_every
            if reader is not None and reader.is_alive():
                skipped += 1
                print(f"[control-room] reading at frame {last_index} skipped — "
                      f"previous one still running", flush=True)
            else:
                reader = threading.Thread(target=live_read, args=(nxt.copy(), last_index),
                                          daemon=True)
                reader.start()

    # the counter is one detector; the after reading must not overlap a live one
    if reader is not None:
        reader.join()
    if motion_log:
        motion_log.close()
    if skipped:
        print(f"[control-room] {skipped} live reading(s) skipped — the detector "
              f"is slower than every {read_every_s:g}s on this machine", flush=True)

    with state_lock:
        state["phase"] = "reading"

    # Train already left in the video and no new one came in: its departure
    # has been sent, and the clip ended on a waiting platform. Ending the clip
    # must not send a second train.
    if not train["present"]:
        with state_lock:
            state["running"] = False
            state["phase"] = "departed"
            state["train_state"] = "gone"
        print(f"[control-room] stopped processing {source_name} — train had already left "
              f"in the video, no further depart", flush=True)
        return

    if not superseded_ev.is_set():
        # the "after" reading, on the last frame the clip actually showed
        with counter_lock:
            take_reading(counter, last_frame, "AFTER - platform as the service leaves", last_index,
                         log_dir=log_dir, kind="after", video_s=last_index / fps)

    with state_lock:
        state["running"] = False
        state["phase"] = "idle"

    print(f"[control-room] stopped processing {source_name}", flush=True)

    reason = "video finished" if finished_naturally else "stopped by operator"
    # live readings already covered the dwell; the post-playback samples are
    # only needed when the clip was read at its ends alone
    hand_off_to_twin(reason, superseded_ev,
                     during=None if read_every else
                     lambda: sample_dwell(video_path, counter, last_index,
                                          stop_ev, superseded_ev, log_dir, fps))
def sample_dwell(video_path: str, counter: PersonCounter, last_index: int,
                 stop_ev: threading.Event, superseded_ev: threading.Event,
                 log_dir: Path | None = None, fps: float = 0.0) -> None:
    """Read a few more frames from across the clip, so the decision engine sees
    the shape of the dwell rather than just its endpoints.

    Two readings measure net flow only: a stop where people get off AND on
    reports the difference and loses both real numbers. These fill in the
    middle so the trajectory can be split into falls (boarding) and rises
    (alighting). Deliberately run after playback, during the announce window,
    where the Twin is already showing STARTING TRAIN and nobody is waiting on
    this."""
    if last_index < 4 or DWELL_SAMPLES < 3:
        return
    cap = cv2.VideoCapture(video_path)
    try:
        for k in range(1, DWELL_SAMPLES - 1):        # the ends are already read
            if stop_ev.is_set() or superseded_ev.is_set():
                return
            idx = round(k * last_index / (DWELL_SAMPLES - 1))
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                continue
            take_reading(counter, resize_max(frame, SOURCE_MAX_DIM),
                         f"DWELL {k}/{DWELL_SAMPLES - 2} frame {idx}", idx,
                         preview=False, log_dir=log_dir, kind="dwell",
                         video_s=idx / fps if fps else None)
    finally:
        cap.release()


def hand_off_to_twin(reason: str, superseded_ev: threading.Event,
                     during=None) -> None:
    """A clip that ran out and a clip the operator cut short are the same
    decision — "this station is done, send the train" — and both hand off on
    whatever the pipeline read. Only a newer clip taking over suppresses it,
    because there the incoming clip owns this station."""
    if superseded_ev.is_set():
        print("[control-room] superseded by a new clip -> no train handoff", flush=True)
        return

    print(f"[control-room] {reason} -> STARTING TRAIN", flush=True)
    publish_control_train("announce", message="STARTING TRAIN")

    # fill in the middle of the dwell while the announce banner is up
    started = time.time()
    if during is not None:
        during()
    remaining = max(0.0, TRAIN_DEPART_DELAY_S - (time.time() - started))
    # Sleeping in this background thread only delays this thread — the server
    # keeps serving requests normally while this waits. Waiting on
    # `superseded_ev` and not the stop flag: after an operator stop that flag
    # is already set, so waiting on it would return instantly and skip the
    # depart, which is exactly the bug where the banner appeared and the
    # train never moved.
    if not superseded_ev.wait(remaining):
        publish_control_train("depart")
        print("[control-room] depart command sent", flush=True)
        # The service has left, so this station's dwell is over and its platform
        # reading is no longer a thing that is true. Holding it was actively
        # wrong, not just stale: COMPARTMENT LOAD adds the queue to the coach
        # occupancy, and the people in that queue are the same people the
        # engine just counted aboard — so the panel read "COACH 3  48/30", a
        # load above the hard capacity, built out of counting one crowd twice.
        # The Twin clears its platform on departure for exactly this reason
        # (see clearPlatform()); this is the console's half of it. The next
        # clip's own BEFORE reading measures whoever is genuinely still there.
        #
        # Dropping the frame too returns the console to its "load a clip"
        # prompt rather than leaving a dead platform on screen looking like
        # the feed is stuck.
        with state_lock:
            state["last_frame_jpeg"] = None
            state["display_frame_index"] = -1
            for zone_id in ZONE_IDS:
                state["zone_counts"][zone_id] = 0
                state["zone_status"][zone_id] = "low"
            # `idle` reads as "nothing has happened here"; the operator has
            # just watched a train leave and needs the console to say so
            state["phase"] = "departed"


@app.get("/")
def index():
    """The console, served fresh every time.

    FileResponse sends ETag and Last-Modified, and with those present a browser
    will happily keep showing the console it already has after the file has
    changed underneath it — which looks exactly like a fix that didn't work.
    That is why the Twin's access log is a trail of ?v=nocache1, ?v=fix3,
    ?v=perf2 cache-busters. no-store alone wasn't enough once a validated copy
    was already in the cache, so the validators go too: read the bytes and
    return them plainly, with nothing for a conditional request to match on.
    It is a 20KB page from localhost; there was never anything to save here."""
    return _page("index.html")


@app.get("/calibrate")
def calibrate_page():
    """Mark each camera's platform and coach seams, and test detection on
    frames from it. Served fresh for the same reason as the console."""
    return _page("calibrate.html")


def _page(name: str) -> Response:
    return Response(
        content=(STATIC_DIR / name).read_bytes(),
        media_type="text/html; charset=utf-8",
        headers={"Cache-Control": "no-store"},
    )


class ChangeTimeQuery(BaseModel):
    date: str   # "YYYY-MM-DD"
    hour: int   # 0-23


@app.post("/api/change-time")
def change_time(q: ChangeTimeQuery):
    """Sets what date/hour the Twin displays and predicts for — not just a
    read-only "what would the baseline be" check, this actually moves the
    Twin's clock. Publishes two things: prediction/baseline (the expected
    crowd level for that date/hour, real BMRCL-data-backed) and control/time
    (tells the Twin to offset its displayed clock to match), both retained
    so a Twin tab that connects later still shows the jumped time."""
    try:
        dt = datetime.strptime(f"{q.date} {q.hour:02d}", "%Y-%m-%d %H").replace(tzinfo=IST)
    except ValueError:
        return JSONResponse({"error": "invalid date or hour"}, status_code=400)

    level, reason = publish_baseline(dt)
    time_payload = {
        "schema": "metropulse.control_time.v1",
        "station_id": STATION_ID,
        "ts": now_iso(),
        "target_iso": dt.replace(tzinfo=None).isoformat(),
        # marks this as ours so the round trip through the broker doesn't
        # come back looking like a fresh request from the Twin
        "origin": "control_room",
    }
    mqtt_client.publish(f"metropulse/{STATION_ID}/control/time", json.dumps(time_payload), qos=1, retain=True)
    with state_lock:
        state["sim_offset_ms"] = int(dt.timestamp() * 1000 - time.time() * 1000)
    return {"expected_level": level, "reason": reason}


@app.post("/api/reset-occupancy")
def reset_occupancy():
    """The only reset for compartment_occupancy's running tally (see
    compartment/occupancy in mqtt_contract.md) — decision_engine.py owns
    that state, so this can't just zero it locally, it has to ask over
    MQTT (control/reset) and wait for the real reset to come back the same
    way every other subscriber finds out. Optimistically zero the local
    mirror too, so this UI doesn't sit stale for the round-trip."""
    with state_lock:
        for zone_id in ZONE_IDS:
            state["compartment_occupancy"][zone_id] = 0
            # the queue half of the compartment picture too — it holds the
            # last clip's telemetry indefinitely once processing stops, and
            # leaving it made a reset console still read a full platform
            state["zone_counts"][zone_id] = 0
            state["zone_status"][zone_id] = "low"
        state["entered_total"] = 0
        state["exited_total"] = 0
    payload = {"schema": "metropulse.control_reset.v1", "station_id": STATION_ID, "ts": now_iso()}
    mqtt_client.publish(f"metropulse/{STATION_ID}/control/reset", json.dumps(payload), qos=1)
    return {"ok": True}


@app.post("/api/load")
async def load(video: UploadFile = File(...), read_every_s: float = Form(READ_EVERY_S)):
    global worker_thread, stop_event, superseded_event

    if worker_thread and worker_thread.is_alive():
        # superseded first, then stop — the worker checks the flag the moment
        # it leaves its loop, so setting them the other way round would race
        # and let the outgoing clip hand off to the Twin anyway
        superseded_event.set()
        stop_event.set()
        worker_thread.join(timeout=5)

    dest = UPLOAD_DIR / "current_video"
    with dest.open("wb") as f:
        shutil.copyfileobj(video.file, f)

    with state_lock:
        state["phase"] = "reading"   # out of `departed` the moment a clip lands
    stop_event = threading.Event()
    superseded_event = threading.Event()
    worker_thread = threading.Thread(
        target=process_video,
        args=(str(dest), video.filename, stop_event, superseded_event, max(0.0, read_every_s)),
        daemon=True,
    )
    worker_thread.start()
    return {"ok": True, "source_name": video.filename}


class TrainEventIn(BaseModel):
    event: str   # "arrival" | "departure"


@app.post("/api/train-event")
def train_event(q: TrainEventIn):
    """The operator saying the train in the video has arrived or left — the
    fallback for when train detection misses, or for a camera with no train
    strip. Unlike STOP the clip keeps playing; the playback loop picks this up
    on its next frame and treats it exactly like a detected event."""
    if q.event not in ("arrival", "departure"):
        return JSONResponse({"error": "event must be arrival or departure"}, status_code=400)
    with state_lock:
        if not state["running"]:
            return JSONResponse({"error": "no clip is playing"}, status_code=409)
        state["manual_train_event"] = q.event
    return {"ok": True}


@app.post("/api/stop")
def stop():
    """Cut the clip short and send the train on what's been read so far.
    Deliberately not a "superseded" stop — see process_video's handoff."""
    stop_event.set()
    return {"ok": True, "departing_in_s": TRAIN_DEPART_DELAY_S}


@app.get("/api/stream.mjpg")
def stream_mjpg():
    """The preview as one long multipart response instead of a polled still.

    The browser used to re-request /api/frame.jpg every 400ms, which is 2.5
    frames a second — so a 230-frame clip played correctly at 30fps on the
    server arrived as about 19 stills and looked like it was racing past. The
    frames were always there; nothing was collecting them. Here the connection
    stays open and frames are pushed as they are produced."""
    def gen():
        last = None
        while True:
            with state_lock:
                jpeg = state["last_frame_jpeg"]
            if jpeg is None:
                last = None      # cleared between clips; next frame is fresh
            elif jpeg is not last:
                last = jpeg
                yield MJPEG_BOUNDARY + jpeg + CRLF
            time.sleep(1.0 / (PREVIEW_FPS * 2))   # poll the slot faster than it fills
    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/frame.jpg")
def frame_jpg():
    with state_lock:
        jpeg = state["last_frame_jpeg"]
    if jpeg is None:
        return Response(status_code=404)
    return Response(content=jpeg, media_type="image/jpeg")


def _stats_snapshot() -> dict:
    with state_lock:
        return {
            "running": state["running"],
            "source_name": state["source_name"],
            "zone_counts": dict(state["zone_counts"]),
            "zone_status": dict(state["zone_status"]),
            "entered_total": state["entered_total"],
            "exited_total": state["exited_total"],
            "compartment_occupancy": dict(state["compartment_occupancy"]),
            "phase": state["phase"],
            "has_frame": state["last_frame_jpeg"] is not None,
            "error": state["error"],
            # the broker link, not this stats stream — see state["mqtt_connected"]
            "mqtt_connected": state["mqtt_connected"],
            # the browser ticks its own clock off this rather than being fed
            # a timestamp twice a second — same offset the Twin holds
            "sim_offset_ms": state["sim_offset_ms"],
            "train_state": state["train_state"],
        }


@app.get("/api/stats")
def stats():
    return JSONResponse(_stats_snapshot())


@app.get("/api/stream")
def stream():
    def gen():
        while True:
            yield f"data: {json.dumps(_stats_snapshot())}\n\n"
            time.sleep(0.5)
    return StreamingResponse(gen(), media_type="text/event-stream")


# --- camera calibration ------------------------------------------------------

class CameraIn(BaseModel):
    id: str
    name: str = ""
    platform: list[list[float]]          # 4 corners, normalised 0..1, in order 1-2-3-4
    boundaries: list[list[float]] = []   # one point on each coach seam, normalised
    coaches: list[str]                   # one label per segment, from corner 1's end
    train_strip: list[list[float]] | None = None   # 4 corners, for train detection
    door_crop: list[list[float]] | None = None     # 4 corners, for the door-state classifier
    reference_jpeg_b64: str | None = None


class ActiveCameraIn(BaseModel):
    id: str | None = None


def _camera_frame_path(cam_id: str) -> Path:
    return CAMERA_FRAMES_DIR / f"{cam_id}.jpg"


def _train_ref_path(cam_id: str, kind: str) -> Path:
    """Reference still for train detection's presence mode: kind "train"
    (a train standing at the platform) or "empty" (no train)."""
    return CAMERA_FRAMES_DIR / f"{cam_id}_{kind}.jpg"


@app.get("/api/cameras")
def list_cameras():
    return {
        "active": cameras.active_id,
        "cameras": [dict(c.to_dict(), has_frame=_camera_frame_path(c.id).exists(),
                         has_train_ref=_train_ref_path(c.id, "train").exists(),
                         has_empty_ref=_train_ref_path(c.id, "empty").exists())
                    for c in cameras.all()],
    }


class TrainRefIn(BaseModel):
    kind: str           # "train" | "empty"
    jpeg_b64: str


@app.post("/api/cameras/{cam_id}/train-ref")
def save_train_ref(cam_id: str, q: TrainRefIn):
    """One of the two reference stills presence mode compares the train strip
    against. Must be from this camera, same framing as the clips."""
    if cameras.get(cam_id) is None:
        return JSONResponse({"error": "save the camera first"}, status_code=404)
    if q.kind not in ("train", "empty"):
        return JSONResponse({"error": "kind must be train or empty"}, status_code=400)
    try:
        data = base64.b64decode(q.jpeg_b64)
    except ValueError:
        return JSONResponse({"error": "not valid base64"}, status_code=400)
    if cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR) is None:
        return JSONResponse({"error": "not a readable image"}, status_code=400)
    _train_ref_path(cam_id, q.kind).write_bytes(data)
    print(f"[control-room] camera {cam_id}: '{q.kind}' reference still saved", flush=True)
    return {"ok": True}


@app.get("/api/cameras/{cam_id}/train-ref/{kind}.jpg")
def get_train_ref(cam_id: str, kind: str):
    path = _train_ref_path(cam_id, kind) if kind in ("train", "empty") else None
    if cameras.get(cam_id) is None or path is None or not path.exists():
        return Response(status_code=404)
    return Response(content=path.read_bytes(), media_type="image/jpeg",
                    headers={"Cache-Control": "no-store"})


@app.post("/api/cameras")
def save_camera(cam: CameraIn):
    try:
        cal = CameraCalibration(cam.id, cam.name, cam.platform, cam.boundaries, cam.coaches,
                                cam.train_strip or None, cam.door_crop or None)
    except (CalibrationError, cv2.error) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    if cam.reference_jpeg_b64:
        try:
            _camera_frame_path(cal.id).write_bytes(base64.b64decode(cam.reference_jpeg_b64))
        except ValueError:
            return JSONResponse({"error": "reference frame is not valid base64"}, status_code=400)
    cameras.put(cal)
    print(f"[control-room] saved camera {cal.id}: seams {[round(s, 3) for s in cal.seams]} "
          f"coaches {cal.coaches}", flush=True)
    return {"ok": True, "camera": cal.to_dict()}


@app.delete("/api/cameras/{cam_id}")
def delete_camera(cam_id: str):
    if not cameras.delete(cam_id):
        return JSONResponse({"error": "no such camera"}, status_code=404)
    _camera_frame_path(cam_id).unlink(missing_ok=True)
    _train_ref_path(cam_id, "train").unlink(missing_ok=True)
    _train_ref_path(cam_id, "empty").unlink(missing_ok=True)
    return {"ok": True}


@app.post("/api/cameras/active")
def set_active_camera(q: ActiveCameraIn):
    """Which camera the next clip is read with. Takes effect on the next
    /api/load — a clip already playing keeps the layout it started with."""
    if q.id is not None and cameras.get(q.id) is None:
        return JSONResponse({"error": "no such camera"}, status_code=404)
    cameras.set_active(q.id)
    print(f"[control-room] active camera -> {q.id or 'none (frame thirds)'}", flush=True)
    return {"ok": True, "active": q.id}


@app.get("/api/cameras/{cam_id}/frame.jpg")
def camera_frame(cam_id: str):
    path = _camera_frame_path(cam_id)
    if cameras.get(cam_id) is None or not path.exists():
        return Response(status_code=404)
    return Response(content=path.read_bytes(), media_type="image/jpeg",
                    headers={"Cache-Control": "no-store"})


def _read_upload_frame(data: bytes, filename: str, at_s: float) -> tuple[np.ndarray, float | None]:
    """An uploaded still or video -> one BGR frame at detection resolution,
    plus the video's length in seconds (None for a still).

    Images decode straight from memory. Video has to go through a file,
    because OpenCV opens containers by path; decoding here rather than in the
    browser also covers codecs a browser won't play, such as the mp4v that
    OpenCV itself writes when frames are stitched into a clip."""
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is not None:
        return resize_max(img, SOURCE_MAX_DIM), None

    fd, tmp = tempfile.mkstemp(dir=UPLOAD_DIR, suffix=Path(filename or "").suffix.lower() or ".bin")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        cap = cv2.VideoCapture(tmp)
        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
            n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
            duration = n / fps if fps > 0 and n > 0 else None
            if at_s > 0 and fps > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(at_s * fps))
            ok, frame = cap.read()
        finally:
            cap.release()
    finally:
        Path(tmp).unlink(missing_ok=True)
    if not ok or frame is None:
        raise ValueError("could not decode an image or a video frame from this file")
    return resize_max(frame, SOURCE_MAX_DIM), duration


@app.post("/api/frame-grab")
def frame_grab(file: UploadFile = File(...), at_s: float = Form(0.0)):
    """One frame from an uploaded image or video, to mark a camera on."""
    try:
        frame, duration = _read_upload_frame(file.file.read(), file.filename, at_s)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        return JSONResponse({"error": "could not encode the frame"}, status_code=500)
    headers = {"Cache-Control": "no-store"}
    if duration is not None:
        headers["X-Video-Duration"] = f"{duration:.2f}"
    return Response(content=buf.tobytes(), media_type="image/jpeg", headers=headers)


@app.post("/api/test-frame")
def test_frame(file: UploadFile = File(...), camera_id: str = Form(""), at_s: float = Form(0.0)):
    """Run detection on one frame and report what was counted, without
    publishing anything — nothing here reaches MQTT, the Twin or the
    decision engine, so testing is safe while a demo is running.

    camera_id "" means no calibration (the frame-thirds split), so the two can
    be compared on the same frame."""
    try:
        frame, _duration = _read_upload_frame(file.file.read(), file.filename, at_s)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    calibration = cameras.get(camera_id) if camera_id else None
    if camera_id and calibration is None:
        return JSONResponse({"error": f"no camera called {camera_id!r}"}, status_code=404)

    t0 = time.time()
    with _test_lock:
        _test_counter.calibration = calibration
        _test_counter.update(frame)
        counts = dict(_test_counter.zone_people)
        detected_total = _test_counter.detected_total
        off_platform = _test_counter.off_platform
        shot = draw_zone_lines(_test_counter.draw_overlay(resize_max(frame, PREVIEW_MAX_DIM)),
                               calibration)
    detect_s = time.time() - t0

    with torch.no_grad():
        density = model(frame_to_tensor(resize_max(frame, MAX_FRAME_DIM)).to(device))[0, 0].cpu().numpy()
    density_counts = density_zone_counts(density, calibration)

    ok, buf = cv2.imencode(".jpg", shot, [cv2.IMWRITE_JPEG_QUALITY, 85])
    h, w = frame.shape[:2]
    print(f"[control-room] TEST {file.filename} @ {at_s:.1f}s camera={camera_id or 'thirds'}: "
          f"{counts} on platform {sum(counts.values())}, off {off_platform}, "
          f"{detect_s:.1f}s", flush=True)
    return {
        "file": file.filename,
        "at_s": at_s,
        "camera_id": calibration.id if calibration else None,
        "frame_size": [w, h],
        "detected_total": detected_total,
        "on_platform": sum(counts.values()),
        "off_platform": off_platform,
        "zones": counts,
        "density_zones": {z: round(v, 1) for z, v in zip(ZONE_IDS, density_counts)},
        "density_total": round(max(0.0, float(density.sum())), 1),
        "detect_s": round(detect_s, 2),
        "image_jpeg_b64": base64.b64encode(buf.tobytes()).decode() if ok else None,
    }


# --- reading logs ------------------------------------------------------------

def _log_run_dir(run: str) -> Path | None:
    """A run folder by name, or None — names come from the URL, so nothing
    with a path separator or '..' is ever joined onto the log directory."""
    if not _LOG_NAME_RE.match(run) or ".." in run:
        return None
    path = READING_LOG_DIR / run
    return path if path.is_dir() else None


def _read_jsonl(path: Path) -> list:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue   # a line cut short by a crash mid-write; skip it
    return out


def _read_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


@app.get("/api/logs")
def list_logs():
    runs = []
    for d in sorted(READING_LOG_DIR.iterdir(), reverse=True):
        if not d.is_dir():
            continue
        meta = _read_json(d / "meta.json", {})
        manual = _read_json(d / "manual.json", {})
        runs.append({
            "run": d.name,
            "clip": meta.get("clip"),
            "started_at": meta.get("started_at"),
            "camera_id": (meta.get("camera") or {}).get("id"),
            "readings": len(_read_jsonl(d / "readings.jsonl")),
            "checked": len(manual),
        })
    return {"runs": runs}


@app.get("/api/logs/{run}")
def get_log(run: str):
    d = _log_run_dir(run)
    if d is None:
        return JSONResponse({"error": "no such log"}, status_code=404)
    readings = sorted(_read_jsonl(d / "readings.jsonl"), key=lambda r: r.get("frame_index", 0))
    return {
        "run": run,
        "meta": _read_json(d / "meta.json", {}),
        "readings": readings,
        "manual": _read_json(d / "manual.json", {}),
    }


@app.get("/api/logs/{run}/files/{name}")
def get_log_file(run: str, name: str):
    d = _log_run_dir(run)
    if d is None or not _LOG_NAME_RE.match(name) or ".." in name or not name.endswith(".jpg"):
        return Response(status_code=404)
    path = d / name
    if not path.exists():
        return Response(status_code=404)
    # a logged frame never changes once written, so the browser may keep it —
    # the log page redraws every card each time a count is saved
    return Response(content=path.read_bytes(), media_type="image/jpeg",
                    headers={"Cache-Control": "private, max-age=86400"})


class ManualCountIn(BaseModel):
    key: str                                   # "<frame_index:06d>_<kind>", as in the file names
    total: int | None = None                   # your own count inside the marked platform
    zones: dict[str, int | None] = {}          # optionally per coach


@app.post("/api/logs/{run}/manual")
def save_manual_count(run: str, q: ManualCountIn):
    """The operator's own count for one logged reading, saved beside it so the
    check survives a reload and can be compared across runs. A total of None
    with no per-coach numbers clears it."""
    d = _log_run_dir(run)
    if d is None or not _LOG_NAME_RE.match(q.key):
        return JSONResponse({"error": "no such log"}, status_code=404)
    manual = _read_json(d / "manual.json", {})
    zones = {z: v for z, v in q.zones.items() if z in ZONE_IDS and v is not None}
    if q.total is None and not zones:
        manual.pop(q.key, None)
    else:
        manual[q.key] = {"total": q.total, "zones": zones}
    (d / "manual.json").write_text(json.dumps(manual, indent=2), encoding="utf-8")
    return {"ok": True}


@app.delete("/api/logs/{run}")
def delete_log(run: str):
    d = _log_run_dir(run)
    if d is None:
        return JSONResponse({"error": "no such log"}, status_code=404)
    shutil.rmtree(d)
    return {"ok": True}
