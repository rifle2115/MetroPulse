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

import json
import os
import shutil
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
import paho.mqtt.client as mqtt
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "mcnn"))
from person_counter import PersonCounter  # noqa: E402
from inference import (  # noqa: E402
    MAX_FRAME_DIM, ZONE_CAPACITY, ZONE_IDS, frame_to_tensor, load_model, make_frame_source,
    now_iso, resize_max, video_fps, zone_counts,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "prediction"))
from baseline_engine import expected_level  # noqa: E402

STATIC_DIR = Path(__file__).parent / "static"
UPLOAD_DIR = Path(__file__).parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

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


def draw_zone_lines(frame: np.ndarray) -> np.ndarray:
    w = frame.shape[1]
    for x in (w // 3, 2 * w // 3):
        cv2.line(frame, (x, 0), (x, frame.shape[0]), (255, 200, 0), 1)
    return frame


def label_frame(frame: np.ndarray, counts: dict, caption: str) -> np.ndarray:
    """Zone lines and the standing counts, without boxes. Used for the frames
    between the two readings: the boxes belong to a frame that is now seconds
    old and drawing them here would put them over empty floor."""
    out = draw_zone_lines(frame.copy())
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


def take_reading(counter: PersonCounter, frame: np.ndarray, label: str,
                 frame_index: int = 0, preview: bool = True) -> dict:
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
    density_counts = zone_counts(density)

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
    if preview:
        shot = draw_zone_lines(counter.draw_overlay(resize_max(frame, PREVIEW_MAX_DIM)))
        cv2.putText(shot, label, (12, shot.shape[0] - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 255, 160), 2, cv2.LINE_AA)
        set_preview(shot)

    print(f"[control-room] {label}: {counts} (total {sum(counts.values())}) "
          f"in {time.time() - t0:.1f}s", flush=True)
    return counts


def process_video(video_path: str, source_name: str,
                  stop_ev: threading.Event, superseded_ev: threading.Event) -> None:
    """Runs _process_video and makes sure a failure reaches the operator.

    Anything raised in here happens on a background thread, so without this the
    only trace is a traceback in a log file while the console sits looking
    idle — which is exactly what an unreadable video did."""
    try:
        _process_video(video_path, source_name, stop_ev, superseded_ev)
    except Exception as exc:                      # noqa: BLE001 — report anything
        print(f"[control-room] FAILED on {source_name}: {exc!r}", flush=True)
        with state_lock:
            state["running"] = False
            state["phase"] = "idle"
            state["error"] = f"{source_name}: {exc}"


def _process_video(video_path: str, source_name: str,
                   stop_ev: threading.Event, superseded_ev: threading.Event) -> None:
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
    counter = PersonCounter()
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

    print(f"[control-room] {source_name} (~{fps:.1f} fps) — reading platform before the dwell", flush=True)
    # `arrive` first, then the reading. The decision engine clears its
    # before/after pair on arrive to mark a fresh service, so a reading
    # published ahead of it gets wiped and both ends of the measurement end up
    # being the same "after" value — boarding then comes out as zero for every
    # coach, which is exactly what happened when this ran the other way round.
    publish_control_train("arrive", source_name=source_name)

    with state_lock:
        state["phase"] = "reading"
    counts = take_reading(counter, frame, "BEFORE - platform as the service arrives", 0)
    with state_lock:
        state["phase"] = "playing"

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
            set_preview(label_frame(resize_max(last_frame, PREVIEW_MAX_DIM), counts,
                                    "PLAYING — next reading when the doors close"))

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

    with state_lock:
        state["phase"] = "reading"

    if not superseded_ev.is_set():
        # the "after" reading, on the last frame the clip actually showed
        take_reading(counter, last_frame, "AFTER - platform as the service leaves", last_index)

    with state_lock:
        state["running"] = False
        state["phase"] = "idle"

    print(f"[control-room] stopped processing {source_name}", flush=True)

    reason = "video finished" if finished_naturally else "stopped by operator"
    hand_off_to_twin(reason, superseded_ev,
                     during=lambda: sample_dwell(video_path, counter, last_index,
                                                 stop_ev, superseded_ev))
def sample_dwell(video_path: str, counter: PersonCounter, last_index: int,
                 stop_ev: threading.Event, superseded_ev: threading.Event) -> None:
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
                         preview=False)
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
    return Response(
        content=(STATIC_DIR / "index.html").read_bytes(),
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
async def load(video: UploadFile = File(...)):
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
        args=(str(dest), video.filename, stop_event, superseded_event),
        daemon=True,
    )
    worker_thread.start()
    return {"ok": True, "source_name": video.filename}


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
