# MetroPulse

**Real-time crowd intelligence for metro platforms.**

Metro systems optimise *train* scheduling but ignore how passengers spread
themselves along a platform — so one coach is crushed while the next is half
empty. MetroPulse watches the platform with a camera, works out how full each
coach is about to be, and tells passengers which coach to stand at.

Everything runs locally on one laptop. No metro CCTV, no edge hardware — but
every component talks over the **same MQTT contract** real hardware would use,
so swapping in real cameras and ESP32 LED strips later changes nothing above
the transport layer.

---

## The three parts

```
                    ┌──────────────────────────┐
   video clip  ───► │  1. CONTROL ROOM  :8000  │   operator's app
                    │  YOLO + MCNN perception  │
                    └────────────┬─────────────┘
                                 │  MQTT (localhost:1883)
                    ┌────────────▼─────────────┐
                    │     DECISION ENGINE      │   thresholds, boarding maths,
                    │  (no UI — a background   │   per-coach occupancy
                    │   process)               │
                    └────────────┬─────────────┘
                                 │  MQTT
              ┌──────────────────┴──────────────────┐
              │                                     │
   ┌──────────▼───────────┐            ┌────────────▼───────────┐
   │ 2. DIGITAL TWIN :8080│  ────────► │ 3. PASSENGER APP :8090 │
   │ 3D station, LED      │  twin/state│ "Board Coach 3"        │
   │ strips, simulation   │            │ + crowd forecast       │
   └──────────────────────┘            └────────────────────────┘
        control room                          what a passenger
        + engineering view                    sees on their phone
```

| # | Part | Who it's for | Where |
|---|------|--------------|-------|
| 1 | **Control Room** | Operator — loads camera footage, watches detection | `pipeline/control_room/` |
| 2 | **Digital Twin** | Control room wall — live 3D model of the platform | `MetroPulse 0/` |
| 3 | **Passenger App** | The public — which coach to board | `user_app/` |

---

## Quick start

### You need

- **Python 3.12**
- **Docker Desktop** (only for the MQTT broker) — must be *running* first
- A modern browser

Install Python dependencies once:

```bash
pip install -r pipeline/control_room/requirements.txt -r pipeline/mcnn/requirements.txt -r pipeline/prediction/requirements.txt -r pipeline/scripts/requirements.txt
```

### Start the five processes

Each one in its own terminal, in this order:

```bash
docker compose -f pipeline/mqtt/docker-compose.yml up -d
```

```bash
cd pipeline/scripts && python -u decision_engine.py
```

```bash
cd pipeline/control_room && python -u -m uvicorn server:app --host 127.0.0.1 --port 8000
```

```bash
cd "MetroPulse 0" && python -u serve.py 8080
```

```bash
cd user_app && python -u serve.py 8090
```

> **The Control Room takes about 40 seconds to start.** It loads and warms up
> the MCNN model and the YOLO detector before it serves anything. That's
> normal — wait for `Uvicorn running on http://127.0.0.1:8000`.

Then open:

| | |
|---|---|
| Digital Twin | <http://127.0.0.1:8080> |
| Control Room | <http://127.0.0.1:8000> |
| Passenger App | <http://127.0.0.1:8090> |

---

## Running a demo

### Option A — the full pipeline (real video → real detection)

1. Open the **Twin** and click **▶ GO LIVE**.
2. Open the **Control Room** and load `Data/Test_data.mp4`.
3. Watch: the train pulls in and **holds**. The clip plays while the detector
   reads the platform. When the clip ends the doors open, everyone boards, and
   the train leaves for the next station.
4. The **Passenger App** follows along the whole time.

Load another clip for the next station. One clip = one station's dwell.

### Option B — no video needed

1. Open the **Twin** (leave it offline — you don't need GO LIVE).
2. Type a number into **SEND BATCH** and press it.

Those passengers walk in, spread themselves across the coaches by what the LED
strips are advising, board, and the train departs. Same result, no camera.

> **The train never leaves on a timer.** It waits at the platform until either
> a clip finishes or a batch is sent, and then until every last passenger is
> physically inside. `▶ DEPART NOW` overrides this.

### Useful before a fresh demo

Coach occupancy is *retained* on the MQTT broker, so it survives restarts. To
start from empty, press **RESET OCCUPANCY** in the Control Room, or:

```bash
curl -X POST http://127.0.0.1:8000/api/reset-occupancy
```

---

## What each part actually does

### 1. Control Room — perception

Loads a video clip and plays it once at true speed. Two models run over it:

- **YOLO11n** — tiled person detection, gives a head count per platform zone
- **MCNN** — a crowd-density network trained on ShanghaiTech Part B

Neither can keep up with 30 fps on a CPU, so the clip is *read* at a handful of
points across the dwell rather than every frame — the endpoints plus four
samples in between. Boarding is then worked out from how the platform crowd
*changed* across the dwell, which needs no person-tracking at all (see
`decompose()` in `pipeline/scripts/decision_engine.py`).

### 2. Digital Twin — simulation and actuation

A 3D station in Three.js. The platform LED strips are the "actuation layer" —
in a real deployment these are ESP32-driven light strips; here they're rendered
in the model, driven by the same MQTT messages.

The Twin can run standalone (SEND BATCH) or be driven by the Control Room
(GO LIVE). Either way it publishes what it's showing on `twin/state`.

### 3. Passenger App — the product

Read-only. Shows the current station, how full each of the three coaches is,
and **which coach to board** — the redistribution advice. If two coaches are
equally empty it offers both, because sending everyone to one of them would
create the imbalance the system exists to prevent.

It also gives a crowd forecast, in priority order:

1. **Significant date** — a real Bangalore event or holiday, named
   ("Diwali", "Republic Day"). 82 dates, 2024–2026.
2. **Rush hour** — 08:00–11:00 and 16:00–19:00 IST.
3. **BMRCL ridership baseline** — the real hourly pattern for that weekday.

---

## Repo layout

```
pipeline/
  control_room/   FastAPI operator app + the perception loop
  mcnn/           MCNN model, training, YOLO person counter
  scripts/        decision_engine.py — thresholds, boarding maths, occupancy
  prediction/     BMRCL ridership baseline + significant dates
  mqtt/           mosquitto broker config (docker-compose)
  docs/
    mqtt_contract.md    ← every topic and payload. Read this first.
MetroPulse 0/     the 3D Digital Twin (single index.html + serve.py)
user_app/         the passenger app
Data/             demo video clips
```

**`pipeline/docs/mqtt_contract.md` is the most useful file in the repo** if you
want to understand how the parts fit together. Every message between stages is
specified there, and nothing changes when the simulated components are swapped
for real hardware.

---

## Data that isn't in this repo

Excluded by `.gitignore` because of GitHub's size limits. None of it is needed
to run the demo.

| What | Size | Where to get it |
|---|---|---|
| `ShanghaiTech.zip` + `pipeline/data/ShanghaiTech/` | 2.3 GB | [ShanghaiTech crowd-counting dataset](https://www.kaggle.com/datasets/tthien/shanghaitech) — only needed to *retrain* MCNN |
| `Data/10287402-*.mp4` | 123 MB | A 4K stock clip; `Data/Test_data.mp4` is committed and is what everything was tested with |
| `pipeline/mcnn/yolo11m.pt`, `yolo11s.pt` | 58 MB | Unused — ultralytics downloads them on demand |
| `tube_data_csv/`, `sydney_data/`, `tube_data/` | 194 MB | Exploratory transit data, not used by any code here |

The trained MCNN weights (`pipeline/mcnn/checkpoints/best.pt`, ~500 KB) and the
BMRCL ridership data **are** committed, so the pipeline runs out of the box.

---

## Troubleshooting

**The Twin/app shows old data after I edit a file.**
Hard-reload once (`Ctrl+Shift+R`). Both static servers send `no-store`, but a
browser that already cached the page needs one forced reload to let go of it.
Use `serve.py`, never `python -m http.server` — the stdlib server caches and
this is exactly the trap it creates.

**The passenger app keeps jumping between stations.**
You have **more than one Digital Twin tab open**. They both publish to the same
topic and fight. The app now warns you when it detects this — close all but
one.

**Control Room says `BROKER DOWN`.**
Docker isn't running, or the broker container is stopped:
`docker compose -f pipeline/mqtt/docker-compose.yml up -d`.
Everything reconnects on its own once it's back — nothing needs restarting.

**Coaches start part-full.**
Retained MQTT state from the last session. Press **RESET OCCUPANCY**.

**The Twin looks blurry.**
It drops render quality automatically if the frame rate can't be held, and
recovers when it can. Resolution is the *last* thing it gives up — if it's
soft, the machine is genuinely struggling.

---

## Credits

- Crowd counting: [ShanghaiTech Part B](https://www.kaggle.com/datasets/tthien/shanghaitech) dataset
- Ridership data: [Vonter/bmrcl-ridership-hourly](https://github.com/Vonter/bmrcl-ridership-hourly) (RTI-sourced, ODbL 1.0)
- Significant dates: [thecont1/namma-metro-ridership-tracker](https://github.com/thecont1/namma-metro-ridership-tracker) (BSD-0)
- Crowd character models: [Quaternius](https://quaternius.com/) (CC0)
- Detection: [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics)
