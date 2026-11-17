"""
Shared pieces for the door-state training scripts: where things live, how
the labels file is read, and what a frame at time t is labelled.

The labels files (Data/train_labels.csv, Data/val_test_labels.csv) have one
row per train:

    video,camera,split,arrive_s,doors_open_s,doors_close_s,depart_s,notes

Times are seconds into the video. arrive_s = 0 means the clip opens with the
train already in; an empty depart_s means the clip ends before it leaves.
Everything outside a train's arrive..depart span is "no train".

Two special rows:
  - notes = TODO: not labelled yet, skipped (the template ships that way);
  - a camera, all four times empty and notes = "no train": a clip in which no
    train ever appears — every frame of it is "no train".
"""

import csv
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
# appended, not prepended: pipeline/mcnn has its own train.py (the MCNN
# trainer), and this folder's train.py must be the one `import train` finds
sys.path.append(str(REPO / "pipeline" / "mcnn"))

from camera_calibration import CalibrationStore  # noqa: E402
from door_state import CLASSES  # noqa: E402

LABELS_DEFAULT = [REPO / "Data" / "train_labels.csv", REPO / "Data" / "val_test_labels.csv"]
# The videos are large and usually live outside the repo (the WuhanMetroCrowd
# set on an external drive): pass --videos, or set METROPULSE_VIDEOS once.
VIDEOS_DEFAULT = Path(os.environ.get("METROPULSE_VIDEOS", REPO / "Data" / "train_videos"))
CAMERAS_DEFAULT = REPO / "pipeline" / "control_room" / "cameras.json"
# where /calibrate keeps each camera's stills, including <id>_empty.jpg (its
# "THIS FRAME = NO TRAIN" reference)
CAMERA_FRAMES_DIR = REPO / "pipeline" / "control_room" / "camera_frames"
DATASET_DEFAULT = HERE / "dataset"
MODEL_DEFAULT = HERE / "models" / "door_state.pt"

SPLITS = ("train", "val", "test")
# Frames this close to a train arriving or leaving are left out of training:
# around those moments the train is half in, and either label would be a
# guess that teaches the model a contradiction.
BOUNDARY_MARGIN_S = 2.0
# The door times stay in the labels file (they could train a door model
# later) but no longer split the classes, so frames near the doors moving are
# kept: a train with its doors mid-way is still a train.


@dataclass
class Train:
    arrive: float | None
    doors_open: float | None
    doors_close: float | None
    depart: float | None


def _num(text: str) -> float | None:
    text = (text or "").strip()
    return float(text) if text else None


def load_labels(paths) -> dict:
    """{video: {"camera": str, "split": str, "trains": [Train]}} from one or
    more labels files (a missing file is skipped), checked: every row of a
    video names the same camera and split, splits are train/val/test, and
    each train's times run in order. TODO rows are skipped."""
    if isinstance(paths, (str, Path)):
        paths = [paths]
    videos: dict = {}
    problems = []
    todo = 0
    for path in map(Path, paths):
        if not path.exists():
            continue
        with open(path, newline="", encoding="utf-8-sig") as f:
            for n, row in enumerate(csv.DictReader(f), start=2):
                where = f"{path.name} line {n}"
                video = (row.get("video") or "").strip()
                if not video:
                    continue
                notes = (row.get("notes") or "").strip()
                if notes.upper().startswith("TODO"):
                    todo += 1
                    continue
                camera = (row.get("camera") or "").strip()
                split = (row.get("split") or "").strip().lower()
                if split not in SPLITS:
                    problems.append(f"{where}: split {split!r} is not one of {SPLITS}")
                if not camera:
                    problems.append(f"{where}: no camera for {video}")
                tr = Train(_num(row.get("arrive_s")), _num(row.get("doors_open_s")),
                           _num(row.get("doors_close_s")), _num(row.get("depart_s")))
                times = [x for x in (tr.arrive, tr.doors_open, tr.doors_close, tr.depart)
                         if x is not None]
                no_train = not times
                if no_train and notes.lower() != "no train":
                    problems.append(f"{where}: {video} has no times — fill them in, or write "
                                    f"notes = no train if no train appears in the clip")
                if times != sorted(times):
                    problems.append(f"{where}: times of {video} are not in order "
                                    f"(arrive <= doors open <= doors close <= depart)")
                info = videos.setdefault(video, {"camera": camera, "split": split, "trains": []})
                if info["camera"] != camera or info["split"] != split:
                    problems.append(f"{where}: {video} has a different camera or split than its "
                                    f"earlier rows")
                if not no_train:
                    info["trains"].append(tr)
    if problems:
        raise ValueError("labels file problems:\n  " + "\n  ".join(problems))
    if todo:
        print(f"({todo} label row(s) still marked TODO — skipped)")
    for info in videos.values():
        info["trains"].sort(key=lambda tr: tr.arrive if tr.arrive is not None else 0.0)
    return videos


def label_at(t: float, trains: list) -> str | None:
    """The class of the frame at t seconds, or None when t is too close to a
    train arriving or leaving to label safely."""
    for tr in trains:
        arrive = tr.arrive or 0.0
        depart = tr.depart if tr.depart is not None else math.inf
        # an arrival at 0 is the clip opening, not a moment of change
        for b in (tr.arrive if tr.arrive else None, tr.depart):
            if b is not None and abs(t - b) < BOUNDARY_MARGIN_S:
                return None
        if arrive <= t <= depart:
            return "train"
    return "no_train"


def true_events(trains: list) -> list:
    """The arrivals and departures that actually happen inside the clip, as
    (kind, seconds) — what event detection is scored against."""
    events = []
    for tr in trains:
        if tr.arrive:                       # not 0 / not missing
            events.append(("arrival", tr.arrive))
        if tr.depart is not None:
            events.append(("departure", tr.depart))
    return sorted(events, key=lambda e: e[1])


def find_video(root: Path, split: str, name: str) -> Path | None:
    """A labelled video under --videos: either straight in that folder, or in
    the dataset's own layout, <root>/<split>/video_format/<name> (train_*.mp4
    under train/, val_*.mp4 under val/, test_*.mp4 under test/)."""
    for path in (root / name, root / split / "video_format" / name, root / split / name):
        if path.exists():
            return path
    return None


def scores(m) -> dict:
    """Accuracy and macro-F1 from a confusion matrix (rows true, columns
    predicted), over the classes that appear in that split."""
    import numpy as np
    f1s = []
    for k in range(len(CLASSES)):
        tp, support, predicted = m[k, k], m[k].sum(), m[:, k].sum()
        if support == 0:
            continue
        precision = tp / predicted if predicted else 0.0
        recall = tp / support
        f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    total = m.sum()
    return {"accuracy": float(np.trace(m) / total) if total else 0.0,
            "macro_f1": float(np.mean(f1s)) if f1s else 0.0}


def print_confusion(m) -> None:
    w = max(len(c) for c in CLASSES) + 2
    print(" " * w + "predicted →")
    print(f"{'true ↓':<{w}}" + "".join(f"{c:>{w}}" for c in CLASSES))
    for k, c in enumerate(CLASSES):
        print(f"{c:<{w}}" + "".join(f"{v:>{w}}" for v in m[k]))


def load_refs(dataset: Path) -> dict:
    """{video: [reference crop paths]} from the dataset's refs.csv."""
    refs: dict = {}
    path = Path(dataset) / "refs.csv"
    if path.exists():
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                refs.setdefault(row["video"], []).append(row["path"])
    return refs


def load_cameras(path: Path) -> CalibrationStore:
    return CalibrationStore(path)


assert CLASSES == ["no_train", "train"]
