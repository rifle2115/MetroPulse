"""
Turn the labelled videos into door-crop images the classifier trains on.

    python build_dataset.py --videos "E:\WuhanMetroCrowd-Final"

Labels come from Data/train_labels.csv and Data/val_test_labels.csv (rows
still marked TODO are skipped); --labels names other files instead.

--videos is either a folder holding the videos, or the dataset root with its
train/, val/ and test/ folders (each video is looked up in its split's
video_format/ folder).

For every video in the labels file: read it start to end, take one frame
every --every seconds of video, cut out that camera's door crop, label it
from the times in the labels file, and save it as
dataset/<split>/<class>/<video>_<frame>.png, plus one manifest.csv row.

Two kinds of frame are left out:
  - frames within a couple of seconds of a train arriving or leaving, or the
    doors opening or closing (common.label_at) — half-in trains are not
    safely either class;
  - repeats. The clips are CCTV stills stretched to video, so one still sits
    on screen for many frames; saving every copy would make a few stills look
    like a large dataset and weight them many times over.

Empty-track references. The model compares each crop with the same camera's
crop of the empty track (door_state.py). For every clip, N_REFS of its
"no train" crops, spread across the clip, are kept as that clip's references
(dataset/refs/, listed in refs.csv) — what the operator's one "NO TRAIN"
still is in use. Those moments, and REF_EXCLUDE_S either side, are taken out
of the scored images, so no frame is ever judged against itself. A clip with
no empty moment at all uses its camera's saved "NO TRAIN" still instead.
"""

import argparse
import csv
import shutil
import sys
from collections import Counter

import cv2
import numpy as np

from common import (CAMERA_FRAMES_DIR, CAMERAS_DEFAULT, DATASET_DEFAULT, LABELS_DEFAULT,
                    VIDEOS_DEFAULT, CLASSES, SPLITS, find_video, label_at, load_cameras,
                    load_labels)
from door_state import DoorCropper

MIN_CHANGE = 1.5     # mean grey-level change, on a 64x32 thumbnail, for a new still
N_REFS = 3           # empty-track references kept per clip
REF_EXCLUDE_S = 2.0  # scored images this close to a reference are dropped


def pick_refs(samples: list) -> list:
    """Up to N_REFS "no train" samples spread evenly across the clip."""
    empty = [s for s in samples if s["label"] == "no_train"]
    if not empty:
        return []
    if len(empty) <= N_REFS:
        return empty
    step = (len(empty) - 1) / (N_REFS - 1)
    return [empty[round(i * step)] for i in range(N_REFS)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", nargs="+", default=[str(p) for p in LABELS_DEFAULT])
    ap.add_argument("--videos", default=str(VIDEOS_DEFAULT))
    ap.add_argument("--cameras", default=str(CAMERAS_DEFAULT))
    ap.add_argument("--out", default=str(DATASET_DEFAULT))
    ap.add_argument("--every", type=float, default=0.5, help="seconds of video between samples")
    ap.add_argument("--skip-missing", action="store_true",
                    help="skip videos whose file or door crop is missing instead of stopping")
    args = ap.parse_args()

    from pathlib import Path
    videos_dir, out = Path(args.videos), Path(args.out)
    labels = load_labels(args.labels)
    cameras = load_cameras(Path(args.cameras))

    # check everything before reading a single frame
    missing = []
    for video, info in labels.items():
        cal = cameras.get(info["camera"])
        if find_video(videos_dir, info["split"], video) is None:
            missing.append(f"{video}: not found in {videos_dir} "
                           f"(nor in {videos_dir / info['split'] / 'video_format'})")
        elif cal is None:
            missing.append(f"{video}: camera {info['camera']!r} is not saved on /calibrate")
        elif cal.door_crop is None:
            missing.append(f"{video}: camera {info['camera']!r} has no door crop (calibrate, step 4)")
    if missing and not args.skip_missing:
        print("Cannot build the dataset yet:\n  " + "\n  ".join(missing))
        print("\nFix these, or pass --skip-missing to build from the rest.")
        return 1

    if out.exists():
        # ours = only the split folders (+ manifest.csv). A run stopped part
        # way leaves the folders without the manifest, which is written last.
        foreign = [p.name for p in out.iterdir()
                   if p.name not in (*SPLITS, "refs", "manifest.csv", "refs.csv")]
        if foreign:
            print(f"{out} has files in it that this script did not make ({', '.join(foreign[:5])}); "
                  f"pick another --out.")
            return 1
        shutil.rmtree(out)
    for split in SPLITS:
        for cls in CLASSES:
            (out / split / cls).mkdir(parents=True, exist_ok=True)
    (out / "refs").mkdir(parents=True, exist_ok=True)

    rows, ref_rows = [], []
    counts = Counter()
    skipped_boundary = skipped_repeat = skipped_ref = 0
    no_ref = []
    for video, info in sorted(labels.items()):
        cal = cameras.get(info["camera"])
        path = find_video(videos_dir, info["split"], video)
        if cal is None or cal.door_crop is None or path is None:
            print(f"  skip {video}")
            continue
        crop = DoorCropper(cal)
        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        if fps <= 0:
            print(f"  skip {video}: unreadable frame rate")
            continue
        stem = Path(video).stem

        # pass 1: the clip's samples, in memory (a crop is ~100 KB)
        samples = []
        index, next_t, last_thumb, last_label = 0, 0.0, None, None
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = index / fps
            index += 1
            if t < next_t:
                continue
            next_t += args.every
            label = label_at(t, info["trains"])
            if label is None:
                skipped_boundary += 1
                continue
            img = crop(frame)
            thumb = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (64, 32)).astype(np.float32)
            if (last_thumb is not None and label == last_label
                    and float(np.mean(np.abs(thumb - last_thumb))) < MIN_CHANGE):
                skipped_repeat += 1
                continue
            last_thumb, last_label = thumb, label
            samples.append({"frame": index - 1, "t": t, "label": label, "img": img})
        cap.release()

        # the clip's empty-track references
        refs = pick_refs(samples)
        ref_imgs = [(f"{stem}_{r['frame']:06d}.png", r["img"]) for r in refs]
        if not ref_imgs:
            saved = cv2.imread(str(CAMERA_FRAMES_DIR / f"{info['camera']}_empty.jpg"))
            if saved is not None:
                ref_imgs = [(f"{stem}_camera_empty.png", crop(saved))]
        if not ref_imgs:
            no_ref.append(video)
            print(f"  skip {video}: no empty-track moment, and camera {info['camera']!r} has no "
                  f"NO TRAIN still")
            continue
        for name, img in ref_imgs:
            cv2.imwrite(str(out / "refs" / name), img)
            ref_rows.append({"video": video, "path": f"refs/{name}"})
        ref_times = [r["t"] for r in refs]

        # pass 2: write the scored images, minus those next to a reference
        kept = 0
        for s in samples:
            if any(abs(s["t"] - rt) < REF_EXCLUDE_S for rt in ref_times):
                skipped_ref += 1
                continue
            rel = Path(info["split"]) / s["label"] / f"{stem}_{s['frame']:06d}.png"
            cv2.imwrite(str(out / rel), s["img"])
            rows.append({"split": info["split"], "class": s["label"], "video": video,
                         "camera": info["camera"], "frame": s["frame"], "t": round(s["t"], 2),
                         "path": rel.as_posix()})
            counts[(info["split"], s["label"])] += 1
            kept += 1
        print(f"  {video:<16} camera {info['camera']:<6} {info['split']:<5} {kept:>4} images, "
              f"{len(ref_imgs)} reference(s)")

    with open(out / "refs.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["video", "path"])
        w.writeheader()
        w.writerows(ref_rows)
    with open(out / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["split", "class", "video", "camera", "frame", "t", "path"])
        w.writeheader()
        w.writerows(rows)

    print(f"\n{'split':<6}" + "".join(f"{c:>14}" for c in CLASSES) + f"{'total':>8}")
    for split in SPLITS:
        n = [counts[(split, c)] for c in CLASSES]
        print(f"{split:<6}" + "".join(f"{x:>14}" for x in n) + f"{sum(n):>8}")
    print(f"\nleft out: {skipped_boundary} near a change, {skipped_repeat} repeated stills, "
          f"{skipped_ref} next to a reference")
    if no_ref:
        print(f"no reference, skipped: {', '.join(no_ref)}")
    print(f"dataset written to {out}")
    if not any(counts[("val", c)] for c in CLASSES):
        print("note: no validation images — add `val` rows to the labels file before training")
    return 0


if __name__ == "__main__":
    sys.exit(main())
