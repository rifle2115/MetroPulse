"""
Score the trained door-state classifier on held-out videos.

    python evaluate.py [--split test]

Two results, both written to models/eval_<split>.json:

1. Per frame: accuracy, macro-F1 and the confusion matrix over that split's
   images in the dataset (made by build_dataset.py).
2. Per event: each video of the split is played through the exact detector
   the Control Room uses (door_state.DoorStateDetector, with its smoothing),
   and every detected arrival and departure is matched to a labelled one
   within --tolerance seconds. Reported: how many were found, missed, or
   fired with no train, and how late or early on average.

Run this on the test split once, at the end, for the numbers you report.
Tune on `val` (python evaluate.py --split val), never on `test`.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from common import (CAMERAS_DEFAULT, CLASSES, DATASET_DEFAULT, LABELS_DEFAULT, MODEL_DEFAULT,
                    VIDEOS_DEFAULT, find_video, load_cameras, load_labels, print_confusion,
                    scores, true_events)
from door_state import DoorStateDetector, DoorStateModel


def frame_scores(model: DoorStateModel, root: Path, split: str):
    manifest = root / "manifest.csv"
    if not manifest.exists():
        return None
    with open(manifest, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == split]
    if not rows:
        return None
    m = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    for r in rows:
        label, _ = model.predict(cv2.imread(str(root / r["path"])))
        m[CLASSES.index(r["class"]), CLASSES.index(label)] += 1
    return m


def video_events(model, cal, path: Path):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if fps <= 0:
        return None
    det = DoorStateDetector(cal, fps, model)
    found, index = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        ev = det.update(frame, index)
        if ev:
            found.append((ev, round(index / fps, 2)))
        index += 1
    cap.release()
    return found


def match(truth: list, found: list, tolerance: float):
    """Greedy: each labelled event takes the nearest unused detection of the
    same kind within tolerance."""
    used, pairs, missed = set(), [], []
    for kind, t in truth:
        best = None
        for i, (k, ft) in enumerate(found):
            if i in used or k != kind or abs(ft - t) > tolerance:
                continue
            if best is None or abs(ft - t) < abs(found[best][1] - t):
                best = i
        if best is None:
            missed.append((kind, t))
        else:
            used.add(best)
            pairs.append((kind, t, found[best][1]))
    false_alarms = [f for i, f in enumerate(found) if i not in used]
    return pairs, missed, false_alarms


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--model", default=str(MODEL_DEFAULT))
    ap.add_argument("--dataset", default=str(DATASET_DEFAULT))
    ap.add_argument("--labels", nargs="+", default=[str(p) for p in LABELS_DEFAULT])
    ap.add_argument("--videos", default=str(VIDEOS_DEFAULT))
    ap.add_argument("--cameras", default=str(CAMERAS_DEFAULT))
    ap.add_argument("--tolerance", type=float, default=10.0,
                    help="seconds a detection may be off and still count")
    args = ap.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"No model at {model_path} — run train.py first.")
        return 1
    model = DoorStateModel(model_path)
    report = {"split": args.split, "model": model.info}

    print(f"=== {args.split}: per frame ===")
    m = frame_scores(model, Path(args.dataset), args.split)
    if m is None:
        print(f"no {args.split} images in the dataset")
    else:
        s = scores(m)
        print(f"accuracy {s['accuracy']:.3f}, macro-F1 {s['macro_f1']:.3f}, {int(m.sum())} images\n")
        print_confusion(m)
        report["frames"] = {**s, "images": int(m.sum()), "confusion": m.tolist(), "classes": CLASSES}

    print(f"\n=== {args.split}: arrivals and departures (tolerance {args.tolerance:g} s) ===")
    labels = load_labels(args.labels)
    cameras = load_cameras(Path(args.cameras))
    all_pairs, all_missed, all_false, per_video = [], [], [], []
    for video, info in sorted(labels.items()):
        if info["split"] != args.split:
            continue
        cal = cameras.get(info["camera"])
        path = find_video(Path(args.videos), info["split"], video)
        if cal is None or cal.door_crop is None or path is None:
            print(f"  skip {video}: missing file, camera or door crop")
            continue
        found = video_events(model, cal, path)
        if found is None:
            print(f"  skip {video}: unreadable")
            continue
        pairs, missed, false_alarms = match(true_events(info["trains"]), found, args.tolerance)
        all_pairs += pairs
        all_missed += missed
        all_false += false_alarms
        per_video.append({"video": video, "found": found, "matched": pairs,
                          "missed": missed, "false_alarms": false_alarms})
        print(f"  {video:<16} matched {len(pairs)}, missed {len(missed)}, "
              f"false alarms {len(false_alarms)}"
              + "".join(f"\n      {k:<9} true {t:>6.1f}s  detected {d:>6.1f}s  ({d - t:+.1f}s)"
                        for k, t, d in pairs))

    events = {}
    for kind in ("arrival", "departure"):
        errs = [d - t for k, t, d in all_pairs if k == kind]
        events[kind] = {
            "matched": len(errs),
            "missed": sum(k == kind for k, _ in all_missed),
            "false_alarms": sum(k == kind for k, _ in all_false),
            "mean_abs_error_s": round(float(np.mean(np.abs(errs))), 2) if errs else None,
            "mean_error_s": round(float(np.mean(errs)), 2) if errs else None,
        }
        e = events[kind]
        print(f"\n{kind}s: {e['matched']} matched, {e['missed']} missed, "
              f"{e['false_alarms']} false alarms"
              + (f"; mean |error| {e['mean_abs_error_s']} s (mean {e['mean_error_s']:+} s, "
                 f"+ = late)" if errs else ""))
    report["events"] = events
    report["videos"] = per_video

    out = model_path.parent / f"eval_{args.split}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nreport written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
