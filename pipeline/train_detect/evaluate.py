"""
Score the trained door-state classifier on held-out videos.

    python evaluate.py [--split test] [--confirm 1.0 1.5 2.0]

Two results, both written to models/eval_<split>.json:

1. Per frame: accuracy, macro-F1 and the confusion matrix over that split's
   images in the dataset (made by build_dataset.py).
2. Per event: each video of the split is classified the way the Control Room
   does it (one frame every door_state.EVERY_S of video, against the clip's
   empty-track reference), and the answers are turned into arrivals and
   departures by the Control Room's own rules (door_state.PresenceTracker).
   Every detection is matched to a labelled event within --tolerance
   seconds. Reported: how many were found, missed, or fired with no train,
   and how late or early on average.

--confirm takes one or more hold times (seconds an answer must last before
it counts). Each clip is classified once and its answers replayed under every
value, so a sweep costs little more than a single run. Tune it on `val`;
then set door_state.CONFIRM_S to the chosen value and run `test` once, at
the end, for the numbers you report.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from common import (CAMERAS_DEFAULT, CLASSES, DATASET_DEFAULT, LABELS_DEFAULT, MODEL_DEFAULT,
                    VIDEOS_DEFAULT, find_video, load_cameras, load_labels, load_refs,
                    print_confusion, scores, true_events)
from door_state import CONFIRM_S, EVERY_S, DoorCropper, DoorStateModel, PresenceTracker


def frame_scores(model: DoorStateModel, root: Path, split: str, refs: dict):
    """Each clip's frames against its first empty-track reference — the one
    the event check uses too."""
    manifest = root / "manifest.csv"
    if not manifest.exists():
        return None
    with open(manifest, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == split and r["video"] in refs]
    if not rows:
        return None
    ref_imgs = {v: cv2.imread(str(root / paths[0])) for v, paths in refs.items()}
    m = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    for r in rows:
        label, _ = model.predict(cv2.imread(str(root / r["path"])), ref_imgs[r["video"]])
        m[CLASSES.index(r["class"]), CLASSES.index(label)] += 1
    return m


def video_answers(model, cal, path: Path, ref_crop):
    """The classifier's answer every EVERY_S of video, as (seconds, label) —
    the same sampling DoorStateDetector does live."""
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if fps <= 0:
        return None
    crop = DoorCropper(cal)
    answers, index, next_t = [], 0, 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = index / fps
        index += 1
        if t < next_t:
            continue
        next_t = t + EVERY_S
        label, _ = model.predict(crop(frame), ref_crop)
        answers.append((t, label))
    cap.release()
    return answers


def replay(answers: list, confirm_s: float) -> list:
    """Arrivals and departures from a clip's answers under one hold time."""
    tracker, found = PresenceTracker(confirm_s), []
    for t, label in answers:
        ev = tracker.observe(t, label)
        if ev:
            found.append((ev, round(t, 2)))
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


def summarise(results: list) -> dict:
    """results: [(pairs, missed, false_alarms)] per video -> per-kind totals."""
    pairs = [p for r in results for p in r[0]]
    missed = [m for r in results for m in r[1]]
    false = [f for r in results for f in r[2]]
    events = {}
    for kind in ("arrival", "departure"):
        errs = [d - t for k, t, d in pairs if k == kind]
        events[kind] = {
            "matched": len(errs),
            "missed": sum(k == kind for k, _ in missed),
            "false_alarms": sum(k == kind for k, _ in false),
            "mean_abs_error_s": round(float(np.mean(np.abs(errs))), 2) if errs else None,
            "mean_error_s": round(float(np.mean(errs)), 2) if errs else None,
        }
    return events


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
    ap.add_argument("--confirm", type=float, nargs="+", default=[CONFIRM_S],
                    help=f"hold time(s) in seconds to score (default {CONFIRM_S}, the live value)")
    args = ap.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"No model at {model_path} — run train.py first.")
        return 1
    model = DoorStateModel(model_path)
    report = {"split": args.split, "model": model.info}

    refs = load_refs(Path(args.dataset))
    print(f"=== {args.split}: per frame ===")
    m = frame_scores(model, Path(args.dataset), args.split, refs)
    if m is None:
        print(f"no {args.split} images in the dataset")
    else:
        s = scores(m)
        print(f"accuracy {s['accuracy']:.3f}, macro-F1 {s['macro_f1']:.3f}, {int(m.sum())} images\n")
        print_confusion(m)
        report["frames"] = {**s, "images": int(m.sum()), "confusion": m.tolist(), "classes": CLASSES}

    # classify every clip once
    labels = load_labels(args.labels)
    cameras = load_cameras(Path(args.cameras))
    clips = []
    for video, info in sorted(labels.items()):
        if info["split"] != args.split:
            continue
        cal = cameras.get(info["camera"])
        path = find_video(Path(args.videos), info["split"], video)
        if cal is None or cal.door_crop is None or path is None:
            print(f"  skip {video}: missing file, camera or door crop")
            continue
        if video not in refs:
            print(f"  skip {video}: no empty-track reference in the dataset")
            continue
        ref_crop = cv2.imread(str(Path(args.dataset) / refs[video][0]))
        answers = video_answers(model, cal, path, ref_crop)
        if answers is None:
            print(f"  skip {video}: unreadable")
            continue
        clips.append((video, true_events(info["trains"]), answers))

    sweep = []
    for confirm in args.confirm:
        results, per_video = [], []
        for video, truth, answers in clips:
            found = replay(answers, confirm)
            pairs, missed, false_alarms = match(truth, found, args.tolerance)
            results.append((pairs, missed, false_alarms))
            per_video.append({"video": video, "found": found, "matched": pairs,
                              "missed": missed, "false_alarms": false_alarms})
        sweep.append({"confirm_s": confirm, "events": summarise(results), "videos": per_video})

    if len(sweep) == 1:
        run = sweep[0]
        print(f"\n=== {args.split}: arrivals and departures "
              f"(hold {run['confirm_s']:g} s, tolerance {args.tolerance:g} s) ===")
        for v in run["videos"]:
            print(f"  {v['video']:<16} matched {len(v['matched'])}, missed {len(v['missed'])}, "
                  f"false alarms {len(v['false_alarms'])}"
                  + "".join(f"\n      {k:<9} true {t:>6.1f}s  detected {d:>6.1f}s  ({d - t:+.1f}s)"
                            for k, t, d in v["matched"]))
        for kind, e in run["events"].items():
            print(f"\n{kind}s: {e['matched']} matched, {e['missed']} missed, "
                  f"{e['false_alarms']} false alarms"
                  + (f"; mean |error| {e['mean_abs_error_s']} s (mean {e['mean_error_s']:+} s, "
                     f"+ = late)" if e["matched"] else ""))
    else:
        print(f"\n=== {args.split}: hold-time sweep (tolerance {args.tolerance:g} s) ===")
        print(f"{'hold':>6} | {'found':>9} {'missed':>7} {'false':>6} | "
              f"{'arr |err|':>9} {'dep |err|':>9}")
        for run in sweep:
            a, d = run["events"]["arrival"], run["events"]["departure"]
            found = a["matched"] + d["matched"]
            total = found + a["missed"] + d["missed"]
            fmt = lambda x: f"{x:.2f} s" if x is not None else "—"  # noqa: E731
            print(f"{run['confirm_s']:>5g}s | {found:>4}/{total:<4} {a['missed'] + d['missed']:>7} "
                  f"{a['false_alarms'] + d['false_alarms']:>6} | "
                  f"{fmt(a['mean_abs_error_s']):>9} {fmt(d['mean_abs_error_s']):>9}")
        print("\nper clip (found / missed / false), one column per hold time:")
        for i, (video, _, _) in enumerate(clips):
            cells = [f"{len(r['videos'][i]['matched'])}/{len(r['videos'][i]['missed'])}/"
                     f"{len(r['videos'][i]['false_alarms'])}" for r in sweep]
            print(f"  {video:<14} " + "  ".join(f"{c:>8}" for c in cells))

    report["sweep"] = sweep
    out = model_path.parent / f"eval_{args.split}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nreport written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
