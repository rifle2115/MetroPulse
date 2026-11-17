"""
Train the door-state classifier on the images build_dataset.py made.

    python train.py [--epochs 20]

MobileNetV3-small, starting from ImageNet weights, with a 3-way head:
no_train / doors_open / doors_closed. Trains on the `train` split, checks the
`val` split after every epoch, and keeps the epoch with the best validation
macro-F1 (the mean of the per-class F1s, so a rare class counts as much as a
common one). The `test` split is never touched here — evaluate.py scores it,
once, at the end.

Writes models/door_state.pt (what the Control Room loads) and
models/train_report.json (every epoch's numbers, for the report).
"""

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from common import CLASSES, DATASET_DEFAULT, MODEL_DEFAULT, print_confusion, scores
from door_state import MEAN, STD, build_model


class Crops(Dataset):
    def __init__(self, root: Path, rows: list, augment: bool):
        from torchvision import transforms as T
        self.root, self.rows = root, rows
        steps = []
        if augment:
            # lighting and a few pixels of camera drift: what really varies
            # between clips of one camera. No flips — door layout is not
            # mirror-symmetric on every camera.
            steps += [T.ColorJitter(0.35, 0.35, 0.25, 0.03),
                      T.RandomAffine(degrees=3, translate=(0.04, 0.04), scale=(0.93, 1.07))]
        steps += [T.ToTensor(), T.Normalize(MEAN.tolist(), STD.tolist())]
        self.tf = T.Compose(steps)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        from PIL import Image
        row = self.rows[i]
        bgr = cv2.imread(str(self.root / row["path"]))
        img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        return self.tf(img), CLASSES.index(row["class"])


def confusion(model, loader) -> np.ndarray:
    m = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            pred = model(x).argmax(1)
            for t, p in zip(y.tolist(), pred.tolist()):
                m[t, p] += 1
    return m


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=str(DATASET_DEFAULT))
    ap.add_argument("--out", default=str(MODEL_DEFAULT))
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    root, out = Path(args.dataset), Path(args.out)
    manifest = root / "manifest.csv"
    if not manifest.exists():
        print(f"No {manifest} — run build_dataset.py first.")
        return 1
    with open(manifest, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "val"]
    if not train_rows:
        print("No training images in the manifest.")
        return 1
    if not val_rows:
        print("warning: no validation images — the last epoch is kept, and there is no honest "
              "estimate of accuracy until you add `val` videos.")

    counts = np.array([sum(r["class"] == c for r in train_rows) for c in CLASSES], dtype=float)
    print("training images: " + ", ".join(f"{c} {int(n)}" for c, n in zip(CLASSES, counts)))
    if (counts == 0).any():
        print("warning: a class has no training images; the model cannot learn it")
    # rarer classes weigh more, so "no train" (most frames) cannot win by default
    weights = torch.tensor(np.where(counts > 0, counts.sum() / (len(CLASSES) * np.maximum(counts, 1)), 0.0),
                           dtype=torch.float32)

    train_loader = DataLoader(Crops(root, train_rows, augment=True), batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(Crops(root, val_rows, augment=False), batch_size=64) if val_rows else None

    try:
        model = build_model(pretrained=True)
    except Exception as exc:  # noqa: BLE001 — no internet for the weights download
        print(f"warning: could not load ImageNet weights ({exc}); training from scratch, "
              "which needs far more data")
        model = build_model(pretrained=False)

    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    history, best, best_state, best_conf = [], -1.0, None, None
    out.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        total, n = 0.0, 0
        for x, y in train_loader:
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            total += loss.item() * len(y)
            n += len(y)
        sched.step()
        entry = {"epoch": epoch, "train_loss": round(total / n, 4)}
        if val_loader:
            m = confusion(model, val_loader)
            entry.update({f"val_{k}": round(v, 4) for k, v in scores(m).items()})
            score = entry["val_macro_f1"]
        else:
            m, score = None, epoch      # no val: keep the latest
        history.append(entry)
        marker = ""
        if score > best:
            best, best_conf = score, m
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            marker = "  ← best"
        print(f"epoch {epoch:>2}/{args.epochs}  " + "  ".join(f"{k} {v}" for k, v in entry.items()
                                                              if k != "epoch")
              + f"  ({time.time() - t0:.0f}s){marker}")

    best_epoch = max(history, key=lambda e: e.get("val_macro_f1", e["epoch"]))["epoch"]
    torch.save({"state_dict": best_state, "classes": CLASSES, "arch": "mobilenet_v3_small",
                "epoch": best_epoch, "val_macro_f1": best if val_loader else None,
                "trained_at": datetime.now().isoformat(timespec="seconds")}, out)
    (out.parent / "train_report.json").write_text(json.dumps({
        "best_epoch": best_epoch,
        "val_confusion": best_conf.tolist() if best_conf is not None else None,
        "classes": CLASSES, "history": history,
        "train_counts": dict(zip(CLASSES, counts.astype(int).tolist())),
    }, indent=2), encoding="utf-8")

    print(f"\nsaved {out} (epoch {best_epoch})")
    if best_conf is not None:
        print(f"validation at that epoch: accuracy {scores(best_conf)['accuracy']:.3f}, "
              f"macro-F1 {scores(best_conf)['macro_f1']:.3f}\n")
        print_confusion(best_conf)
    print("\nNext: python evaluate.py — scores the held-out test videos.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
