"""
Train the train / no-train classifier on the images build_dataset.py made.

    python train.py [--epochs 15]

MobileNetV3-small from ImageNet weights, shown each door crop together with
an empty-track crop of the same camera (door_state.py). Trains on the `train`
split, checks `val` after every epoch, keeps the epoch with the best
validation macro-F1, and stops once validation has not improved for
--patience epochs: the first model peaked at epoch 1 and only got worse as it
memorised the training cameras. The `test` split is never touched here —
evaluate.py scores it, once, at the end.

Writes models/door_state.pt (what the Control Room loads) and
models/train_report.json (every epoch's numbers, for the report).
"""

import argparse
import csv
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from common import CLASSES, DATASET_DEFAULT, MODEL_DEFAULT, load_refs, print_confusion, scores
from door_state import IN_CHANNELS, MEAN, STD, build_model


class Pairs(Dataset):
    """(current crop, this clip's empty-track crop) -> class.

    Augmentation moves both crops by the same small affine — they come from
    one fixed camera, so they always line up — and changes their lighting
    independently, because the reference is taken at a different time of day
    from the frame it is compared with. Training draws a random one of the
    clip's references each time; validation always uses the first."""

    def __init__(self, root: Path, rows: list, refs: dict, augment: bool):
        self.root, self.rows, self.refs, self.augment = root, rows, refs, augment

    def __len__(self):
        return len(self.rows)

    def _load(self, rel: str):
        from PIL import Image
        bgr = cv2.imread(str(self.root / rel))
        return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    def __getitem__(self, i):
        from torchvision import transforms as T
        from torchvision.transforms import functional as F
        row = self.rows[i]
        refs = self.refs[row["video"]]
        cur = self._load(row["path"])
        ref = self._load(random.choice(refs) if self.augment else refs[0])
        if self.augment:
            angle, translate, scale, shear = T.RandomAffine.get_params(
                (-3, 3), (0.04, 0.04), (0.93, 1.07), None, list(cur.size))
            cur = F.affine(cur, angle, list(translate), scale, list(shear))
            ref = F.affine(ref, angle, list(translate), scale, list(shear))
            jitter = T.ColorJitter(0.45, 0.4, 0.3, 0.03)
            cur, ref = jitter(cur), jitter(ref)
            if random.random() < 0.2:
                cur, ref = F.to_grayscale(cur, 3), F.to_grayscale(ref, 3)
        norm = T.Compose([T.ToTensor(), T.Normalize(MEAN.tolist(), STD.tolist())])
        return torch.cat([norm(cur), norm(ref)], dim=0), CLASSES.index(row["class"])


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
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--patience", type=int, default=4,
                    help="stop after this many epochs without a better validation score")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    root, out = Path(args.dataset), Path(args.out)
    manifest = root / "manifest.csv"
    if not manifest.exists() or not (root / "refs.csv").exists():
        print(f"No dataset with references in {root} — run build_dataset.py first.")
        return 1
    with open(manifest, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    refs = load_refs(root)
    unreferenced = sorted({r["video"] for r in rows if r["video"] not in refs})
    if unreferenced:
        print(f"skipping images of clips with no reference: {', '.join(unreferenced)}")
        rows = [r for r in rows if r["video"] in refs]
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

    train_loader = DataLoader(Pairs(root, train_rows, refs, augment=True), batch_size=args.batch,
                              shuffle=True)
    val_loader = (DataLoader(Pairs(root, val_rows, refs, augment=False), batch_size=64)
                  if val_rows else None)

    try:
        model = build_model(pretrained=True)
    except Exception as exc:  # noqa: BLE001 — no internet for the weights download
        print(f"warning: could not load ImageNet weights ({exc}); training from scratch, "
              "which needs far more data")
        model = build_model(pretrained=False)

    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    history, best, best_state, best_conf, best_epoch, stale = [], -1.0, None, None, 0, 0
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
            best, best_conf, best_epoch, stale = score, m, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            marker = "  ← best"
        else:
            stale += 1
        print(f"epoch {epoch:>2}/{args.epochs}  " + "  ".join(f"{k} {v}" for k, v in entry.items()
                                                              if k != "epoch")
              + f"  ({time.time() - t0:.0f}s){marker}")
        if val_loader and stale >= args.patience:
            print(f"no improvement for {args.patience} epochs — stopping")
            break

    torch.save({"state_dict": best_state, "classes": CLASSES, "in_channels": IN_CHANNELS,
                "arch": "mobilenet_v3_small+empty_reference",
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
    print("\nNext: python evaluate.py --split val")
    return 0


if __name__ == "__main__":
    sys.exit(main())
