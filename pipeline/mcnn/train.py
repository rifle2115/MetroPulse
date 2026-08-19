"""
MCNN training on ShanghaiTech Part B. Runs on CPU or GPU automatically
(same script for a local CPU run or a Colab GPU session). Batch size 1 —
images vary in size, and padding/cropping to a fixed batch shape isn't worth
the complexity for a Phase-I proof of concept on 400 training images.
Run prepare_density_maps.py first.
"""

import argparse
import csv
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import ShanghaiTechB
from model import MCNN


def evaluate(model, loader, device):
    model.eval()
    mae, mse, n = 0.0, 0.0, 0
    with torch.no_grad():
        for img, den in loader:
            img, den = img.to(device), den.to(device)
            pred = model(img)
            pred_count = pred.sum().item()
            gt_count = den.sum().item()
            mae += abs(pred_count - gt_count)
            mse += (pred_count - gt_count) ** 2
            n += 1
    model.train()
    return mae / n, (mse / n) ** 0.5


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="../data/ShanghaiTech/part_B")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--out-dir", default="./checkpoints")
    ap.add_argument("--resume", action="store_true",
                     help="continue from checkpoints/resume.pt if it exists")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)
    train_set = ShanghaiTechB(args.data_root, "train")
    test_set = ShanghaiTechB(args.data_root, "test")
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    model = MCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "metrics.csv"
    resume_path = out_dir / "resume.pt"
    best_mae = float("inf")
    start_epoch = 1

    # a laptop can sleep, lose power, or just get closed mid-run — checkpoint
    # every epoch (not just on eval improvement) so at most one epoch of
    # progress is ever at risk, and --resume can pick back up from it
    if args.resume and resume_path.exists():
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        best_mae = ckpt["best_mae"]
        start_epoch = ckpt["epoch"] + 1
        print(f"resuming from epoch {start_epoch} (best_mae so far: {best_mae:.2f})", flush=True)
    else:
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "train_loss", "mae", "rmse", "seconds"])

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        total_loss = 0.0
        for img, den in train_loader:
            img, den = img.to(device), den.to(device)
            opt.zero_grad()
            pred = model(img)
            loss = loss_fn(pred, den)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        elapsed = time.time() - t0

        mae = rmse = float("nan")
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            mae, rmse = evaluate(model, test_loader, device)
            if mae < best_mae:
                best_mae = mae
                torch.save(model.state_dict(), out_dir / "best.pt")

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{avg_loss:.4f}", f"{mae:.2f}", f"{rmse:.2f}", f"{elapsed:.1f}"])
        print(f"epoch {epoch:3d}/{args.epochs}  loss={avg_loss:.4f}  mae={mae:.2f}  rmse={rmse:.2f}  {elapsed:.1f}s", flush=True)

        torch.save({
            "epoch": epoch, "model": model.state_dict(),
            "optimizer": opt.state_dict(), "best_mae": best_mae,
        }, resume_path)

    torch.save(model.state_dict(), out_dir / "last.pt")


if __name__ == "__main__":
    main()
