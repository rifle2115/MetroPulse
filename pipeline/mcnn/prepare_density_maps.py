"""
Precomputes and caches ground-truth density maps as .npy files, once, so
training doesn't redo the scipy Gaussian filtering on every epoch. Run this
before train.py.
"""

import argparse
from pathlib import Path

import numpy as np
import scipy.io as sio
from PIL import Image

from density_map import points_to_density_map


def process_split(root: Path, split: str) -> None:
    img_dir = root / f"{split}_data" / "images"
    gt_dir = root / f"{split}_data" / "ground-truth"
    out_dir = root / f"{split}_data" / "density_maps"
    out_dir.mkdir(exist_ok=True)

    img_paths = sorted(img_dir.glob("*.jpg"))
    if not img_paths:
        raise FileNotFoundError(f"no images found in {img_dir}")

    made = 0
    for img_path in img_paths:
        out_path = out_dir / f"{img_path.stem}.npy"
        if out_path.exists():
            continue
        gt_path = gt_dir / f"GT_{img_path.stem}.mat"
        with Image.open(img_path) as img:
            w, h = img.size
        mat = sio.loadmat(gt_path)
        points = mat["image_info"][0, 0][0, 0][0]
        density = points_to_density_map(points, (h, w))
        np.save(out_path, density)
        made += 1

    print(f"{split}: {len(img_paths)} images, {made} density maps written -> {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="../data/ShanghaiTech/part_B")
    args = ap.parse_args()
    root = Path(args.data_root)

    process_split(root, "train")
    process_split(root, "test")


if __name__ == "__main__":
    main()
