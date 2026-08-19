"""
Loads a ShanghaiTech Part B split. Expects density_maps/ to already exist
(run prepare_density_maps.py first) — this class doesn't generate them
itself, so a missing cache fails loudly instead of silently regenerating
slow ground truth mid-epoch.
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class ShanghaiTechB(Dataset):
    def __init__(self, root, split):
        root = Path(root)
        self.img_dir = root / f"{split}_data" / "images"
        self.den_dir = root / f"{split}_data" / "density_maps"
        self.paths = sorted(self.img_dir.glob("*.jpg"))
        if not self.paths:
            raise FileNotFoundError(f"no images found in {self.img_dir}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        den_path = self.den_dir / f"{img_path.stem}.npy"
        if not den_path.exists():
            raise FileNotFoundError(f"{den_path} missing — run prepare_density_maps.py first")

        img = Image.open(img_path).convert("RGB")
        img_t = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)

        density = np.load(den_path)
        # crop to a multiple of 4 so the image and the model's 1/4-resolution
        # output line up, then downsample the density map by summing 4x4
        # blocks — summing (not averaging) preserves total head count
        h, w = density.shape
        h4, w4 = h - h % 4, w - w % 4
        density = density[:h4, :w4].reshape(h4 // 4, 4, w4 // 4, 4).sum(axis=(1, 3))
        den_t = torch.from_numpy(density).unsqueeze(0)
        img_t = img_t[:, :h4, :w4]

        return img_t, den_t
