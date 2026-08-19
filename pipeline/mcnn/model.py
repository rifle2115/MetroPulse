"""
Multi-Column CNN (Zhang et al., "Single-Image Crowd Counting via Multi-Column
Convolutional Neural Network", CVPR 2016). Three parallel columns with
different receptive field sizes (large/medium/small filters) capture heads at
different scales from perspective distortion, fused into a single-channel
density map. Two stride-2 maxpools per column mean the output is 1/4 the
input resolution in each dimension — the ground truth density maps built by
prepare_density_maps.py are downsampled to match.
"""

import torch
import torch.nn as nn


class _Column(nn.Module):
    def __init__(self, c1, c2, c3, c4, k1, k2, k3, k4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, c1, k1, padding=k1 // 2), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(c1, c2, k2, padding=k2 // 2), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(c2, c3, k3, padding=k3 // 2), nn.ReLU(inplace=True),
            nn.Conv2d(c3, c4, k4, padding=k4 // 2), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class MCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.large = _Column(16, 32, 16, 8, 9, 7, 7, 7)
        self.medium = _Column(20, 40, 20, 10, 7, 5, 5, 5)
        self.small = _Column(24, 48, 24, 12, 5, 3, 3, 3)
        self.fuse = nn.Conv2d(8 + 10 + 12, 1, kernel_size=1)

    def forward(self, x):
        feats = torch.cat([self.large(x), self.medium(x), self.small(x)], dim=1)
        return self.fuse(feats)
