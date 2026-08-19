"""
Ground-truth density map generation from head-point annotations. Part B is
street scenes with sparse, relatively uniform crowds (unlike Part A's dense,
perspective-heavy scenes), so the original MCNN paper uses one fixed-sigma
Gaussian per head rather than Part A's geometry-adaptive (kNN-distance)
kernel — sigma=15 is the value used across common reference implementations
for Part B.
"""

import numpy as np
from scipy.ndimage import gaussian_filter

SIGMA = 15.0


def points_to_density_map(points, shape, sigma=SIGMA):
    """points: (N, 2) array of (x, y) head locations. shape: (h, w).
    Summing a delta per point then blurring once is equivalent to summing N
    individually-blurred Gaussians, and is far cheaper than N separate
    convolutions over the full image."""
    density = np.zeros(shape, dtype=np.float32)
    h, w = shape
    if points is None or len(points) == 0:
        return density
    for x, y in points:
        xi = min(w - 1, max(0, int(round(float(x)))))
        yi = min(h - 1, max(0, int(round(float(y)))))
        density[yi, xi] += 1.0
    # mode='constant' truncates the kernel at the image border, so a small
    # amount of mass from heads near the edge is lost rather than reflected
    # back in — matches standard practice in MCNN reference implementations.
    return gaussian_filter(density, sigma=sigma, mode="constant")
