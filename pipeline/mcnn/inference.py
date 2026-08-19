"""
Shared MCNN inference pieces — frame preprocessing, zone splitting, model
loading, and frame sourcing. Used by both zone_telemetry_publisher.py (CLI)
and control_room/server.py (web backend) so there's one source of truth for
how a frame becomes zone counts.
"""

from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch

from model import MCNN

ZONE_IDS = ["A", "B", "C"]
ZONE_CAPACITY = 40  # matches CFG.zoneCapacity in MetroPulse 0/index.html
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# Real source footage varies wildly in resolution — a 4K phone-shot clip is
# ~11x the pixels of what MCNN was trained on, and both MCNN and the entry/
# exit tracker's background subtraction cost scale with pixel count. Without
# a cap, a single MCNN call on a 4K frame takes minutes (measured), not
# seconds, which stalls the whole processing loop. 768px matches the
# resolution ShanghaiTech Part B images are already close to, so this isn't
# a meaningful accuracy trade — it's just not doing 10x the work for the
# same answer.
MAX_FRAME_DIM = 768


def resize_max(frame: np.ndarray, max_dim: int = MAX_FRAME_DIM) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = max_dim / max(h, w)
    if scale >= 1.0:
        return frame
    return cv2.resize(frame, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def load_model(checkpoint: str, device: torch.device) -> MCNN:
    model = MCNN().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    return model


def frame_to_tensor(frame_bgr: np.ndarray) -> torch.Tensor:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    h, w = rgb.shape[:2]
    h4, w4 = h - h % 4, w - w % 4          # must match model's 1/4-resolution output
    rgb = rgb[:h4, :w4]
    return torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)


def zone_counts(density: np.ndarray) -> list:
    """density: (h, w) predicted map. Splits into 3 equal-width vertical
    strips left-to-right — the same A/B/C convention as the simulator's
    ZONES layout, for a single fixed camera looking down the platform."""
    w = density.shape[1]
    edges = [0, w // 3, 2 * w // 3, w]
    return [max(0.0, float(density[:, edges[i]:edges[i + 1]].sum())) for i in range(3)]


def make_frame_source(source: str, loop: bool = True, max_dim: int = MAX_FRAME_DIM):
    """Returns a zero-arg callable that yields the next BGR frame (capped to
    `max_dim` — see resize_max), or None when the source is exhausted (webcams
    never do). `loop` only affects video files — set False to play a source
    once and stop (e.g. so a "video ended" event means something, like the
    Control Room's train-departure trigger).

    `max_dim` is a parameter and not just the constant because the two models
    want opposite things. MCNN is happiest near the resolution it was trained
    at and gets slow fast with pixels, so 768 suits it. Person detection wants
    every pixel it can get: capping frames at 768 before detection threw away
    the distant half of a platform crowd outright, and no amount of inference
    resolution downstream brings back detail that was already discarded. A
    caller that intends to tile a full-resolution frame passes a bigger cap
    and downsizes for the other consumers itself."""
    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
        if not cap.isOpened():
            raise RuntimeError(f"couldn't open webcam index {source}")

        def next_frame():
            ok, frame = cap.read()
            return resize_max(frame, max_dim) if ok else None
        return next_frame

    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"source not found: {source}")

    if path.suffix.lower() in IMAGE_EXTS:
        frame = cv2.imread(str(path))
        if frame is None:
            raise RuntimeError(f"couldn't decode image: {source}")
        frame = resize_max(frame, max_dim)
        return lambda: frame  # same static frame every tick

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"couldn't open video: {source}")

    def next_frame():
        ok, frame = cap.read()
        if not ok and loop:  # loop back to the start instead of stopping
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()
        return resize_max(frame, max_dim) if ok else None
    return next_frame


def video_fps(source: str, default: float = 25.0) -> float:
    """Best-effort native FPS of a video file, for pacing playback. Webcams
    and still images fall back to `default`."""
    if source.isdigit():
        return default
    path = Path(source)
    if path.suffix.lower() in IMAGE_EXTS:
        return default
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps if fps and fps > 1 else default
