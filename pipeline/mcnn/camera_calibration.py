"""
Per-camera platform calibration — which coach a detected person is queuing
for, worked out from where they are standing rather than which third of the
frame they happen to fall in.

Why. Splitting the frame into three equal vertical strips only works for a
camera looking square-on at the train. A real platform camera looks down the
platform at an angle: the near coach fills most of the frame and the far one
is squeezed into a corner, people seen through the carriage windows land in
"zone A", and one camera rarely sees all three coaches anyway.

What gets marked. A CCTV camera is bolted to the ceiling, so its view never
changes and the operator marks it once (Control Room -> /calibrate):

  - the 4 corners of the platform waiting area, in this order
        1  platform edge (screen doors), one end
        2  platform edge, the other end
        3  back of the platform, behind corner 2
        4  back of the platform, behind corner 1
  - the seams where one coach ends and the next begins, clicked on the floor

The 4 corners give a homography from the image onto a flat unit rectangle
laid over the platform floor: u runs along the platform (0 at corner 1, 1 at
corner 2) and v from the edge (0) to the back (1). A person's feet — the
bottom-centre of their box — map to one (u, v). Outside the rectangle means
they are not waiting on this platform (inside the train, behind the glass, on
the stairs) and they are not counted; inside, u against the seams says which
coach. The homography is what absorbs the perspective.

Optionally, a train strip: 4 corners around a band of the train body seen
through the upper half of the screen-door glass, above people's heads —
  1  top edge, corner-1 end of the platform
  2  top edge, corner-2 end
  3  bottom edge, under 2
  4  bottom edge, under 1
It is flattened into a long rectangle so a moving train becomes a sideways
slide along it, which is what train_motion.py watches for to tell when a
train arrives and leaves.

Points are stored normalised to 0..1 of the frame, so a calibration marked on
a 1920x1080 still applies unchanged to the 960-wide preview of the same feed.
"""

import bisect
import json
import os
import re
import threading
from pathlib import Path

import cv2
import numpy as np

ZONE_IDS = ["A", "B", "C"]

_UNIT_SQUARE = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])
_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,40}$")
# BGR, matching the coach colours used on the calibration page
COACH_BGR = {"A": (255, 170, 60), "B": (120, 220, 80), "C": (60, 160, 255)}


class CalibrationError(ValueError):
    """A calibration that cannot be used — reported back to the operator
    rather than saved and discovered later as nonsense counts."""


class CameraCalibration:
    def __init__(self, cam_id: str, name: str, platform: list, boundaries: list, coaches: list,
                 train_strip: list | None = None, door_crop: list | None = None):
        self.id = cam_id
        self.name = name or cam_id
        self.platform = [[float(x), float(y)] for x, y in platform]
        self.boundaries = [[float(x), float(y)] for x, y in boundaries]
        self.coaches = [str(c) for c in coaches]
        self.train_strip = [[float(x), float(y)] for x, y in train_strip] if train_strip else None
        # 4 corners around 2-3 screen doors: what the door-state classifier
        # (door_state.py) looks at to tell no train / doors open / doors closed
        self.door_crop = [[float(x), float(y)] for x, y in door_crop] if door_crop else None
        self._validate_shape()

        src = np.float32(self.platform)
        self._to_floor = cv2.getPerspectiveTransform(src, _UNIT_SQUARE)
        self._to_image = cv2.getPerspectiveTransform(_UNIT_SQUARE, src)

        # Seams as positions along the platform, sorted: the operator can click
        # them in any order, and coaches[i] always names the i-th segment
        # counting from corner 1.
        floor = self.to_floor(np.float32(self.boundaries)) if self.boundaries else np.empty((0, 2))
        for u, v in floor:
            if not (0.0 < u < 1.0 and -0.05 <= v <= 1.05):
                raise CalibrationError("every coach seam must be clicked inside the platform area")
        self.seams = sorted(float(u) for u, _v in floor)
        self._mask_cache: dict = {}

    def _validate_shape(self) -> None:
        if not _ID_RE.match(self.id or ""):
            raise CalibrationError("camera id must be 1-40 letters, digits, '-' or '_'")
        if len(self.platform) != 4:
            raise CalibrationError("the platform needs exactly 4 corners")
        if any(not (0.0 <= c <= 1.0) for pt in self.platform + self.boundaries for c in pt):
            raise CalibrationError("points must lie inside the frame")
        poly = np.float32(self.platform).reshape(-1, 1, 2)
        if cv2.contourArea(poly) < 1e-3:
            raise CalibrationError("the platform area is too small or its corners are in a line")
        if not cv2.isContourConvex(poly):
            raise CalibrationError("platform corners cross over — click them in order 1-2-3-4 "
                                   "around the edge of the area")
        if len(self.coaches) != len(self.boundaries) + 1:
            raise CalibrationError("there must be one coach label per segment (seams + 1)")
        bad = [c for c in self.coaches if c not in ZONE_IDS]
        if bad:
            raise CalibrationError(f"unknown coach label(s) {bad}; use {ZONE_IDS}")
        for quad, what in ((self.train_strip, "train strip"), (self.door_crop, "door crop")):
            if quad is None:
                continue
            if len(quad) != 4:
                raise CalibrationError(f"the {what} needs exactly 4 corners")
            if any(not (0.0 <= c <= 1.0) for pt in quad for c in pt):
                raise CalibrationError(f"{what} corners must lie inside the frame")
            poly = np.float32(quad).reshape(-1, 1, 2)
            if cv2.contourArea(poly) < 2e-4 or not cv2.isContourConvex(poly):
                raise CalibrationError(f"{what} corners cross over or are in a line — click "
                                       "them 1-2 along the top edge, then 3-4 along the bottom")

    # --- geometry -----------------------------------------------------------

    def to_floor(self, pts_norm: np.ndarray) -> np.ndarray:
        """Normalised image points (N, 2) -> platform (u, v) (N, 2)."""
        pts = np.asarray(pts_norm, dtype=np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self._to_floor).reshape(-1, 2)

    def to_image(self, pts_floor: np.ndarray) -> np.ndarray:
        """Platform (u, v) (N, 2) -> normalised image points (N, 2)."""
        pts = np.asarray(pts_floor, dtype=np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self._to_image).reshape(-1, 2)

    def coach_at(self, u: float) -> str:
        return self.coaches[bisect.bisect_right(self.seams, u)]

    def zones_for_points(self, points_px: list, w: int, h: int) -> list:
        """Pixel points (feet) -> coach label per point, or None when the point
        is off the marked platform and should not be counted."""
        if not points_px:
            return []
        norm = np.float32(points_px) / np.float32([w, h])
        out = []
        for u, v in self.to_floor(norm):
            out.append(self.coach_at(float(u)) if 0.0 <= u <= 1.0 and 0.0 <= v <= 1.0 else None)
        return out

    def zone_masks(self, h: int, w: int) -> dict:
        """Boolean mask per coach over an (h, w) grid — used to split MCNN's
        density map the same way the detector's people are split. Cached per
        size, since the density map is the same size every reading.

        Density sits on bodies, not feet, so this leans each person slightly
        towards the far end of the platform. It is the secondary reading (see
        publish_zone_telemetry); the detection count is the one that drives
        decisions."""
        key = (h, w)
        if key not in self._mask_cache:
            ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
            norm = np.stack([(xs + 0.5) / w, (ys + 0.5) / h], axis=-1).reshape(-1, 2)
            uv = self.to_floor(norm)
            inside = (uv[:, 0] >= 0) & (uv[:, 0] <= 1) & (uv[:, 1] >= 0) & (uv[:, 1] <= 1)
            seg = np.searchsorted(np.float32(self.seams), uv[:, 0], side="right")
            labels = np.array(self.coaches)[np.clip(seg, 0, len(self.coaches) - 1)]
            self._mask_cache[key] = {
                z: (inside & (labels == z)).reshape(h, w) for z in ZONE_IDS
            }
        return self._mask_cache[key]

    def strip_transform(self, w: int, h: int, out_w: int, out_h: int):
        """Perspective transform taking the train strip in a (w, h) frame to a
        flat out_w x out_h rectangle, corner 1 at the left — or None when this
        camera has no strip marked."""
        return _quad_transform(self.train_strip, w, h, out_w, out_h)

    def door_transform(self, w: int, h: int, out_w: int, out_h: int):
        """Same for the door crop — None when this camera has none marked."""
        return _quad_transform(self.door_crop, w, h, out_w, out_h)

    # --- drawing ------------------------------------------------------------

    def draw(self, frame: np.ndarray) -> np.ndarray:
        """Platform outline, coach seams, coach labels and the train strip, in
        place."""
        h, w = frame.shape[:2]
        scale = np.float32([w, h])
        if self.train_strip is not None:
            strip = (np.float32(self.train_strip) * scale).astype(np.int32)
            cv2.polylines(frame, [strip], True, (255, 80, 220), 1, cv2.LINE_AA)
        if self.door_crop is not None:
            crop = (np.float32(self.door_crop) * scale).astype(np.int32)
            cv2.polylines(frame, [crop], True, (160, 224, 56), 1, cv2.LINE_AA)

        def px(pts_floor):
            return (self.to_image(np.float32(pts_floor)) * scale).astype(np.int32)

        poly = (np.float32(self.platform) * scale).astype(np.int32)
        cv2.polylines(frame, [poly], True, (255, 200, 0), 2, cv2.LINE_AA)
        for u in self.seams:
            a, b = px([[u, 0.0], [u, 1.0]])
            cv2.line(frame, tuple(map(int, a)), tuple(map(int, b)), (255, 200, 0), 2, cv2.LINE_AA)

        edges = [0.0, *self.seams, 1.0]
        for i, label in enumerate(self.coaches):
            (cx, cy), = px([[(edges[i] + edges[i + 1]) / 2.0, 0.5]])
            cv2.putText(frame, label, (int(cx) - 9, int(cy) + 9), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(frame, label, (int(cx) - 9, int(cy) + 9), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, COACH_BGR.get(label, (255, 255, 255)), 2, cv2.LINE_AA)
        return frame

    # --- persistence --------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "platform": self.platform,
            "boundaries": self.boundaries,
            "coaches": self.coaches,
            "train_strip": self.train_strip,
            "door_crop": self.door_crop,
            "seams": self.seams,   # derived; sent for display, ignored on load
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CameraCalibration":
        return cls(d["id"], d.get("name", ""), d["platform"], d.get("boundaries", []), d["coaches"],
                   d.get("train_strip"), d.get("door_crop"))


def _quad_transform(quad, w: int, h: int, out_w: int, out_h: int):
    """Perspective transform from a marked quad (normalised corners 1-2 top,
    3 under 2, 4 under 1) in a (w, h) frame to a flat out_w x out_h image."""
    if quad is None:
        return None
    src = np.float32(quad) * np.float32([w, h])
    dst = np.float32([[0, 0], [out_w, 0], [out_w, out_h], [0, out_h]])
    return cv2.getPerspectiveTransform(src, dst)


class CalibrationStore:
    """All cameras' calibrations plus which one the Control Room is using, in
    one JSON file. Small, read rarely, written only when the operator saves —
    a file is all this needs."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._cameras: dict = {}
        self._active: str | None = None
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        data = json.loads(self.path.read_text(encoding="utf-8"))
        for d in data.get("cameras", []):
            try:
                cal = CameraCalibration.from_dict(d)
            except (CalibrationError, KeyError, cv2.error) as exc:
                # one bad entry (hand-edited, older format) shouldn't take the
                # whole Control Room down at startup
                print(f"[calibration] skipping camera {d.get('id')!r}: {exc}", flush=True)
                continue
            self._cameras[cal.id] = cal
        active = data.get("active")
        self._active = active if active in self._cameras else None

    def _save(self) -> None:
        data = {
            "active": self._active,
            "cameras": [{k: v for k, v in c.to_dict().items() if k != "seams"}
                        for c in self._cameras.values()],
        }
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        os.replace(tmp, self.path)   # never leaves a half-written file behind

    def all(self) -> list:
        with self._lock:
            return list(self._cameras.values())

    def get(self, cam_id: str):
        with self._lock:
            return self._cameras.get(cam_id)

    @property
    def active_id(self) -> str | None:
        with self._lock:
            return self._active

    def active(self):
        with self._lock:
            return self._cameras.get(self._active) if self._active else None

    def put(self, cal: CameraCalibration) -> None:
        with self._lock:
            self._cameras[cal.id] = cal
            self._save()

    def delete(self, cam_id: str) -> bool:
        with self._lock:
            if self._cameras.pop(cam_id, None) is None:
                return False
            if self._active == cam_id:
                self._active = None
            self._save()
            return True

    def set_active(self, cam_id: str | None) -> None:
        with self._lock:
            self._active = cam_id if cam_id in self._cameras else None
            self._save()
