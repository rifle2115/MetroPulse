"""
Door-state classifier: is there a train standing at the platform — from one
still, no motion needed.

Two classes, train / no train. The first model also split "train" into doors
open and doors closed; on validation it confused a train standing with its
doors shut for an empty track (4 of 27 right), which made arrivals late and
departures early. Arrival and departure only need presence, so the two
train classes are merged: more examples of "train", and no class the
detector depends on being learnt from 70 images.

Each camera's own empty track as a reference. Trained on single crops, the
model learnt a shortcut from the training stations — dark wall behind the
glass = empty, bright = train — and at a validation station with a light
track wall it called every empty frame "train" at 99%. So it is shown two
crops: the current one, and the same camera's door crop with no train
(taken once at calibration, "THIS FRAME = NO TRAIN"), stacked as 6 channels.
The question becomes "does this look different from this camera's empty
track", which holds for dark walls, light walls and reflections alike. The
extra input weights start as the negative of the pretrained ones, so the
untrained network already sees current minus empty — a change detector to
refine rather than one to discover.

Why a classifier. The project's footage is CCTV stills taken seconds apart,
stitched into clips. A moving train is never seen, so detectors that watch
the train slide (train_motion.py) have nothing to work with. A small CNN that
looks at one crop of the screen doors and names what it sees works the same
on a still, a 3fps clip or a live stream. This follows the door-state work
in the literature (a lightweight MobileNet recognising train doors opening
and closing), adapted to wide platform cameras by cropping each camera's
view to 2-3 screen doors first (camera_calibration.py, the door crop).

One module for all three users — the dataset builder, training, evaluation
(pipeline/train_detect/) and the Control Room — so the crop, the
preprocessing and the model are identical in training and in use.
"""

from pathlib import Path

import cv2
import numpy as np
import torch

CLASSES = ["no_train", "train"]
CROP_W, CROP_H = 256, 128        # the flattened door crop the model sees
MEAN = np.float32([0.485, 0.456, 0.406])   # ImageNet, which the backbone was
STD = np.float32([0.229, 0.224, 0.225])    # pretrained on

EVERY_S = 0.25        # classify a frame this often, in video seconds
CONFIRM_S = 1.0       # an answer must hold this long before it counts


IN_CHANNELS = 6      # current crop (3) + this camera's empty-track crop (3)


def build_model(pretrained: bool = False) -> torch.nn.Module:
    """MobileNetV3-small taking current + empty reference, with a train /
    no-train head. pretrained=True starts from ImageNet weights (downloaded
    once by torchvision) — used for training; the Control Room loads trained
    weights over an untrained skeleton."""
    from torchvision import models
    weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.mobilenet_v3_small(weights=weights)
    first = model.features[0][0]
    wide = torch.nn.Conv2d(IN_CHANNELS, first.out_channels, first.kernel_size,
                           stride=first.stride, padding=first.padding, bias=False)
    with torch.no_grad():
        # [w, -w]: the pretrained filters applied to (current - reference)
        wide.weight[:, :3] = first.weight
        wide.weight[:, 3:] = -first.weight
    model.features[0][0] = wide
    model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, len(CLASSES))
    return model


class DoorCropper:
    """Flattens a camera's door crop out of a frame. The transform is cached
    per frame size, since a clip's frames are all one size."""

    def __init__(self, calibration):
        if calibration.door_crop is None:
            raise ValueError(f"camera {calibration.id!r} has no door crop marked")
        self.calibration = calibration
        self._size = None
        self._M = None

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        if self._size != (w, h):
            self._M = self.calibration.door_transform(w, h, CROP_W, CROP_H)
            self._size = (w, h)
        return cv2.warpPerspective(frame, self._M, (CROP_W, CROP_H), flags=cv2.INTER_AREA)


def to_tensor(crop_bgr: np.ndarray) -> torch.Tensor:
    """BGR crop -> normalised (3, H, W) tensor. Matches training's eval
    transform (ToTensor + Normalize) exactly."""
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - MEAN) / STD
    return torch.from_numpy(rgb.transpose(2, 0, 1).copy())


def pair_tensor(crop_bgr: np.ndarray, ref_bgr: np.ndarray) -> torch.Tensor:
    """Current crop + the camera's empty-track crop -> (6, H, W)."""
    return torch.cat([to_tensor(crop_bgr), to_tensor(ref_bgr)], dim=0)


class DoorStateModel:
    """A trained checkpoint (train_detect/train.py) ready to classify crops."""

    def __init__(self, path: Path, device: str = "cpu"):
        ckpt = torch.load(str(path), map_location=device, weights_only=False)
        if list(ckpt.get("classes", [])) != CLASSES or ckpt.get("in_channels") != IN_CHANNELS:
            raise ValueError(f"{path} is an older model ({ckpt.get('classes')}, "
                             f"{ckpt.get('in_channels', 3)} input channels); retrain it with "
                             f"train_detect/train.py")
        self.model = build_model(pretrained=False)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval().to(device)
        self.device = device
        self.info = {k: ckpt.get(k) for k in ("epoch", "val_macro_f1", "trained_at")}

    @torch.no_grad()
    def predict(self, crop_bgr: np.ndarray, ref_bgr: np.ndarray) -> tuple[str, list]:
        """Classify a door crop against the same camera's empty-track crop."""
        logits = self.model(pair_tensor(crop_bgr, ref_bgr).unsqueeze(0).to(self.device))
        probs = torch.softmax(logits, dim=1)[0].cpu().tolist()
        return CLASSES[int(np.argmax(probs))], probs


class PresenceTracker:
    """The timing rules that turn per-frame answers into arrivals and
    departures, apart from the model so evaluation can replay one clip's
    answers under several confirm times without classifying it again.

    "Train" against "no train" has to hold for confirm_s before it counts,
    which rides over a misread still — a person standing in front of the
    doors, a reflection. The first confirmed answer of a clip only sets the
    state: a clip that opens on an empty platform has not seen a train leave.
    """

    def __init__(self, confirm_s: float = CONFIRM_S):
        self.confirm_s = confirm_s
        self.state = "unknown"
        self._seen = None
        self._seen_since = None

    def observe(self, t: float, label: str):
        """One classified frame at video time t -> "arrival", "departure" or None."""
        seen = "present" if label == "train" else "absent"
        if seen != self._seen:
            self._seen, self._seen_since = seen, t
        elif t - self._seen_since >= self.confirm_s:
            if seen == "present" and self.state != "standing":
                event = "arrival" if self.state == "gone" else None
                self.state = "standing"
                return event
            if seen == "absent" and self.state != "gone":
                event = "departure" if self.state == "standing" else None
                self.state = "gone"
                return event
        return None


class DoorStateDetector:
    """Feed every frame to update(); it returns "arrival", "departure" or
    None — the same interface as train_motion.TrainMotionDetector, so the
    Control Room can use either. Every EVERY_S of video one frame is
    classified, and PresenceTracker decides when that is an event.
    """

    mode = "classifier"

    def __init__(self, calibration, fps: float, model: DoorStateModel, ref_crop: np.ndarray,
                 every_s: float = EVERY_S, confirm_s: float = CONFIRM_S):
        """ref_crop: this camera's door crop with no train (DoorCropper
        applied to its "NO TRAIN" still)."""
        self.fps = max(1e-3, float(fps))
        self.model = model
        self.crop = DoorCropper(calibration)
        self.ref = ref_crop
        self.every_s = every_s
        self.tracker = PresenceTracker(confirm_s)
        self._next_t = 0.0
        self.last = self._record(0.0, 0, None, 0.0, None)

    @property
    def state(self) -> str:
        return self.tracker.state

    @state.setter
    def state(self, value: str) -> None:
        # the Control Room's TRAIN ARRIVED / TRAIN LEFT buttons set this
        self.tracker.state = value

    def _record(self, t, frame_index, label, prob, event) -> dict:
        # same keys train_motion writes, so the Control Room's motion log and
        # departure anchor work unchanged; `response` carries the confidence
        return {"t": round(t, 3), "frame_index": frame_index, "moving": False, "slide": False,
                "change": 0.0, "shift": 0.0, "response": round(prob, 3), "speed": 0.0,
                "state": self.state, "event": event, "label": label or "",
                "anchor": label == "train"}

    def update(self, frame: np.ndarray, frame_index: int):
        t = frame_index / self.fps
        if t < self._next_t:
            self.last = dict(self.last, t=round(t, 3), frame_index=frame_index, event=None)
            return None
        self._next_t = t + self.every_s

        label, probs = self.model.predict(self.crop(frame), self.ref)
        event = self.tracker.observe(t, label)
        self.last = self._record(t, frame_index, label, max(probs), event)
        return event
