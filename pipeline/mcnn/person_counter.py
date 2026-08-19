"""
Person detection on full-resolution frames — how many people are stood at each
coach's doors. This replaced entry_exit_counter.py's background-subtraction
blob counter, which asked "what moved?" rather than "where are the people":
measured on real footage it churned 34 tracker IDs through 120 frames for a
scene holding ~20 people, because overlapping bodies merged into one blob,
anyone standing still was absorbed into the background model, and reflections
off the train shell registered as movement.

Why tiled. A detector only sees what reaches it, and the pipeline used to cap
every frame at 768px (inference.MAX_FRAME_DIM) before anything looked at it,
which on a 1920x1080 platform shot discards the far half of the crowd. Counts
came out around 35 on a frame holding closer to 80, and no amount of inference
resolution afterwards recovers detail already thrown away. Slicing the
full-resolution frame into overlapping tiles and detecting in each at native
scale is the standard fix:

    768px downsized, whole frame     33 people    2755 ms
    full-res, whole frame            32 people     230 ms
    full-res, 2x2 tiles              48 people     479 ms
    full-res, 3x3 tiles              65 people     968 ms   <- default
    full-res, 4x3 tiles              64 people    1136 ms

Bigger models find FEWER, not more (yolo11n 81, yolo11s 69, yolo11m 64 on the
same frame), and a 4x4 grid's higher number is visibly inflated by duplicate
boxes where tiles cut bodies in half. So the tuning here is not "turn it up
until the number looks big" — past this point extra detections are false
positives, not recovered people.

Known limit: the zones are thirds of the whole frame, so a detection inside
the carriage or on the far platform still counts toward that zone. Restricting
each zone to a polygon over the platform waiting area is the fix, and it needs
per-camera calibration.

There is no tracking here. There used to be — a centroid tracker and a
line-crossing counter — but boarding is now measured as the change in platform
crowd across a dwell (see handle_train in decision_engine.py), which needs two
head counts and no identities at all. Tracking survived a while as code
nothing called; it is gone.
"""

import argparse

import cv2
import numpy as np

ZONE_IDS = ["A", "B", "C"]

MODEL_NAME = "yolo11n.pt"
TILE_GRID = (3, 3)      # columns, rows — see the measurements above
TILE_OVERLAP = 0.25     # so a body on a seam is whole in at least one tile
TILE_IMG_SIZE = 960     # inference size per tile
CONF = 0.12             # a real person at the far end of this platform scores
                        # 0.15-0.2; the stock 0.25 drops half the platform
NMS_IOU = 0.55
PERSON_CLASS = 0        # COCO "person"


def _zone_of_x(x: float, w: int) -> str:
    if x < w / 3:
        return "A"
    if x < 2 * w / 3:
        return "B"
    return "C"


def _nms(boxes: list, scores: list, iou_thr: float = NMS_IOU) -> list:
    """Standard greedy NMS, needed because overlapping tiles deliberately show
    the same person to more than one tile."""
    if not boxes:
        return []
    b = np.asarray(boxes, dtype=np.float32)
    s = np.asarray(scores, dtype=np.float32)
    x1, y1, x2, y2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    area = (x2 - x1) * (y2 - y1)
    order = s.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (area[i] + area[order[1:]] - inter + 1e-9)
        order = order[1:][iou < iou_thr]
    return keep


class PersonCounter:
    """Counts people per zone in a frame. `update()` reads one frame and fills
    `zone_people` / `people_total`; `draw_overlay()` renders the boxes onto a
    frame of any size."""

    def __init__(self, model_name: str = MODEL_NAME, grid: tuple = TILE_GRID,
                 overlap: float = TILE_OVERLAP, imgsz: int = TILE_IMG_SIZE,
                 conf: float = CONF):
        from ultralytics import YOLO   # imported here so this module is cheap to import

        self.model = YOLO(model_name)
        self.grid = grid
        self.overlap = overlap
        self.imgsz = imgsz
        self.conf = conf

        self.zone_people = {z: 0 for z in ZONE_IDS}
        self.people_total = 0
        self.boxes = np.empty((0, 4), dtype=np.float32)
        self.scores = np.empty((0,), dtype=np.float32)
        # resolution the boxes were measured at, so draw_overlay can rescale
        # them onto a smaller preview frame
        self._src_w = 0
        self._src_h = 0

    def detect(self, frame: np.ndarray):
        """Full-resolution tiled detection. Returns (boxes Nx4, scores N)."""
        h, w = frame.shape[:2]
        gx, gy = self.grid
        tw, th = w // gx, h // gy
        ox, oy = int(tw * self.overlap), int(th * self.overlap)

        # A whole-frame pass alongside the tiles: someone close to the camera
        # can be larger than a single tile and would otherwise only ever be
        # seen in pieces. One extra pass, and it costs nothing in accuracy
        # because NMS collapses the duplicates it creates.
        panes = [(0, 0, w, h)]
        for iy in range(gy):
            for ix in range(gx):
                panes.append((max(0, ix * tw - ox), max(0, iy * th - oy),
                              min(w, (ix + 1) * tw + ox), min(h, (iy + 1) * th + oy)))

        boxes, scores = [], []
        for (x0, y0, x1, y1) in panes:
            pane = frame[y0:y1, x0:x1]
            if pane.size == 0:
                continue
            res = self.model.predict(pane, classes=[PERSON_CLASS], conf=self.conf,
                                     imgsz=self.imgsz, verbose=False)[0]
            if res.boxes is None or len(res.boxes) == 0:
                continue
            for bb, sc in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy()):
                boxes.append([bb[0] + x0, bb[1] + y0, bb[2] + x0, bb[3] + y0])
                scores.append(float(sc))

        keep = _nms(boxes, scores)
        if not keep:
            return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)
        return (np.asarray([boxes[i] for i in keep], dtype=np.float32),
                np.asarray([scores[i] for i in keep], dtype=np.float32))

    def update(self, frame: np.ndarray) -> dict:
        """Reads one frame. Returns the per-zone head count."""
        h, w = frame.shape[:2]
        self._src_w, self._src_h = w, h
        self.boxes, self.scores = self.detect(frame)

        # feet, not box centre: someone's ground position is what puts them in
        # a zone, and for a camera looking down at an angle the two differ by
        # most of a body
        counts = {z: 0 for z in ZONE_IDS}
        for x1, _y1, x2, y2 in self.boxes:
            counts[_zone_of_x((x1 + x2) / 2.0, w)] += 1

        self.zone_people = counts
        self.people_total = int(len(self.boxes))
        return counts

    def draw_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Boxes are in full-resolution coordinates; the preview frame may be
        smaller, so they are scaled to whatever this is handed."""
        out = frame.copy()
        h, w = out.shape[:2]
        sx = w / float(self._src_w) if self._src_w else 1.0
        sy = h / float(self._src_h) if self._src_h else 1.0

        for (x1, y1, x2, y2), sc in zip(self.boxes, self.scores):
            col = (0, 230, 0) if sc >= 0.25 else (0, 165, 255)
            cv2.rectangle(out, (int(x1 * sx), int(y1 * sy)), (int(x2 * sx), int(y2 * sy)), col, 2)

        cv2.putText(out, f"PEOPLE {self.people_total}", (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(out, "  ".join(f"{z}:{self.zone_people[z]}" for z in ZONE_IDS), (12, 54),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 1, cv2.LINE_AA)
        return out


def main() -> None:
    from inference import make_frame_source

    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0", help="webcam index or video path")
    ap.add_argument("--stride", type=int, default=30, help="read every Nth frame")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    get_frame = make_frame_source(args.source, loop=False, max_dim=1920)
    counter = PersonCounter()

    n = 0
    while True:
        frame = get_frame()
        if frame is None:
            break
        n += 1
        if (n - 1) % args.stride:
            continue
        counts = counter.update(frame)
        print(f"frame {n}: {counts} (total {counter.people_total})", flush=True)
        if args.show:
            cv2.imshow("person counter", counter.draw_overlay(frame))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    if args.show:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
