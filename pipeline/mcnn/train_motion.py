"""
When does the train arrive, and when does it leave — read from the video.

Why not a detector. YOLO has a "train" class, but on these cameras the train
is mostly behind screen-door glass and a crowd, and a detection costs ~1.6s
of CPU. The question is narrower than "is that a train": it is "is the thing
behind the glass sliding sideways", and that is cheap to answer on every
frame.

How. The operator marks a train strip per camera (camera_calibration.py): a
band of the train body seen through the upper glass, above people's heads.
Every frame that band is flattened into a STRIP_W x STRIP_H rectangle running
along the platform, so a moving train becomes one coherent sideways slide of
the whole rectangle. Two measurements, against the frame LAG_S of video time
earlier:

  - shift: phase correlation gives the dominant sideways offset and how
    strongly the image agrees on it (response). People walking past or
    moving about inside a carriage change patches; they don't slide the
    whole strip.
  - change: the fraction of the strip whose brightness changed. A train
    sliding past changes nearly all of it; a fast train blurs enough that
    phase correlation loses it, and this still catches it.

Arriving or leaving. Trains run one way on a track, so direction says
nothing. Speed does: an arrival comes in fast and slows to a stop; a
departure starts from still and speeds up, and the movement ends abruptly as
the last car clears the strip. Once one event has been seen the train's
presence is known and the next movement is simply the other kind; only the
first movement in a clip has to be judged by its speed profile.

Every threshold is in video seconds or strip fractions, never frames or
pixels, so the same numbers hold for a 3fps clip and a 30fps stream.

Presence mode. All of the above needs the train to be seen moving, and
footage stitched together from CCTV stills taken seconds apart never shows
that: between two stills the train has simply gone. Measured on such a clip,
85% of the strip changed at every new still and the motion state machine sat
in "arriving" for the whole clip. When the camera has two reference stills —
one with a train standing, one with the track empty — the question becomes
"which of the two does the strip look like now", which a still answers as
well as a video does. A change of answer that holds for PRESENCE_CONFIRM_S
is an arrival or a departure. Behind screen-door glass the two looks are far
apart (a lit carriage against a dark trackside wall), so this is the more
reliable mode wherever references exist, and it is used in preference.
"""

from collections import deque

import cv2
import numpy as np

STRIP_W, STRIP_H = 256, 48
LAG_S = 0.25              # compare against the strip this long ago
DIFF_THR = 18             # grey levels — a pixel that moved more than noise
MOVE_FRAC = 0.30          # strip fraction changed that means "train moving"
MIN_RESPONSE = 0.12       # phase-correlation agreement for a real slide
MIN_SPEED = 0.03          # strip lengths per second, for that slide to count
GAP_S = 0.5               # a pause shorter than this doesn't end a movement
DEPART_CONFIRM_S = 1.0    # movement this long from standing = departure
ARRIVE_MIN_S = 1.5        # shorter movement than this is not a train arriving
STILL_CONFIRM_S = 1.5     # still this long = the train has stopped / cleared
# A departure starts slow, and slow is exactly when phase correlation sees a
# clean slide. Requiring it on part of the movement keeps a crowd shuffling
# behind the glass during boarding — lots of change, no common slide — from
# sending the train away.
DEPART_MIN_SLIDE = 0.3    # fraction of the movement's frames with a real slide
# presence mode: score is +1 "exactly the train reference", -1 "exactly the
# empty one"; inside +/-PRESENCE_MARGIN it's too close to call and the last
# answer stands (someone's head in the strip, a reflection)
PRESENCE_MARGIN = 0.10
PRESENCE_CONFIRM_S = 2.0


class TrainMotionDetector:
    """Feed every frame to update(); it returns "arrival", "departure" or None.

    `state` is one of:
      unknown   — clip start, nothing seen yet
      standing  — a train is at the platform, still
      leaving   — departure fired, train still sliding out
      gone      — no train; the platform is waiting
      arriving  — a train is sliding in, not yet stopped
    """

    def __init__(self, calibration, fps: float, initial: str = "unknown",
                 ref_train: np.ndarray | None = None, ref_empty: np.ndarray | None = None):
        self.calibration = calibration
        self.fps = max(1e-3, float(fps))
        self.lag = max(1, round(LAG_S * self.fps))
        self.state = initial
        self._strips = deque(maxlen=self.lag + 1)
        self._M = None
        self._M_size = None
        # presence mode when both reference stills are there (see module doc)
        self.mode = "presence" if ref_train is not None and ref_empty is not None else "motion"
        self._ref_train = self._strip(ref_train) if self.mode == "presence" else None
        self._ref_empty = self._strip(ref_empty) if self.mode == "presence" else None
        self._M = None                # the refs may differ in size from the clip
        self._seen = None             # "present"/"absent" being confirmed
        self._seen_since = None
        self._window = cv2.createHanningWindow((STRIP_W, STRIP_H), cv2.CV_32F)
        self._moving_since = None     # video seconds the current movement began
        self._last_moving = None      # video seconds movement was last seen
        self._still_since = None
        self._run = []                # change fraction per frame of this movement
        self._slides = []             # whether each of those frames was a clean slide
        self.last = {}                # latest measurement, for logging/overlay

    def _strip(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        if self._M is None or self._M_size != (w, h):
            self._M = self.calibration.strip_transform(w, h, STRIP_W, STRIP_H)
            self._M_size = (w, h)
        flat = cv2.warpPerspective(frame, self._M, (STRIP_W, STRIP_H), flags=cv2.INTER_AREA)
        grey = cv2.cvtColor(flat, cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(grey, (3, 3), 0).astype(np.float32)

    def measure(self, frame: np.ndarray) -> dict:
        cur = self._strip(frame)
        self._strips.append(cur)
        if len(self._strips) <= self.lag:
            return {"moving": False, "slide": False, "change": 0.0, "shift": 0.0,
                    "response": 0.0, "speed": 0.0}
        prev = self._strips[0]
        change = float(np.mean(np.abs(cur - prev) > DIFF_THR))
        (dx, _dy), response = cv2.phaseCorrelate(prev, cur, self._window)
        speed = abs(dx) / STRIP_W / (self.lag / self.fps)
        slide = bool(response >= MIN_RESPONSE and speed >= MIN_SPEED and abs(dx) >= 0.5)
        return {"moving": bool(change >= MOVE_FRAC or slide), "slide": slide, "change": change,
                "shift": float(dx), "response": float(response), "speed": float(speed)}

    @staticmethod
    def _accelerating(run: list) -> bool:
        """Starts low and builds — a train pulling away. An arriving train is
        already at speed when it enters the strip."""
        if len(run) < 2:
            return True
        head = run[: max(1, len(run) // 3)]
        return float(np.mean(head)) < 0.6 * max(run) and run[-1] >= run[0]

    def _presence(self, frame: np.ndarray, frame_index: int):
        """Which reference the strip looks like, confirmed over
        PRESENCE_CONFIRM_S. Returns "arrival", "departure", or None. The
        first confirmed answer of a clip only sets the state: a clip that
        opens on an empty track hasn't seen a train leave."""
        t = frame_index / self.fps
        cur = self._strip(frame)
        d_train = float(np.mean(np.abs(cur - self._ref_train)))
        d_empty = float(np.mean(np.abs(cur - self._ref_empty)))
        score = (d_empty - d_train) / (d_empty + d_train + 1e-6)
        seen = "present" if score > PRESENCE_MARGIN else "absent" if score < -PRESENCE_MARGIN else None

        event = None
        if seen is not None:
            if seen != self._seen:
                self._seen, self._seen_since = seen, t
            elif t - self._seen_since >= PRESENCE_CONFIRM_S:
                if seen == "present" and self.state != "standing":
                    event = "arrival" if self.state == "gone" else None
                    self.state = "standing"
                elif seen == "absent" and self.state != "gone":
                    event = "departure" if self.state == "standing" else None
                    self.state = "gone"

        self.last = {"t": round(t, 3), "frame_index": frame_index, "moving": False,
                     "slide": False, "change": 0.0, "shift": 0.0, "response": round(score, 3),
                     "speed": 0.0, "state": self.state, "event": event,
                     # a frame that shows the train standing — the one to read
                     # the people left behind on if it turns out to be the last
                     "anchor": seen == "present"}
        return event

    def update(self, frame: np.ndarray, frame_index: int):
        if self.mode == "presence":
            return self._presence(frame, frame_index)
        t = frame_index / self.fps
        m = self.measure(frame)
        m.update(t=round(t, 3), frame_index=frame_index)
        event = None

        if m["moving"]:
            self._last_moving = t
            self._still_since = None
            if self._moving_since is None:
                self._moving_since = t
                self._run = []
                self._slides = []
            self._run.append(m["change"])
            self._slides.append(m["slide"])
            moved_for = t - self._moving_since
            slid = sum(self._slides) / len(self._slides)

            if self.state in ("standing", "unknown") and moved_for >= DEPART_CONFIRM_S:
                if (self.state == "standing" or self._accelerating(self._run)) \
                        and slid >= DEPART_MIN_SLIDE:
                    self.state, event = "leaving", "departure"
                elif self.state == "unknown" and not self._accelerating(self._run):
                    self.state = "arriving"
                # standing with lots of change but no common slide: a crowd
                # behind the glass, not the train — keep waiting
            elif self.state == "gone" and moved_for >= ARRIVE_MIN_S:
                self.state = "arriving"
        else:
            if self._still_since is None:
                self._still_since = t
            # a single still frame mid-movement (3fps, a gap between cars)
            # doesn't end it
            if self._last_moving is not None and t - self._last_moving >= GAP_S:
                self._moving_since = None
            if t - self._still_since >= STILL_CONFIRM_S:
                if self.state == "arriving":
                    self.state, event = "standing", "arrival"
                elif self.state == "leaving":
                    self.state = "gone"

        m.update(state=self.state, event=event, anchor=not m["moving"])
        self.last = m
        return event
