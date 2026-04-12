"""
gestures/hand_tracker.py
========================
MediaPipe Hand Landmarker (Tasks API) wrapper that processes webcam frames
and returns structured HandData objects with landmarks in both normalised
and pixel space.

MediaPipe version note
----------------------
This module targets MediaPipe ≥ 0.10 (Tasks API).  The legacy
`mp.solutions.hands` API was removed in 0.10.x.  The model file
(hand_landmarker.task) must exist at config.MEDIAPIPE_MODEL_PATH;
run the download once with:

    curl -sL https://storage.googleapis.com/mediapipe-models/ \
         hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task \
         -o assets/models/hand_landmarker.task

Design notes
------------
- Single-hand mode only (num_hands=1, per Q1).
- VIDEO RunningMode is used for synchronous per-frame processing.
  Timestamps are strictly monotonically increasing milliseconds.
- Wrist velocity is computed by diffing landmark[0] between consecutive
  frames in normalised space and dividing by dt (seconds).
- `landmarks_px` are pre-computed pixel tuples — all drawing code should
  use these.  `landmarks_norm` is kept for proportional geometry checks.
- Finger extension booleans are computed once here so downstream modules
  never touch raw landmark objects.
- HAND_CONNECTIONS is defined locally since the Tasks API does not expose it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

import config

# ── Hand skeleton connections (wrist + 5 fingers, 21 landmarks) ───────────────
HAND_CONNECTIONS: List[Tuple[int, int]] = [
    (0, 1),  (1, 2),  (2, 3),  (3, 4),       # thumb
    (0, 5),  (5, 6),  (6, 7),  (7, 8),        # index
    (5, 9),  (9, 10), (10, 11), (11, 12),      # middle
    (9, 13), (13, 14), (14, 15), (15, 16),     # ring
    (13, 17), (17, 18), (18, 19), (19, 20),    # pinky
    (0, 17),                                   # palm base
]


@dataclass
class HandData:
    """
    All processed information about a detected hand for one frame.

    Attributes:
        landmarks_norm:   List of 21 NormalizedLandmark objects in [0, 1] space.
                          Access .x, .y, .z.  Use for proportional geometry.
        landmarks_px:     (x, y) pixel coordinates, same ordering.
                          Use for all drawing operations.
        handedness:       "Left" or "Right" as classified by MediaPipe.
                          Because frames are mirrored (flip=1), "Right" means
                          the user's right hand.
        fingers_extended: [thumb, index, middle, ring, pinky] — True = extended.
        wrist_velocity:   (vx, vy) wrist displacement in normalised units/second.
                          Positive vx = moving right, positive vy = moving down.
        wrist_speed:      Euclidean magnitude of wrist_velocity.
        frame_width:      Source frame width in pixels.
        frame_height:     Source frame height in pixels.
    """

    landmarks_norm:   list                   # List[NormalizedLandmark]
    landmarks_px:     List[Tuple[int, int]]
    handedness:       str
    fingers_extended: List[bool]             # [T, I, M, R, P]
    wrist_velocity:   Tuple[float, float]    # (vx, vy) norm/s
    wrist_speed:      float
    frame_width:      int
    frame_height:     int


class HandTracker:
    """
    Wraps the MediaPipe Hand Landmarker (Tasks API) for single-hand detection.

    The detector runs in VIDEO mode, which provides synchronous per-frame
    results with temporal smoothing.  Each call to process() must supply
    a monotonically increasing timestamp.

    Usage (explicit close):
        tracker = HandTracker()
        hand = tracker.process(bgr_frame)   # HandData or None
        tracker.close()

    Context manager:
        with HandTracker() as tracker:
            hand = tracker.process(frame)
    """

    def __init__(self) -> None:
        """Initialise the MediaPipe Hand Landmarker with settings from config."""
        options = mp_vision.HandLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(
                model_asset_path=config.MEDIAPIPE_MODEL_PATH,
            ),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=config.MEDIAPIPE_DETECTION_CONFIDENCE,
            min_hand_presence_confidence=config.MEDIAPIPE_PRESENCE_CONFIDENCE,
            min_tracking_confidence=config.MEDIAPIPE_TRACKING_CONFIDENCE,
        )
        self._detector   = mp_vision.HandLandmarker.create_from_options(options)
        self._start_time = time.time()
        self._last_ts_ms = -1                               # for monotonic guard
        self._prev_wrist: Optional[Tuple[float, float]] = None
        self._prev_time:  Optional[float]               = None

    def process(self, frame: np.ndarray) -> Optional[HandData]:
        """
        Detect a hand in a BGR frame and return structured HandData.

        The frame must already be mirrored (cv2.flip(frame, 1)) so
        left/right handedness matches the user's perspective.

        Args:
            frame: BGR image (H×W×3 numpy array) from OpenCV.

        Returns:
            HandData if a hand is detected, None otherwise.
            When None is returned the internal velocity state is cleared.
        """
        h, w = frame.shape[:2]
        now  = time.time()

        # Build a monotonically increasing millisecond timestamp for the
        # MediaPipe VIDEO-mode detector.
        ts_ms = int((now - self._start_time) * 1000)
        if ts_ms <= self._last_ts_ms:
            ts_ms = self._last_ts_ms + 1
        self._last_ts_ms = ts_ms

        # Convert to RGB for MediaPipe
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = self._detector.detect_for_video(mp_img, ts_ms)

        if not result.hand_landmarks:
            self._prev_wrist = None
            self._prev_time  = None
            return None

        # Single-hand mode — always use the first detection.
        lm_norm    = result.hand_landmarks[0]           # List[NormalizedLandmark]
        handedness = result.handedness[0][0].category_name  # "Left" | "Right"

        lm_px = [(int(lm.x * w), int(lm.y * h)) for lm in lm_norm]

        # ── Wrist velocity ─────────────────────────────────────────────────────
        wx, wy = lm_norm[0].x, lm_norm[0].y
        if self._prev_wrist is not None and self._prev_time is not None:
            dt = max(now - self._prev_time, 1e-6)
            vx = (wx - self._prev_wrist[0]) / dt
            vy = (wy - self._prev_wrist[1]) / dt
        else:
            vx, vy = 0.0, 0.0

        self._prev_wrist = (wx, wy)
        self._prev_time  = now
        speed = float(np.hypot(vx, vy))

        fingers = self._compute_extensions(lm_norm, handedness)

        return HandData(
            landmarks_norm   = lm_norm,
            landmarks_px     = lm_px,
            handedness       = handedness,
            fingers_extended = fingers,
            wrist_velocity   = (vx, vy),
            wrist_speed      = speed,
            frame_width      = w,
            frame_height     = h,
        )

    def draw_landmarks(
        self,
        frame:            np.ndarray,
        hand:             HandData,
        landmark_color:   Tuple[int, int, int] = config.TEAL_DARK,
        connection_color: Tuple[int, int, int] = config.TEAL_MID,
    ) -> None:
        """
        Draw the hand skeleton onto a BGR frame in-place using the project
        teal palette.

        Args:
            frame:            BGR frame to draw onto (modified in-place).
            hand:             HandData from process().
            landmark_color:   BGR colour for joint dots.
            connection_color: BGR colour for bone lines.
        """
        for start, end in HAND_CONNECTIONS:
            cv2.line(
                frame,
                hand.landmarks_px[start],
                hand.landmarks_px[end],
                connection_color, 1, cv2.LINE_AA,
            )
        for px, py in hand.landmarks_px:
            cv2.circle(frame, (px, py), 3, landmark_color, -1, cv2.LINE_AA)

    def close(self) -> None:
        """Release MediaPipe detector resources."""
        self._detector.close()

    def __enter__(self) -> "HandTracker":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ── Private ────────────────────────────────────────────────────────────────

    def _compute_extensions(
        self,
        landmarks: list,
        handedness: str,
    ) -> List[bool]:
        """
        Determine which fingers are extended.

        Non-thumb (index through pinky):
            Extended when tip.y < pip.y − FINGER_EXTENSION_Y_MARGIN.
            (y increases downward, so tip above PIP joint = extended.)

        Thumb:
            Compare tip.x vs joint-3.x; direction depends on handedness to
            handle mirror symmetry.

        Args:
            landmarks:  List of 21 NormalizedLandmark objects.
            handedness: "Left" or "Right".

        Returns:
            [thumb_ext, index_ext, middle_ext, ring_ext, pinky_ext]
        """
        tips   = [4, 8, 12, 16, 20]
        pips   = [3, 6, 10, 14, 18]
        margin = config.FINGER_EXTENSION_Y_MARGIN

        # ── Thumb extension ────────────────────────────────────────────────────
        # Debug: print key x values every frame so we can verify in live use.
        print(
            f"[THUMB] lm1.x={landmarks[1].x:.3f}  lm2.x={landmarks[2].x:.3f}"
            f"  lm3.x={landmarks[3].x:.3f}  lm4.x={landmarks[4].x:.3f}"
            f"  hand={handedness}",
            flush=True,
        )

        # Check 1 — lateral x-axis abduction: tip (lm4) vs MCP base (lm2).
        # lm2 is more stable than lm3 (IP) because it's further from the tip.
        if handedness == "Right":
            x_abducted = landmarks[4].x < landmarks[2].x
        else:
            x_abducted = landmarks[4].x > landmarks[2].x

        # Check 2 — spread distance: tip (lm4) vs middle-finger base (lm9).
        # When the thumb is genuinely abducted it is far from the centre of the
        # palm regardless of hand rotation — this catches cases where the x-axis
        # comparison alone fails due to hand tilt.
        import math as _math
        tip_to_palm_dist = _math.hypot(
            landmarks[4].x - landmarks[9].x,
            landmarks[4].y - landmarks[9].y,
        )
        dist_abducted = tip_to_palm_dist > config.THUMB_ABDUCTION_MIN_DIST

        # Either check passing is sufficient.
        thumb_ext = x_abducted or dist_abducted

        result = [thumb_ext]
        for tip_i, pip_i in zip(tips[1:], pips[1:]):
            result.append(landmarks[tip_i].y < landmarks[pip_i].y - margin)

        return result
