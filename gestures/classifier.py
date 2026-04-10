"""
gestures/classifier.py
======================
Maps per-frame HandData into discrete gesture events.

Architecture
------------
The classifier operates in two conceptual layers:

  1. Shape detection   — What does the hand look like right now?
                         Returns a string label such as "open_palm", "fist",
                         "knife_hand", "index_only", "two_finger_v",
                         "two_finger_pt", "horns", or "unknown".

  2. Gesture evaluation — Given the shape + hold duration + position +
                          geometry + motion, should a gesture event fire?

The two layers are kept separate so the debug overlay (test_gestures.py) can
show the raw shape even when no event is ready to fire.

Debounce policy
---------------
Continuous gestures (OPEN_PALM, FIST): emit only on the LEADING EDGE of the
shape transition (i.e. the first frame the shape appears).  They are NOT
re-emitted on every subsequent frame.

Hold gestures (TAKT, MES, HORNS): emit exactly once per hold session when
hold_elapsed ≥ hold_seconds.  Reset if the shape breaks before the hold
completes.

Release gesture (GAMMA_KNIFE): arm when hold completes, fire on the first
frame the shape breaks.  To prevent the resulting fist from accidentally
collapsing ROOM, FIST emission is suppressed for GAMMA_KNIFE_FIST_SUPPRESS
seconds after GAMMA_KNIFE fires.

Motion gestures (AMPUTATE, SHAMBLES):
  - AMPUTATE fires on any frame where the knife-hand shape is present,
    the hand is oriented horizontally, AND horizontal wrist speed exceeds
    SWIPE_VELOCITY_THRESHOLD.
  - SHAMBLES fires in two stages: (a) arm when the two-finger-point shape
    is held for GESTURE_HOLD_SHAMBLES_STEP1 seconds; (b) fire when any wrist
    swipe exceeds SWIPE_VELOCITY_THRESHOLD within SHAMBLES_FLICK_WINDOW
    seconds of arming.

MES vs FIST disambiguation
---------------------------
Both use a fist shape.  MES is checked first; if the fist is in the chest
region (wrist y-norm > CHEST_REGION_Y_MIN), we enter the MES hold path and
never emit FIST.  A chest fist that doesn't reach the hold threshold simply
emits nothing — it will not collapse ROOM.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import config
from gestures.definitions import GESTURES, K_ROOM_STEPS
from gestures.hand_tracker import HandData


class GestureType(Enum):
    """
    Discrete gesture types recognised by the classifier and sequence buffer.

    Mapping to core.events.EventType is deferred to the main loop (Phase 4)
    so this module stays independent of the state machine.
    """
    NONE            = "none"
    HAND_LOST       = "hand_lost"
    OPEN_PALM       = "open_palm"        # all 5 extended — leading-edge
    FIST            = "fist"             # all non-thumb curled — leading-edge
    AMPUTATE        = "amputate"         # knife-hand horizontal swipe
    GAMMA_KNIFE     = "gamma_knife"      # index hold → release
    MES             = "mes"             # chest-fist hold
    TAKT            = "takt"            # V-sign upward hold
    SHAMBLES        = "shambles"         # two-finger arm + flick
    HORNS           = "horns"           # index+pinky hold — K-ROOM step only
    K_ROOM_COMPLETE = "k_room_complete"  # emitted by SequenceBuffer


@dataclass
class ClassificationResult:
    """
    Output of GestureClassifier.classify() for one frame.

    Attributes:
        shape:         Raw hand shape label this frame (for debug display).
        fired:         Gesture event to forward, or None.  Non-None only on
                       the frame a gesture triggers.
        fingers:       [thumb, index, middle, ring, pinky] extension bools.
        hold_progress: 0.0–1.0 fraction toward current hold gesture's
                       required duration.  0.0 when no hold is in progress.
        gamma_armed:   True when GAMMA_KNIFE hold is satisfied; awaiting release.
        shambles_armed:True when SHAMBLES two-finger hold is satisfied; awaiting flick.
        wrist_speed:   Normalised wrist speed for debug overlay.
        hand_detected: False when HandTracker returned None.
    """
    shape:          str
    fired:          Optional[GestureType]
    fingers:        List[bool]
    hold_progress:  float
    gamma_armed:    bool
    shambles_armed: bool
    wrist_speed:    float
    hand_detected:  bool


class GestureClassifier:
    """
    Stateful gesture classifier.  One instance should live for the lifetime
    of the application and receive every processed frame via classify().

    The classifier is deliberately decoupled from the state machine — it does
    not know which techniques are on cooldown or whether ROOM is active.
    Filtering by game state happens in the main loop.
    """

    def __init__(self) -> None:
        """Initialise all hold / arm tracking state to clean defaults."""
        # ── Hold state ──────────────────────────────────────────────────────────
        self._hold_shape:  Optional[str]  = None   # shape currently being held
        self._hold_start:  float          = 0.0    # when current hold began
        self._hold_fired:  bool           = False  # did this hold session fire?

        # ── GAMMA_KNIFE ─────────────────────────────────────────────────────────
        self._gamma_armed:       bool  = False
        self._gamma_fist_suppress_until: float = 0.0  # suppress FIST until this time

        # ── SHAMBLES ────────────────────────────────────────────────────────────
        self._shambles_armed:    bool  = False
        self._shambles_arm_time: float = 0.0

        # ── Debounce for continuous gestures ────────────────────────────────────
        self._last_continuous: Optional[GestureType] = None

        # ── HAND_LOST debounce ──────────────────────────────────────────────────
        self._hand_was_present: bool = False

    # ── Public API ─────────────────────────────────────────────────────────────

    def classify(
        self,
        hand: Optional[HandData],
        now:  Optional[float] = None,
    ) -> ClassificationResult:
        """
        Classify hand state for one frame.

        Args:
            hand: HandData from HandTracker.process(), or None if no hand.
            now:  Current timestamp.  Defaults to time.time(); supply an
                  explicit value in tests for deterministic behaviour.

        Returns:
            ClassificationResult describing the current shape and any fired event.
        """
        if now is None:
            now = time.time()

        if hand is None:
            return self._no_hand(now)

        self._hand_was_present = True
        fingers = hand.fingers_extended

        # 1. Detect raw shape.
        shape = self._detect_shape(hand, fingers)

        # 2. Update hold timer (reset if shape changed).
        if shape != self._hold_shape:
            self._hold_shape = shape
            self._hold_start = now
            self._hold_fired = False

        hold_elapsed = now - self._hold_start

        # 3. Evaluate whether a gesture fires this frame.
        fired = self._evaluate(hand, shape, hold_elapsed, now)

        # 4. Compute progress toward current hold target.
        hold_progress = self._hold_progress(shape, hold_elapsed)

        return ClassificationResult(
            shape          = shape,
            fired          = fired,
            fingers        = fingers,
            hold_progress  = hold_progress,
            gamma_armed    = self._gamma_armed,
            shambles_armed = self._shambles_armed,
            wrist_speed    = hand.wrist_speed,
            hand_detected  = True,
        )

    # ── No-hand path ───────────────────────────────────────────────────────────

    def _no_hand(self, now: float) -> ClassificationResult:
        """Return a HAND_LOST event on the first no-hand frame; None thereafter."""
        fired = None
        if self._hand_was_present:
            fired = GestureType.HAND_LOST
            self._hand_was_present = False
            self._reset_all()

        return ClassificationResult(
            shape="none", fired=fired, fingers=[False] * 5,
            hold_progress=0.0, gamma_armed=False, shambles_armed=False,
            wrist_speed=0.0, hand_detected=False,
        )

    # ── Shape detection ────────────────────────────────────────────────────────

    def _detect_shape(self, hand: HandData, fingers: List[bool]) -> str:
        """
        Map finger extension state and basic geometry to a named shape string.

        Shapes (checked in priority order):
            "open_palm"     — all 5 extended
            "fist"          — index+middle+ring+pinky all curled (thumb free)
            "horns"         — index + pinky extended; middle + ring + thumb curled
            "index_only"    — only index extended; others curled (thumb free)
            "knife_hand"    — index+middle+ring+pinky extended; thumb curled
            "two_finger_v"  — index+middle extended, V-spread, thumb curled
            "two_finger_pt" — index+middle extended, close together, thumb curled
            "unknown"       — anything else

        Note on "two_finger_v" vs "two_finger_pt": spread angle is checked
        here to produce distinct shape labels.  Orientation (upward/any) is
        validated later in _evaluate() so the debug overlay shows the correct
        sub-shape even when orientation is wrong.
        """
        T, I, M, R, P = fingers

        if T and I and M and R and P:
            return "open_palm"

        if not I and not M and not R and not P:
            return "fist"

        if I and not M and not R and P and not T:
            return "horns"

        if I and not M and not R and not P:
            return "index_only"

        if I and M and R and P and not T:
            return "knife_hand"

        if I and M and not R and not P and not T:
            angle = self._v_spread_angle(hand)
            if angle >= config.V_SIGN_MIN_SPREAD_DEGREES:
                return "two_finger_v"
            return "two_finger_pt"

        return "unknown"

    # ── Gesture evaluation ─────────────────────────────────────────────────────

    def _evaluate(
        self,
        hand:         HandData,
        shape:        str,
        hold_elapsed: float,
        now:          float,
    ) -> Optional[GestureType]:
        """
        Decide whether a gesture event should fire this frame.

        Priority order:
            1. GAMMA_KNIFE release (check before FIST so armed state wins)
            2. OPEN_PALM continuous (leading edge)
            3. FIST continuous (leading edge, with post-Gamma suppression)
            4. AMPUTATE (knife hand + horizontal + swipe)
            5. SHAMBLES flick (if armed) or SHAMBLES arm
            6. TAKT hold
            7. MES hold (fist in chest region — must come before FIST path)
            8. HORNS hold
            9. GAMMA_KNIFE arm (index_only hold complete → set armed)
        """
        # ── 1. GAMMA_KNIFE release ─────────────────────────────────────────────
        if self._gamma_armed and shape != "index_only":
            self._gamma_armed = False
            self._gamma_fist_suppress_until = now + config.GAMMA_KNIFE_FIST_SUPPRESS
            return GestureType.GAMMA_KNIFE

        # ── 2. OPEN_PALM ───────────────────────────────────────────────────────
        if shape == "open_palm":
            return self._leading_edge(GestureType.OPEN_PALM)

        # ── 3. FIST ────────────────────────────────────────────────────────────
        if shape == "fist":
            # MES takes priority when the hand is in chest position.
            if self._check_position(hand, "chest"):
                # MES hold path
                if not self._hold_fired:
                    if hold_elapsed >= GESTURES["mes"].hold_seconds:
                        self._hold_fired = True
                        return GestureType.MES
                return None   # fist at chest emits nothing until MES fires

            # Regular FIST — respect post-Gamma suppression.
            if now < self._gamma_fist_suppress_until:
                return None
            self._last_continuous = None   # reset continuous so any non-fist resets
            return self._leading_edge(GestureType.FIST)

        # Not a continuous gesture — reset continuous debounce.
        self._last_continuous = None

        # ── 4. AMPUTATE ────────────────────────────────────────────────────────
        if shape == "knife_hand":
            if (self._check_orientation(hand, "horizontal") and
                    self._check_motion(hand, "horizontal_swipe")):
                return GestureType.AMPUTATE

        # ── 5. SHAMBLES ────────────────────────────────────────────────────────
        if shape == "two_finger_pt":
            if self._shambles_armed:
                elapsed_since_arm = now - self._shambles_arm_time
                if elapsed_since_arm <= config.SHAMBLES_FLICK_WINDOW:
                    if self._check_motion(hand, "any_swipe"):
                        self._shambles_armed = False
                        return GestureType.SHAMBLES
                else:
                    self._shambles_armed = False   # window expired
            else:
                # Attempt to arm
                if not self._hold_fired and hold_elapsed >= GESTURES["shambles"].hold_seconds:
                    self._shambles_armed    = True
                    self._shambles_arm_time = now
                    self._hold_fired        = True

        # ── 6. TAKT ────────────────────────────────────────────────────────────
        if shape == "two_finger_v":
            if self._check_orientation(hand, "upward") and not self._hold_fired:
                if hold_elapsed >= GESTURES["takt"].hold_seconds:
                    self._hold_fired = True
                    return GestureType.TAKT

        # ── 7. HORNS ───────────────────────────────────────────────────────────
        if shape == "horns":
            if not self._hold_fired:
                if hold_elapsed >= GESTURES["horns"].hold_seconds:
                    self._hold_fired = True
                    return GestureType.HORNS

        # ── 8. GAMMA_KNIFE arm ─────────────────────────────────────────────────
        if shape == "index_only" and not self._gamma_armed:
            if not self._hold_fired and hold_elapsed >= GESTURES["gamma_knife"].hold_seconds:
                self._gamma_armed = True
                self._hold_fired  = True   # suppress re-arming in same session

        return None

    # ── Constraint checks ──────────────────────────────────────────────────────

    def _check_position(self, hand: HandData, constraint: str) -> bool:
        """
        Return True if the hand satisfies the positional constraint.

        Args:
            hand:       Current HandData.
            constraint: "any" or "chest".
        """
        if constraint == "any":
            return True
        if constraint == "chest":
            # Use wrist y (landmark 0) in normalised space.
            return hand.landmarks_norm[0].y > config.CHEST_REGION_Y_MIN
        return True

    def _check_orientation(self, hand: HandData, constraint: str) -> bool:
        """
        Return True if the hand's major axis satisfies the orientation constraint.

        "horizontal": angle from wrist (landmark 0) to middle fingertip
                      (landmark 12) is within AMPUTATE_MAX_ANGLE_DEG of 0° or 180°.
        "upward":     middle fingertip is at least TAKT_MIN_UPWARD_RATIO ×
                      frame_height pixels above the wrist (in pixel space).

        Args:
            hand:       Current HandData.
            constraint: "any", "horizontal", or "upward".
        """
        if constraint == "any":
            return True

        wrist_px = hand.landmarks_px[0]
        tip_px   = hand.landmarks_px[12]   # middle finger tip

        if constraint == "horizontal":
            dx = tip_px[0] - wrist_px[0]
            dy = tip_px[1] - wrist_px[1]
            if dx == 0 and dy == 0:
                return False
            angle_deg = abs(math.degrees(math.atan2(dy, dx)))
            # Horizontal: angle near 0° or near 180°
            return (angle_deg <= config.AMPUTATE_MAX_ANGLE_DEG or
                    angle_deg >= 180.0 - config.AMPUTATE_MAX_ANGLE_DEG)

        if constraint == "upward":
            # In pixel space, y increases downward.  Upward = tip_y < wrist_y.
            dy_px = wrist_px[1] - tip_px[1]   # positive when tip is above wrist
            threshold_px = config.TAKT_MIN_UPWARD_RATIO * hand.frame_height
            return dy_px >= threshold_px

        return True

    def _check_motion(self, hand: HandData, constraint: str) -> bool:
        """
        Return True if the wrist velocity satisfies the motion constraint.

        Args:
            hand:       Current HandData.
            constraint: "none", "horizontal_swipe", or "any_swipe".
        """
        if constraint == "none":
            return True
        if constraint == "horizontal_swipe":
            return abs(hand.wrist_velocity[0]) > config.SWIPE_VELOCITY_THRESHOLD
        if constraint == "any_swipe":
            return hand.wrist_speed > config.SWIPE_VELOCITY_THRESHOLD
        return True

    # ── Geometry helpers ───────────────────────────────────────────────────────

    def _v_spread_angle(self, hand: HandData) -> float:
        """
        Compute the angle (degrees) between the index and middle finger
        vectors, measured from the palm centre (landmark 9).

        Returns 0.0 if either vector has zero magnitude.

        Args:
            hand: Current HandData.
        """
        origin = hand.landmarks_norm[9]
        idx    = hand.landmarks_norm[8]   # index tip
        mid    = hand.landmarks_norm[12]  # middle tip

        v1x = idx.x - origin.x;  v1y = idx.y - origin.y
        v2x = mid.x - origin.x;  v2y = mid.y - origin.y

        mag1 = math.hypot(v1x, v1y)
        mag2 = math.hypot(v2x, v2y)
        if mag1 < 1e-9 or mag2 < 1e-9:
            return 0.0

        dot   = v1x * v2x + v1y * v2y
        cos_a = max(-1.0, min(1.0, dot / (mag1 * mag2)))
        return math.degrees(math.acos(cos_a))

    # ── Debounce helpers ───────────────────────────────────────────────────────

    def _leading_edge(self, gesture: GestureType) -> Optional[GestureType]:
        """
        Emit `gesture` only on the first frame of a continuous gesture run.
        Subsequent frames with the same shape return None.

        Args:
            gesture: The continuous GestureType to debounce.

        Returns:
            gesture on leading edge; None on continuation frames.
        """
        if self._last_continuous == gesture:
            return None
        self._last_continuous = gesture
        return gesture

    # ── Hold progress ──────────────────────────────────────────────────────────

    def _hold_progress(self, shape: str, hold_elapsed: float) -> float:
        """
        Return 0.0–1.0 progress toward the current shape's hold requirement.
        Returns 0.0 for shapes that have no hold requirement.

        Args:
            shape:        Current detected shape label.
            hold_elapsed: Seconds since the current shape began.
        """
        shape_to_gesture = {
            "two_finger_v":  "takt",
            "two_finger_pt": "shambles",
            "fist":          "mes",    # approximation; valid in chest region
            "horns":         "horns",
            "index_only":    "gamma_knife",
        }
        if shape not in shape_to_gesture:
            return 0.0
        key = shape_to_gesture[shape]
        required = GESTURES[key].hold_seconds
        if required <= 0.0:
            return 0.0
        return min(1.0, hold_elapsed / required)

    # ── Reset helpers ──────────────────────────────────────────────────────────

    def _reset_all(self) -> None:
        """Reset all hold/arm state (called when hand is lost)."""
        self._hold_shape         = None
        self._hold_start         = 0.0
        self._hold_fired         = False
        self._gamma_armed        = False
        self._shambles_armed     = False
        self._shambles_arm_time  = 0.0
        self._last_continuous    = None
