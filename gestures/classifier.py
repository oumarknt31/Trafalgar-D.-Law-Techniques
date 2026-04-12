"""
gestures/classifier.py
======================
Maps per-frame HandData into discrete gesture events.

Changes in this revision
------------------------
Issue 1  — open_palm priority: shape detection now checks open_palm first and
           relies on the improved thumb detection in hand_tracker.py.
Issue 2  — GAMMA_KNIFE: added terminal debug print of hold_elapsed while
           index_only is active; added HAND_LOSS_GRACE_FRAMES so a brief
           tracking drop does not clear the armed flag.
Issue 3  — TAKT: _takt_stable counter requires TAKT_CONSECUTIVE_FRAMES
           consecutive orientation-passing frames before hold timer counts.
Issue 4  — SHAMBLES: SHAMBLES_FLICK_WINDOW and SWIPE_VELOCITY_THRESHOLD
           are now lower/wider (changed in config.py).  wrist_vx/wrist_vy
           are surfaced in ClassificationResult for the debug overlay.
Issue 5  — MES: CHEST_REGION_Y_MIN lowered (config.py).  hand_y_norm
           surfaced in ClassificationResult for the debug overlay.
Issue 6  — K-ROOM: added GestureType.FIST_HELD — a non-chest fist held for
           GESTURE_HOLD_K_ROOM_STEP seconds.  This is the new K-ROOM step 0,
           distinct from the leading-edge FIST that collapses ROOM.

Debounce policy
---------------
Continuous (OPEN_PALM, FIST): leading-edge only.
Hold-then-fire (TAKT, MES, HORNS, FIST_HELD): once per hold session.
Release-based (GAMMA_KNIFE): armed on hold, fires on shape break.
Motion (AMPUTATE, SHAMBLES): shape + velocity threshold per frame / window.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

import config
from gestures.definitions import GESTURES
from gestures.hand_tracker import HandData


class GestureType(Enum):
    """All gesture types produced by the classifier and sequence buffer."""
    NONE            = "none"
    HAND_LOST       = "hand_lost"
    OPEN_PALM       = "open_palm"
    FIST            = "fist"          # leading-edge; collapses ROOM
    FIST_HELD       = "fist_held"     # non-chest fist held 0.4 s; K-ROOM step 0
    AMPUTATE        = "amputate"
    GAMMA_KNIFE     = "gamma_knife"
    MES             = "mes"
    TAKT            = "takt"
    SHAMBLES        = "shambles"
    HORNS           = "horns"         # K-ROOM step 2
    K_ROOM_COMPLETE = "k_room_complete"


@dataclass
class ClassificationResult:
    """
    Output of GestureClassifier.classify() for one frame.

    Attributes:
        shape:          Raw hand shape label this frame.
        fired:          Gesture to forward this frame, or None.
        fingers:        [thumb, index, middle, ring, pinky] booleans.
        hold_progress:  0.0–1.0 toward current hold gesture's required duration.
        gamma_armed:    True when GAMMA_KNIFE hold is satisfied.
        shambles_armed: True when SHAMBLES two-finger hold is satisfied.
        wrist_speed:    Magnitude of wrist velocity (normalised units/s).
        wrist_vx:       Horizontal wrist velocity component (+ = right).
        wrist_vy:       Vertical wrist velocity component (+ = down).
        hand_y_norm:    Wrist y in normalised [0,1] space (for Mes tuning).
        hand_detected:  False when HandTracker returned None.
    """
    shape:          str
    fired:          Optional[GestureType]
    fingers:        List[bool]
    hold_progress:  float
    gamma_armed:    bool
    shambles_armed: bool
    wrist_speed:    float
    wrist_vx:       float
    wrist_vy:       float
    hand_y_norm:    float
    hand_detected:  bool


class GestureClassifier:
    """
    Stateful gesture classifier.  One instance lives for the application
    lifetime; call classify() once per frame.
    """

    def __init__(self) -> None:
        # ── Hold tracking ────────────────────────────────────────────────────────
        self._hold_shape:  Optional[str] = None
        self._hold_start:  float         = 0.0
        self._hold_fired:  bool          = False

        # ── GAMMA_KNIFE ──────────────────────────────────────────────────────────
        self._gamma_armed:              bool  = False
        self._gamma_fist_suppress_until: float = 0.0

        # ── SHAMBLES ────────────────────────────────────────────────────────────
        self._shambles_armed:    bool  = False
        self._shambles_arm_time: float = 0.0

        # ── TAKT consecutive-frame stability counter ─────────────────────────────
        # Counts frames where two_finger_v + upward orientation are both met.
        # Hold timer only progresses once this reaches TAKT_CONSECUTIVE_FRAMES.
        self._takt_stable: int = 0

        # ── Debounce ─────────────────────────────────────────────────────────────
        self._last_continuous: Optional[GestureType] = None

        # ── Hand-loss grace period ───────────────────────────────────────────────
        # Arm/hold state is only reset after HAND_LOSS_GRACE_FRAMES consecutive
        # no-hand frames, preventing brief tracking drops from aborting gestures.
        self._no_hand_frames:   int  = 0
        self._hand_was_present: bool = False

    # ── Public API ──────────────────────────────────────────────────────────────

    def classify(
        self,
        hand: Optional[HandData],
        now:  Optional[float] = None,
    ) -> ClassificationResult:
        """
        Classify hand state for one frame.

        Args:
            hand: HandData from HandTracker.process(), or None.
            now:  Timestamp. Defaults to time.time().

        Returns:
            ClassificationResult with shape, fired event, and debug fields.
        """
        if now is None:
            now = time.time()

        if hand is None:
            return self._no_hand(now)

        self._no_hand_frames   = 0
        self._hand_was_present = True

        fingers = hand.fingers_extended

        # 1. Raw shape detection.
        shape = self._detect_shape(hand, fingers)

        # 2. Update hold timer — reset on shape change.
        if shape != self._hold_shape:
            self._hold_shape  = shape
            self._hold_start  = now
            self._hold_fired  = False
            self._takt_stable = 0   # reset Takt stability on any shape change

        hold_elapsed = now - self._hold_start

        # 3. Evaluate gesture event.
        fired = self._evaluate(hand, shape, hold_elapsed, now)

        # 4. Hold progress for current shape.
        hold_progress = self._hold_progress(shape, hold_elapsed)

        return ClassificationResult(
            shape          = shape,
            fired          = fired,
            fingers        = fingers,
            hold_progress  = hold_progress,
            gamma_armed    = self._gamma_armed,
            shambles_armed = self._shambles_armed,
            wrist_speed    = hand.wrist_speed,
            wrist_vx       = hand.wrist_velocity[0],
            wrist_vy       = hand.wrist_velocity[1],
            hand_y_norm    = hand.landmarks_norm[0].y,
            hand_detected  = True,
        )

    # ── No-hand path ────────────────────────────────────────────────────────────

    def _no_hand(self, now: float) -> ClassificationResult:
        """
        Handle missing hand detection.

        Increments the grace-period counter.  State is reset only after
        HAND_LOSS_GRACE_FRAMES consecutive no-hand frames, so brief tracking
        drops (common at frame edges or fast motion) don't abort gestures.
        """
        fired = None

        if self._hand_was_present:
            self._no_hand_frames += 1
            if self._no_hand_frames >= config.HAND_LOSS_GRACE_FRAMES:
                fired = GestureType.HAND_LOST
                self._hand_was_present = False
                self._reset_all()

        return ClassificationResult(
            shape="none", fired=fired, fingers=[False] * 5,
            hold_progress=0.0, gamma_armed=False, shambles_armed=False,
            wrist_speed=0.0, wrist_vx=0.0, wrist_vy=0.0,
            hand_y_norm=0.0, hand_detected=False,
        )

    # ── Shape detection ─────────────────────────────────────────────────────────

    def _detect_shape(self, hand: HandData, fingers: List[bool]) -> str:
        """
        Map finger extension state and geometry to a named shape string.

        Priority order (highest first):
            open_palm     — all 5 extended (checked first, Issue 1 fix)
            fist          — index+middle+ring+pinky all curled
            horns         — index + pinky; middle + ring + thumb curled
            index_only    — only index extended
            knife_hand    — index+middle+ring+pinky extended, thumb curled
            two_finger_v  — index+middle, V-spread ≥ V_SIGN_MIN_SPREAD_DEGREES
            two_finger_pt — index+middle, spread below threshold
            unknown
        """
        T, I, M, R, P = fingers

        # open_palm MUST be checked before knife_hand — thumb fix in hand_tracker
        # ensures this only fires when the thumb is genuinely abducted.
        # Spread guard: mean x-distance between adjacent fingertips must exceed
        # MIN_PALM_FINGER_SPREAD, rejecting fists where all fingers accidentally
        # register as "extended" but remain physically bunched together.
        if T and I and M and R and P:
            lm = hand.landmarks_norm
            mean_spread = (
                abs(lm[8].x - lm[12].x) +
                abs(lm[12].x - lm[16].x) +
                abs(lm[16].x - lm[20].x)
            ) / 3.0
            if mean_spread >= config.MIN_PALM_FINGER_SPREAD:
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
            return "two_finger_v" if angle >= config.V_SIGN_MIN_SPREAD_DEGREES else "two_finger_pt"

        return "unknown"

    # ── Gesture evaluation ───────────────────────────────────────────────────────

    def _evaluate(
        self,
        hand:         HandData,
        shape:        str,
        hold_elapsed: float,
        now:          float,
    ) -> Optional[GestureType]:
        """
        Decide which gesture event (if any) fires this frame.

        Priority:
            1. GAMMA_KNIFE release (before FIST so armed state wins)
            2. OPEN_PALM (continuous leading-edge)
            3. FIST / MES / FIST_HELD (fist variants)
            4. AMPUTATE (motion)
            5. SHAMBLES (arm + flick)
            6. TAKT (hold with stability gate)
            7. HORNS (hold)
            8. GAMMA_KNIFE arm
        """

        # ── 1. GAMMA_KNIFE release ───────────────────────────────────────────────
        if self._gamma_armed and shape != "index_only":
            self._gamma_armed = False
            self._gamma_fist_suppress_until = now + config.GAMMA_KNIFE_FIST_SUPPRESS
            return GestureType.GAMMA_KNIFE

        # ── 2. OPEN_PALM ─────────────────────────────────────────────────────────
        # Hold-then-fire: user must keep the palm open for ROOM_CHARGE_HOLD
        # seconds before the event fires.  This gives hold_progress a meaningful
        # value to display and prevents accidental one-frame triggers.
        if shape == "open_palm":
            self._last_continuous = None   # clear FIST debounce if transitioning
            if not self._hold_fired and hold_elapsed >= config.ROOM_CHARGE_HOLD:
                self._hold_fired = True
                return GestureType.OPEN_PALM
            return None

        # Clear continuous debounce when leaving continuous gestures.
        self._last_continuous = None

        # ── 3. FIST variants ─────────────────────────────────────────────────────
        if shape == "fist":
            if self._check_position(hand, "chest"):
                # ── MES hold (chest fist, takes priority over FIST) ──────────────
                if not self._hold_fired:
                    if hold_elapsed >= GESTURES["mes"].hold_seconds:
                        self._hold_fired = True
                        return GestureType.MES
                return None   # silent while building toward MES
            else:
                # Non-chest fist — respect post-Gamma suppression.
                if now < self._gamma_fist_suppress_until:
                    return None

                # ── FIST_HELD (for K-ROOM step 0) ────────────────────────────────
                # Fires once after GESTURE_HOLD_K_ROOM_STEP seconds.
                if not self._hold_fired and hold_elapsed >= config.GESTURE_HOLD_K_ROOM_STEP:
                    self._hold_fired = True
                    return GestureType.FIST_HELD

                # ── Regular FIST (leading-edge, collapses ROOM) ──────────────────
                return self._leading_edge(GestureType.FIST)

        # ── 4. AMPUTATE ──────────────────────────────────────────────────────────
        if shape == "knife_hand":
            if (self._check_orientation(hand, "horizontal") and
                    self._check_motion(hand, "horizontal_swipe")):
                return GestureType.AMPUTATE

        # ── 5. SHAMBLES ──────────────────────────────────────────────────────────
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
                if not self._hold_fired and hold_elapsed >= GESTURES["shambles"].hold_seconds:
                    self._shambles_armed    = True
                    self._shambles_arm_time = now
                    self._hold_fired        = True

        # ── 6. TAKT (with consecutive-frame stability gate) ───────────────────────
        if shape == "two_finger_v":
            if self._check_orientation(hand, "upward"):
                self._takt_stable = min(
                    self._takt_stable + 1,
                    config.TAKT_CONSECUTIVE_FRAMES + 1,
                )
            else:
                self._takt_stable = 0

            if (self._takt_stable >= config.TAKT_CONSECUTIVE_FRAMES
                    and not self._hold_fired):
                if hold_elapsed >= GESTURES["takt"].hold_seconds:
                    self._hold_fired  = True
                    self._takt_stable = 0
                    return GestureType.TAKT
        else:
            # Shape left two_finger_v — reset stability counter.
            self._takt_stable = 0

        # ── 7. HORNS hold ─────────────────────────────────────────────────────────
        if shape == "horns":
            if not self._hold_fired:
                if hold_elapsed >= GESTURES["horns"].hold_seconds:
                    self._hold_fired = True
                    return GestureType.HORNS

        # ── 8. GAMMA_KNIFE arm ────────────────────────────────────────────────────
        if shape == "index_only" and not self._gamma_armed:
            # Debug: print hold duration every frame so we can verify timing.
            print(
                f"[GAMMA_KNIFE] index_only | hold={hold_elapsed:.3f}s / "
                f"{GESTURES['gamma_knife'].hold_seconds:.1f}s | "
                f"armed={self._gamma_armed} | fired={self._hold_fired}",
                flush=True,
            )
            if not self._hold_fired and hold_elapsed >= GESTURES["gamma_knife"].hold_seconds:
                self._gamma_armed = True
                self._hold_fired  = True
                print("[GAMMA_KNIFE] *** ARMED — curl finger to fire ***", flush=True)

        return None

    # ── Constraint checks ────────────────────────────────────────────────────────

    def _check_position(self, hand: HandData, constraint: str) -> bool:
        """Return True if the hand satisfies the positional constraint."""
        if constraint == "any":
            return True
        if constraint == "chest":
            return hand.landmarks_norm[0].y > config.CHEST_REGION_Y_MIN
        return True

    def _check_orientation(self, hand: HandData, constraint: str) -> bool:
        """
        Return True if the hand axis satisfies the orientation constraint.

        "horizontal": wrist→middle-tip angle within AMPUTATE_MAX_ANGLE_DEG of 0°/180°.
        "upward":     middle tip ≥ TAKT_MIN_UPWARD_RATIO × frame_height above wrist.
        """
        if constraint == "any":
            return True

        wrist = hand.landmarks_px[0]
        tip   = hand.landmarks_px[12]   # middle finger tip

        if constraint == "horizontal":
            dx = tip[0] - wrist[0]
            dy = tip[1] - wrist[1]
            if dx == 0 and dy == 0:
                return False
            angle = abs(math.degrees(math.atan2(dy, dx)))
            return (angle <= config.AMPUTATE_MAX_ANGLE_DEG or
                    angle >= 180.0 - config.AMPUTATE_MAX_ANGLE_DEG)

        if constraint == "upward":
            dy_px         = wrist[1] - tip[1]   # positive when tip is above wrist
            threshold_px  = config.TAKT_MIN_UPWARD_RATIO * hand.frame_height
            return dy_px >= threshold_px

        return True

    def _check_motion(self, hand: HandData, constraint: str) -> bool:
        """Return True if the wrist velocity satisfies the motion constraint."""
        if constraint == "none":
            return True
        if constraint == "horizontal_swipe":
            return abs(hand.wrist_velocity[0]) > config.SWIPE_VELOCITY_THRESHOLD
        if constraint == "any_swipe":
            return hand.wrist_speed > config.SWIPE_VELOCITY_THRESHOLD
        return True

    # ── Geometry helpers ─────────────────────────────────────────────────────────

    def _v_spread_angle(self, hand: HandData) -> float:
        """
        Angle (degrees) between index and middle finger vectors from palm centre.
        """
        origin = hand.landmarks_norm[9]
        idx    = hand.landmarks_norm[8]
        mid    = hand.landmarks_norm[12]

        v1x = idx.x - origin.x;  v1y = idx.y - origin.y
        v2x = mid.x - origin.x;  v2y = mid.y - origin.y

        mag1 = math.hypot(v1x, v1y)
        mag2 = math.hypot(v2x, v2y)
        if mag1 < 1e-9 or mag2 < 1e-9:
            return 0.0

        cos_a = max(-1.0, min(1.0, (v1x * v2x + v1y * v2y) / (mag1 * mag2)))
        return math.degrees(math.acos(cos_a))

    # ── Debounce ─────────────────────────────────────────────────────────────────

    def _leading_edge(self, gesture: GestureType) -> Optional[GestureType]:
        """Emit `gesture` only on the first frame of a continuous run."""
        if self._last_continuous == gesture:
            return None
        self._last_continuous = gesture
        return gesture

    # ── Hold progress ─────────────────────────────────────────────────────────────

    def _hold_progress(self, shape: str, hold_elapsed: float) -> float:
        """0.0–1.0 progress toward current shape's hold requirement."""
        if shape == "open_palm":
            required = config.ROOM_CHARGE_HOLD
            return 0.0 if required <= 0.0 else min(1.0, hold_elapsed / required)
        shape_to_gesture = {
            "two_finger_v":  "takt",
            "two_finger_pt": "shambles",
            "fist":          "mes",
            "horns":         "horns",
            "index_only":    "gamma_knife",
        }
        if shape not in shape_to_gesture:
            return 0.0
        required = GESTURES[shape_to_gesture[shape]].hold_seconds
        return 0.0 if required <= 0.0 else min(1.0, hold_elapsed / required)

    # ── Reset ─────────────────────────────────────────────────────────────────────

    def _reset_all(self) -> None:
        """Reset all state (called after hand-loss grace period expires)."""
        self._hold_shape              = None
        self._hold_start              = 0.0
        self._hold_fired              = False
        self._gamma_armed             = False
        self._shambles_armed          = False
        self._shambles_arm_time       = 0.0
        self._takt_stable             = 0
        self._last_continuous         = None
        self._no_hand_frames          = 0
