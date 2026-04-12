"""
test_gestures.py
================
Live webcam debug tool for the gesture perception layer.

Purpose
-------
Validate that all 7 gestures detect reliably BEFORE connecting the gesture
engine to the state machine.  This script is purely observational — it does
not fire any StateMachine events.

Controls
--------
    Q  — Quit
    D  — Toggle debug overlay (finger state, wrist speed, hold bar, K-ROOM step)
    R  — Reset the sequence buffer (clear in-progress K-ROOM sequence)

Debug overlay elements
----------------------
    TOP-CENTER  Large gesture name label.  Displays the last *fired* gesture
                for LABEL_PERSIST_S seconds, then falls back to the current
                raw shape.
    TOP-LEFT    Finger extension indicators: T I M R P  (cyan = extended,
                dim = curled).
    TOP-RIGHT   Wrist speed (normalised units/s).
    MID-LEFT    Current raw shape label.
    HOLD BAR    Thin bar at bottom of frame; fills left-to-right as hold
                progress advances toward 1.0.
    STATUS      "ARMED" badge when GAMMA_KNIFE is charged; flashes cyan.
    K-ROOM      Sequence step indicator  ●●○  below the main label.

Run with:
    python test_gestures.py           (if venv is active)
    venv/bin/python test_gestures.py  (explicit venv)
"""

from __future__ import annotations

import time
from collections import deque
from typing import Optional

import cv2
import numpy as np

import config
from gestures.classifier import ClassificationResult, GestureClassifier, GestureType  # noqa: F401
from gestures.hand_tracker import HandTracker
from gestures.sequence_buffer import SequenceBuffer

# ── Layout constants ───────────────────────────────────────────────────────────
LABEL_PERSIST_S   = 1.2    # seconds to keep fired-gesture label visible
FONT_MAIN         = cv2.FONT_HERSHEY_DUPLEX
FONT_SMALL        = cv2.FONT_HERSHEY_SIMPLEX
BANNER_H          = 48     # top HUD banner height in pixels
HOLD_BAR_H        = 6      # hold-progress bar thickness

# ── Colors ─────────────────────────────────────────────────────────────────────
TEAL_BRIGHT = config.TEAL_BRIGHT
TEAL_MID    = config.TEAL_MID
TEAL_DARK   = config.TEAL_DARK
WHITE       = config.WHITE
BLACK       = config.BLACK
DIM_GREY    = (60, 60, 60)
AMBER       = (0, 165, 255)   # BGR amber for "ARMED"

# ── Gesture → display label ────────────────────────────────────────────────────
GESTURE_LABELS: dict[str, str] = {
    "open_palm":       "OPEN PALM — ROOM",
    "fist":            "FIST — Deactivate",
    "amputate":        "AMPUTATE  切断",
    "gamma_knife":     "GAMMA KNIFE  ガンマナイフ",
    "mes":             "MES  メス",
    "takt":            "TAKT  タクト",
    "shambles":        "SHAMBLES  シャンブルズ",
    "horns":           "HORNS (K-ROOM step 2)",
    "fist_held":       "FIST HELD (K-ROOM step 0)",
    "k_room_complete": "★ K-ROOM AWAKENING ★",
    "hand_lost":       "HAND LOST",
    "none":            "",
}


# ── Drawing helpers ────────────────────────────────────────────────────────────

def draw_banner(frame: np.ndarray, h: int, w: int) -> None:
    """Draw a semi-transparent dark banner at the top of the frame."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, BANNER_H), BLACK, -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)


def draw_main_label(
    frame:   np.ndarray,
    label:   str,
    w:       int,
    color:   tuple = TEAL_BRIGHT,
) -> None:
    """Draw the main gesture name centred in the top banner."""
    if not label:
        return
    scale      = 0.75
    thickness  = 1
    (tw, _), _ = cv2.getTextSize(label, FONT_MAIN, scale, thickness)
    x = (w - tw) // 2
    cv2.putText(frame, label, (x, 32), FONT_MAIN, scale, color, thickness, cv2.LINE_AA)


def draw_finger_state(
    frame:   np.ndarray,
    fingers: list[bool],
    debug:   bool,
) -> None:
    """Draw T I M R P indicators in the top-left corner."""
    if not debug:
        return
    labels  = ["T", "I", "M", "R", "P"]
    x_start = 12
    for i, (lbl, ext) in enumerate(zip(labels, fingers)):
        color = TEAL_BRIGHT if ext else DIM_GREY
        cv2.putText(
            frame, lbl,
            (x_start + i * 22, 30),
            FONT_SMALL, 0.6, color, 1, cv2.LINE_AA,
        )


def draw_wrist_velocity(
    frame:  np.ndarray,
    result: "ClassificationResult",
    w:      int,
    debug:  bool,
) -> None:
    """
    Draw wrist velocity readout in the top-right corner.
    Shows speed magnitude + vx/vy components.
    Components turn bright cyan when above SWIPE_VELOCITY_THRESHOLD (Shambles tuning).
    """
    if not debug:
        return

    threshold = config.SWIPE_VELOCITY_THRESHOLD
    vx_color  = TEAL_BRIGHT if abs(result.wrist_vx) > threshold else TEAL_MID
    vy_color  = TEAL_BRIGHT if abs(result.wrist_vy) > threshold else TEAL_MID
    spd_color = TEAL_BRIGHT if result.wrist_speed > threshold else TEAL_MID

    lines = [
        (f"spd {result.wrist_speed:.3f}", spd_color),
        (f"vx  {result.wrist_vx:+.3f}",  vx_color),
        (f"vy  {result.wrist_vy:+.3f}",  vy_color),
    ]
    scale = 0.40
    for i, (text, color) in enumerate(lines):
        (tw, _), _ = cv2.getTextSize(text, FONT_SMALL, scale, 1)
        cv2.putText(
            frame, text,
            (w - tw - 10, 26 + i * 18),
            FONT_SMALL, scale, color, 1, cv2.LINE_AA,
        )


def draw_hand_y_norm(
    frame:      np.ndarray,
    result:     "ClassificationResult",
    debug:      bool,
) -> None:
    """
    Draw the wrist y-norm value when a fist is detected (Mes tuning).
    Turns bright cyan when y_norm exceeds CHEST_REGION_Y_MIN.
    """
    if not debug or not result.hand_detected:
        return
    if result.shape != "fist":
        return
    y    = result.hand_y_norm
    over = y > config.CHEST_REGION_Y_MIN
    col  = TEAL_BRIGHT if over else TEAL_MID
    text = f"y_norm {y:.3f}  {'[CHEST]' if over else '[above chest]'}"
    cv2.putText(frame, text, (12, 80), FONT_SMALL, 0.44, col, 1, cv2.LINE_AA)


def draw_shape_label(
    frame: np.ndarray,
    shape: str,
    debug: bool,
) -> None:
    """Draw the raw shape name below the main banner."""
    if not debug:
        return
    cv2.putText(
        frame, f"shape: {shape}",
        (12, BANNER_H + 22),
        FONT_SMALL, 0.45, TEAL_DARK, 1, cv2.LINE_AA,
    )


def draw_hold_bar(
    frame:    np.ndarray,
    progress: float,
    h:        int,
    w:        int,
) -> None:
    """Fill a thin bar along the bottom of the frame to show hold progress."""
    if progress <= 0.0:
        return
    bar_w = int(w * progress)
    y     = h - HOLD_BAR_H
    cv2.rectangle(frame, (0, y), (bar_w, h), TEAL_BRIGHT, -1)
    # Dim unfilled portion
    cv2.rectangle(frame, (bar_w, y), (w, h), DIM_GREY, -1)


def draw_armed_badge(
    frame: np.ndarray,
    armed: bool,
    h:     int,
    w:     int,
    now:   float,
) -> None:
    """Draw a pulsing ARMED badge in the bottom-right when Gamma Knife is charged."""
    if not armed:
        return
    pulse   = 0.6 + 0.4 * abs(np.sin(now * 6.0))
    color   = tuple(int(c * pulse) for c in AMBER)
    text    = "ARMED"
    scale   = 0.7
    (tw, _), _ = cv2.getTextSize(text, FONT_MAIN, scale, 1)
    cv2.putText(
        frame, text,
        (w - tw - 16, h - HOLD_BAR_H - 12),
        FONT_MAIN, scale, color, 1, cv2.LINE_AA,
    )


def draw_shambles_armed_badge(
    frame: np.ndarray,
    armed: bool,
    h:     int,
    now:   float,
) -> None:
    """Draw a pulsing FLICK! badge when Shambles is armed and awaiting swipe."""
    if not armed:
        return
    pulse = 0.6 + 0.4 * abs(np.sin(now * 8.0))
    color = tuple(int(c * pulse) for c in TEAL_BRIGHT)
    cv2.putText(
        frame, "FLICK!",
        (16, h - HOLD_BAR_H - 12),
        FONT_MAIN, 0.7, color, 1, cv2.LINE_AA,
    )


def draw_k_room_progress(
    frame:     np.ndarray,
    step:      int,
    w:         int,
    debug:     bool,
) -> None:
    """
    Draw K-ROOM sequence step indicator  ●●○  below the main label.
    Each dot is filled (●) when that step has been completed.
    """
    if not debug:
        return
    total  = 3
    radius = 7
    gap    = 24
    total_w = total * radius * 2 + (total - 1) * gap
    x_start = (w - total_w) // 2
    y       = BANNER_H + 20

    for i in range(total):
        cx     = x_start + i * (radius * 2 + gap) + radius
        filled = i < step
        color  = TEAL_BRIGHT if filled else DIM_GREY
        cv2.circle(frame, (cx, y), radius, color, -1 if filled else 1, cv2.LINE_AA)

    label = "K-ROOM sequence"
    (tw, _), _ = cv2.getTextSize(label, FONT_SMALL, 0.38, 1)
    cv2.putText(
        frame, label,
        ((w - tw) // 2, y + radius + 14),
        FONT_SMALL, 0.38, TEAL_DARK, 1, cv2.LINE_AA,
    )


def draw_hint(frame: np.ndarray, h: int, w: int) -> None:
    """Draw key-hint line at the very bottom."""
    hint = "Q=quit  D=toggle debug  R=reset sequence"
    cv2.putText(
        frame, hint,
        (16, h - HOLD_BAR_H - 36),
        FONT_SMALL, 0.38,
        tuple(int(c * 0.5) for c in TEAL_MID),
        1, cv2.LINE_AA,
    )


# ── Main loop ──────────────────────────────────────────────────────────────────

def main() -> None:
    """Open the webcam and run the gesture debug loop."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: cannot open webcam (index 0).")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    tracker    = HandTracker()
    classifier = GestureClassifier()
    seq_buf    = SequenceBuffer()

    debug_overlay      = True
    last_fired_gesture: Optional[str]  = None
    last_fired_time:    float          = 0.0

    print("Gesture debug tool running.  Q=quit  D=toggle debug  R=reset K-ROOM")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]
        now   = time.time()

        # ── Perception ─────────────────────────────────────────────────────────
        hand   = tracker.process(frame)
        result = classifier.classify(hand, now)

        # Feed fired event to sequence buffer
        seq_result: Optional[GestureType] = None
        if result.fired is not None:
            seq_result = seq_buf.update(result.fired, now)

        # The final "event" to display (sequence result takes priority)
        display_fired = seq_result or result.fired

        if display_fired is not None and display_fired != GestureType.NONE:
            last_fired_gesture = display_fired.value
            last_fired_time    = now

        # ── Draw landmarks ─────────────────────────────────────────────────────
        if hand is not None:
            tracker.draw_landmarks(frame, hand)

        # ── HUD ────────────────────────────────────────────────────────────────
        draw_banner(frame, h, w)

        # Main label: show last fired gesture for LABEL_PERSIST_S, then shape.
        # Special case: while open_palm is held, show "ROOM XX%" charge progress.
        if last_fired_gesture and (now - last_fired_time) < LABEL_PERSIST_S:
            label = GESTURE_LABELS.get(last_fired_gesture, last_fired_gesture.upper())
            label_color = WHITE if last_fired_gesture == "k_room_complete" else TEAL_BRIGHT
        elif result.hand_detected and result.shape == "open_palm":
            pct = int(result.hold_progress * 100)
            label = f"ROOM {pct}%"
            label_color = TEAL_BRIGHT if result.hold_progress >= 1.0 else TEAL_MID
        else:
            label = result.shape if result.hand_detected else "— no hand —"
            label_color = TEAL_MID

        draw_main_label(frame, label, w, label_color)

        if debug_overlay:
            draw_finger_state(frame, result.fingers, debug=True)
            draw_wrist_velocity(frame, result, w, debug=True)
            draw_shape_label(frame, result.shape, debug=True)
            draw_hand_y_norm(frame, result, debug=True)
            draw_k_room_progress(frame, seq_buf.k_room_step, w, debug=True)

        draw_hold_bar(frame, result.hold_progress, h, w)
        draw_armed_badge(frame, result.gamma_armed, h, w, now)
        draw_shambles_armed_badge(frame, result.shambles_armed, h, now)
        draw_hint(frame, h, w)

        cv2.imshow("Gesture Debug — Trafalgar Law", frame)

        # ── Key handling ───────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("d"):
            debug_overlay = not debug_overlay
            print(f"Debug overlay: {'ON' if debug_overlay else 'OFF'}")
        elif key == ord("r"):
            seq_buf.reset()
            print("K-ROOM sequence buffer reset.")

    tracker.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
