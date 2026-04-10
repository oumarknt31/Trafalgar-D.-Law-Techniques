"""
gestures/sequence_buffer.py
============================
Time-windowed gesture sequence detector for multi-step gestures.

Currently tracks one sequence: K-ROOM Awakening (覚醒).

K-ROOM sequence
---------------
    Step 0 — HORNS  (index + pinky held 0.4 s, fired by classifier)
    Step 1 — OPEN_PALM (all five extended, leading-edge from classifier)
    Step 2 — HORNS  (same as step 0)

All three steps must complete within config.SEQUENCE_WINDOW seconds of step 0
firing.  On completion, update() returns GestureType.K_ROOM_COMPLETE.

Gesture choice rationale
------------------------
Using HORNS → OPEN_PALM → HORNS avoids any conflict with the FIST gesture
(which collapses ROOM) and produces a visually distinct, memorable sequence.
OPEN_PALM in ROOM_ACTIVE is a no-op in the StateMachine, so forwarding it
during the sequence causes no unintended state transitions.

Reset policy
------------
The buffer resets to step 0 when:
    - The SEQUENCE_WINDOW elapses without completion.
    - HAND_LOST is received.
    - Any unexpected gesture is received mid-sequence (except NONE).

This intentionally strict reset policy makes accidental K-ROOM activation
extremely unlikely.

Extension note
--------------
If future gestures require sequences, add a new `_track_<name>` method and
call it from update().  The sequential step-index pattern used here scales
cleanly to any sequence length.
"""

from __future__ import annotations

import time
from typing import List, Optional

import config
from gestures.classifier import GestureType


# The ordered GestureType steps that form the K-ROOM sequence.
_K_ROOM_STEPS: List[GestureType] = [
    GestureType.HORNS,
    GestureType.OPEN_PALM,
    GestureType.HORNS,
]

# GestureTypes that are silently ignored mid-sequence (not treated as resets).
_IGNORED_GESTURES: frozenset[GestureType] = frozenset({
    GestureType.NONE,
})


class SequenceBuffer:
    """
    Monitors the gesture event stream for multi-step sequences.

    One instance lives for the application lifetime and receives every
    GestureType emitted by GestureClassifier.

    Usage (per frame):
        result = classifier.classify(hand, now)
        if result.fired:
            seq_result = buffer.update(result.fired, now)
            if seq_result == GestureType.K_ROOM_COMPLETE:
                # fire K-ROOM Awakening
    """

    def __init__(self) -> None:
        """Initialise K-ROOM tracking state."""
        self._k_step:        int   = 0      # next expected step index (0, 1, 2)
        self._k_start_time:  float = 0.0    # when step 0 fired

    def update(
        self,
        gesture: GestureType,
        now: Optional[float] = None,
    ) -> Optional[GestureType]:
        """
        Feed a gesture event into the buffer.

        Args:
            gesture: The GestureType emitted by GestureClassifier this frame.
            now:     Current timestamp.  Defaults to time.time(); supply an
                     explicit value in tests for deterministic behaviour.

        Returns:
            GestureType.K_ROOM_COMPLETE when the full K-ROOM sequence is
            completed within the time window, None otherwise.
        """
        if now is None:
            now = time.time()

        return self._track_k_room(gesture, now)

    def reset(self) -> None:
        """
        Force-reset all sequence tracking (e.g. on app force-reset).
        """
        self._k_step       = 0
        self._k_start_time = 0.0

    @property
    def k_room_step(self) -> int:
        """Current K-ROOM step index (0 = not started, 1–2 = in progress)."""
        return self._k_step

    # ── K-ROOM tracking ────────────────────────────────────────────────────────

    def _track_k_room(
        self,
        gesture: GestureType,
        now: float,
    ) -> Optional[GestureType]:
        """
        Advance (or reset) the K-ROOM sequence state machine.

        The sequence window is measured from when step 0 fires.  If any step
        takes too long or an unexpected gesture appears, we reset to step 0.

        Args:
            gesture: Incoming gesture type.
            now:     Current timestamp.

        Returns:
            GestureType.K_ROOM_COMPLETE on sequence completion, None otherwise.
        """
        # Silently ignore NONE and other non-meaningful gestures.
        if gesture in _IGNORED_GESTURES:
            return None

        # ── Window expiry check ────────────────────────────────────────────────
        if self._k_step > 0:
            elapsed = now - self._k_start_time
            if elapsed > config.SEQUENCE_WINDOW:
                self._k_step = 0   # window expired; reset silently

        # ── Step matching ──────────────────────────────────────────────────────
        expected = _K_ROOM_STEPS[self._k_step]

        if gesture == expected:
            if self._k_step == 0:
                # Sequence starting; record the start time.
                self._k_start_time = now
            self._k_step += 1

            if self._k_step == len(_K_ROOM_STEPS):
                # Sequence complete!
                self._k_step = 0
                return GestureType.K_ROOM_COMPLETE

        elif gesture == GestureType.HAND_LOST:
            # Hand lost mid-sequence — hard reset.
            self._k_step = 0

        else:
            # Unexpected gesture during an active sequence — reset.
            # If we're at step 0, this is just a non-matching gesture and
            # there's nothing to reset.
            if self._k_step > 0:
                self._k_step = 0

        return None

    # ── Debug representation ───────────────────────────────────────────────────

    def __repr__(self) -> str:
        steps_total = len(_K_ROOM_STEPS)
        return (
            f"SequenceBuffer("
            f"k_room={self._k_step}/{steps_total}, "
            f"window_remaining="
            f"{max(0.0, config.SEQUENCE_WINDOW - (time.time() - self._k_start_time)):.1f}s"
            f")"
        )
