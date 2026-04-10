"""
core/events.py
==============
Defines all event types that flow through the application and the
GestureEvent dataclass that carries them.

Design notes:
    - EventType uses auto() so integer values are stable but not meaningful.
    - GestureEvent is intentionally lightweight; it carries only the type and
      a timestamp.  Landmark data stays in the gesture layer and is never
      passed downstream — the state machine and renderers operate on abstract
      events, not raw MediaPipe output.
    - TECHNIQUE_COMPLETE, COLLAPSE_COMPLETE, and AWAKENING_COMPLETE are
      "internal" system events fired by effect renderers back into the state
      machine when their animation finishes.  They are listed here so there
      is a single canonical event vocabulary.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto


class EventType(Enum):
    """Exhaustive list of events recognised by the state machine."""

    # ── Gesture events (produced by gestures/classifier.py) ───────────────────
    OPEN_PALM       = auto()   # all 5 fingers extended — initiates ROOM charge
    FIST            = auto()   # all fingers curled — deactivates ROOM
    SHAMBLES        = auto()   # two-finger point + wrist flick
    AMPUTATE        = auto()   # horizontal knife-hand swipe
    GAMMA_KNIFE     = auto()   # single index-finger hold then release
    MES             = auto()   # chest-region fist hold
    TAKT            = auto()   # upward V-sign hold
    K_ROOM_SEQUENCE = auto()   # completed fist→palm→horns sequence
    HAND_LOST       = auto()   # MediaPipe no longer detects a hand

    # ── System / internal events (produced by effect renderers) ───────────────
    TECHNIQUE_COMPLETE  = auto()   # active technique animation has finished
    COLLAPSE_COMPLETE   = auto()   # ROOM collapse animation has finished
    AWAKENING_COMPLETE  = auto()   # K-ROOM cinematic has finished


# Convenience set: all events that trigger a technique while ROOM is active.
# K_ROOM_SEQUENCE is intentionally excluded — it transitions to its own state.
TECHNIQUE_EVENTS: frozenset[EventType] = frozenset({
    EventType.SHAMBLES,
    EventType.AMPUTATE,
    EventType.GAMMA_KNIFE,
    EventType.MES,
    EventType.TAKT,
})


@dataclass
class GestureEvent:
    """
    Immutable record of a single gesture or system event.

    Attributes:
        type:      The EventType that occurred.
        timestamp: Unix timestamp of when the event was created.  Defaults to
                   the current time so callers rarely need to supply it
                   explicitly; pass an explicit value in tests for
                   deterministic behaviour.
    """

    type: EventType
    timestamp: float = field(default_factory=time.time)

    def __repr__(self) -> str:
        return f"GestureEvent({self.type.name}, t={self.timestamp:.3f})"
