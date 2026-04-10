"""
core/state_machine.py
=====================
Central state machine for the Trafalgar Law Ope Ope no Mi demo.

Architecture:
    The state machine is the single source of truth for application state.
    Neither the gesture engine nor the renderers mutate state directly.
    Gesture events flow IN via process_event(); the rendered state is READ
    via the public properties.

State graph (top-level):

    IDLE
      └─ OPEN_PALM held ──▶ ROOM_CHARGING
              │
              ├─ FIST / HAND_LOST ──▶ IDLE
              └─ charge timer done ──▶ ROOM_ACTIVE
                          │
                          ├─ FIST ──▶ ROOM_COLLAPSING ──▶ IDLE
                          ├─ technique gesture ──▶ TECHNIQUE_FIRING
                          │                              │
                          │         TECHNIQUE_COMPLETE ──┘ (back to ROOM_ACTIVE)
                          │         FIST ──▶ ROOM_COLLAPSING
                          └─ K_ROOM_SEQUENCE ──▶ K_ROOM_AWAKENING
                                                        │
                                          AWAKENING_COMPLETE ──▶ ROOM_ACTIVE

Key design decisions:
    - ROOM stays active after a technique completes (Q3).
    - ROOM stays active even if the hand is lost (Q3).
    - Only an explicit FIST event collapses ROOM.
    - K-ROOM Awakening is a locked one-shot cinematic; no other events are
      processed while it plays (Q7).
    - The state machine owns cooldown checking; it will silently ignore a
      technique event if the cooldown has not elapsed.
    - update(now) must be called once per frame to handle time-based
      transitions (charge timer).
    - Listeners registered via add_listener() are called on every transition
      with (new_state, active_technique) so renderers and audio can react
      without polling.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, List, Optional

import config
from core.cooldown import CooldownRegistry
from core.events import EventType, GestureEvent, TECHNIQUE_EVENTS


# ── Enums ──────────────────────────────────────────────────────────────────────

class AppState(Enum):
    """Top-level application states."""
    IDLE             = "idle"
    ROOM_CHARGING    = "room_charging"
    ROOM_ACTIVE      = "room_active"
    TECHNIQUE_FIRING = "technique_firing"
    ROOM_COLLAPSING  = "room_collapsing"
    K_ROOM_AWAKENING = "k_room_awakening"


class ActiveTechnique(Enum):
    """Which technique is currently firing (NONE when idle or ROOM-only)."""
    NONE         = "none"
    SHAMBLES     = "shambles"
    AMPUTATE     = "amputate"
    GAMMA_KNIFE  = "gamma_knife"
    MES          = "mes"
    TAKT         = "takt"
    K_ROOM       = "k_room"


# ── Mapping helpers ────────────────────────────────────────────────────────────

# Maps gesture EventType → ActiveTechnique for the five standard techniques.
_EVENT_TO_TECHNIQUE: dict[EventType, ActiveTechnique] = {
    EventType.SHAMBLES:    ActiveTechnique.SHAMBLES,
    EventType.AMPUTATE:    ActiveTechnique.AMPUTATE,
    EventType.GAMMA_KNIFE: ActiveTechnique.GAMMA_KNIFE,
    EventType.MES:         ActiveTechnique.MES,
    EventType.TAKT:        ActiveTechnique.TAKT,
}


# ── Transition record ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Transition:
    """
    Immutable record of a single state transition.

    Attributes:
        from_state:       The state before the transition.
        to_state:         The state after the transition.
        trigger:          The EventType (or None for time-based transitions).
        technique_before: ActiveTechnique before the transition.
        technique_after:  ActiveTechnique after the transition.
        timestamp:        When the transition occurred.
    """
    from_state:       AppState
    to_state:         AppState
    trigger:          Optional[EventType]
    technique_before: ActiveTechnique
    technique_after:  ActiveTechnique
    timestamp:        float

    def __str__(self) -> str:
        trigger_str = self.trigger.name if self.trigger else "TIME"
        tech_str = ""
        if self.technique_after != ActiveTechnique.NONE:
            tech_str = f" [{self.technique_after.value}]"
        elif self.technique_before != ActiveTechnique.NONE:
            tech_str = f" [{self.technique_before.value} → none]"
        return (
            f"{self.from_state.value} ──{trigger_str}──▶ "
            f"{self.to_state.value}{tech_str}"
        )


# ── State machine ──────────────────────────────────────────────────────────────

# Type alias for listener callbacks
TransitionListener = Callable[[AppState, ActiveTechnique, Transition], None]


class StateMachine:
    """
    Event-driven state machine managing ROOM and all Ope Ope no Mi techniques.

    Public interface:
        process_event(event)  — feed a GestureEvent into the machine.
        update(now)           — call once per frame for time-based transitions.
        state                 — current AppState (read-only property).
        active_technique      — current ActiveTechnique (read-only property).
        room_active           — True whenever the ROOM sphere should render.
        add_listener(fn)      — register a callback for state transitions.
        history               — list of all Transition records (for debugging).
        cooldowns             — direct access to the CooldownRegistry.
    """

    def __init__(self) -> None:
        """Initialise to IDLE with all cooldowns cleared."""
        self._state:             AppState        = AppState.IDLE
        self._active_technique:  ActiveTechnique = ActiveTechnique.NONE
        self._cooldowns:         CooldownRegistry = CooldownRegistry()
        self._listeners:         List[TransitionListener] = []
        self._history:           List[Transition] = []

        # Timestamp of when we entered the current state (for charge timer)
        self._state_enter_time:  float = time.time()

    # ── Public read-only properties ────────────────────────────────────────────

    @property
    def state(self) -> AppState:
        """Current top-level application state."""
        return self._state

    @property
    def active_technique(self) -> ActiveTechnique:
        """The technique currently animating, or ActiveTechnique.NONE."""
        return self._active_technique

    @property
    def room_active(self) -> bool:
        """
        True whenever the ROOM sphere overlay should be rendered.
        ROOM renders in all states except IDLE.
        """
        return self._state not in (AppState.IDLE, AppState.ROOM_CHARGING)

    @property
    def history(self) -> List[Transition]:
        """Ordered list of all state transitions since startup (read-only view)."""
        return list(self._history)

    @property
    def cooldowns(self) -> CooldownRegistry:
        """Direct access to the cooldown registry (e.g. for force-reset)."""
        return self._cooldowns

    # ── Listener registration ──────────────────────────────────────────────────

    def add_listener(self, fn: TransitionListener) -> None:
        """
        Register a callback that fires on every state transition.

        The callback receives:
            new_state  (AppState)       — the state just entered
            technique  (ActiveTechnique) — the active technique after transition
            transition (Transition)     — full transition record

        Args:
            fn: Callable matching TransitionListener signature.
        """
        self._listeners.append(fn)

    # ── Core update methods ────────────────────────────────────────────────────

    def process_event(self, event: GestureEvent, now: float | None = None) -> bool:
        """
        Feed a gesture or system event into the state machine.

        Returns True if a state transition occurred, False if the event was
        a no-op in the current state (e.g. technique gesture while IDLE).

        Args:
            event: The GestureEvent to process.
            now:   Current timestamp.  Defaults to time.time(); pass an
                   explicit value in tests for deterministic behaviour.
        """
        if now is None:
            now = time.time()

        current = self._state
        et = event.type

        # ── IDLE ───────────────────────────────────────────────────────────────
        if current == AppState.IDLE:
            if et == EventType.OPEN_PALM:
                return self._transition(AppState.ROOM_CHARGING,
                                        ActiveTechnique.NONE, et, now)
            return False

        # ── ROOM_CHARGING ──────────────────────────────────────────────────────
        elif current == AppState.ROOM_CHARGING:
            if et in (EventType.FIST, EventType.HAND_LOST):
                return self._transition(AppState.IDLE,
                                        ActiveTechnique.NONE, et, now)
            # Other events (stray technique gestures, etc.) are ignored here —
            # the user must complete the charge before techniques are available.
            return False

        # ── ROOM_ACTIVE ────────────────────────────────────────────────────────
        elif current == AppState.ROOM_ACTIVE:
            if et == EventType.FIST:
                return self._transition(AppState.ROOM_COLLAPSING,
                                        ActiveTechnique.NONE, et, now)

            if et == EventType.K_ROOM_SEQUENCE:
                if self._cooldowns.is_ready("k_room", now):
                    self._cooldowns.activate("k_room", now)
                    return self._transition(AppState.K_ROOM_AWAKENING,
                                            ActiveTechnique.K_ROOM, et, now)
                return False   # on cooldown — silently ignore

            if et in TECHNIQUE_EVENTS:
                technique = _EVENT_TO_TECHNIQUE[et]
                key = technique.value   # e.g. "amputate"
                if self._cooldowns.is_ready(key, now):
                    self._cooldowns.activate(key, now)
                    return self._transition(AppState.TECHNIQUE_FIRING,
                                            technique, et, now)
                return False   # on cooldown — silently ignore

            # HAND_LOST in ROOM_ACTIVE is intentionally a no-op (Q3: ROOM
            # stays active; only an explicit FIST deactivates it).
            return False

        # ── TECHNIQUE_FIRING ───────────────────────────────────────────────────
        elif current == AppState.TECHNIQUE_FIRING:
            if et == EventType.FIST:
                return self._transition(AppState.ROOM_COLLAPSING,
                                        ActiveTechnique.NONE, et, now)
            if et == EventType.TECHNIQUE_COMPLETE:
                return self._transition(AppState.ROOM_ACTIVE,
                                        ActiveTechnique.NONE, et, now)
            # Any other event (new technique gestures, OPEN_PALM, etc.) is
            # ignored — a technique must finish or be cancelled before another
            # can start.
            return False

        # ── K_ROOM_AWAKENING ───────────────────────────────────────────────────
        elif current == AppState.K_ROOM_AWAKENING:
            if et == EventType.AWAKENING_COMPLETE:
                return self._transition(AppState.ROOM_ACTIVE,
                                        ActiveTechnique.NONE, et, now)
            # K-ROOM cinematic is fully locked — no other events are processed.
            return False

        # ── ROOM_COLLAPSING ────────────────────────────────────────────────────
        elif current == AppState.ROOM_COLLAPSING:
            if et == EventType.COLLAPSE_COMPLETE:
                return self._transition(AppState.IDLE,
                                        ActiveTechnique.NONE, et, now)
            # Locked during collapse; ignore everything else.
            return False

        return False  # unreachable, but satisfies type checker

    def update(self, now: float | None = None) -> bool:
        """
        Process time-based transitions.  Must be called once per frame.

        Currently handles:
            ROOM_CHARGING → ROOM_ACTIVE  when ROOM_CHARGE_HOLD seconds have
                                          elapsed since entering the charge state.

        Returns True if a transition occurred, False otherwise.

        Args:
            now: Current timestamp.  Defaults to time.time(); pass an
                 explicit value in tests for deterministic behaviour.
        """
        if now is None:
            now = time.time()

        if self._state == AppState.ROOM_CHARGING:
            elapsed = now - self._state_enter_time
            if elapsed >= config.ROOM_CHARGE_HOLD:
                self._cooldowns.activate("room", now)
                return self._transition(AppState.ROOM_ACTIVE,
                                        ActiveTechnique.NONE,
                                        trigger=None,   # time-based
                                        now=now)
        return False

    def force_reset(self, now: float | None = None) -> None:
        """
        Immediately reset the machine to IDLE and clear all cooldowns.
        Bound to the 'R' key in the main loop.

        Args:
            now: Timestamp for the reset transition record.
        """
        if now is None:
            now = time.time()
        self._cooldowns.reset_all()
        if self._state != AppState.IDLE:
            self._transition(AppState.IDLE, ActiveTechnique.NONE,
                             trigger=None, now=now)

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _transition(
        self,
        new_state:     AppState,
        new_technique: ActiveTechnique,
        trigger:       Optional[EventType],
        now:           float,
    ) -> bool:
        """
        Execute a state transition, record it in history, and notify listeners.

        Args:
            new_state:     Target AppState.
            new_technique: ActiveTechnique after the transition.
            trigger:       EventType that caused the transition (None = timer).
            now:           Transition timestamp.

        Returns:
            True (always, so callers can return self._transition(...) directly).
        """
        record = Transition(
            from_state       = self._state,
            to_state         = new_state,
            trigger          = trigger,
            technique_before = self._active_technique,
            technique_after  = new_technique,
            timestamp        = now,
        )

        self._state            = new_state
        self._active_technique = new_technique
        self._state_enter_time = now
        self._history.append(record)

        for listener in self._listeners:
            listener(new_state, new_technique, record)

        return True

    # ── String representation ──────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"StateMachine(state={self._state.value}, "
            f"technique={self._active_technique.value}, "
            f"room_active={self.room_active})"
        )
