"""
gestures/definitions.py
=======================
Pure data definitions for all gesture schemas.

This module contains NO logic — only data structures that describe what each
gesture requires.  GestureClassifier reads these definitions and contains all
evaluation logic.  Keeping data and logic separate means thresholds and
descriptions can be tuned here without touching classifier.py.

Gesture taxonomy
----------------
Continuous (leading-edge debounce in classifier):
    open_palm  — all 5 extended; ROOM charge timer managed by StateMachine
    fist       — all non-thumb fingers curled

Hold-then-fire (emitted once when hold duration satisfied):
    takt       — V-sign (spread index + middle) pointing upward
    mes        — fist in lower-half of frame (chest region)
    horns      — index + pinky extended; used as K-ROOM sequence step only

Release-based (armed during hold, fires on shape break):
    gamma_knife — index only, hold GESTURE_HOLD_GAMMA_KNIFE s, fire on release

Motion-based (shape present + velocity threshold crossed):
    amputate   — knife-hand (4 fingers, thumb tucked) + horizontal swipe
    shambles   — two-finger close point, arm for GESTURE_HOLD_SHAMBLES_STEP1 s,
                 then any wrist flick within SHAMBLES_FLICK_WINDOW s fires event

Sequence (assembled by SequenceBuffer):
    k_room_complete — horns → open_palm → horns within SEQUENCE_WINDOW seconds

K-ROOM gesture choice rationale
---------------------------------
Using  horns → open_palm → horns  (index+pinky / all-five / index+pinky) avoids
any conflict with FIST (which collapses ROOM) and uses a visually memorable
sequence.  OPEN_PALM in ROOM_ACTIVE is a no-op in the state machine, so the
intermediate step is safe to forward.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import config


@dataclass(frozen=True)
class GestureDefinition:
    """
    Data-only description of a single gesture's detection requirements.

    All string constraint fields are sentinels the classifier maps to concrete
    check functions — no geometry lives here.

    Attributes:
        name:         Internal key; must match the corresponding GestureType
                      enum value in classifier.py.
        display_name: English label shown on the debug HUD.
        japanese:     Technique name in Japanese for name-card rendering.

        thumb / index / middle / ring / pinky:
                      Required extension state (True=extended, False=curled,
                      None=don't-care).

        hold_seconds: Minimum continuous-hold duration before firing.
                      0.0 = instantaneous / continuous.

        position:     "any"   — no positional constraint.
                      "chest" — wrist y-norm must exceed CHEST_REGION_Y_MIN.

        orientation:  "any"        — unconstrained.
                      "horizontal" — hand axis within AMPUTATE_MAX_ANGLE_DEG
                                     of horizontal.
                      "upward"     — fingertip above wrist by TAKT_MIN_UPWARD_RATIO
                                     × frame_height pixels.

        spread:       "any"    — unconstrained finger spread.
                      "wide"   — index/middle vector angle ≥ V_SIGN_MIN_SPREAD_DEGREES.
                      "narrow" — index/middle angle < V_SIGN_MIN_SPREAD_DEGREES
                                 (distinguishes Shambles from Takt V-sign).

        motion:       "none"             — static gesture.
                      "horizontal_swipe" — wrist horizontal speed > SWIPE_VELOCITY_THRESHOLD.
                      "any_swipe"        — wrist speed (any direction) > SWIPE_VELOCITY_THRESHOLD.

        fire_on_release: When True, event fires when the shape *breaks* after the
                         hold has been satisfied (GAMMA_KNIFE: charge then release).

        continuous:   When True, emit on leading edge every time the shape
                      transitions into this state (OPEN_PALM, FIST).
    """

    name:         str
    display_name: str
    japanese:     str

    thumb:  Optional[bool]
    index:  Optional[bool]
    middle: Optional[bool]
    ring:   Optional[bool]
    pinky:  Optional[bool]

    hold_seconds:    float
    position:        str    # "any" | "chest"
    orientation:     str    # "any" | "horizontal" | "upward"
    spread:          str    # "any" | "wide" | "narrow"
    motion:          str    # "none" | "horizontal_swipe" | "any_swipe"
    fire_on_release: bool
    continuous:      bool


# ── Definitions ────────────────────────────────────────────────────────────────

GESTURES: dict[str, GestureDefinition] = {

    "open_palm": GestureDefinition(
        name="open_palm",    display_name="ROOM",          japanese="ルーム",
        thumb=True, index=True, middle=True, ring=True, pinky=True,
        hold_seconds=0.0, position="any",   orientation="any",
        spread="any",     motion="none",
        fire_on_release=False, continuous=True,
    ),

    "fist": GestureDefinition(
        name="fist",         display_name="Deactivate",    japanese="解除",
        thumb=None, index=False, middle=False, ring=False, pinky=False,
        hold_seconds=0.0, position="any",   orientation="any",
        spread="any",     motion="none",
        fire_on_release=False, continuous=True,
    ),

    "amputate": GestureDefinition(
        name="amputate",     display_name="Amputate",      japanese="切断",
        thumb=False, index=True, middle=True, ring=True, pinky=True,
        hold_seconds=0.0, position="any",   orientation="horizontal",
        spread="any",     motion="horizontal_swipe",
        fire_on_release=False, continuous=False,
    ),

    "gamma_knife": GestureDefinition(
        name="gamma_knife",  display_name="Gamma Knife",   japanese="ガンマナイフ",
        thumb=None, index=True, middle=False, ring=False, pinky=False,
        hold_seconds=config.GESTURE_HOLD_GAMMA_KNIFE,
        position="any",     orientation="any",
        spread="any",       motion="none",
        fire_on_release=True, continuous=False,
    ),

    "mes": GestureDefinition(
        name="mes",          display_name="Mes",            japanese="メス",
        thumb=None, index=False, middle=False, ring=False, pinky=False,
        hold_seconds=config.GESTURE_HOLD_MES,
        position="chest",   orientation="any",
        spread="any",       motion="none",
        fire_on_release=False, continuous=False,
    ),

    "takt": GestureDefinition(
        name="takt",         display_name="Takt",           japanese="タクト",
        thumb=False, index=True, middle=True, ring=False, pinky=False,
        hold_seconds=config.GESTURE_HOLD_TAKT,
        position="any",     orientation="upward",
        spread="wide",      motion="none",
        fire_on_release=False, continuous=False,
    ),

    "shambles": GestureDefinition(
        name="shambles",     display_name="Shambles",       japanese="シャンブルズ",
        thumb=False, index=True, middle=True, ring=False, pinky=False,
        hold_seconds=config.GESTURE_HOLD_SHAMBLES_STEP1,
        position="any",     orientation="any",
        spread="narrow",    motion="any_swipe",
        fire_on_release=False, continuous=False,
    ),

    # HORNS is an intermediate step for the K-ROOM sequence only.
    # It does NOT map to a technique — the SequenceBuffer consumes it.
    "horns": GestureDefinition(
        name="horns",        display_name="K-ROOM Step",   japanese="覚醒準備",
        thumb=False, index=True, middle=False, ring=False, pinky=True,
        hold_seconds=config.GESTURE_HOLD_K_ROOM_STEP,
        position="any",     orientation="any",
        spread="any",       motion="none",
        fire_on_release=False, continuous=False,
    ),
}


# ── K-ROOM sequence ────────────────────────────────────────────────────────────

# Ordered list of gesture names the SequenceBuffer must observe to trigger
# K-ROOM Awakening.  Each step fires from the classifier after its hold_seconds;
# all three must complete within config.SEQUENCE_WINDOW seconds.
K_ROOM_STEPS: list[str] = ["horns", "open_palm", "horns"]


# ── Convenience lookups ────────────────────────────────────────────────────────

def get(name: str) -> GestureDefinition:
    """
    Retrieve a GestureDefinition by name, raising KeyError on unknown names.

    Args:
        name: Gesture name key (e.g. "amputate").

    Returns:
        The corresponding GestureDefinition.
    """
    if name not in GESTURES:
        raise KeyError(f"Unknown gesture '{name}'.  Known: {list(GESTURES)}")
    return GESTURES[name]
