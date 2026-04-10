"""
core/cooldown.py
================
Per-technique cooldown tracking.

The CooldownRegistry maps technique names (strings matching ActiveTechnique
enum values) to the Unix timestamp at which they were last activated.  It
answers a single question per frame: "is this technique available to fire?"

All cooldown durations are sourced from config.py — this module contains no
magic numbers.
"""

from __future__ import annotations

import time
from typing import Dict

import config


# Map from technique name (str) → cooldown duration (seconds).
# Keys must match ActiveTechnique enum values exactly.
_COOLDOWN_DURATIONS: Dict[str, float] = {
    "room":        config.COOLDOWN_ROOM,
    "shambles":    config.COOLDOWN_SHAMBLES,
    "amputate":    config.COOLDOWN_AMPUTATE,
    "gamma_knife": config.COOLDOWN_GAMMA_KNIFE,
    "mes":         config.COOLDOWN_MES,
    "takt":        config.COOLDOWN_TAKT,
    "k_room":      config.COOLDOWN_K_ROOM,
}


class CooldownRegistry:
    """
    Tracks the last-activation timestamp for each technique and reports
    whether a technique is currently available to fire.

    Usage:
        registry = CooldownRegistry()
        if registry.is_ready("amputate"):
            registry.activate("amputate")
    """

    def __init__(self) -> None:
        """Initialise all techniques with a last-activated time of 0.0 so
        every technique is ready immediately on startup."""
        self._last_activated: Dict[str, float] = {
            name: 0.0 for name in _COOLDOWN_DURATIONS
        }

    def is_ready(self, technique: str, now: float | None = None) -> bool:
        """
        Return True if the technique's cooldown has fully elapsed.

        Args:
            technique: Technique name string (e.g. "amputate").
            now:       Current timestamp.  Defaults to time.time(); pass an
                       explicit value in tests for deterministic behaviour.

        Raises:
            KeyError: If `technique` is not a known technique name.
        """
        if technique not in _COOLDOWN_DURATIONS:
            raise KeyError(f"Unknown technique: '{technique}'.  "
                           f"Valid names: {list(_COOLDOWN_DURATIONS)}")
        if now is None:
            now = time.time()
        elapsed = now - self._last_activated[technique]
        return elapsed >= _COOLDOWN_DURATIONS[technique]

    def activate(self, technique: str, now: float | None = None) -> None:
        """
        Record that a technique has just been activated, starting its cooldown.

        Args:
            technique: Technique name string.
            now:       Timestamp of activation.  Defaults to time.time().

        Raises:
            KeyError: If `technique` is not a known technique name.
        """
        if technique not in _COOLDOWN_DURATIONS:
            raise KeyError(f"Unknown technique: '{technique}'.  "
                           f"Valid names: {list(_COOLDOWN_DURATIONS)}")
        if now is None:
            now = time.time()
        self._last_activated[technique] = now

    def remaining(self, technique: str, now: float | None = None) -> float:
        """
        Return the number of seconds remaining on a technique's cooldown.
        Returns 0.0 if the technique is already ready.

        Args:
            technique: Technique name string.
            now:       Current timestamp.  Defaults to time.time().
        """
        if technique not in _COOLDOWN_DURATIONS:
            raise KeyError(f"Unknown technique: '{technique}'.  "
                           f"Valid names: {list(_COOLDOWN_DURATIONS)}")
        if now is None:
            now = time.time()
        elapsed = now - self._last_activated[technique]
        return max(0.0, _COOLDOWN_DURATIONS[technique] - elapsed)

    def reset(self, technique: str) -> None:
        """
        Force a technique back to ready state by clearing its activation time.

        Useful for the 'R' force-reset keybind and for test setup.

        Args:
            technique: Technique name string.
        """
        if technique not in _COOLDOWN_DURATIONS:
            raise KeyError(f"Unknown technique: '{technique}'.  "
                           f"Valid names: {list(_COOLDOWN_DURATIONS)}")
        self._last_activated[technique] = 0.0

    def reset_all(self) -> None:
        """Force all techniques back to ready state."""
        for name in self._last_activated:
            self._last_activated[name] = 0.0

    def status_snapshot(self, now: float | None = None) -> Dict[str, float]:
        """
        Return a dict of {technique_name: seconds_remaining} for all techniques.
        A value of 0.0 means the technique is ready.

        Useful for debug overlays and test assertions.
        """
        if now is None:
            now = time.time()
        return {name: self.remaining(name, now) for name in _COOLDOWN_DURATIONS}

    def __repr__(self) -> str:
        snapshot = self.status_snapshot()
        parts = ", ".join(f"{k}={v:.2f}s" for k, v in snapshot.items())
        return f"CooldownRegistry({parts})"
