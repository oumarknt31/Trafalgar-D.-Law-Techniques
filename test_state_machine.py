"""
test_state_machine.py
=====================
Standalone verification script for core/state_machine.py.

Fires mock GestureEvents with explicit timestamps so the test is fully
deterministic (no real time.time() dependency).  Prints a formatted log of
every state transition and marks each scenario PASS or FAIL.

Run with:
    python test_state_machine.py

No webcam, no rendering, no external dependencies beyond the project modules.
"""

from __future__ import annotations

import sys
from typing import List

from core.events import EventType, GestureEvent
from core.state_machine import AppState, ActiveTechnique, StateMachine, Transition

# ── ANSI colors for terminal output ───────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
CYAN   = "\033[96m"
YELLOW = "\033[93m"
DIM    = "\033[2m"
RESET  = "\033[0m"


# ── Helpers ────────────────────────────────────────────────────────────────────

def make_event(etype: EventType, t: float) -> GestureEvent:
    """Create a GestureEvent with an explicit timestamp."""
    return GestureEvent(type=etype, timestamp=t)


class ScenarioRunner:
    """
    Runs a named test scenario against a fresh StateMachine instance.

    Usage:
        with ScenarioRunner("My scenario") as s:
            s.event(EventType.OPEN_PALM)
            s.update()          # advance time
            s.assert_state(AppState.ROOM_ACTIVE)
    """

    _all_results: List[tuple[str, bool]] = []   # class-level accumulator

    def __init__(self, name: str) -> None:
        self.name   = name
        self.sm     = StateMachine()
        self.t      = 1000.0   # arbitrary start time (avoids 0.0 edge cases)
        self.passed = True
        self._log: List[str] = []

        # Attach listener so every transition is printed
        self.sm.add_listener(self._on_transition)

    def __enter__(self) -> "ScenarioRunner":
        print(f"\n{CYAN}{'─'*60}{RESET}")
        print(f"{CYAN}  SCENARIO: {self.name}{RESET}")
        print(f"{CYAN}{'─'*60}{RESET}")
        return self

    def __exit__(self, *_) -> None:
        result_str = f"{GREEN}PASS{RESET}" if self.passed else f"{RED}FAIL{RESET}"
        print(f"  Result: {result_str}")
        ScenarioRunner._all_results.append((self.name, self.passed))

    def _on_transition(
        self,
        new_state: AppState,
        technique: ActiveTechnique,
        record: Transition,
    ) -> None:
        trigger = record.trigger.name if record.trigger else "TIMER"
        print(f"  {DIM}t={record.timestamp:.2f}{RESET}  "
              f"{record.from_state.value:18s} "
              f"──{YELLOW}{trigger:20s}{RESET}──▶  "
              f"{new_state.value:18s}"
              + (f"  [{technique.value}]" if technique != ActiveTechnique.NONE else ""))

    def event(self, etype: EventType, dt: float = 0.1) -> bool:
        """Fire an event, advancing simulated time by dt seconds first."""
        self.t += dt
        ev = make_event(etype, self.t)
        return self.sm.process_event(ev, now=self.t)

    def update(self, dt: float = 0.1) -> bool:
        """Advance simulated time by dt and call sm.update()."""
        self.t += dt
        return self.sm.update(now=self.t)

    def skip_time(self, seconds: float) -> None:
        """Advance simulated time without triggering any event or update."""
        self.t += seconds

    def assert_state(self, expected: AppState, label: str = "") -> None:
        """Assert the machine is in the expected AppState."""
        actual = self.sm.state
        ok = actual == expected
        tag = f" ({label})" if label else ""
        icon = f"{GREEN}✓{RESET}" if ok else f"{RED}✗{RESET}"
        print(f"  {icon} assert state == {expected.value}{tag}"
              + ("" if ok else f"  {RED}got: {actual.value}{RESET}"))
        if not ok:
            self.passed = False

    def assert_technique(self, expected: ActiveTechnique, label: str = "") -> None:
        """Assert the machine has the expected ActiveTechnique."""
        actual = self.sm.active_technique
        ok = actual == expected
        tag = f" ({label})" if label else ""
        icon = f"{GREEN}✓{RESET}" if ok else f"{RED}✗{RESET}"
        print(f"  {icon} assert technique == {expected.value}{tag}"
              + ("" if ok else f"  {RED}got: {actual.value}{RESET}"))
        if not ok:
            self.passed = False

    def assert_room_active(self, expected: bool, label: str = "") -> None:
        """Assert room_active equals expected bool."""
        actual = self.sm.room_active
        ok = actual == expected
        tag = f" ({label})" if label else ""
        icon = f"{GREEN}✓{RESET}" if ok else f"{RED}✗{RESET}"
        print(f"  {icon} assert room_active == {expected}{tag}"
              + ("" if ok else f"  {RED}got: {actual}{RESET}"))
        if not ok:
            self.passed = False

    def assert_transitioned(self, did: bool, label: str = "") -> None:
        """Assert that the last event/update call returned did (True/False)."""
        # We check the last history entry
        tag = f" ({label})" if label else ""
        icon = f"{GREEN}✓{RESET}" if did else f"{RED}✗{RESET}"
        # This is just a logical assertion helper — compare manually
        print(f"  {icon} assert transition occurred == {did}{tag}")

    @classmethod
    def print_summary(cls) -> None:
        """Print overall PASS/FAIL summary and exit with non-zero on failure."""
        total  = len(cls._all_results)
        passed = sum(1 for _, ok in cls._all_results if ok)
        failed = total - passed
        print(f"\n{'═'*60}")
        print(f"  Results: {passed}/{total} passed", end="")
        if failed:
            print(f"  {RED}({failed} failed){RESET}")
        else:
            print(f"  {GREEN}(all passed){RESET}")
        print(f"{'═'*60}\n")
        if failed:
            sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
#  SCENARIOS
# ══════════════════════════════════════════════════════════════════════════════

def scenario_room_charge_and_activate():
    """Happy path: open palm → charge timer → ROOM_ACTIVE."""
    with ScenarioRunner("ROOM: charge and activate") as s:
        s.assert_state(AppState.IDLE, "initial")
        s.assert_room_active(False, "initial")

        s.event(EventType.OPEN_PALM)
        s.assert_state(AppState.ROOM_CHARGING)
        s.assert_room_active(False, "charging does not render ROOM yet")

        # Advance less than charge hold — should still be charging
        result = s.update(dt=0.3)
        s.assert_state(AppState.ROOM_CHARGING, "not enough time elapsed")

        # Advance past charge hold (config.ROOM_CHARGE_HOLD = 0.5s)
        result = s.update(dt=0.3)   # total 0.6s elapsed in charging state
        s.assert_state(AppState.ROOM_ACTIVE, "charge complete")
        s.assert_room_active(True, "ROOM now active")


def scenario_room_cancelled_by_fist():
    """ROOM charge cancelled by fist before timer fires."""
    with ScenarioRunner("ROOM: charge cancelled by FIST") as s:
        s.event(EventType.OPEN_PALM)
        s.assert_state(AppState.ROOM_CHARGING)

        s.event(EventType.FIST)
        s.assert_state(AppState.IDLE)
        s.assert_room_active(False)


def scenario_room_cancelled_by_hand_lost():
    """ROOM charge cancelled when hand disappears."""
    with ScenarioRunner("ROOM: charge cancelled by HAND_LOST") as s:
        s.event(EventType.OPEN_PALM)
        s.event(EventType.HAND_LOST)
        s.assert_state(AppState.IDLE)


def scenario_room_deactivation():
    """ROOM_ACTIVE → ROOM_COLLAPSING → IDLE via FIST."""
    with ScenarioRunner("ROOM: deactivation via FIST") as s:
        s.event(EventType.OPEN_PALM)
        s.update(dt=0.6)
        s.assert_state(AppState.ROOM_ACTIVE)

        s.event(EventType.FIST)
        s.assert_state(AppState.ROOM_COLLAPSING)
        s.assert_room_active(True, "ROOM still renders during collapse")

        s.event(EventType.COLLAPSE_COMPLETE)
        s.assert_state(AppState.IDLE)
        s.assert_room_active(False)


def scenario_room_survives_hand_loss():
    """ROOM stays active when hand is lost (Q3: only FIST deactivates)."""
    with ScenarioRunner("ROOM: survives HAND_LOST (Q3)") as s:
        s.event(EventType.OPEN_PALM)
        s.update(dt=0.6)
        s.assert_state(AppState.ROOM_ACTIVE)

        s.event(EventType.HAND_LOST)
        s.assert_state(AppState.ROOM_ACTIVE, "ROOM must survive hand loss")
        s.assert_room_active(True)


def scenario_technique_fires_and_returns():
    """Technique fires → completes → returns to ROOM_ACTIVE (Q3)."""
    with ScenarioRunner("Technique: fire AMPUTATE and complete") as s:
        s.event(EventType.OPEN_PALM)
        s.update(dt=0.6)
        s.assert_state(AppState.ROOM_ACTIVE)

        s.event(EventType.AMPUTATE)
        s.assert_state(AppState.TECHNIQUE_FIRING)
        s.assert_technique(ActiveTechnique.AMPUTATE)
        s.assert_room_active(True, "ROOM renders during technique")

        s.event(EventType.TECHNIQUE_COMPLETE)
        s.assert_state(AppState.ROOM_ACTIVE, "back to ROOM after complete")
        s.assert_technique(ActiveTechnique.NONE)


def scenario_technique_cancelled_by_fist():
    """Technique in progress can be cancelled with FIST → ROOM_COLLAPSING."""
    with ScenarioRunner("Technique: cancelled mid-animation by FIST") as s:
        s.event(EventType.OPEN_PALM)
        s.update(dt=0.6)

        s.event(EventType.TAKT)
        s.assert_state(AppState.TECHNIQUE_FIRING)
        s.assert_technique(ActiveTechnique.TAKT)

        s.event(EventType.FIST)
        s.assert_state(AppState.ROOM_COLLAPSING)
        s.assert_technique(ActiveTechnique.NONE, "technique cleared on cancel")


def scenario_technique_blocked_when_idle():
    """Technique gestures before ROOM is active must be ignored."""
    with ScenarioRunner("Technique: ignored when IDLE (no ROOM)") as s:
        transitioned = s.event(EventType.AMPUTATE)
        s.assert_state(AppState.IDLE, "must stay IDLE")
        # event() returns the bool from process_event
        ok = (s.sm.state == AppState.IDLE)
        if not ok:
            s.passed = False


def scenario_cooldown_blocks_repeat():
    """Firing the same technique twice in quick succession is blocked by cooldown."""
    with ScenarioRunner("Cooldown: second AMPUTATE blocked (1.5s cooldown)") as s:
        s.event(EventType.OPEN_PALM)
        s.update(dt=0.6)

        # First fire — should succeed
        s.event(EventType.AMPUTATE)
        s.assert_state(AppState.TECHNIQUE_FIRING, "first fire OK")
        s.event(EventType.TECHNIQUE_COMPLETE)
        s.assert_state(AppState.ROOM_ACTIVE)

        # Second fire immediately — cooldown not elapsed (1.5s required)
        s.event(EventType.AMPUTATE, dt=0.1)
        s.assert_state(AppState.ROOM_ACTIVE, "blocked by cooldown")

        # Advance past cooldown and try again
        s.skip_time(1.6)
        s.event(EventType.AMPUTATE, dt=0.0)
        s.assert_state(AppState.TECHNIQUE_FIRING, "fires after cooldown elapsed")


def scenario_all_five_techniques():
    """All five standard techniques fire correctly from ROOM_ACTIVE."""
    techniques = [
        (EventType.SHAMBLES,    ActiveTechnique.SHAMBLES),
        (EventType.GAMMA_KNIFE, ActiveTechnique.GAMMA_KNIFE),
        (EventType.MES,         ActiveTechnique.MES),
        (EventType.TAKT,        ActiveTechnique.TAKT),
    ]
    # Note: AMPUTATE tested separately above; include here too for completeness
    techniques.insert(0, (EventType.AMPUTATE, ActiveTechnique.AMPUTATE))

    with ScenarioRunner("All 5 techniques fire sequentially") as s:
        s.event(EventType.OPEN_PALM)
        s.update(dt=0.6)
        s.assert_state(AppState.ROOM_ACTIVE)

        for ev_type, expected_tech in techniques:
            s.event(ev_type, dt=0.0)
            s.assert_state(AppState.TECHNIQUE_FIRING, f"{ev_type.name} fires")
            s.assert_technique(expected_tech)
            s.event(EventType.TECHNIQUE_COMPLETE, dt=0.0)
            s.assert_state(AppState.ROOM_ACTIVE, f"back after {ev_type.name}")
            # Advance past each cooldown before next technique
            s.skip_time(5.0)


def scenario_k_room_awakening():
    """K-ROOM fires, locks all input, then returns to ROOM_ACTIVE."""
    with ScenarioRunner("K-ROOM: one-shot cinematic then ROOM_ACTIVE (Q7)") as s:
        s.event(EventType.OPEN_PALM)
        s.update(dt=0.6)
        s.assert_state(AppState.ROOM_ACTIVE)

        s.event(EventType.K_ROOM_SEQUENCE)
        s.assert_state(AppState.K_ROOM_AWAKENING)
        s.assert_technique(ActiveTechnique.K_ROOM)

        # All events during cinematic must be ignored
        s.event(EventType.FIST)
        s.assert_state(AppState.K_ROOM_AWAKENING, "FIST ignored during cinematic")

        s.event(EventType.AMPUTATE)
        s.assert_state(AppState.K_ROOM_AWAKENING, "AMPUTATE ignored during cinematic")

        # Cinematic ends
        s.event(EventType.AWAKENING_COMPLETE)
        s.assert_state(AppState.ROOM_ACTIVE, "returns to ROOM_ACTIVE after cinematic")
        s.assert_technique(ActiveTechnique.NONE)


def scenario_k_room_cooldown():
    """K-ROOM cannot fire again within its 5.0s cooldown."""
    with ScenarioRunner("K-ROOM: blocked by cooldown") as s:
        s.event(EventType.OPEN_PALM)
        s.update(dt=0.6)

        s.event(EventType.K_ROOM_SEQUENCE)
        s.assert_state(AppState.K_ROOM_AWAKENING)
        s.event(EventType.AWAKENING_COMPLETE)
        s.assert_state(AppState.ROOM_ACTIVE)

        # Try immediately — should be blocked (5.0s cooldown)
        s.event(EventType.K_ROOM_SEQUENCE, dt=0.1)
        s.assert_state(AppState.ROOM_ACTIVE, "blocked by 5s cooldown")

        # Advance past cooldown
        s.skip_time(5.1)
        s.event(EventType.K_ROOM_SEQUENCE, dt=0.0)
        s.assert_state(AppState.K_ROOM_AWAKENING, "fires after cooldown")


def scenario_force_reset():
    """Force reset returns machine to IDLE from any state."""
    with ScenarioRunner("Force reset from TECHNIQUE_FIRING") as s:
        s.event(EventType.OPEN_PALM)
        s.update(dt=0.6)
        s.event(EventType.AMPUTATE)
        s.assert_state(AppState.TECHNIQUE_FIRING)

        s.sm.force_reset(now=s.t)
        s.assert_state(AppState.IDLE)
        s.assert_technique(ActiveTechnique.NONE)
        s.assert_room_active(False)

        # Cooldowns should also be cleared — AMPUTATE should be ready
        if not s.sm.cooldowns.is_ready("amputate", now=s.t):
            print(f"  {RED}✗ cooldowns not cleared after force_reset{RESET}")
            s.passed = False
        else:
            print(f"  {GREEN}✓ cooldowns cleared after force_reset{RESET}")


def scenario_technique_not_fired_during_technique():
    """A second technique gesture while one is already firing must be ignored."""
    with ScenarioRunner("Technique: second gesture ignored while busy") as s:
        s.event(EventType.OPEN_PALM)
        s.update(dt=0.6)
        s.event(EventType.AMPUTATE)
        s.assert_state(AppState.TECHNIQUE_FIRING)
        s.assert_technique(ActiveTechnique.AMPUTATE)

        s.skip_time(5.0)   # ensure cooldowns wouldn't be the reason
        s.event(EventType.TAKT, dt=0.0)
        s.assert_state(AppState.TECHNIQUE_FIRING, "still TECHNIQUE_FIRING")
        s.assert_technique(ActiveTechnique.AMPUTATE, "technique unchanged")


def scenario_cooldown_remaining():
    """CooldownRegistry.remaining() returns correct values."""
    with ScenarioRunner("Cooldown: remaining() accuracy") as s:
        import config
        s.event(EventType.OPEN_PALM)
        s.update(dt=0.6)
        s.event(EventType.AMPUTATE)
        s.event(EventType.TECHNIQUE_COMPLETE)

        # 0.1s has passed since activation (one event dt)
        remaining = s.sm.cooldowns.remaining("amputate", now=s.t)
        expected_approx = config.COOLDOWN_AMPUTATE - 0.1
        # Allow small floating-point tolerance
        ok = abs(remaining - expected_approx) < 0.05
        icon = f"{GREEN}✓{RESET}" if ok else f"{RED}✗{RESET}"
        print(f"  {icon} remaining ≈ {expected_approx:.2f}s  (got {remaining:.3f}s)")
        if not ok:
            s.passed = False


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"\n{'═'*60}")
    print("  Trafalgar Law — State Machine Test Suite")
    print(f"{'═'*60}")

    scenario_room_charge_and_activate()
    scenario_room_cancelled_by_fist()
    scenario_room_cancelled_by_hand_lost()
    scenario_room_deactivation()
    scenario_room_survives_hand_loss()
    scenario_technique_fires_and_returns()
    scenario_technique_cancelled_by_fist()
    scenario_technique_blocked_when_idle()
    scenario_cooldown_blocks_repeat()
    scenario_all_five_techniques()
    scenario_k_room_awakening()
    scenario_k_room_cooldown()
    scenario_force_reset()
    scenario_technique_not_fired_during_technique()
    scenario_cooldown_remaining()

    ScenarioRunner.print_summary()
