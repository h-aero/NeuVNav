"""
LapsInterface — Abstract interface between NeuVNav AI modules and flight controller.

Public (this file): MockLaps for standalone testing and reproducibility.
Private (not published): Real implementation via LAPS AI_Control.py.

ENFIELD WP2 — funded by the European Union, grant No 101120657.
"""


class LapsInterface:
    """Abstract flight controller interface. Implement for your platform."""

    def orient(self, orit: float) -> None:
        """Set target heading. orit = compass degrees 0-359."""
        raise NotImplementedError

    def thrust(self, wert: float) -> None:
        """Set thrust level. wert = 0.0 (off) to 1.0 (full)."""
        raise NotImplementedError

    def riref(self, enable: bool) -> None:
        """Enable/disable heading hold. Holds last orient() value +-3 deg."""
        raise NotImplementedError

    def get_distance(self) -> float:
        """Return ultrasonic sensor distance in meters."""
        raise NotImplementedError

    def get_heading(self) -> float:
        """Return current compass heading 0-359."""
        raise NotImplementedError

    def get_thrust(self) -> float:
        """Return current thrust level 0.0-1.0."""
        raise NotImplementedError


class MockLaps(LapsInterface):
    """
    Mock implementation for standalone testing and development.
    No hardware required. Use in unit tests and CI.
    """

    def __init__(self):
        self._heading = 0.0
        self._thrust = 0.0
        self._riref_active = False
        self._distance = 1.0
        self._log = []

    def orient(self, orit: float) -> None:
        self._heading = float(orit) % 360.0
        self._log.append(f"orient({orit})")

    def thrust(self, wert: float) -> None:
        self._thrust = max(0.0, min(1.0, float(wert)))
        self._log.append(f"thrust({wert})")

    def riref(self, enable: bool) -> None:
        self._riref_active = bool(enable)
        self._log.append(f"riref({enable})")

    def get_distance(self) -> float:
        return self._distance

    def get_heading(self) -> float:
        return self._heading

    def get_thrust(self) -> float:
        return self._thrust

    def set_mock_distance(self, d: float) -> None:
        """Test helper: simulate ultrasonic sensor reading."""
        self._distance = float(d)

    def get_log(self) -> list:
        """Return command history for test assertions."""
        return self._log.copy()

    def reset(self) -> None:
        """Reset all state. Call between test cases."""
        self.__init__()
