"""
laps_snn.py -- Spiking Neural Network reactive controller for LTA boundary patrol.

Implements two independent threshold loops that solve the LTA timing gap problem:
the delay between visual boundary detection and effective actuation on high-inertia
buoyant platforms (identified by Hufen, 2019; addressed here with timed impulses).

Loop 1 -- Boundary maneuver:
    Line detected -> wait t1 -> reduce thrust to T -> wait t2 -> orient(+180 deg)
    -> wait t3 -> restore thrust to F

Loop 2 -- Altitude hold:
    altitude_error > alt_threshold -> thrust correction

Parameters t1, t2, T, t3, F are found offline by laps_ne.py (Neuro-Evolution).

ENFIELD WP2 -- funded by the European Union, grant No 101120657.
"""

import time


class SNNParams:
    """
    Parameter set for the SNN controller.
    Produced by laps_ne.py Neuro-Evolution optimizer.

    Parameters
    ----------
    t1 : float
        Seconds from line detection to thrust reduction.
    t2 : float
        Seconds from thrust reduction to heading change (orient +180 deg).
    T : float
        Thrust level during braking phase [0.0 - 1.0].
    t3 : float
        Seconds from heading change to thrust restoration.
    F : float
        Forward cruise thrust after maneuver [0.0 - 1.0].
    alt_target : float
        Target hover altitude in meters.
    alt_threshold : float
        Altitude error (meters) that triggers a correction.
    alt_gain : float
        Proportional gain for altitude correction thrust.
    """

    def __init__(
        self,
        t1=1.0,
        t2=1.5,
        T=0.2,
        t3=1.0,
        F=0.5,
        alt_target=1.0,
        alt_threshold=0.15,
        alt_gain=0.3,
    ):
        self.t1 = float(t1)
        self.t2 = float(t2)
        self.T = float(T)
        self.t3 = float(t3)
        self.F = float(F)
        self.alt_target = float(alt_target)
        self.alt_threshold = float(alt_threshold)
        self.alt_gain = float(alt_gain)


# Internal state machine phases for Loop 1
_PHASE_CRUISE = "cruise"
_PHASE_BRAKING = "braking"
_PHASE_TURNING = "turning"
_PHASE_RESTORE = "restore"


class SNNController:
    """
    Two-loop threshold controller for LTA boundary patrol.

    Typical usage::

        from python.laps_interface import MockLaps
        from python.laps_snn import SNNController, SNNParams

        params = SNNParams(t1=1.2, t2=1.8, T=0.15, t3=0.8, F=0.45)
        laps = MockLaps()
        snn = SNNController(laps, params)

        snn.start()
        # In your sensor loop:
        while running:
            line_detected = line_detector.detect(frame)[0]
            altitude = laps.get_distance()
            snn.step(line_detected, altitude)
            time.sleep(0.05)
    """

    def __init__(self, laps, params=None):
        """
        Parameters
        ----------
        laps : LapsInterface
            Flight controller interface (real or MockLaps).
        params : SNNParams or None
            Controller parameters. Defaults to SNNParams() if None.
        """
        self.laps = laps
        self.params = params if params is not None else SNNParams()

        self._phase = _PHASE_CRUISE
        self._phase_start = 0.0
        self._maneuver_heading = 0.0
        self._running = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self):
        """Initialise cruise: enable heading hold and set forward thrust."""
        self._phase = _PHASE_CRUISE
        self._phase_start = time.monotonic()
        self.laps.riref(True)
        self.laps.thrust(self.params.F)
        self._running = True

    def stop(self):
        """Cut thrust and disable heading hold."""
        self.laps.thrust(0.0)
        self.laps.riref(False)
        self._running = False

    def step(self, line_detected, altitude):
        """
        Advance the controller by one sensor cycle.

        Call this at a fixed rate (e.g. every 50 ms) from your camera loop.

        Parameters
        ----------
        line_detected : bool
            True when laps_line.py reports a boundary line in the frame.
        altitude : float
            Current altitude in meters from the ultrasonic sensor.
        """
        if not self._running:
            return

        now = time.monotonic()
        elapsed = now - self._phase_start

        self._loop1(line_detected, elapsed, now)
        self._loop2(altitude)

    # ------------------------------------------------------------------
    # Loop 1 -- Boundary maneuver state machine
    # ------------------------------------------------------------------

    def _loop1(self, line_detected, elapsed, now):
        p = self.params

        if self._phase == _PHASE_CRUISE:
            if line_detected:
                # Boundary detected: begin timed braking sequence
                self._maneuver_heading = self.laps.get_heading()
                self._phase = _PHASE_BRAKING
                self._phase_start = now

        elif self._phase == _PHASE_BRAKING:
            if elapsed < p.t1:
                # Coast at reduced thrust -- do not change heading yet
                self.laps.thrust(p.T)
            else:
                # t1 elapsed: issue heading reversal
                new_heading = (self._maneuver_heading + 180.0) % 360.0
                self.laps.orient(new_heading)
                self._phase = _PHASE_TURNING
                self._phase_start = now

        elif self._phase == _PHASE_TURNING:
            if elapsed >= p.t2:
                # t2 elapsed: heading change complete, begin restore countdown
                self._phase = _PHASE_RESTORE
                self._phase_start = now

        elif self._phase == _PHASE_RESTORE:
            if elapsed >= p.t3:
                # t3 elapsed: restore cruise thrust
                self.laps.thrust(self.params.F)
                self.laps.riref(True)
                self._phase = _PHASE_CRUISE
                self._phase_start = now

    # ------------------------------------------------------------------
    # Loop 2 -- Altitude hold (proportional correction)
    # ------------------------------------------------------------------

    def _loop2(self, altitude):
        p = self.params
        error = p.alt_target - altitude
        if abs(error) > p.alt_threshold:
            correction = p.F + p.alt_gain * error
            correction = max(0.0, min(1.0, correction))
            self.laps.thrust(correction)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_phase(self):
        """Return current Loop 1 phase name (for logging/debug)."""
        return self._phase
