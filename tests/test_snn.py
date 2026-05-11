"""Tests for SNNController — state machine and altitude hold. No hardware needed."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from python.laps_interface import MockLaps
from python.laps_snn import SNNController, SNNParams


def make(t1=0.05, t2=0.05, T=0.2, t3=0.05, F=0.5):
    laps = MockLaps()
    params = SNNParams(t1=t1, t2=t2, T=T, t3=t3, F=F)
    snn = SNNController(laps, params)
    return snn, laps


def test_start_sets_cruise():
    snn, laps = make()
    snn.start()
    assert snn.get_phase() == "cruise"
    assert "thrust(0.5)" in laps.get_log()
    assert "riref(True)" in laps.get_log()


def test_line_triggers_braking():
    snn, laps = make()
    snn.start()
    snn.step(line_detected=True, altitude=1.0)
    assert snn.get_phase() == "braking"


def test_no_line_stays_cruise():
    snn, laps = make()
    snn.start()
    snn.step(line_detected=False, altitude=1.0)
    assert snn.get_phase() == "cruise"


def test_braking_to_turning_after_t1():
    snn, laps = make(t1=0.05)
    snn.start()
    snn.step(line_detected=True, altitude=1.0)   # enters braking
    time.sleep(0.08)
    snn.step(line_detected=False, altitude=1.0)  # t1 elapsed → turning
    assert snn.get_phase() == "turning"
    log = laps.get_log()
    orient_calls = [e for e in log if e.startswith("orient(")]
    assert len(orient_calls) == 1
    # heading reversal: started at 0 deg → should orient to 180
    assert orient_calls[0] == "orient(180.0)"


def test_turning_to_restore_after_t2():
    snn, laps = make(t1=0.05, t2=0.05)
    snn.start()
    snn.step(line_detected=True, altitude=1.0)
    time.sleep(0.08)
    snn.step(line_detected=False, altitude=1.0)  # → turning
    time.sleep(0.08)
    snn.step(line_detected=False, altitude=1.0)  # → restore
    assert snn.get_phase() == "restore"


def test_restore_to_cruise_after_t3():
    snn, laps = make(t1=0.05, t2=0.05, t3=0.05)
    snn.start()
    snn.step(line_detected=True, altitude=1.0)
    time.sleep(0.08)
    snn.step(line_detected=False, altitude=1.0)
    time.sleep(0.08)
    snn.step(line_detected=False, altitude=1.0)
    time.sleep(0.08)
    snn.step(line_detected=False, altitude=1.0)  # → cruise
    assert snn.get_phase() == "cruise"


def test_altitude_correction_applied():
    snn, laps = make(F=0.5)
    snn.start()
    laps.reset()
    # altitude too low (target=1.0, current=0.5) → thrust should increase
    snn.step(line_detected=False, altitude=0.5)
    thrust_calls = [e for e in laps.get_log() if e.startswith("thrust(")]
    assert len(thrust_calls) > 0
    thrust_val = float(thrust_calls[-1].replace("thrust(", "").replace(")", ""))
    assert thrust_val > 0.5


def test_stop_cuts_thrust():
    snn, laps = make()
    snn.start()
    snn.stop()
    log = laps.get_log()
    assert "thrust(0.0)" in log
    assert "riref(False)" in log


if __name__ == "__main__":
    test_start_sets_cruise()
    test_line_triggers_braking()
    test_no_line_stays_cruise()
    test_braking_to_turning_after_t1()
    test_turning_to_restore_after_t2()
    test_restore_to_cruise_after_t3()
    test_altitude_correction_applied()
    test_stop_cuts_thrust()
    print("All SNN tests passed.")
