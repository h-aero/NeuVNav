"""Smoke tests for MockLaps — verifies LapsInterface contract."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

from laps_interface import MockLaps


def test_orient_wraps():
    m = MockLaps()
    m.orient(370)
    assert m.get_heading() == 10.0

def test_thrust_clamps():
    m = MockLaps()
    m.thrust(2.0)
    assert m.get_thrust() == 1.0

def test_log_records():
    m = MockLaps()
    m.orient(90)
    m.thrust(0.5)
    assert len(m.get_log()) == 2

def test_reset():
    m = MockLaps()
    m.orient(180)
    m.reset()
    assert m.get_heading() == 0.0

if __name__ == '__main__':
    test_orient_wraps()
    test_thrust_clamps()
    test_log_records()
    test_reset()
    print("All tests passed.")
