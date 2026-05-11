# NeuVNav Architecture

## Interface Design

All AI modules communicate with the flight controller exclusively via `LapsInterface`.

Benefits:
- Standalone testing with `MockLaps` (no hardware required)
- Platform independence
- Clean separation from proprietary flight software

## AI Pipeline

Camera → laps_line.py (HSV) → laps_snn.py (2 loops) → laps_ne.py (evolve) → LapsInterface

## NE Parameters

| Parameter | Role |
|-----------|------|
| t1 | Line-detection threshold → thrust reduction |
| t2 | Thrust reduction threshold → orient(heading + 180 deg) |
| T  | Thrust level during turn maneuver |
| t3 | Post-orient delay before thrust restore |
| F  | Forward thrust magnitude after maneuver |
