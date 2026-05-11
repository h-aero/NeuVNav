# NeuVNav — Neuro-Evolving Visual Navigation for Low SWaP LTA UAVs

AI navigation modules for autonomous, GPS-denied indoor navigation of
Lighter-Than-Air (LTA) unmanned platforms.

> **Funding:** This work is funded by the European Union under grant agreement No 101120657
> (ENFIELD Innovation Schemes OC1-2025-TIS-01-165).
> Views and opinions expressed are those of the authors only and do not necessarily reflect
> those of the European Union or the European Research Executive Agency.

---

## Overview

NeuVNav addresses a fundamental challenge in LTA navigation: **the timing gap between visual
detection and effective actuation**. Due to the high inertia of buoyant platforms, conventional
reactive controllers fail — a correction command issued at detection time arrives too late.

We solve this with a two-layer architecture:

1. **SNN reactive controller** — two threshold loops that trigger heading and thrust corrections
   at precisely timed intervals relative to boundary detection
2. **Neuro-Evolution optimizer** — an evolutionary strategy that searches the timing parameter
   space offline, using a full physics simulation as fitness oracle

The result is a closed-loop autonomous patrol system demonstrated on the **h-aero® semirigid
LTA platform** (indoor, GPS-denied).

---

## Scientific Context

### Development Stages

| Stage | Method | Year | Role |
|-------|--------|------|------|
| 1 | Binary thresholding → boundary line detection | Hufen, 2019 | Baseline |
| 2 | HSV masks → zone-based object detection | H-AERO intern, 2025 | Improvement |
| 3 | SNN reactive control + NE optimizer → Bounded Area Patrol | H-AERO, 2026 | **Contribution** |

Hufen (2019) demonstrated the first application of computer vision line detection to the
h-aero® LTA platform, identifying the timing gap as the core control challenge.
NeuVNav Stage 3 addresses this directly with an impulse-based controller whose parameters
are optimized by neuro-evolution.

### TRL

**TRL 5** — System validated in simulation; hardware demonstrates individual subsystems
(laser ranging, camera-based tracking). Full closed-loop hardware demonstration in progress.

---

## Modules

| Module | Description | Status |
|--------|-------------|--------|
| `python/laps_interface.py` | Abstract flight controller interface + `MockLaps` for standalone testing | ✅ |
| `python/laps_line.py` | HSV-based boundary line detection (after Hufen, 2019) | 🔄 in development |
| `python/laps_snn.py` | Spiking Neural Network reactive controller — 2 threshold loops | 🔄 in development |
| `python/laps_ne.py` | Neuro-Evolution optimizer — physics-simulation-based parameter search | 🔄 in development |

---

## Neuro-Evolution: Parameters and Fitness

The NE optimizer searches a parameter space using an evolutionary strategy
(configurable generations × candidates per generation).

### Search Space

| Parameter | Role |
|-----------|------|
| `F` | Forward thrust magnitude after boundary maneuver |
| `theta` | Turn angle (orbit geometry) |
| `n` | Thrust profile exponent (ramp shape) |
| `ts` | Turn setup time before heading change |
| `bd` | Brake distance threshold |

Additional rotation parameters: `Frot` (rotation thrust), `rotTh` (rotation threshold).

### Fitness Function

The fitness oracle runs a headless physics simulation and scores each candidate
on up to 9 components (weighted sum or product mode):

| Component | Formula | Optimizes for |
|-----------|---------|---------------|
| `f_tdrin` | `t_inside / t_total` | Time inside patrol area |
| `f_tover` | `1 / (1 + t_over · 50)` | Minimize boundary overshoot |
| `f_tger` | `t_straight / t_total` | Maximize straight-line flight ratio |
| `f_stab` | `1 / (1 + σ²_z · 200)` | Altitude variance |
| `f_track` | `1 / (1 + Δz̄ · 20)` | Altitude mean deviation |
| `f_komf` | `1 / (1 + A_z · 10)` | Altitude oscillation amplitude |
| `f_eff` | `1 / (1 + r_imp · 3)` | Impulse efficiency |
| `f_boost` | `1 / (1 + n_boost · 3)` | Penalize boost events |
| `f_rot_z` | `1 / (1 + Δz_max · 20)` | Altitude stability during rotation |

---

## Platform

- **Vehicle:** h-aero® semirigid LTA platform (motorized free balloon)
- **Compute:** Embedded single-board computer (ARM, 1 GB RAM)
- **Sensors:** Camera (HSV line detection), ultrasonic rangefinder (altitude), compass (heading)
- **No external infrastructure:** GPS-denied, no motion-capture, no beacons
- **Python:** 3.7 compatible (embedded), 3.9+ for development

The flight controller interface is abstracted via `LapsInterface`.
`MockLaps` enables full algorithm development and testing without hardware.

---

## Installation

```bash
pip install opencv-python-headless numpy
```

## Quick Start

```python
from python.laps_interface import MockLaps

laps = MockLaps()
laps.orient(90)
laps.thrust(0.5)
print(laps.get_log())  # ['orient(90)', 'thrust(0.5)']
```

---

## Citation

*Paper under review. Citation and DOI will be added upon publication.*

<!-- CITATION_PLACEHOLDER
@article{singer2026neuvnav,
  title   = {NeuVNav: Neuro-Evolving Visual Navigation for Low SWaP LTA UAVs},
  author  = {Singer, Csaba},
  journal = {[venue]},
  year    = {2026},
  doi     = {[doi]}
}
-->

### References

- Hufen, F. (2019). *Entwicklung einer autonomen Steuerung für Unmanned Aerial Vehicles
  mithilfe von Methoden der Bildverarbeitung*. Master's thesis, Hochschule Anhalt.

---

## License

[Apache 2.0](LICENSE) — Copyright 2026 Hybrid Airplane Technologies GmbH
