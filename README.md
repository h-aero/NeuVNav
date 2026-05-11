# NeuVNav — Neuro-Evolving Visual Navigation for Low SWaP UAVs

AI modules for vision-based autonomous navigation of Lighter-Than-Air (LTA) unmanned platforms.

> **Funding:** This work is funded by the European Union under grant agreement No 101120657
> (ENFIELD Innovation Schemes OC1-2025-TIS-01-165).
> Views and opinions expressed are those of the authors only and do not necessarily reflect
> those of the European Union or the European Research Executive Agency.

---

## Modules

| Module | Description | Status |
|--------|-------------|--------|
| `laps_interface.py` | Abstract flight controller interface + `MockLaps` for standalone testing | ✅ |
| `laps_line.py` | HSV-based boundary line detection (OpenCV) | 🔄 in development |
| `laps_snn.py` | Spiking Neural Network reactive controller — 2 threshold loops | 🔄 in development |
| `laps_ne.py` | Neuro-Evolution optimizer — 5 parameters: t1, t2, T, t3, F | 🔄 in development |

## Platform

Tested on: Raspberry Pi 3 B+ (Python 3.7), compatible with Python 3.9+.

Hardware interface abstracted via `LapsInterface` — implement for your platform.
`MockLaps` enables full development and testing without physical hardware.

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

## Citation

*Paper under review. Citation and DOI will be added upon publication.*

<!-- CITATION_PLACEHOLDER
@article{singer2026neuvnav,
  title   = {NeuVNav: Neuro-Evolving Visual Navigation for Low SWaP LTA UAVs},
  author  = {Singer, Csaba},
  journal = {[journal]},
  year    = {2026},
  doi     = {[doi]}
}
-->

## License

[Apache 2.0](LICENSE)
