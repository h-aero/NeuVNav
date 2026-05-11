# Interactive Physics Simulation

The Neuro-Evolution optimizer in NeuVNav requires a **fitness oracle** — a model
that scores a candidate set of controller parameters without running hardware.

We provide a publicly accessible browser-based simulation of the h-aero® LTA
platform physics as the fitness oracle:

**[https://laps.h-aero.com/NeuVNav/](https://laps.h-aero.com/NeuVNav/)**

## What it demonstrates

The simulator models the physical behaviour of a buoyant lighter-than-air vehicle:

- Thrust-to-heading coupling (pitch-roll interaction unique to LTA)
- Altitude hold under varying buoyancy
- Impulse-based manoeuvre timing (the core challenge identified by Hufen, 2019)
- The 9-component combined fitness function used by `laps_ne.py`

## How it connects to the code

```
laps_ne.py  →  FitnessOracle (interface)
                    │
                    ├── MockOracle          (unit tests, no physics)
                    └── SimulationOracle    (wraps the live sim endpoint)
```

`SimulationOracle` posts a candidate `SNNParams` set to the simulation API
and receives a scalar fitness score. The NE optimizer treats this as a black box.

## Running a fitness evaluation

```python
from python.laps_ne import optimize, MockOracle

oracle = MockOracle()          # local physics model for development
result = optimize(oracle, n_gen=20, lam=10)
print(result.best_params)
```

> **Note:** For production use, replace `MockOracle` with a `SimulationOracle`
> wrapping a live fitness endpoint. The reference implementation uses a
> server-side PHP physics model as the oracle backend.

## Reference

The simulation implements the vehicle dynamics described in:

> Hufen, F. (2019). *Autonomous Indoor Navigation of h-aero Using Computer Vision.*
> Master's Thesis, Hochschule Anhalt.

The timing-gap problem identified by Hufen — actuation delay between visual
detection and effective heading change — is the primary challenge that the
SNN + NE architecture in NeuVNav addresses.
