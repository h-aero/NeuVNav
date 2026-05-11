# NeuVNav Architecture

## Design Principle

All AI modules communicate with the flight controller exclusively via `LapsInterface`.

```
Camera → laps_line.py → laps_snn.py → laps_ne.py (offline) → LapsInterface → Platform
```

Benefits:
- Standalone testing with `MockLaps` (no hardware required)
- Platform independence — swap `LapsInterface` implementation for any flight controller
- Clean IP separation from proprietary flight software

---

## Physics Model (Simulator)

The NE optimizer uses a full physics simulation as its fitness oracle.
The simulator models the complete flight envelope of a buoyant LTA platform.

### Flight Phases

| Phase | Description |
|-------|-------------|
| `rising` / `rising2` | Altitude gain to target hover height |
| `settling` | Post-maneuver altitude stabilization (with `settlePostRot` flag for post-rotation settling) |
| `cruising` | Forward flight inside patrol area |
| `braking` | Thrust reduction before boundary maneuver |
| `rotating` | Impulse-based heading change (pulse mode) |
| `rot_cont` | Continuous rotation sub-phases: `ramp_in` → `main_fwd` → `coast_out` |
| `turning_rise` | Combined turn + altitude correction |

### Key Physical Properties

- **Inertia dominance:** heading and altitude corrections take effect with significant delay
- **Coupling:** thrust changes affect both horizontal speed and vertical position
- **riref principle:** once a heading is set via `orient()`, the platform holds course
  automatically within ±3° — the controller only decides *when* to issue corrections

---

## SNN Reactive Controller (`laps_snn.py`)

Two independent threshold loops operating on sensor input:

### Loop 1 — Boundary Detection → Heading

```
IF line_detected AND timer_1 >= t1:
    thrust_reduce()
IF thrust_reduced AND timer_2 >= t2:
    orient(current_heading + 180°)
```

### Loop 2 — Altitude Control

```
IF altitude_error > threshold:
    thrust(correction)
```

The SNN does not implement traditional spiking neural network dynamics.
The name reflects the event-driven, threshold-triggered nature of the control law —
sparse firing only when sensor thresholds are crossed.

---

## Neuro-Evolution Optimizer (`laps_ne.py`)

Offline parameter search using an evolutionary strategy.

### Algorithm

```
Initialize: random candidate in search space
For each generation (N_GEN):
    Sample LAMBDA candidates (Gaussian mutation around current best)
    Evaluate each via runHeadless(candidate) → fitness score
    Update best if improved
    Adapt step size (success-based)
Return: best parameter set
```

### Search Space

| Parameter | Symbol | Role |
|-----------|--------|------|
| Forward thrust | `F` | Thrust magnitude during straight flight |
| Turn angle | `theta` | Heading change geometry |
| Thrust exponent | `n` | Ramp profile shape (n=0: constant, n=1: linear, n>1: steep start) |
| Setup time | `ts` | Time before heading change is issued |
| Brake distance | `bd` | Boundary proximity threshold for braking |
| Rotation thrust | `Frot` | Thrust during rotation maneuver |
| Rotation threshold | `rotTh` | Angular threshold for rotation completion |

### Fitness Function

Weighted sum (or product) of 9 components — all in [0, 1]:

| Symbol | Formula | What it rewards |
|--------|---------|-----------------|
| `f_tdrin` | `t_inside / t_total` | Time inside patrol boundary |
| `f_tover` | `1 / (1 + t_overshoot · 50)` | Low boundary overshoot |
| `f_tger` | `t_straight / t_total` | Straight-line flight fraction |
| `f_stab` | `1 / (1 + σ²_z · 200)` | Altitude variance |
| `f_track` | `1 / (1 + Δz̄ · 20)` | Altitude tracking accuracy |
| `f_komf` | `1 / (1 + A_z · 10)` | Low altitude oscillation |
| `f_eff` | `1 / (1 + r_impulse · 3)` | Actuator efficiency |
| `f_boost` | `1 / (1 + n_boost · 3)` | Low boost event count |
| `f_rot_z` | `1 / (1 + Δz_max · 20)` | Altitude stability during rotation |

---

## AI Pipeline — Data Flow

```
┌─────────────┐     frame      ┌──────────────┐   line_detected   ┌──────────────┐
│   Camera    │ ────────────►  │ laps_line.py │ ────────────────► │ laps_snn.py  │
└─────────────┘                └──────────────┘                   └──────┬───────┘
                                                                          │ orient() / thrust()
┌─────────────────────────────────────────────────────────────────────────▼───────┐
│                            LapsInterface                                         │
│              orient(heading)  ·  thrust(value)  ·  get_sensor()                 │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                               ┌────────▼────────┐
                               │  Flight Platform │
                               └─────────────────┘

laps_ne.py runs OFFLINE before deployment:
Physics Sim (headless) → Fitness → Evolutionary Search → Optimal Parameters → laps_snn.py config
```

---

## Interface Reference

```python
class LapsInterface:
    def orient(self, heading: float) -> None: ...  # Set target heading [degrees]
    def thrust(self, value: float) -> None: ...    # Set thrust [-1.0 ... 1.0]
    def get_sensor(self) -> dict: ...              # altitude, heading, line_detected

class MockLaps(LapsInterface):
    """Logs all calls. No hardware required."""
    def get_log(self) -> list[str]: ...
```
