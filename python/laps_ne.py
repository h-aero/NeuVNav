"""
laps_ne.py -- Neuro-Evolution optimizer for LTA boundary patrol parameters.

Evolutionary Strategy (ES) that searches the SNN controller parameter space
offline, using a physics simulation as fitness oracle. Ported from the
NeuVNav JavaScript simulator (sim.php).

Algorithm: (1+lambda)-ES with per-generation sigma adaptation (1/5 success rule).
Fitness oracle is pluggable -- implement FitnessOracle for your simulation.

ENFIELD WP2 -- funded by the European Union, grant No 101120657.
"""

import math
import random

from python.laps_snn import SNNParams


# ---------------------------------------------------------------------------
# Parameter bounds (mirrors sim.php ES search space defaults)
# ---------------------------------------------------------------------------

BOUNDS = {
    "F":     (0.1,  1.0),   # forward thrust
    "theta": (0.0, 90.0),   # turn angle [deg]
    "n":     (0.0,  3.0),   # thrust profile exponent
    "ts":    (0.0,  5.0),   # setup time [s]
    "bd":    (0.0,  1.0),   # brake distance [m]
    "t1":    (0.1,  5.0),   # detection -> thrust reduction [s]
    "t2":    (0.1,  5.0),   # thrust reduction -> orient [s]
    "T":     (0.0,  0.5),   # thrust during braking
    "t3":    (0.1,  5.0),   # post-orient -> restore [s]
}

# Initial mutation step sizes (sigma) -- from sim.php
SIGMA_INIT = {
    "F":     3.0,
    "theta": 3.0,
    "n":     0.3,
    "ts":    0.4,
    "bd":    0.1,
    "t1":    0.4,
    "t2":    0.4,
    "T":     0.05,
    "t3":    0.4,
}

SIGMA_MIN = {k: b * 0.01 for k, b in {
    "F": 0.2, "theta": 0.5, "n": 0.05, "ts": 0.05, "bd": 0.01,
    "t1": 0.05, "t2": 0.05, "T": 0.01, "t3": 0.05,
}.items()}

SIGMA_MAX = {
    "F": 8.0, "theta": 10.0, "n": 1.0, "ts": 1.5, "bd": 0.5,
    "t1": 2.0, "t2": 2.0, "T": 0.2, "t3": 2.0,
}


# ---------------------------------------------------------------------------
# Fitness oracle interface
# ---------------------------------------------------------------------------

class FitnessOracle:
    """
    Abstract fitness oracle. Implement this with your physics simulation.

    The NE optimizer calls evaluate() for each candidate parameter set.
    Return a score in [0, 1] where 1 is perfect.

    Example implementation::

        class MyOracle(FitnessOracle):
            def evaluate(self, params):
                result = my_physics_sim(params)
                return result.t_inside / result.t_total
    """

    def evaluate(self, params):
        """
        Parameters
        ----------
        params : dict
            Candidate parameter dict with keys: F, theta, n, ts, bd, t1, t2, T, t3.

        Returns
        -------
        float
            Fitness score in [0, 1]. Higher is better.
        """
        raise NotImplementedError


class MockOracle(FitnessOracle):
    """
    Simple mock oracle for unit testing. Returns random fitness.
    Replace with a real physics simulation for production use.
    """

    def __init__(self, seed=42):
        random.seed(seed)

    def evaluate(self, params):
        # Prefer moderate thrust and short delays -- crude heuristic for testing
        f_score = 1.0 - abs(params["F"] - 0.5) / 0.5
        t_score = 1.0 - (params["t1"] + params["t2"]) / 10.0
        return max(0.0, min(1.0, 0.6 * f_score + 0.4 * t_score + random.gauss(0, 0.05)))


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

class NEResult:
    """Holds the outcome of one NE optimization run."""

    def __init__(self, best_params, best_fitness, history):
        self.best_params = best_params      # dict of best parameter values
        self.best_fitness = best_fitness    # float in [0, 1]
        self.history = history              # list of (gen, fitness) tuples

    def to_snn_params(self, alt_target=1.0, alt_threshold=0.15, alt_gain=0.3):
        """Convert NE result to SNNParams for direct use with SNNController."""
        p = self.best_params
        return SNNParams(
            t1=p["t1"],
            t2=p["t2"],
            T=p["T"],
            t3=p["t3"],
            F=p["F"],
            alt_target=alt_target,
            alt_threshold=alt_threshold,
            alt_gain=alt_gain,
        )

    def __repr__(self):
        p = self.best_params
        return (
            f"NEResult(fitness={self.best_fitness:.4f}, "
            f"F={p['F']:.3f}, t1={p['t1']:.2f}, t2={p['t2']:.2f}, "
            f"T={p['T']:.3f}, t3={p['t3']:.2f})"
        )


# ---------------------------------------------------------------------------
# ES optimizer
# ---------------------------------------------------------------------------

def _gauss():
    """Box-Muller Gaussian sample (no numpy dependency)."""
    u1 = random.random() or 1e-10
    u2 = random.random()
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _mutate(best, sigma, active_keys):
    candidate = dict(best)
    for key in active_keys:
        lo, hi = BOUNDS[key]
        candidate[key] = _clamp(best[key] + sigma[key] * _gauss(), lo, hi)
    return candidate


def _adapt_sigma(sigma, p_success, active_keys):
    """1/5 success rule: scale up if success rate > 20%, down otherwise."""
    scale = 1.22 if p_success > 0.2 else 0.82
    for key in active_keys:
        sigma[key] = _clamp(sigma[key] * scale, SIGMA_MIN[key], SIGMA_MAX[key])


def optimize(
    oracle,
    n_gen=20,
    lam=10,
    active_keys=None,
    init_params=None,
    seed=None,
    verbose=False,
    on_generation=None,
):
    """
    Run the Neuro-Evolution optimizer.

    Parameters
    ----------
    oracle : FitnessOracle
        Physics simulation or other fitness evaluator.
    n_gen : int
        Number of generations. Default 20.
    lam : int
        Candidates evaluated per generation (lambda). Default 10.
    active_keys : list of str or None
        Parameters to optimize. None = all keys in BOUNDS.
    init_params : dict or None
        Starting parameter values. None = midpoint of each bound.
    seed : int or None
        Random seed for reproducibility.
    verbose : bool
        Print progress each generation.
    on_generation : callable or None
        Called after each generation: on_generation(gen, best_fitness, best_params).
        Useful for progress bars or early stopping.

    Returns
    -------
    NEResult
    """
    if seed is not None:
        random.seed(seed)

    if active_keys is None:
        active_keys = list(BOUNDS.keys())

    # Initialise best from init_params or bound midpoints
    if init_params is not None:
        best = dict(init_params)
    else:
        best = {k: (BOUNDS[k][0] + BOUNDS[k][1]) / 2.0 for k in BOUNDS}

    sigma = {k: SIGMA_INIT.get(k, 0.1) for k in BOUNDS}

    best_fit = oracle.evaluate(best)
    history = [(0, best_fit)]

    for gen in range(1, n_gen + 1):
        gen_best = dict(best)
        gen_best_fit = best_fit
        successes = 0

        for _ in range(lam):
            candidate = _mutate(best, sigma, active_keys)
            fit = oracle.evaluate(candidate)
            if fit > gen_best_fit:
                gen_best = candidate
                gen_best_fit = fit
                successes += 1

        _adapt_sigma(sigma, successes / lam, active_keys)

        if gen_best_fit > best_fit:
            best = gen_best
            best_fit = gen_best_fit

        history.append((gen, best_fit))

        if verbose:
            print(f"gen {gen:3d}/{n_gen}  fitness={best_fit:.4f}  "
                  f"F={best['F']:.3f}  t1={best['t1']:.2f}  t2={best['t2']:.2f}")

        if on_generation is not None:
            on_generation(gen, best_fit, dict(best))

    return NEResult(best, best_fit, history)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("NeuVNav laps_ne.py -- Neuro-Evolution optimizer")
    print("Running demo with MockOracle (replace with real physics sim)\n")

    result = optimize(
        oracle=MockOracle(),
        n_gen=15,
        lam=8,
        verbose=True,
        seed=0,
    )

    print(f"\nBest result: {result}")
    snn_params = result.to_snn_params()
    print(f"SNNParams:   t1={snn_params.t1:.2f}  t2={snn_params.t2:.2f}  "
          f"T={snn_params.T:.3f}  t3={snn_params.t3:.2f}  F={snn_params.F:.3f}")
