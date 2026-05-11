"""Tests for laps_ne.py — ES optimizer. No hardware needed."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from python.laps_ne import optimize, MockOracle, BOUNDS
from python.laps_snn import SNNParams


def test_optimizer_returns_result():
    result = optimize(MockOracle(), n_gen=5, lam=4, seed=42)
    assert result.best_fitness >= 0.0
    assert result.best_fitness <= 1.0


def test_params_within_bounds():
    result = optimize(MockOracle(), n_gen=5, lam=4, seed=42)
    for key, (lo, hi) in BOUNDS.items():
        assert lo <= result.best_params[key] <= hi, f"{key} out of bounds"


def test_fitness_improves_or_holds():
    result = optimize(MockOracle(), n_gen=10, lam=6, seed=0)
    fitnesses = [f for _, f in result.history]
    for i in range(1, len(fitnesses)):
        assert fitnesses[i] >= fitnesses[i - 1] - 1e-9


def test_to_snn_params():
    result = optimize(MockOracle(), n_gen=5, lam=4, seed=42)
    params = result.to_snn_params()
    assert isinstance(params, SNNParams)
    assert params.F == result.best_params["F"]
    assert params.t1 == result.best_params["t1"]


def test_history_length():
    result = optimize(MockOracle(), n_gen=7, lam=3, seed=1)
    assert len(result.history) == 8  # gen 0 (baseline) + 7 generations


def test_reproducible_with_seed():
    r1 = optimize(MockOracle(), n_gen=5, lam=4, seed=99)
    r2 = optimize(MockOracle(), n_gen=5, lam=4, seed=99)
    assert r1.best_fitness == r2.best_fitness


if __name__ == "__main__":
    test_optimizer_returns_result()
    test_params_within_bounds()
    test_fitness_improves_or_holds()
    test_to_snn_params()
    test_history_length()
    test_reproducible_with_seed()
    print("All NE tests passed.")
