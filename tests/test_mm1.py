import numpy as np
import pytest

from src.queueing.mm1 import (
    compare_scenarios,
    estimate_parameters,
    littles_law,
    mm1_theoretical,
    simulate_fifo,
)


def test_simulate_fifo_no_wait_when_spaced_out():
    arrivals = np.array([0.0, 10.0, 20.0])
    service = np.array([1.0, 1.0, 1.0])
    result = simulate_fifo(arrivals, service)
    assert (result["wait"] == 0).all()


def test_simulate_fifo_queue_builds_up():
    arrivals = np.array([0.0, 0.0, 0.0])
    service = np.array([1.0, 1.0, 1.0])
    result = simulate_fifo(arrivals, service)
    assert list(result["wait"]) == [0.0, 1.0, 2.0]


def test_estimate_parameters_matches_known_inputs():
    arrivals = np.arange(0, 100, 2.0)  # 50 arrivals over 98 units -> lambda ~ 0.51
    service = np.full(len(arrivals), 1.0)  # mu = 1.0
    params = estimate_parameters(arrivals, service)
    assert params["mu"] == pytest.approx(1.0)
    assert params["stable"] is True


def test_mm1_theoretical_matches_hand_calculation():
    metrics = mm1_theoretical(lam=4.0, mu=5.0)
    assert metrics["rho"] == pytest.approx(0.8)
    assert metrics["L"] == pytest.approx(4.0)
    assert metrics["Lq"] == pytest.approx(3.2)
    assert metrics["W"] == pytest.approx(1.0)
    assert metrics["Wq"] == pytest.approx(0.8)


def test_mm1_theoretical_unstable_when_overloaded():
    metrics = mm1_theoretical(lam=6.0, mu=5.0)
    assert metrics["stable"] is False
    assert metrics["L"] == float("inf")


def test_littles_law_close_to_theory_on_average():
    # Large n needed for stable convergence near rho=0.8; M/M/1 queue
    # length variance grows quickly as utilization increases.
    rng = np.random.default_rng(3)
    lam, mu = 4.0, 5.0
    n = 30000
    arrivals = np.cumsum(rng.exponential(1 / lam, size=n))
    service = rng.exponential(1 / mu, size=n)

    result = simulate_fifo(arrivals, service)
    empirical = littles_law(result, lam)
    theory = mm1_theoretical(lam, mu)

    assert empirical["L"] == pytest.approx(theory["L"], rel=0.1)


def test_compare_scenarios_flags_unstable_scenario():
    scenarios = {"base": (4.0, 5.0), "overload": (6.0, 5.0)}
    table = compare_scenarios(scenarios)
    assert table.set_index("scenario").loc["base", "stable"] == True
    assert table.set_index("scenario").loc["overload", "stable"] == False
