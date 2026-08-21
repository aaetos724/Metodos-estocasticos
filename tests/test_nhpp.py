import numpy as np
import pandas as pd
import pytest

from src.poisson_process.nhpp import (
    chi_square_homogeneity_test,
    chi_square_poisson_fit,
    fit_exponential_interarrival,
    hourly_rates,
    poisson_rate_and_dispersion,
    simulate_nhpp,
    wait_time_probabilities,
)


def _synthetic_poisson_events(rate=5.0, duration=500, seed=0):
    rng = np.random.default_rng(seed)
    intervals = rng.exponential(1 / rate, size=int(rate * duration * 1.5))
    times = np.cumsum(intervals)
    return times[times < duration]


def test_poisson_rate_and_dispersion_near_one_for_homogeneous_process():
    events = _synthetic_poisson_events(rate=5.0, duration=1000)
    result = poisson_rate_and_dispersion(events, bin_hours=1.0)
    assert result["lambda"] == pytest.approx(5.0, rel=0.15)
    assert result["dispersion_ratio"] == pytest.approx(1.0, abs=0.4)


def test_chi_square_poisson_fit_accepts_true_poisson_data():
    pytest.importorskip("scipy")
    events = _synthetic_poisson_events(rate=8.0, duration=2000)
    result = poisson_rate_and_dispersion(events, bin_hours=1.0)
    chi2, p_value = chi_square_poisson_fit(result["counts"], result["lambda"])
    assert p_value > 0.01  # should not reject a true Poisson fit


def test_fit_exponential_interarrival_recovers_rate():
    pytest.importorskip("scipy")
    rng = np.random.default_rng(1)
    true_rate = 0.5
    intervals = rng.exponential(1 / true_rate, size=5000)
    fit = fit_exponential_interarrival(intervals)
    assert fit["rate"] == pytest.approx(true_rate, rel=0.1)
    assert fit["ks_p_value"] > 0.01


def test_wait_time_probabilities_basic_cases():
    probs = wait_time_probabilities(rate=0.1, thresholds={"gt30": (30, None), "lt5": (0, 5)})
    assert probs["gt30"] == pytest.approx(np.exp(-3))
    assert probs["lt5"] == pytest.approx(1 - np.exp(-0.5))


def test_hourly_rates_uniform_when_events_spread_evenly():
    hours = np.tile(np.arange(24) + 0.5, 10)
    rates = hourly_rates(hours, n_days=10)
    assert len(rates) == 24
    assert rates.std() < 1e-6


def test_chi_square_homogeneity_detects_uneven_rates():
    pytest.importorskip("scipy")
    uneven = pd.Series([20] * 4 + [1] * 20)
    chi2, p_value = chi_square_homogeneity_test(uneven)
    assert p_value < 0.01


def test_simulate_nhpp_produces_sorted_events_within_duration():
    rates = pd.Series([3.0] * 24)
    events = simulate_nhpp(rates, duration_hours=72, seed=5)
    assert len(events) > 0
    assert events.max() < 72
    assert (np.diff(events) >= 0).all()
