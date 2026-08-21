"""FIFO queue simulation and M/M/1 theory, ported from a coursework
notebook analyzing a single-server queue (arrivals, service times,
utilization, and scenario comparisons).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def simulate_fifo(arrivals: np.ndarray, service_times: np.ndarray) -> pd.DataFrame:
    """Simulate a single-server FIFO queue.

    start_i  = max(arrival_i, finish_{i-1})
    finish_i = start_i + service_i
    wait_i   = start_i - arrival_i
    system_i = finish_i - arrival_i
    """
    n = len(arrivals)
    start = np.zeros(n)
    finish = np.zeros(n)

    prev_finish = 0.0
    for i in range(n):
        start[i] = max(arrivals[i], prev_finish)
        finish[i] = start[i] + service_times[i]
        prev_finish = finish[i]

    df = pd.DataFrame(
        {"arrival": arrivals, "service": service_times, "start": start, "finish": finish}
    )
    df["wait"] = df["start"] - df["arrival"]
    df["system"] = df["finish"] - df["arrival"]
    return df


def estimate_parameters(arrivals: np.ndarray, service_times: np.ndarray) -> dict:
    """Estimate lambda, mu, rho from observed data.

    lambda = N / T (arrival rate over the observed window)
    mu     = 1 / mean(service_time)
    rho    = lambda / mu
    """
    n = len(arrivals)
    duration = arrivals.max() - arrivals.min()
    lam = n / duration
    mu = 1.0 / service_times.mean()
    return {"lambda": lam, "mu": mu, "rho": lam / mu, "stable": bool(lam < mu)}


def mm1_theoretical(lam: float, mu: float) -> dict:
    """Closed-form M/M/1 metrics: rho, L, Lq, W, Wq."""
    rho = lam / mu
    if rho >= 1:
        return {"rho": rho, "stable": False, "L": np.inf, "Lq": np.inf, "W": np.inf, "Wq": np.inf}
    return {
        "rho": rho,
        "stable": True,
        "L": rho / (1 - rho),
        "Lq": rho**2 / (1 - rho),
        "W": 1 / (mu - lam),
        "Wq": lam / (mu * (mu - lam)),
    }


def littles_law(results: pd.DataFrame, lam: float) -> dict:
    """L = lambda * W and Lq = lambda * Wq, from simulated results."""
    return {
        "L": lam * results["system"].mean(),
        "Lq": lam * results["wait"].mean(),
    }


def compare_scenarios(scenarios: dict[str, tuple[float, float]]) -> pd.DataFrame:
    """Compare M/M/1 metrics across named (lambda, mu) scenarios.

    Args:
        scenarios: mapping scenario name -> (lambda, mu).

    Returns:
        DataFrame with one row per scenario and the M/M/1 metrics.
    """
    rows = []
    for name, (lam, mu) in scenarios.items():
        metrics = mm1_theoretical(lam, mu)
        rows.append({"scenario": name, "lambda": lam, "mu": mu, **metrics})
    return pd.DataFrame(rows)
