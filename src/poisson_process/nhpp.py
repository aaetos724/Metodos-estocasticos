"""Fit and simulate Poisson / Non-Homogeneous Poisson event processes.

Ported and generalized from a coursework notebook that analyzed network
failure events: are they well described by a homogeneous Poisson
process, or does the event rate vary by time of day (NHPP)? The
functions here are domain-agnostic - they take arrays of event
timestamps, not a specific CSV schema - so the same code applies to any
counting process (failures, arrivals, claims, etc.).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def poisson_rate_and_dispersion(event_times_hours: np.ndarray, bin_hours: float = 1.0) -> dict:
    """Bin events into `bin_hours`-wide windows and estimate the Poisson
    rate lambda, plus the variance/mean dispersion ratio.

    A ratio near 1.0 is consistent with a homogeneous Poisson process;
    materially above 1.0 suggests over-dispersion (e.g. a rate that
    varies over time, or clustering).
    """
    bins = np.arange(0, event_times_hours.max() + bin_hours, bin_hours)
    counts, _ = np.histogram(event_times_hours, bins=bins)
    lam = counts.mean()
    return {
        "lambda": float(lam),
        "variance": float(counts.var()),
        "dispersion_ratio": float(counts.var() / lam) if lam > 0 else float("nan"),
        "counts": counts,
    }


def chi_square_poisson_fit(
    counts: np.ndarray, lam: float, min_expected: float = 5.0
) -> tuple[float, float]:
    """Chi-square goodness-of-fit test: do `counts` follow Poisson(lam)?

    Bins the observed counts by value (0, 1, 2, ...), with the top category
    absorbing the whole upper tail (>= k_max) so the expected probabilities
    sum to 1. Adjacent categories whose expected frequency is below
    `min_expected` are pooled, since the chi-square approximation is
    unreliable for sparse cells - a single rare high count against a tiny
    expected frequency would otherwise blow up the statistic. The degrees
    of freedom are reduced by one extra for the rate parameter estimated
    from the data.

    Returns:
        (chi2_statistic, p_value). A small p-value rejects the Poisson
        homogeneous hypothesis.
    """
    from scipy import stats  # lazy import

    counts = np.asarray(counts)
    n = len(counts)
    k_max = int(counts.max())
    ks = np.arange(0, k_max + 1)

    observed = np.array([(counts == k).sum() for k in ks], dtype=float)
    probs = stats.poisson.pmf(ks, lam)
    # top category captures the entire upper tail P(X >= k_max)
    probs[-1] = max(0.0, 1.0 - stats.poisson.cdf(k_max - 1, lam))
    expected = probs * n

    # pool adjacent categories until every expected frequency >= min_expected
    obs_pooled, exp_pooled = [], []
    o_acc = e_acc = 0.0
    for o, e in zip(observed, expected):
        o_acc += o
        e_acc += e
        if e_acc >= min_expected:
            obs_pooled.append(o_acc)
            exp_pooled.append(e_acc)
            o_acc = e_acc = 0.0
    if e_acc > 0:  # fold any leftover tail into the last bin
        if exp_pooled:
            obs_pooled[-1] += o_acc
            exp_pooled[-1] += e_acc
        else:
            obs_pooled.append(o_acc)
            exp_pooled.append(e_acc)

    obs_pooled = np.array(obs_pooled)
    exp_pooled = np.array(exp_pooled) * obs_pooled.sum() / np.sum(exp_pooled)

    # ddof=1 for the estimated lambda, as long as df stays positive
    ddof = 1 if len(obs_pooled) - 2 >= 1 else 0
    chi2, p_value = stats.chisquare(obs_pooled, exp_pooled, ddof=ddof)
    return float(chi2), float(p_value)


def fit_exponential_interarrival(intervals: np.ndarray) -> dict:
    """Fit an exponential distribution to inter-event intervals and test
    the fit with a Kolmogorov-Smirnov test.

    Args:
        intervals: Non-negative inter-event times (any consistent unit).
    """
    from scipy import stats  # lazy import

    intervals = intervals[intervals >= 0]
    mean_interval = intervals.mean()
    rate = 1.0 / mean_interval
    ks_stat, ks_p = stats.kstest(intervals, "expon", args=(0, mean_interval))
    return {
        "mean_interval": float(mean_interval),
        "rate": float(rate),
        "ks_statistic": float(ks_stat),
        "ks_p_value": float(ks_p),
    }


def wait_time_probabilities(rate: float, thresholds: dict[str, tuple[float, float | None]]) -> dict:
    """Compute exponential wait-time probabilities.

    Args:
        rate: Exponential rate parameter (events per unit time).
        thresholds: mapping name -> (low, high). `high=None` means
            "P(wait > low)"; `low=0` and a value means "P(wait < high)".

    Returns:
        dict of name -> probability.
    """
    results = {}
    for name, (low, high) in thresholds.items():
        if high is None:
            results[name] = float(np.exp(-rate * low))
        elif low == 0:
            results[name] = float(1 - np.exp(-rate * high))
        else:
            results[name] = float(np.exp(-rate * low) - np.exp(-rate * high))
    return results


def hourly_rates(event_hours: np.ndarray, n_days: int) -> pd.Series:
    """Average event count per hour-of-day bucket (0-23), across n_days."""
    hour_bucket = np.floor(event_hours).astype(int) % 24
    counts = pd.Series(hour_bucket).value_counts().reindex(range(24), fill_value=0)
    return counts / n_days


def chi_square_homogeneity_test(rates_by_hour: pd.Series) -> tuple[float, float]:
    """Test whether the rate is uniform across hours (homogeneous) or
    varies by hour (non-homogeneous)."""
    from scipy import stats  # lazy import

    observed = rates_by_hour.values
    expected = np.full_like(observed, observed.mean(), dtype=float)
    chi2, p_value = stats.chisquare(observed, expected)
    return float(chi2), float(p_value)


def simulate_nhpp(rates_by_hour: pd.Series, duration_hours: float, seed: int | None = None) -> np.ndarray:
    """Simulate a Non-Homogeneous Poisson Process via thinning.

    Args:
        rates_by_hour: Series indexed 0..23, rate for that hour-of-day.
        duration_hours: Total duration to simulate.
        seed: RNG seed.

    Returns:
        Sorted array of event times in [0, duration_hours).
    """
    rng = np.random.default_rng(seed)
    lam_max = rates_by_hour.max()
    if lam_max <= 0:
        return np.array([])

    events = []
    t = 0.0
    while t < duration_hours:
        t += rng.exponential(1.0 / lam_max)
        if t >= duration_hours:
            break
        hour = int(t % 24)
        if rng.random() < rates_by_hour.loc[hour] / lam_max:
            events.append(t)

    return np.array(events)
