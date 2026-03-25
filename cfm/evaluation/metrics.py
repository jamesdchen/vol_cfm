"""Evaluation metrics for conditional flow matching of intraday RV proportions."""

from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.spatial.distance import cdist


def daily_consistency_error(generated_48: np.ndarray, daily_rv: np.ndarray) -> float:
    """Mean absolute error between summed intraday RV and daily RV.

    Parameters
    ----------
    generated_48 : np.ndarray, shape (N, 48)
        Absolute intraday RV values (not proportions).
    daily_rv : np.ndarray, shape (N,)
        Daily realized volatility.

    Returns
    -------
    float
        Mean absolute error.
    """
    return float(np.mean(np.abs(generated_48.sum(axis=1) - daily_rv)))


def marginal_ks_test(real: np.ndarray, generated: np.ndarray) -> dict:
    """Per-period two-sample KS tests between real and generated proportions.

    Parameters
    ----------
    real : np.ndarray, shape (N, 48)
    generated : np.ndarray, shape (M, 48)

    Returns
    -------
    dict
        ks_stats, p_values (arrays of 48), mean_ks, frac_significant.
    """
    ks_stats = np.empty(48)
    p_values = np.empty(48)
    for i in range(48):
        stat, pval = stats.ks_2samp(real[:, i], generated[:, i])
        ks_stats[i] = stat
        p_values[i] = pval
    return {
        "ks_stats": ks_stats,
        "p_values": p_values,
        "mean_ks": float(ks_stats.mean()),
        "frac_significant": float((p_values < 0.05).mean()),
    }


def _acf(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Compute autocorrelation for lags 1..max_lag."""
    x = x - x.mean()
    var = np.dot(x, x)
    if var == 0:
        return np.zeros(max_lag)
    acf = np.empty(max_lag)
    for lag in range(1, max_lag + 1):
        acf[lag - 1] = np.dot(x[:-lag], x[lag:]) / var
    return acf


def acf_comparison(
    real: np.ndarray, generated: np.ndarray, max_lag: int = 10
) -> dict:
    """Compare autocorrelation structure of flattened real vs generated series.

    Parameters
    ----------
    real : np.ndarray, shape (N, 48)
    generated : np.ndarray, shape (M, 48)
    max_lag : int
        Number of lags to compute.

    Returns
    -------
    dict
        real_acf, gen_acf (arrays of length max_lag), acf_mae.
    """
    real_flat = real.ravel()
    gen_flat = generated.ravel()
    real_acf = _acf(real_flat, max_lag)
    gen_acf = _acf(gen_flat, max_lag)
    return {
        "real_acf": real_acf,
        "gen_acf": gen_acf,
        "acf_mae": float(np.mean(np.abs(real_acf - gen_acf))),
    }


def diurnal_pattern_error(real: np.ndarray, generated: np.ndarray) -> dict:
    """Compare mean diurnal patterns.

    Parameters
    ----------
    real : np.ndarray, shape (N, 48)
    generated : np.ndarray, shape (M, 48)

    Returns
    -------
    dict
        real_pattern, gen_pattern (arrays of 48), l2_error, max_error.
    """
    real_pattern = real.mean(axis=0)
    gen_pattern = generated.mean(axis=0)
    diff = real_pattern - gen_pattern
    return {
        "real_pattern": real_pattern,
        "gen_pattern": gen_pattern,
        "l2_error": float(np.linalg.norm(diff)),
        "max_error": float(np.max(np.abs(diff))),
    }


def energy_distance(real: np.ndarray, generated: np.ndarray) -> float:
    """Energy distance between two sets of 48-dimensional samples.

    Uses the formula: 2*E[||X-Y||] - E[||X-X'||] - E[||Y-Y'||].
    Subsamples to at most 1000 observations per set for efficiency.

    Parameters
    ----------
    real : np.ndarray, shape (N, 48)
    generated : np.ndarray, shape (M, 48)

    Returns
    -------
    float
        Energy distance.
    """
    rng = np.random.default_rng(42)
    max_n = 1000
    if len(real) > max_n:
        real = real[rng.choice(len(real), max_n, replace=False)]
    if len(generated) > max_n:
        generated = generated[rng.choice(len(generated), max_n, replace=False)]

    xy = cdist(real, generated, metric="euclidean").mean()
    xx = cdist(real, real, metric="euclidean").mean()
    yy = cdist(generated, generated, metric="euclidean").mean()
    return float(2.0 * xy - xx - yy)


def evaluate_all(
    real: np.ndarray,
    generated: np.ndarray,
    daily_rv: np.ndarray | None = None,
) -> dict:
    """Run all evaluation metrics and return a flat summary dict.

    Parameters
    ----------
    real : np.ndarray, shape (N, 48)
        Real proportion vectors.
    generated : np.ndarray, shape (M, 48)
        Generated proportion vectors.
    daily_rv : np.ndarray or None, shape (M,)
        If provided, generated is treated as absolute RV for consistency check.

    Returns
    -------
    dict
        Flat dictionary of scalar summary metrics.
    """
    results: dict = {}

    if daily_rv is not None:
        results["daily_consistency_mae"] = daily_consistency_error(
            generated, daily_rv
        )

    ks = marginal_ks_test(real, generated)
    results["marginal_mean_ks"] = ks["mean_ks"]
    results["marginal_frac_significant"] = ks["frac_significant"]

    acf = acf_comparison(real, generated)
    results["acf_mae"] = acf["acf_mae"]

    diurnal = diurnal_pattern_error(real, generated)
    results["diurnal_l2_error"] = diurnal["l2_error"]
    results["diurnal_max_error"] = diurnal["max_error"]

    results["energy_distance"] = energy_distance(real, generated)

    return results
