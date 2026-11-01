"""Hellings and Downs (1983) Angular Correlation Curve for Stochastic GWB."""

import numpy as np
from typing import Union


def hellings_downs_correlation(theta_rad: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Compute Hellings-Downs spatial correlation coefficient between pulsar pairs.
    Gamma(theta) = 0.5 - 0.25 * x + 1.5 * x * ln(x)
    where x = (1 - cos(theta)) / 2.
    """
    is_scalar = np.isscalar(theta_rad)
    theta = np.atleast_1d(theta_rad)

    cos_th = np.clip(np.cos(theta), -1.0, 1.0)
    x = 0.5 * (1.0 - cos_th)

    # Where x -> 0, x * ln(x) -> 0
    gamma = np.zeros_like(x, dtype=np.float64)

    # Mask for x > 0
    pos_mask = x > 1e-15
    gamma[pos_mask] = 0.5 - 0.25 * x[pos_mask] + 1.5 * x[pos_mask] * np.log(x[pos_mask])
    # For x == 0 (collinear / zero separation)
    gamma[~pos_mask] = 0.5

    if is_scalar:
        return float(gamma[0])
    return gamma


def compute_hd_correlation_matrix(sky_positions: list) -> np.ndarray:
    """
    Compute full N x N Hellings-Downs correlation matrix for a list of SkyPosition objects.
    """
    n = len(sky_positions)
    matrix = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        matrix[i, i] = 0.5  # Auto-correlation
        for j in range(i + 1, n):
            theta = sky_positions[i].angular_separation(sky_positions[j])
            val = hellings_downs_correlation(theta)
            matrix[i, j] = val
            matrix[j, i] = val

    return matrix
