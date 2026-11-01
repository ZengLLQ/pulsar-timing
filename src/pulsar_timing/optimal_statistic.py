"""Optimal Statistic and Cross-Correlation Estimator for Pulsar Timing Arrays."""

import numpy as np
from typing import Tuple, Dict, List
from .hellings_downs import hellings_downs_correlation


class OptimalStatistic:
    """Estimates stochastic gravitational wave background amplitude from cross-correlations."""

    def __init__(self, angular_separations_rad: np.ndarray, cross_correlations: np.ndarray, errors: np.ndarray):
        self.thetas = np.asarray(angular_separations_rad, dtype=np.float64)
        self.rhos = np.asarray(cross_correlations, dtype=np.float64)
        self.sigmas = np.asarray(errors, dtype=np.float64)

        if len(self.thetas) != len(self.rhos) or len(self.rhos) != len(self.sigmas):
            raise ValueError("Arrays must have identical lengths")

        self.gamma_hd = hellings_downs_correlation(self.thetas)

    def compute_amplitude_and_snr(self) -> Tuple[float, float, float]:
        """
        Compute A_hat^2, sigma_A2, and SNR.
        Returns: (a2_hat, sigma_a2, snr)
        """
        weights = 1.0 / (self.sigmas ** 2)

        numerator = np.sum(self.rhos * self.gamma_hd * weights)
        denominator = np.sum((self.gamma_hd ** 2) * weights)

        if denominator <= 0.0:
            return 0.0, 1.0, 0.0

        a2_hat = float(numerator / denominator)
        sigma_a2 = float(1.0 / np.sqrt(denominator))
        snr = float(a2_hat / sigma_a2)

        return a2_hat, sigma_a2, snr

    def angular_binned_correlation(self, num_bins: int = 6) -> List[Dict[str, float]]:
        """Group cross-correlations into angular bins for plotting against the HD curve."""
        bins = np.linspace(0.0, np.pi, num_bins + 1)
        binned_results = []

        for i in range(num_bins):
            th_min = bins[i]
            th_max = bins[i + 1]
            mask = (self.thetas >= th_min) & (self.thetas < th_max)

            if not np.any(mask):
                continue

            th_center = 0.5 * (th_min + th_max)
            w = 1.0 / (self.sigmas[mask] ** 2)
            weighted_rho = np.sum(self.rhos[mask] * w) / np.sum(w)
            weighted_err = 1.0 / np.sqrt(np.sum(w))
            model_hd = hellings_downs_correlation(th_center)

            binned_results.append({
                "theta_deg": float(np.degrees(th_center)),
                "rho_mean": float(weighted_rho),
                "rho_err": float(weighted_err),
                "model_hd": float(model_hd),
                "pair_count": int(np.sum(mask)),
            })

        return binned_results
