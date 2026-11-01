"""Pulsar Timing Array (PTA) data container and cross-correlation analysis."""

import numpy as np
from typing import List, Tuple
from .astrometry import SkyPosition
from .optimal_statistic import OptimalStatistic


class Pulsar:
    """Millisecond pulsar timed by an astronomical observatory."""

    def __init__(
        self,
        name: str,
        position: SkyPosition,
        period_seconds: float,
        pdot: float,
        dm_pc_cm3: float,
        residuals_seconds: np.ndarray = None,
    ):
        self.name = name
        self.position = position
        self.period = float(period_seconds)
        self.pdot = float(pdot)
        self.dm = float(dm_pc_cm3)
        self.residuals = (
            np.asarray(residuals_seconds, dtype=np.float64)
            if residuals_seconds is not None
            else np.empty(0, dtype=np.float64)
        )


class PulsarTimingArray:
    """Collection of millisecond pulsars monitored for correlated GWB signals."""

    def __init__(self, pulsars: List[Pulsar] = None):
        self.pulsars = pulsars if pulsars is not None else []

    def add_pulsar(self, pulsar: Pulsar):
        self.pulsars.append(pulsar)

    @property
    def num_pulsars(self) -> int:
        return len(self.pulsars)

    def extract_pairs(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute angular separation, cross-correlation of residuals, and uncertainties for all pairs.
        Returns: (thetas, rhos, sigmas)
        """
        thetas = []
        rhos = []
        sigmas = []

        n = len(self.pulsars)
        for i in range(n):
            for j in range(i + 1, n):
                p1 = self.pulsars[i]
                p2 = self.pulsars[j]

                theta = p1.position.angular_separation(p2.position)

                # Ensure aligned residual length
                min_len = min(len(p1.residuals), len(p2.residuals))
                if min_len < 10:
                    continue

                r1 = p1.residuals[:min_len]
                r2 = p2.residuals[:min_len]

                # Normalized cross-correlation
                norm1 = np.std(r1)
                norm2 = np.std(r2)
                if norm1 < 1e-15 or norm2 < 1e-15:
                    continue

                rho = np.mean(r1 * r2) / (norm1 * norm2)
                # Standard error of cross-correlation
                err = 1.0 / np.sqrt(min_len)

                thetas.append(theta)
                rhos.append(rho)
                sigmas.append(err)

        return (
            np.array(thetas, dtype=np.float64),
            np.array(rhos, dtype=np.float64),
            np.array(sigmas, dtype=np.float64),
        )

    def analyze_gwb(self) -> OptimalStatistic:
        """Run Optimal Statistic analysis across all pairs in the timing array."""
        thetas, rhos, sigmas = self.extract_pairs()
        return OptimalStatistic(thetas, rhos, sigmas)
