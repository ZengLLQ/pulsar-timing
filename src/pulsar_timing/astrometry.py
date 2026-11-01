"""Pulsar Astrometry and Celestial Coordinate Transformations."""

import numpy as np


class SkyPosition:
    """Celestial coordinate on the sky sphere."""

    def __init__(self, ra_rad: float, dec_rad: float):
        self.ra = float(ra_rad) % (2.0 * np.pi)
        self.dec = float(np.clip(dec_rad, -np.pi / 2.0, np.pi / 2.0))

    @classmethod
    def from_degrees(cls, ra_deg: float, dec_deg: float) -> "SkyPosition":
        return cls(np.radians(ra_deg), np.radians(dec_deg))

    @property
    def unit_vector(self) -> np.ndarray:
        """Unit vector pointing from observer towards the pulsar."""
        cos_d = np.cos(self.dec)
        return np.array([
            cos_d * np.cos(self.ra),
            cos_d * np.sin(self.ra),
            np.sin(self.dec)
        ], dtype=np.float64)

    def angular_separation(self, other: "SkyPosition") -> float:
        """Angular separation theta in radians between two sky positions."""
        n1 = self.unit_vector
        n2 = other.unit_vector
        cos_theta = np.clip(np.dot(n1, n2), -1.0, 1.0)
        return float(np.arccos(cos_theta))
