"""PulsarTiming: High-precision PTA astrometry, relativistic delays, and Hellings-Downs correlation."""

from .astrometry import SkyPosition
from .delays import (
    SPEED_OF_LIGHT,
    G_CONST,
    SOLAR_MASS,
    T_SUN,
    DM_CONST,
    roemer_delay,
    shapiro_delay,
    dispersion_delay,
    correct_topocentric_to_barycentric,
)
from .hellings_downs import hellings_downs_correlation, compute_hd_correlation_matrix
from .optimal_statistic import OptimalStatistic
from .pulsar import Pulsar, PulsarTimingArray

__version__ = "0.1.0"
__all__ = [
    "SkyPosition",
    "SPEED_OF_LIGHT",
    "G_CONST",
    "SOLAR_MASS",
    "T_SUN",
    "DM_CONST",
    "roemer_delay",
    "shapiro_delay",
    "dispersion_delay",
    "correct_topocentric_to_barycentric",
    "hellings_downs_correlation",
    "compute_hd_correlation_matrix",
    "OptimalStatistic",
    "Pulsar",
    "PulsarTimingArray",
]
