"""Relativistic and Interstellar Propagation Delays for Pulsar Timing."""

import numpy as np

# SI and Astronomical Constants
SPEED_OF_LIGHT = 299792458.0  # m / s
G_CONST = 6.67430e-11         # m^3 / (kg * s^2)
SOLAR_MASS = 1.98847e30       # kg
T_SUN = G_CONST * SOLAR_MASS / (SPEED_OF_LIGHT ** 3)  # ~4.9255 microseconds
DM_CONST = 4.148808e3         # MHz^2 * pc^-1 * cm^3 * s


def roemer_delay(r_obs_ssb_meters: np.ndarray, pulsar_unit_vector: np.ndarray) -> float:
    """
    Geometric Roemer delay between topocentric observatory and Solar System Barycenter (SSB).
    Delta_R = - (r_obs . n_pulsar) / c
    """
    return float(-np.dot(r_obs_ssb_meters, pulsar_unit_vector) / SPEED_OF_LIGHT)


def shapiro_delay(r_obs_sun_meters: np.ndarray, pulsar_unit_vector: np.ndarray) -> float:
    """
    General relativistic Shapiro delay due to the Sun's spacetime curvature.
    Delta_S = - 2 * T_sun * ln(1 - s_sun . n_pulsar)
    where s_sun is unit vector from Sun to Earth.
    """
    r_norm = np.linalg.norm(r_obs_sun_meters)
    if r_norm < 1e-6:
        return 0.0

    s_sun = r_obs_sun_meters / r_norm
    cos_angle = np.clip(np.dot(s_sun, pulsar_unit_vector), -1.0, 1.0)

    # Avoid singularity near superior conjunction
    arg = max(1e-12, 1.0 - cos_angle)
    return float(-2.0 * T_SUN * np.log(arg))


def dispersion_delay(dm_pc_cm3: float, frequency_mhz: float) -> float:
    """
    Interstellar plasma cold-plasma dispersion delay in seconds.
    Delta_t_DM = D * (DM / f^2)
    """
    if frequency_mhz <= 0.0:
        raise ValueError("Observing frequency must be positive")
    return float(DM_CONST * (dm_pc_cm3 / (frequency_mhz ** 2)))


def correct_topocentric_to_barycentric(
    t_topo_seconds: float,
    r_obs_ssb_meters: np.ndarray,
    r_obs_sun_meters: np.ndarray,
    pulsar_unit_vector: np.ndarray,
    dm_pc_cm3: float,
    frequency_mhz: float,
) -> float:
    """
    Convert topocentric pulse time-of-arrival (TOA) to barycentric coordinate arrival time (TCB).
    """
    delta_r = roemer_delay(r_obs_ssb_meters, pulsar_unit_vector)
    delta_s = shapiro_delay(r_obs_sun_meters, pulsar_unit_vector)
    delta_dm = dispersion_delay(dm_pc_cm3, frequency_mhz)

    return t_topo_seconds + delta_r + delta_s - delta_dm
