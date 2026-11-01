import numpy as np
import pytest
from pulsar_timing import (
    SkyPosition,
    SPEED_OF_LIGHT,
    roemer_delay,
    shapiro_delay,
    dispersion_delay,
    correct_topocentric_to_barycentric,
    hellings_downs_correlation,
    compute_hd_correlation_matrix,
    OptimalStatistic,
    Pulsar,
    PulsarTimingArray,
)


def test_sky_position_and_separations():
    p1 = SkyPosition.from_degrees(0.0, 0.0)
    p2 = SkyPosition.from_degrees(90.0, 0.0)
    p3 = SkyPosition.from_degrees(180.0, 0.0)

    # 90 degrees separation
    assert np.isclose(p1.angular_separation(p2), np.pi / 2.0, atol=1e-12)
    # 180 degrees separation (antipodal)
    assert np.isclose(p1.angular_separation(p3), np.pi, atol=1e-12)
    # Self-separation
    assert np.isclose(p1.angular_separation(p1), 0.0, atol=1e-12)


def test_roemer_delay():
    # 1 Astronomical Unit = 149597870700 meters
    au_meters = 149597870700.0
    r_earth = np.array([au_meters, 0.0, 0.0])
    n_pulsar = np.array([1.0, 0.0, 0.0])

    delay = roemer_delay(r_earth, n_pulsar)
    # Time for light to travel 1 AU is ~499.00478 seconds
    expected = -au_meters / SPEED_OF_LIGHT
    assert np.isclose(delay, expected, atol=1e-8)


def test_dispersion_delay():
    dm = 10.0  # pc cm^-3
    f1 = 1400.0  # MHz (L-band)
    f2 = 2800.0  # MHz (S-band)

    delay_l = dispersion_delay(dm, f1)
    delay_s = dispersion_delay(dm, f2)

    assert delay_l > 0.0
    # Quad inverse frequency law: f2 = 2 * f1 -> delay_s = delay_l / 4
    assert np.isclose(delay_s, delay_l / 4.0, rtol=1e-10)

    with pytest.raises(ValueError):
        dispersion_delay(dm, 0.0)


def test_hellings_downs_analytical_values():
    # Collinear theta = 0 -> Gamma = 0.5
    gamma_0 = hellings_downs_correlation(0.0)
    assert np.isclose(gamma_0, 0.5, atol=1e-12)

    # Antipodal theta = pi -> Gamma = 0.25
    gamma_pi = hellings_downs_correlation(np.pi)
    assert np.isclose(gamma_pi, 0.25, atol=1e-12)

    # Perpendicular theta = pi / 2 -> Gamma = 0.5 - 0.25*(0.5) + 1.5*(0.5)*ln(0.5)
    # = 0.375 - 0.75 * ln(2) ~ -0.144883
    gamma_half_pi = hellings_downs_correlation(np.pi / 2.0)
    expected_half_pi = 0.375 - 0.75 * np.log(2.0)
    assert np.isclose(gamma_half_pi, expected_half_pi, atol=1e-6)

    # Array input check
    angles = np.array([0.0, np.pi / 2.0, np.pi])
    gammas = hellings_downs_correlation(angles)
    assert len(gammas) == 3
    assert np.isclose(gammas[0], 0.5)
    assert np.isclose(gammas[2], 0.25)


def test_hd_correlation_matrix():
    positions = [
        SkyPosition.from_degrees(0.0, 0.0),
        SkyPosition.from_degrees(90.0, 0.0),
        SkyPosition.from_degrees(180.0, 0.0),
    ]

    mat = compute_hd_correlation_matrix(positions)
    assert mat.shape == (3, 3)
    assert np.all(np.diag(mat) == 0.5)
    # Symmetric
    assert np.allclose(mat, mat.T)


def test_optimal_statistic_gwb_detection():
    np.random.seed(42)
    num_pairs = 100
    thetas = np.random.uniform(0.1, np.pi - 0.1, num_pairs)

    true_a2 = 4.0
    hd_model = hellings_downs_correlation(thetas)
    # Inject signal + small white noise
    noise_sigma = 0.2
    rhos = true_a2 * hd_model + np.random.normal(0.0, noise_sigma, num_pairs)
    errors = np.full(num_pairs, noise_sigma)

    opt = OptimalStatistic(thetas, rhos, errors)
    a2_hat, sigma_a2, snr = opt.compute_amplitude_and_snr()

    # Recovery should be close to 4.0 with high SNR
    assert np.isclose(a2_hat, true_a2, atol=0.5)
    assert snr > 5.0

    # Test angular binning
    binned = opt.angular_binned_correlation(num_bins=5)
    assert len(binned) > 0
    for b in binned:
        assert "theta_deg" in b
        assert "rho_mean" in b
        assert b["pair_count"] > 0


def test_pta_integration():
    np.random.seed(123)
    pta = PulsarTimingArray()

    # Generate 4 synthetic pulsars
    positions = [
        SkyPosition.from_degrees(0.0, 0.0),
        SkyPosition.from_degrees(45.0, 30.0),
        SkyPosition.from_degrees(120.0, -20.0),
        SkyPosition.from_degrees(180.0, 10.0),
    ]

    # Compute HD covariance matrix for the 4 positions
    hd_mat = compute_hd_correlation_matrix(positions)
    # Regularize with small diagonal white noise to ensure strict positive-definiteness
    cov = 1.0 * hd_mat + 0.2 * np.eye(4)
    cholesky_l = np.linalg.cholesky(cov)

    n_epochs = 200
    # Generate HD-correlated epoch series
    white_noise = np.random.normal(0.0, 1e-7, (4, n_epochs))
    correlated_residuals = cholesky_l @ white_noise

    for i in range(4):
        p = Pulsar(
            name=f"J0{i}00+0000",
            position=positions[i],
            period_seconds=0.005,
            pdot=1e-20,
            dm_pc_cm3=15.0,
            residuals_seconds=correlated_residuals[i],
        )
        pta.add_pulsar(p)

    assert pta.num_pulsars == 4
    thetas, rhos, sigmas = pta.extract_pairs()
    assert len(thetas) == 6
    assert len(rhos) == 6

    stat = pta.analyze_gwb()
    a2, _, snr = stat.compute_amplitude_and_snr()
    assert a2 > 0.0
    assert snr > 2.0
