import numpy as np
import pyssht as ssht
from numpy.testing import assert_allclose, assert_array_equal, assert_equal
from pys2let import pys2let_j_max
from sleplet.functions.flm.axisymmetric_wavelet_coefficients_earth import (
    AxisymmetricWaveletCoefficientsEarth,
)
from sleplet.functions.flm.axisymmetric_wavelet_coefficients_south_america import (
    AxisymmetricWaveletCoefficientsSouthAmerica,
)
from sleplet.utils.slepian_methods import slepian_forward
from sleplet.utils.vars import SAMPLING_SCHEME
from sleplet.utils.wavelet_methods import (
    axisymmetric_wavelet_inverse,
    compute_slepian_wavelet_covariance,
    compute_wavelet_covariance,
    create_kappas,
    find_non_zero_wavelet_coefficients,
    slepian_wavelet_forward,
    slepian_wavelet_inverse,
)

B = 2
J_MIN = 0
L_LARGE = 128
L_SMALL = 16
VAR_SIGNAL = 1


def test_synthesis_polar(slepian_wavelets_polar_cap, earth_polar_cap) -> None:
    """
    tests that Slepian polar wavelet synthesis matches the coefficients
    """
    coefficients = slepian_forward(
        slepian_wavelets_polar_cap.L,
        slepian_wavelets_polar_cap.slepian,
        flm=earth_polar_cap.coefficients,
    )
    wav_coeffs = slepian_wavelet_forward(
        coefficients,
        slepian_wavelets_polar_cap.wavelets,
        slepian_wavelets_polar_cap.slepian.N,
    )
    f_p = slepian_wavelet_inverse(
        wav_coeffs,
        slepian_wavelets_polar_cap.wavelets,
        slepian_wavelets_polar_cap.slepian.N,
    )
    assert_allclose(np.abs(f_p - coefficients).mean(), 0, atol=1e-14)


def test_synthesis_lim_lat_lon(slepian_wavelets_lim_lat_lon, earth_lim_lat_lon) -> None:
    """
    tests that Slepian lim_lat_lon wavelet synthesis matches the coefficients
    """
    coefficients = slepian_forward(
        slepian_wavelets_lim_lat_lon.L,
        slepian_wavelets_lim_lat_lon.slepian,
        flm=earth_lim_lat_lon.coefficients,
    )
    wav_coeffs = slepian_wavelet_forward(
        coefficients,
        slepian_wavelets_lim_lat_lon.wavelets,
        slepian_wavelets_lim_lat_lon.slepian.N,
    )
    f_p = slepian_wavelet_inverse(
        wav_coeffs,
        slepian_wavelets_lim_lat_lon.wavelets,
        slepian_wavelets_lim_lat_lon.slepian.N,
    )
    assert_allclose(np.abs(f_p - coefficients).mean(), 0, atol=0)


def test_axisymmetric_synthesis_earth() -> None:
    """
    tests that the axisymmetric wavelet synthesis recoveres the coefficients
    """
    awc = AxisymmetricWaveletCoefficientsEarth(L_SMALL, B=B, j_min=J_MIN)
    flm = axisymmetric_wavelet_inverse(L_SMALL, awc.wavelet_coefficients, awc.wavelets)
    assert_allclose(np.abs(flm - awc.earth.coefficients).mean(), 0, atol=1e-13)


def test_axisymmetric_synthesis_south_america() -> None:
    """
    tests that the axisymmetric wavelet synthesis recoveres the coefficients
    """
    awc = AxisymmetricWaveletCoefficientsSouthAmerica(L_SMALL, B=B, j_min=J_MIN)
    flm = axisymmetric_wavelet_inverse(L_SMALL, awc.wavelet_coefficients, awc.wavelets)
    assert_allclose(np.abs(flm - awc.south_america.coefficients).mean(), 0, atol=1e-14)


def test_only_wavelet_coefficients_within_shannon_returned() -> None:
    """
    verifies that only the non-zero wavelet coefficients are returned
    """
    coeffs_in = np.array([[3], [2], [1], [0]])
    coeffs_out = np.array([[3], [2], [1]])
    shannon_coeffs = find_non_zero_wavelet_coefficients(coeffs_in, axis=1)
    assert_array_equal(shannon_coeffs, coeffs_out)


def test_create_kappas() -> None:
    """
    checks that the method creates the scaling function and wavelets
    """
    wavelets = create_kappas(L_LARGE**2, B, J_MIN)
    j_max = pys2let_j_max(B, L_LARGE**2, J_MIN)
    assert_equal(j_max - J_MIN + 2, wavelets.shape[0])
    assert_equal(L_LARGE**2, wavelets.shape[1])


def test_wavelet_covariance(random_nd_flm) -> None:
    """
    checks that sigma^j is computed for the axisymmetric case
    """
    covariance = compute_wavelet_covariance(
        L_SMALL, random_nd_flm, var_signal=VAR_SIGNAL
    )
    assert_equal(random_nd_flm.shape[0], covariance.shape[0])


def test_slepian_wavelet_covariance(slepian_wavelets_polar_cap) -> None:
    """
    checks that sigma^j is computed for the Slepian case
    """
    covariance = compute_slepian_wavelet_covariance(
        slepian_wavelets_polar_cap.L,
        slepian_wavelets_polar_cap.wavelets,
        slepian_wavelets_polar_cap.slepian,
        var_signal=VAR_SIGNAL,
    )
    assert_equal(slepian_wavelets_polar_cap.wavelets.shape[0], covariance.shape[0])
    assert_equal(
        ssht.sample_shape(slepian_wavelets_polar_cap.L, Method=SAMPLING_SCHEME),
        covariance.shape[1:],
    )
