import numpy as np
import pyssht as ssht
from numpy.testing import assert_allclose, assert_array_equal, assert_equal
from pys2let import pys2let_j_max

from pys2sleplet.functions.flm.axisymmetric_wavelet_coefficients_earth import (
    AxisymmetricWaveletCoefficientsEarth,
)
from pys2sleplet.test.constants import J_MIN, L_LARGE, L_SMALL, VAR_SIGNAL, B
from pys2sleplet.utils.slepian_methods import slepian_forward
from pys2sleplet.utils.vars import SAMPLING_SCHEME
from pys2sleplet.utils.wavelet_methods import (
    axisymmetric_wavelet_inverse,
    compute_slepian_wavelet_covariance,
    compute_wavelet_covariance,
    create_slepian_wavelets,
    find_non_zero_wavelet_coefficients,
    slepian_wavelet_forward,
    slepian_wavelet_inverse,
)


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
    assert_allclose(np.abs(f_p - coefficients).mean(), 0, atol=65)


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


def test_axisymmetric_synthesis() -> None:
    """
    tests that the axisymmetric wavelet synthesis recoveres the coefficients
    """
    awc = AxisymmetricWaveletCoefficientsEarth(L_SMALL)
    flm = axisymmetric_wavelet_inverse(L_SMALL, awc.wavelet_coefficients, awc.wavelets)
    assert_allclose(np.abs(flm - awc.earth.coefficients).mean(), 0, atol=1e-13)


def test_only_wavelet_coefficients_within_shannon_returned() -> None:
    """
    verifies that only the non-zero wavelet coefficients are returned
    """
    p_idx = 1
    coeffs_in = np.array([[3], [2], [1], [0]])
    coeffs_out = np.array([[3], [2], [1]])
    shannon_coeffs = find_non_zero_wavelet_coefficients(coeffs_in, p_idx)
    assert_array_equal(shannon_coeffs, coeffs_out)


def test_create_slepian_wavelets() -> None:
    """
    checks that the method creates the scaling function and wavelets
    """
    wavelets = create_slepian_wavelets(L_LARGE, B, J_MIN)
    j_max = pys2let_j_max(B, L_LARGE ** 2, J_MIN)
    assert_equal(j_max - J_MIN + 2, wavelets.shape[0])
    assert_equal(L_LARGE ** 2, wavelets.shape[1])


def test_wavelet_covariance(random_nd_flm) -> None:
    """
    checks that sigma^j is computed for the axisymmetric case
    """
    covariance = compute_wavelet_covariance(random_nd_flm, VAR_SIGNAL)
    assert_equal(random_nd_flm.shape[0], covariance.shape[0])


def test_slepian_wavelet_covariance(slepian_wavelets_polar_cap) -> None:
    """
    checks that sigma^j is computed for the Slepian case
    """
    covariance = compute_slepian_wavelet_covariance(
        slepian_wavelets_polar_cap.wavelets,
        VAR_SIGNAL,
        slepian_wavelets_polar_cap.L,
        slepian_wavelets_polar_cap.slepian,
    )
    assert_equal(slepian_wavelets_polar_cap.wavelets.shape[0], covariance.shape[0])
    assert_equal(
        ssht.sample_shape(slepian_wavelets_polar_cap.L, Method=SAMPLING_SCHEME),
        covariance.shape[1:],
    )
