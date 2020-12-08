import numpy as np
from numpy.testing import assert_allclose, assert_array_less

from pys2sleplet.functions.flm.axisymmetric_wavelet_coefficients_earth import (
    AxisymmetricWaveletCoefficientsEarth,
)
from pys2sleplet.test.constants import L_SMALL
from pys2sleplet.utils.slepian_methods import slepian_forward
from pys2sleplet.utils.wavelet_methods import (
    axisymmetric_wavelet_inverse,
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


def test_axisymmetric_synthesis() -> None:
    """
    tests that the axisymmetric wavelet synthesis recoveres the coefficients
    """
    awc = AxisymmetricWaveletCoefficientsEarth(L_SMALL)
    flm = axisymmetric_wavelet_inverse(L_SMALL, awc.wavelet_coefficients, awc.wavelets)
    assert_allclose(np.abs(flm - awc.earth.coefficients).mean(), 0, atol=1e-13)


def test_only_wavelet_coefficients_within_shannon_returned(
    slepian_wavelet_coefficients_south_america,
) -> None:
    """
    verifies that only the non-zero wavelet coefficients are returned
    """
    within_shannon_wavelets = find_non_zero_wavelet_coefficients(
        slepian_wavelet_coefficients_south_america.wavelet_coefficients
    )
    assert_array_less(
        within_shannon_wavelets.shape[0],
        slepian_wavelet_coefficients_south_america.wavelet_coefficients.shape[0],
    )
