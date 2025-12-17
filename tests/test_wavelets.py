from __future__ import annotations

import numpy as np

import pys2let

import sleplet

B = 2
J_MIN = 0
L_LARGE = 128
L_SMALL = 16
VAR_SIGNAL = 1


def test_synthesis_polar(
    slepian_wavelets_polar_cap: sleplet.functions.slepian_wavelets.SlepianWavelets,
    earth_polar_cap: sleplet.functions.earth.Earth,
) -> None:
    """Test that Slepian polar wavelet synthesis matches the coefficients."""
    coefficients = sleplet.slepian_methods.slepian_forward(
        slepian_wavelets_polar_cap.L,
        slepian_wavelets_polar_cap.slepian,
        flm=earth_polar_cap.coefficients,
    )
    wav_coeffs = sleplet.wavelet_methods.slepian_wavelet_forward(
        coefficients,
        slepian_wavelets_polar_cap.wavelets,
        slepian_wavelets_polar_cap.slepian.N,
    )
    f_p = sleplet.wavelet_methods.slepian_wavelet_inverse(
        wav_coeffs,
        slepian_wavelets_polar_cap.wavelets,
        slepian_wavelets_polar_cap.slepian.N,
    )
    np.testing.assert_allclose(np.abs(f_p - coefficients).mean(), 0, atol=1e-14)


def test_synthesis_lim_lat_lon(
    slepian_wavelets_lim_lat_lon: sleplet.functions.slepian_wavelets.SlepianWavelets,
    earth_lim_lat_lon: sleplet.functions.earth.Earth,
) -> None:
    """Test that Slepian lim_lat_lon wavelet synthesis matches the coefficients."""
    coefficients = sleplet.slepian_methods.slepian_forward(
        slepian_wavelets_lim_lat_lon.L,
        slepian_wavelets_lim_lat_lon.slepian,
        flm=earth_lim_lat_lon.coefficients,
    )
    wav_coeffs = sleplet.wavelet_methods.slepian_wavelet_forward(
        coefficients,
        slepian_wavelets_lim_lat_lon.wavelets,
        slepian_wavelets_lim_lat_lon.slepian.N,
    )
    f_p = sleplet.wavelet_methods.slepian_wavelet_inverse(
        wav_coeffs,
        slepian_wavelets_lim_lat_lon.wavelets,
        slepian_wavelets_lim_lat_lon.slepian.N,
    )
    np.testing.assert_allclose(np.abs(f_p - coefficients).mean(), 0, atol=0)


def test_axisymmetric_synthesis_earth() -> None:
    """Test that the axisymmetric wavelet synthesis recovers the coefficients."""
    awc = sleplet.functions.AxisymmetricWaveletCoefficientsEarth(
        L_SMALL,
        B=B,
        j_min=J_MIN,
    )
    flm = sleplet.wavelet_methods.axisymmetric_wavelet_inverse(
        L_SMALL,
        awc.wavelet_coefficients,
        awc.wavelets,
    )
    np.testing.assert_allclose(
        np.abs(flm - awc._earth.coefficients).mean(),
        0,
        atol=1e-13,
    )


def test_axisymmetric_synthesis_south_america() -> None:
    """Test that the axisymmetric wavelet synthesis recovers the coefficients."""
    awc = sleplet.functions.AxisymmetricWaveletCoefficientsSouthAmerica(
        L_SMALL,
        B=B,
        j_min=J_MIN,
    )
    flm = sleplet.wavelet_methods.axisymmetric_wavelet_inverse(
        L_SMALL,
        awc.wavelet_coefficients,
        awc.wavelets,
    )
    np.testing.assert_allclose(
        np.abs(flm - awc._south_america.coefficients).mean(),
        0,
        atol=1e-14,
    )


def test_only_wavelet_coefficients_within_shannon_returned() -> None:
    """Verify that only the non-zero wavelet coefficients are returned."""
    coeffs_in = np.array([[3], [2], [1], [0]])
    coeffs_out = np.array([[3], [2], [1]])
    shannon_coeffs = sleplet.wavelet_methods.find_non_zero_wavelet_coefficients(
        coeffs_in,
        axis=1,
    )
    np.testing.assert_array_equal(shannon_coeffs, coeffs_out)


def test_create_kappas() -> None:
    """Check that the method creates the scaling function and wavelets."""
    wavelets = sleplet.wavelet_methods.create_kappas(L_LARGE**2, B, J_MIN)
    j_max = pys2let.pys2let_j_max(B, L_LARGE**2, J_MIN)
    np.testing.assert_equal(j_max - J_MIN + 2, wavelets.shape[0])
    np.testing.assert_equal(L_LARGE**2, wavelets.shape[1])
