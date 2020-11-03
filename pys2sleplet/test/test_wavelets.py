import numpy as np
import pytest
from numpy.testing import assert_allclose

from pys2sleplet.functions.flm.axisymmetric_wavelet_coefficients_earth import (
    AxisymmetricWaveletCoefficientsEarth,
)
from pys2sleplet.functions.fp.slepian_wavelets import SlepianWavelets
from pys2sleplet.test.constants import L_SMALL as L
from pys2sleplet.utils.slepian_methods import slepian_forward
from pys2sleplet.utils.wavelet_methods import (
    axisymmetric_wavelet_inverse,
    slepian_wavelet_forward,
    slepian_wavelet_inverse,
)


@pytest.mark.slow
def test_synthesis_polar(slepian_polar_cap, earth_polar_cap) -> None:
    """
    tests that Slepian polar wavelet synthesis matches the coefficients
    """
    sw = SlepianWavelets(L, region=slepian_polar_cap.region)
    coefficients = slepian_forward(L, earth_polar_cap.coefficients, slepian_polar_cap)
    wav_coeffs = slepian_wavelet_forward(coefficients, sw.wavelets, slepian_polar_cap.N)
    f_p = slepian_wavelet_inverse(wav_coeffs, sw.wavelets, slepian_polar_cap.N)
    assert_allclose(np.abs(f_p - coefficients).mean(), 0, atol=1e-14)


def test_synthesis_lim_lat_lon(slepian_lim_lat_lon, earth_lim_lat_lon) -> None:
    """
    tests that Slepian lim_lat_lon wavelet synthesis matches the coefficients
    """
    sw = SlepianWavelets(L, region=slepian_lim_lat_lon.region)
    coefficients = slepian_forward(
        L, earth_lim_lat_lon.coefficients, slepian_lim_lat_lon
    )
    wav_coeffs = slepian_wavelet_forward(
        coefficients, sw.wavelets, slepian_lim_lat_lon.N
    )
    f_p = slepian_wavelet_inverse(wav_coeffs, sw.wavelets, slepian_lim_lat_lon.N)
    assert_allclose(np.abs(f_p - coefficients).mean(), 0)


def test_axisymmetric_synthesis() -> None:
    """
    tests that the axisymmetric wavelet synthesis recoveres the coefficients
    """
    awc = AxisymmetricWaveletCoefficientsEarth(L)
    flm = axisymmetric_wavelet_inverse(L, awc.wavelet_coefficients, awc.wavelets)
    assert_allclose(np.abs(flm - awc.earth.coefficients).mean(), 0, atol=1e-13)
