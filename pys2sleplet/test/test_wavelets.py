import numpy as np
import pyssht as ssht
import pytest
from numpy.testing import assert_allclose

from pys2sleplet.functions.fp.slepian_wavelets import SlepianWavelets
from pys2sleplet.test.constants import L_SMALL as L
from pys2sleplet.utils.mask_methods import create_mask_region
from pys2sleplet.utils.slepian_methods import slepian_forward, slepian_inverse
from pys2sleplet.utils.wavelet_methods import (
    slepian_wavelet_forward,
    slepian_wavelet_inverse,
)


@pytest.mark.slow
def test_synthesis_polar(slepian_polar_cap, earth_polar_cap) -> None:
    """
    tests that Slepian polar wavelet synthesis matches the real space
    """
    sw = SlepianWavelets(L, region=slepian_polar_cap.region)
    coefficients = slepian_forward(L, earth_polar_cap.coefficients, slepian_polar_cap)
    wav_coeffs = slepian_wavelet_forward(coefficients, sw.wavelets)
    f_p = slepian_wavelet_inverse(wav_coeffs, sw.wavelets)
    f_wavelets = slepian_inverse(L, f_p, slepian_polar_cap)
    f_harmonic = ssht.inverse(earth_polar_cap.coefficients, L)
    mask = create_mask_region(L, slepian_polar_cap.region)
    assert_allclose(np.abs(f_wavelets - f_harmonic)[mask].mean(), 0, atol=271)


def test_synthesis_lim_lat_lon(slepian_lim_lat_lon, earth_lim_lat_lon) -> None:
    """
    tests that Slepian lim_lat_lon wavelet synthesis matches the real space
    """
    sw = SlepianWavelets(L, region=slepian_lim_lat_lon.region)
    coefficients = slepian_forward(
        L, earth_lim_lat_lon.coefficients, slepian_lim_lat_lon
    )
    wav_coeffs = slepian_wavelet_forward(coefficients, sw.wavelets)
    f_p = slepian_wavelet_inverse(wav_coeffs, sw.wavelets)
    f_wavelets = slepian_inverse(L, f_p, slepian_lim_lat_lon)
    f_harmonic = ssht.inverse(earth_lim_lat_lon.coefficients, L)
    mask = create_mask_region(L, slepian_lim_lat_lon.region)
    assert_allclose(np.abs(f_wavelets - f_harmonic)[mask].mean(), 0, atol=1e3)
