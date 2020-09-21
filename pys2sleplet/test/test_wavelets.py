import numpy as np
import pyssht as ssht
import pytest
from numpy.testing import assert_allclose

from pys2sleplet.flm.kernels.slepian_wavelets import SlepianWavelets
from pys2sleplet.test.constants import L_SMALL as L
from pys2sleplet.utils.mask_methods import create_mask_region
from pys2sleplet.utils.wavelet_methods import (
    slepian_wavelet_forward,
    slepian_wavelet_inverse,
)


@pytest.mark.slow
def test_synthesis_polar(polar_cap_decomposition) -> None:
    """
    tests that Slepian polar wavelet synthesis matches the real space
    """
    sw = SlepianWavelets(L, region=polar_cap_decomposition.function.region)
    wav_coeffs = slepian_wavelet_forward(
        polar_cap_decomposition.function.multipole, sw.wavelets
    )
    flm = slepian_wavelet_inverse(wav_coeffs, sw.wavelets)
    f_wavelets = ssht.inverse(flm, L)
    f_harmonic = ssht.inverse(polar_cap_decomposition.flm, L)
    mask = create_mask_region(L, polar_cap_decomposition.function.region)
    assert_allclose(np.abs(f_wavelets - f_harmonic)[mask].mean(), 0, atol=271)


def test_synthesis_lim_lat_lon(lim_lat_lon_decomposition) -> None:
    """
    tests that Slepian lim_lat_lon wavelet synthesis matches the real space
    """
    sw = SlepianWavelets(L, region=lim_lat_lon_decomposition.function.region)
    wav_coeffs = slepian_wavelet_forward(
        lim_lat_lon_decomposition.function.multipole, sw.wavelets
    )
    flm = slepian_wavelet_inverse(wav_coeffs, sw.wavelets)
    f_wavelets = ssht.inverse(flm, L)
    f_harmonic = ssht.inverse(lim_lat_lon_decomposition.flm, L)
    mask = create_mask_region(L, lim_lat_lon_decomposition.function.region)
    assert_allclose(np.abs(f_wavelets - f_harmonic)[mask].mean(), 0, atol=1e3)
