import numpy as np
import pyssht as ssht
import pytest
from numpy.testing import assert_allclose

from pys2sleplet.flm.kernels.slepian_wavelets import SlepianWavelets
from pys2sleplet.test.constants import L_SMALL as L
from pys2sleplet.utils.mask_methods import create_mask_region
from pys2sleplet.utils.vars import SAMPLING_SCHEME
from pys2sleplet.utils.wavelet_methods import wavelet_inverse


@pytest.mark.slow
def test_synthesis_polar(polar_cap_decomposition) -> None:
    """
    tests that Slepian polar wavelet synthesis matches the real space
    """
    scaling = SlepianWavelets(L, region=polar_cap_decomposition.function.region)
    flm = wavelet_inverse(polar_cap_decomposition.function, scaling.multipole)
    for j in range(scaling.j_max - scaling.j_min):
        wavelet = SlepianWavelets(
            L, j=j, region=polar_cap_decomposition.function.region
        )
        flm += wavelet_inverse(polar_cap_decomposition.function, wavelet.multipole)
    f_wavelets = ssht.inverse(flm, L, Method=SAMPLING_SCHEME)
    f_harmonic = ssht.inverse(polar_cap_decomposition.flm, L, Method=SAMPLING_SCHEME)
    mask = create_mask_region(L, polar_cap_decomposition.function.region)
    assert_allclose(np.abs(f_wavelets - f_harmonic)[mask].mean(), 0, atol=343)


# @pytest.mark.slow
def test_synthesis_lim_lat_lon(lim_lat_lon_decomposition) -> None:
    """
    tests that Slepian lim_lat_lon wavelet synthesis matches the real space
    """
    scaling = SlepianWavelets(L, region=lim_lat_lon_decomposition.function.region)
    flm = wavelet_inverse(lim_lat_lon_decomposition.function, scaling.multipole)
    for j in range(scaling.j_max - scaling.j_min):
        wavelet = SlepianWavelets(
            L, j=j, region=lim_lat_lon_decomposition.function.region
        )
        flm += wavelet_inverse(lim_lat_lon_decomposition.function, wavelet.multipole)
    f_wavelets = ssht.inverse(flm, L, Method=SAMPLING_SCHEME)
    f_harmonic = ssht.inverse(lim_lat_lon_decomposition.flm, L, Method=SAMPLING_SCHEME)
    mask = create_mask_region(L, lim_lat_lon_decomposition.function.region)
    assert_allclose(np.abs(f_wavelets - f_harmonic)[mask].mean(), 0, atol=936)
