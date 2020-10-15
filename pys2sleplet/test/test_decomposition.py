import numpy as np
import pyssht as ssht
from numpy.testing import assert_allclose, assert_raises

from pys2sleplet.slepian.slepian_decomposition import SlepianDecomposition
from pys2sleplet.test.constants import L_SMALL as L
from pys2sleplet.utils.mask_methods import create_mask_region
from pys2sleplet.utils.slepian_methods import slepian_forward, slepian_inverse


def test_decompose_all_polar(slepian_polar_cap, earth_polar_cap) -> None:
    """
    tests that all three methods produce the same coefficients for polar cap
    """
    f_p = slepian_forward(
        L, earth_polar_cap.coefficients, slepian_polar_cap, method="integrate_region"
    )
    g_p = slepian_forward(
        L, earth_polar_cap.coefficients, slepian_polar_cap, method="integrate_sphere"
    )
    h_p = slepian_forward(
        L, earth_polar_cap.coefficients, slepian_polar_cap, method="harmonic_sum"
    )
    assert_allclose(np.abs(f_p - h_p)[: slepian_polar_cap.N].mean(), 0, atol=1.1)
    assert_allclose(np.abs(g_p - h_p)[: slepian_polar_cap.N].mean(), 0, atol=0.8)


def test_decompose_all_lim_lat_lon(slepian_lim_lat_lon, earth_lim_lat_lon) -> None:
    """
    tests that all three methods produce the same coefficients for
    limited latitude longitude region
    """
    f_p = slepian_forward(
        L,
        earth_lim_lat_lon.coefficients,
        slepian_lim_lat_lon,
        method="integrate_region",
    )
    g_p = slepian_forward(
        L,
        earth_lim_lat_lon.coefficients,
        slepian_lim_lat_lon,
        method="integrate_sphere",
    )
    h_p = slepian_forward(
        L, earth_lim_lat_lon.coefficients, slepian_lim_lat_lon, method="harmonic_sum"
    )
    assert_allclose(np.abs(f_p - h_p)[: slepian_lim_lat_lon.N].mean(), 0, atol=36)
    assert_allclose(np.abs(g_p - h_p)[: slepian_lim_lat_lon.N].mean(), 0, atol=1e-2)


def test_equality_to_harmonic_transform_polar(
    slepian_polar_cap, earth_polar_cap
) -> None:
    """
    tests that fp*Sp up to N is roughly equal to flm*Ylm
    """
    f_p = slepian_forward(L, earth_polar_cap.coefficients, slepian_polar_cap)
    f_slepian = slepian_inverse(L, f_p, slepian_polar_cap)
    f_harmonic = ssht.inverse(earth_polar_cap.coefficients, L)
    mask = create_mask_region(L, slepian_polar_cap.region)
    assert_allclose(np.abs(f_slepian - f_harmonic)[mask].mean(), 0, atol=15)


def test_equality_to_harmonic_transform_lim_lat_lon(
    slepian_lim_lat_lon, earth_lim_lat_lon
) -> None:
    """
    tests that fp*Sp up to N is roughly equal to flm*Ylm
    """
    f_p = slepian_forward(L, earth_lim_lat_lon.coefficients, slepian_lim_lat_lon)
    f_slepian = slepian_inverse(L, f_p, slepian_lim_lat_lon)
    f_harmonic = ssht.inverse(earth_lim_lat_lon.coefficients, L)
    mask = create_mask_region(L, slepian_lim_lat_lon.region)
    assert_allclose(np.abs(f_slepian - f_harmonic)[mask].mean(), 0, atol=95)


def test_pass_rank_higher_than_available(slepian_polar_cap, earth_polar_cap) -> None:
    """
    tests that asking for a Slepian coefficients above the limit fails
    """
    sd = SlepianDecomposition(L, earth_polar_cap.coefficients, slepian_polar_cap)
    assert_raises(ValueError, sd.decompose, L ** 2)
