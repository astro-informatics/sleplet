import numpy as np
from numpy.testing import assert_allclose, assert_raises

from pys2sleplet.flm.maps.earth import Earth
from pys2sleplet.slepian.slepian_decomposition import SlepianDecomposition
from pys2sleplet.test.constants import L_SMALL as L


def test_decompose_all_polar(polar_cap_decomposition) -> None:
    """
    tests that all three methods produce the same coefficients for polar cap
    """
    f_p = polar_cap_decomposition.decompose_all(method="integrate_region")
    g_p = polar_cap_decomposition.decompose_all(method="integrate_sphere")
    h_p = polar_cap_decomposition.decompose_all(method="harmonic_sum")
    assert_allclose(
        np.abs(f_p - h_p)[: polar_cap_decomposition.shannon].mean(), 0, atol=7
    )
    assert_allclose(
        np.abs(g_p - h_p)[: polar_cap_decomposition.shannon].mean(), 0, atol=0.04
    )


def test_decompose_all_lim_lat_lon(lim_lat_lon_decomposition) -> None:
    """
    tests that all three methods produce the same coefficients for
    limited latitude longitude region
    """
    f_p = lim_lat_lon_decomposition.decompose_all(method="integrate_region")
    g_p = lim_lat_lon_decomposition.decompose_all(method="integrate_sphere")
    h_p = lim_lat_lon_decomposition.decompose_all(method="harmonic_sum")
    assert_allclose(
        np.abs(f_p - h_p)[: lim_lat_lon_decomposition.shannon].mean(), 0, atol=18
    )
    assert_allclose(
        np.abs(g_p - h_p)[: lim_lat_lon_decomposition.shannon].mean(), 0, atol=1e-2
    )


def test_pass_function_without_region() -> None:
    """
    tests that the class throws an exception if no Region is passed to the function
    """
    earth = Earth(L)
    assert_raises(AttributeError, SlepianDecomposition, earth)


def test_pass_rank_higher_than_available(polar_cap_decomposition) -> None:
    """
    tests that asking for a Slepian coefficient above the limit fails
    """
    assert_raises(ValueError, polar_cap_decomposition.decompose, L * L)
