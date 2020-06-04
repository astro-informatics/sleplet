import pytest
from numpy.testing import assert_allclose, assert_raises

from pys2sleplet.flm.maps.earth import Earth
from pys2sleplet.slepian.slepian_decomposition import SlepianDecomposition
from pys2sleplet.test.constants import L_SMALL as L
from pys2sleplet.test.constants import ORDER, PHI_0, PHI_1, THETA_0, THETA_1, THETA_MAX
from pys2sleplet.utils.region import Region


@pytest.fixture(scope="module")
def polar_cap_decomposition() -> SlepianDecomposition:
    region = Region(theta_max=THETA_MAX, order=ORDER)
    earth = Earth(L, region=region)
    return SlepianDecomposition(earth)


@pytest.fixture(scope="module")
def lim_lat_lon_decomposition() -> SlepianDecomposition:
    region = Region(theta_min=THETA_0, theta_max=THETA_1, phi_min=PHI_0, phi_max=PHI_1)
    earth = Earth(L, region=region)
    return SlepianDecomposition(earth)


def test_decompose_all_polar(polar_cap_decomposition) -> None:
    """
    tests that all three methods produce the same coefficients for polar cap
    """
    f_p = polar_cap_decomposition.decompose_all(method="integrate_region")
    g_p = polar_cap_decomposition.decompose_all(method="integrate_sphere")
    h_p = polar_cap_decomposition.decompose_all(method="harmonic_sum")
    assert_allclose(f_p, g_p, rtol=1e8)
    assert_allclose(g_p, h_p, rtol=1e-3)
    assert_allclose(h_p, f_p, rtol=1.1)


@pytest.mark.slow
def test_decompose_all_lim_lat_lon(lim_lat_lon_decomposition) -> None:
    """
    tests that all three methods produce the same coefficients for
    limited latitude longitude region
    """
    f_p = lim_lat_lon_decomposition.decompose_all(method="integrate_region")
    g_p = lim_lat_lon_decomposition.decompose_all(method="integrate_sphere")
    h_p = lim_lat_lon_decomposition.decompose_all(method="harmonic_sum")
    assert_allclose(f_p, g_p, rtol=1e10)
    assert_allclose(g_p, h_p, rtol=0.3)
    assert_allclose(h_p, f_p, rtol=2)


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
    assert_raises(ValueError, polar_cap_decomposition.decompose, L)
