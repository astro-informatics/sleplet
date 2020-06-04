import pytest
from hypothesis import given, seed, settings
from hypothesis.strategies import SearchStrategy, integers
from numpy.testing import assert_allclose, assert_raises

from pys2sleplet.flm.maps.earth import Earth
from pys2sleplet.slepian.slepian_decomposition import SlepianDecomposition
from pys2sleplet.test.constants import L_SMALL as L
from pys2sleplet.test.constants import (
    ORDER,
    PHI_0,
    PHI_1,
    RANDOM_SEED,
    THETA_0,
    THETA_1,
    THETA_MAX,
)
from pys2sleplet.utils.region import Region


@pytest.fixture(scope="module")
def polar_cap_decomposition() -> SlepianDecomposition:
    region = Region(theta_max=THETA_MAX, order=ORDER)
    earth = Earth(L, region=region)
    return SlepianDecomposition(earth)


def valid_polar_ranks() -> SearchStrategy[int]:
    """
    must be less than L-m
    """
    return integers(min_value=0, max_value=(L - ORDER) - 1)


@pytest.fixture(scope="module")
def lim_lat_lon_decomposition() -> SlepianDecomposition:
    region = Region(theta_min=THETA_0, theta_max=THETA_1, phi_min=PHI_0, phi_max=PHI_1)
    earth = Earth(L, region=region)
    return SlepianDecomposition(earth)


def valid_lim_lat_lon_ranks() -> SearchStrategy[int]:
    """
    must be less than L
    """
    return integers(min_value=0, max_value=L - 1)


@seed(RANDOM_SEED)
@settings(max_examples=8, deadline=None)
@given(rank=valid_polar_ranks())
def test_integrate_region_and_harmonic_sum_agree_polar(
    polar_cap_decomposition, rank
) -> None:
    """
    tests integrate_region and harmonic_sum are in agreement for a polar region
    """
    f_p = polar_cap_decomposition.decompose(rank, method="integrate_region")
    g_p = polar_cap_decomposition.decompose(rank, method="harmonic_sum")
    assert_allclose(f_p, g_p, rtol=1e8)


@seed(RANDOM_SEED)
@settings(max_examples=8, deadline=None)
@given(rank=valid_lim_lat_lon_ranks())
def test_integrate_region_and_harmonic_sum_agree_lim_lat_lon(
    lim_lat_lon_decomposition, rank
) -> None:
    """
    tests integrate_region and harmonic_sum are in agreement for a
    limited latitude longitude region
    """
    f_p = lim_lat_lon_decomposition.decompose(rank, method="integrate_region")
    g_p = lim_lat_lon_decomposition.decompose(rank, method="harmonic_sum")
    assert_allclose(f_p, g_p, rtol=20.7)


@seed(RANDOM_SEED)
@settings(max_examples=8, deadline=None)
@given(rank=valid_polar_ranks())
def test_integrate_sphere_and_harmonic_sum_agree_polar(
    polar_cap_decomposition, rank
) -> None:
    """
    tests integrate_sphere and harmonic_sum are in agreement for a polar region
    """
    f_p = polar_cap_decomposition.decompose(rank, method="integrate_sphere")
    g_p = polar_cap_decomposition.decompose(rank, method="harmonic_sum")
    assert_allclose(f_p, g_p, rtol=1e-3)


@seed(RANDOM_SEED)
@settings(max_examples=8, deadline=None)
@given(rank=valid_lim_lat_lon_ranks())
def test_integrate_sphere_and_harmonic_sum_agree_lim_lat_lon(
    lim_lat_lon_decomposition, rank
) -> None:
    """
    tests integrate_sphere and harmonic_sum are in agreement for a
    limited latitude longitude region
    """
    f_p = lim_lat_lon_decomposition.decompose(rank, method="integrate_sphere")
    g_p = lim_lat_lon_decomposition.decompose(rank, method="harmonic_sum")
    assert_allclose(f_p, g_p, rtol=0.3)


@seed(RANDOM_SEED)
@settings(max_examples=8, deadline=None)
@given(rank=valid_polar_ranks())
def test_integrate_region_and_integrate_sphere_agree_polar(
    polar_cap_decomposition, rank
) -> None:
    """
    tests integrate_region and integrate_sphere are in agreement for a polar region
    """
    f_p = polar_cap_decomposition.decompose(rank, method="integrate_region")
    g_p = polar_cap_decomposition.decompose(rank, method="integrate_sphere")
    assert_allclose(f_p, g_p, rtol=1e8)


@seed(RANDOM_SEED)
@settings(max_examples=8, deadline=None)
@given(rank=valid_lim_lat_lon_ranks())
def test_integrate_region_and_integrate_sphere_agree_lim_lat_lon(
    lim_lat_lon_decomposition, rank
) -> None:
    """
    tests integrate_region and integrate_sphere are in agreement for
    a limited latitude longitude region
    """
    f_p = lim_lat_lon_decomposition.decompose(rank, method="integrate_region")
    g_p = lim_lat_lon_decomposition.decompose(rank, method="integrate_sphere")
    assert_allclose(f_p, g_p, rtol=21.5)


def test_pass_function_without_region() -> None:
    """
    tests that the class throws an exception if no Region is passed to the function
    """
    earth = Earth(L)
    assert_raises(AttributeError, SlepianDecomposition, earth)
