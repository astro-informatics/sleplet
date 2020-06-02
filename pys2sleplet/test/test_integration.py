import numpy as np
import pytest
from hypothesis import given, seed, settings
from hypothesis.strategies import SearchStrategy, integers
from numpy.testing import assert_allclose

from pys2sleplet.slepian.slepian_region.slepian_limit_lat_lon import SlepianLimitLatLon
from pys2sleplet.slepian.slepian_region.slepian_polar_cap import SlepianPolarCap
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
from pys2sleplet.utils.integration_methods import integrate_whole_sphere


@pytest.fixture(scope="module")
def polar_cap_evecs() -> np.ndarray:
    """
    retrieve the eigenvectors of a Slepian polar cap
    """
    slepian = SlepianPolarCap(L, THETA_MAX, order=ORDER)
    return slepian.eigenvectors


@pytest.fixture(scope="module")
def lim_lat_lon_evecs() -> np.ndarray:
    """
    retrieve the eigenvectors of a Slepian limited latitude longitude region
    """
    slepian = SlepianLimitLatLon(
        L, theta_min=THETA_0, theta_max=THETA_1, phi_min=PHI_0, phi_max=PHI_1
    )
    return slepian.eigenvectors


def valid_polar_ranks() -> SearchStrategy[int]:
    """
    must be less than L-m
    """
    return integers(min_value=0, max_value=(L - ORDER) - 1)


def valid_lim_lat_lon_ranks() -> SearchStrategy[int]:
    """
    must be less than L
    """
    return integers(min_value=0, max_value=L - 1)


@seed(RANDOM_SEED)
@settings(max_examples=8, deadline=None)
@given(rank1=valid_polar_ranks(), rank2=valid_polar_ranks())
def test_integrate_two_slepian_polar_functions_whole_sphere(
    polar_cap_evecs, rank1, rank2
) -> None:
    """
    tests that integration two slepian functions over the
    whole sphere gives the Kronecker delta
    """
    flm = polar_cap_evecs[rank1]
    glm = polar_cap_evecs[rank2]
    output = integrate_whole_sphere(L, flm, glm, glm_conj=True)
    if rank1 == rank2:
        assert_allclose(output, 1, rtol=1e-3)
    else:
        assert_allclose(output, 0, atol=1e-4)


@seed(RANDOM_SEED)
@settings(max_examples=8, deadline=None)
@given(rank1=valid_lim_lat_lon_ranks(), rank2=valid_lim_lat_lon_ranks())
def test_integrate_two_slepian_lim_lat_lon_functions_whole_sphere(
    lim_lat_lon_evecs, rank1, rank2
) -> None:
    """
    tests that integration two slepian functions over the
    whole sphere gives the Kronecker delta
    """
    flm = lim_lat_lon_evecs[rank1]
    glm = lim_lat_lon_evecs[rank2]
    output = integrate_whole_sphere(L, flm, glm, glm_conj=True)
    if rank1 == rank2:
        assert_allclose(output, 1, rtol=1e-5)
    else:
        assert_allclose(output, 0, atol=1e-5)
