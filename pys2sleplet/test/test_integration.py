import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis.strategies import SearchStrategy, integers
from numpy.testing import assert_allclose

from pys2sleplet.slepian.slepian_region.slepian_polar_cap import SlepianPolarCap
from pys2sleplet.utils.integration_methods import integrate_whole_sphere


@pytest.fixture(scope="module")
def L() -> int:
    """
    needs to be small for speed
    """
    return 8


@pytest.fixture(scope="module")
def theta_max() -> int:
    """
    theta can be in the range [0, 180]
    """
    return 40


@pytest.fixture(scope="module")
def order() -> int:
    """
    the order of the Dm matrix, needs to be less than L
    """
    return 0


@pytest.fixture(scope="module")
def polar_cap_evecs(L, theta_max, order):
    """
    retrieve the eigenvectors of a sample Slepian polar cap
    """
    slepian = SlepianPolarCap(L, np.deg2rad(theta_max), order=order)
    return slepian.eigenvectors


def valid_ranks() -> SearchStrategy[int]:
    """
    must be less than L-m
    """
    return integers(min_value=0, max_value=7)


@settings(max_examples=8, derandomize=True, deadline=None)
@given(rank1=valid_ranks(), rank2=valid_ranks())
def test_integrate_two_slepian_functions_whole_sphere(
    L, polar_cap_evecs, rank1, rank2
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
        assert_allclose(output, 0, atol=1e-3)
