import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis.strategies import SearchStrategy, integers
from numpy.testing import assert_allclose

from pys2sleplet.slepian.slepian_region.slepian_polar_cap import SlepianPolarCap
from pys2sleplet.test.constants import L_SMALL as L
from pys2sleplet.test.constants import ORDER, THETA_MAX
from pys2sleplet.utils.integration_methods import integrate_whole_sphere


@pytest.fixture(scope="module")
def polar_cap_evecs():
    """
    retrieve the eigenvectors of a sample Slepian polar cap
    """
    slepian = SlepianPolarCap(L, np.deg2rad(THETA_MAX), order=ORDER)
    return slepian.eigenvectors


def valid_ranks() -> SearchStrategy[int]:
    """
    must be less than L-m
    """
    return integers(min_value=0, max_value=(L - ORDER) - 1)


@settings(max_examples=8, derandomize=True, deadline=None)
@given(rank1=valid_ranks(), rank2=valid_ranks())
def test_integrate_two_slepian_functions_whole_sphere(
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
        assert_allclose(output, 0, atol=1e-3)
