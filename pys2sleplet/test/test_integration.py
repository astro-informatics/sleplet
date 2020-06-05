from typing import Optional

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
from pys2sleplet.utils.array_methods import fill_upper_triangle_of_hermitian_matrix
from pys2sleplet.utils.integration_methods import integrate_sphere
from pys2sleplet.utils.region import Region


@pytest.fixture(scope="module")
def slepian_polar_cap() -> SlepianPolarCap:
    """
    retrieve the eigenvectors of a Slepian polar cap
    """
    return SlepianPolarCap(L, THETA_MAX, order=ORDER)


@pytest.fixture(scope="module")
def polar_cap_region() -> Region:
    """
    creates a polar cap region
    """
    return Region(theta_max=THETA_1)


def valid_polar_ranks() -> SearchStrategy[int]:
    """
    must be less than L-m
    """
    return integers(min_value=0, max_value=(L - ORDER) - 1)


@pytest.fixture(scope="module")
def slepian_lim_lat_lon() -> SlepianLimitLatLon:
    """
    retrieve the eigenvectors of a Slepian limited latitude longitude region
    """
    return SlepianLimitLatLon(
        L, theta_min=THETA_0, theta_max=THETA_1, phi_min=PHI_0, phi_max=PHI_1
    )


@pytest.fixture(scope="module")
def lim_lat_lon_region() -> Region:
    """
    creates a limited latitude longitude region
    """
    return Region(theta_min=THETA_0, theta_max=THETA_1, phi_min=PHI_0, phi_max=PHI_1)


def valid_lim_lat_lon_ranks() -> SearchStrategy[int]:
    """
    must be less than L
    """
    return integers(min_value=0, max_value=L - 1)


@seed(RANDOM_SEED)
@settings(max_examples=8, deadline=None)
@given(rank1=valid_polar_ranks(), rank2=valid_polar_ranks())
def test_integrate_two_slepian_polar_functions_whole_sphere_per_rank(
    slepian_polar_cap, rank1, rank2
) -> None:
    """
    tests that integration of two slepian poalr functions over the
    whole sphere gives the Kronecker delta
    """
    output = _integrate_two_functions_per_rank_helper(
        slepian_polar_cap.eigenvectors, rank1, rank2
    )
    if rank1 == rank2:
        assert_allclose(output, 1, atol=1e-3)
    else:
        assert_allclose(output, 0, atol=1e-4)


@seed(RANDOM_SEED)
@settings(max_examples=8, deadline=None)
@given(rank1=valid_lim_lat_lon_ranks(), rank2=valid_lim_lat_lon_ranks())
def test_integrate_two_slepian_lim_lat_lon_functions_whole_sphere_per_rank(
    slepian_lim_lat_lon, rank1, rank2
) -> None:
    """
    tests that integration of two slepian lim lat lon functions over the
    whole sphere gives the Kronecker delta
    """
    output = _integrate_two_functions_per_rank_helper(
        slepian_lim_lat_lon.eigenvectors, rank1, rank2
    )
    if rank1 == rank2:
        assert_allclose(output, 1, atol=1e-5)
    else:
        assert_allclose(output, 0, atol=1e-5)


@seed(RANDOM_SEED)
@settings(max_examples=8, deadline=None)
@given(rank1=valid_polar_ranks(), rank2=valid_polar_ranks())
def test_integrate_two_slepian_polar_functions_region_sphere_per_rank(
    slepian_polar_cap, polar_cap_region, rank1, rank2
) -> None:
    """
    tests that integration of two slepian polar functions over a region on
    the sphere gives the Kronecker delta multiplied by the eigenvalue
    """
    output = _integrate_two_functions_per_rank_helper(
        slepian_polar_cap.eigenvectors, rank1, rank2, region=polar_cap_region
    )
    lambda_p = slepian_polar_cap.eigenvalues[rank1]
    if rank1 == rank2:
        assert_allclose(output, lambda_p, atol=1e-3)
    else:
        assert_allclose(output, 0, atol=0.2)


@seed(RANDOM_SEED)
@settings(max_examples=8, deadline=None)
@given(rank1=valid_lim_lat_lon_ranks(), rank2=valid_lim_lat_lon_ranks())
def test_integrate_two_slepian_lim_lat_lon_functions_region_sphere_per_rank(
    slepian_lim_lat_lon, lim_lat_lon_region, rank1, rank2
) -> None:
    """
    tests that integration of two slepian lim lat lon functions over a region on
    the sphere gives the Kronecker delta multiplied by the eigenvalue
    """
    output = _integrate_two_functions_per_rank_helper(
        slepian_lim_lat_lon.eigenvectors, rank1, rank2, region=lim_lat_lon_region
    )
    lambda_p = slepian_lim_lat_lon.eigenvalues[rank1]
    if rank1 == rank2:
        assert_allclose(output, lambda_p, atol=0.4)
    else:
        assert_allclose(output, 0, atol=0.7)


@pytest.mark.slow
def test_integrate_two_slepian_polar_functions_whole_sphere_matrix(
    slepian_polar_cap,
) -> None:
    """
    tests that integration of two slepian polar functions over the
    whole sphere gives the identity matrix
    """
    output = _integrate_whole_matrix_helper(slepian_polar_cap.eigenvectors)
    desired = np.identity(output.shape[0])
    test = np.abs(output - desired).mean()
    assert_allclose(test, 0, atol=1e-4)


@pytest.mark.slow
def test_integrate_two_slepian_lim_lat_lon_functions_whole_sphere_matrix(
    slepian_lim_lat_lon,
) -> None:
    """
    tests that integration of two slepian lim lat lon functions over the
    whole sphere gives the identity matrix
    """
    output = _integrate_whole_matrix_helper(slepian_lim_lat_lon.eigenvectors)
    desired = np.identity(output.shape[0])
    test = np.abs(output - desired).mean()
    assert_allclose(test, 0, atol=1e-6)


@pytest.mark.slow
def test_integrate_two_slepian_polar_functions_region_sphere_matrix(
    slepian_polar_cap, polar_cap_region
) -> None:
    """
    tests that integration of two slepian polar functions over a region on
    the sphere gives the identity matrix multiplied by the eigenvalue
    """
    output = _integrate_whole_matrix_helper(
        slepian_polar_cap.eigenvectors, region=polar_cap_region
    )
    desired = slepian_polar_cap.eigenvalues * np.identity(output.shape[0])
    test = np.abs(output - desired).mean()
    assert_allclose(test, 0, atol=0.02)


@pytest.mark.slow
def test_integrate_two_slepian_lim_lat_lon_functions_region_sphere_matrix(
    slepian_lim_lat_lon, lim_lat_lon_region
) -> None:
    """
    tests that integration of two slepian lim lat lon functions over a region on
    the sphere gives the identity matrix multiplied by the eigenvalue
    """
    output = _integrate_whole_matrix_helper(
        slepian_lim_lat_lon.eigenvectors, region=lim_lat_lon_region
    )
    desired = slepian_lim_lat_lon.eigenvalues * np.identity(output.shape[0])
    test = np.abs(output - desired).mean()
    assert_allclose(test, 0, atol=1e-5)


def _integrate_two_functions_per_rank_helper(
    eigenvectors: np.ndarray, rank1: int, rank2: int, region: Optional[Region] = None
) -> complex:
    """
    helper function which integrates two slepian functions of given ranks
    """
    flm = eigenvectors[rank1]
    glm = eigenvectors[rank2]
    output = integrate_sphere(L, flm, glm, region=region, glm_conj=True)
    return output


def _integrate_whole_matrix_helper(
    eigenvectors: np.ndarray, region: Optional[Region] = None
) -> np.ndarray:
    """
    helper function which integrates all of the slepian functionss
    """
    N = len(eigenvectors)
    output = np.zeros((N, N), dtype=complex)
    for i, flm in enumerate(eigenvectors):
        for j, glm in enumerate(eigenvectors):
            # Hermitian matrix so can use symmetry
            if i <= j:
                output[i][j] = integrate_sphere(
                    L, flm, glm, region=region, glm_conj=True
                )
    fill_upper_triangle_of_hermitian_matrix(output)
    return output
