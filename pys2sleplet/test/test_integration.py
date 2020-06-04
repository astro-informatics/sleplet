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
    tests that integration two slepian functions over the
    whole sphere gives the Kronecker delta
    """
    flm = slepian_polar_cap.eigenvectors[rank1]
    glm = slepian_polar_cap.eigenvectors[rank2]
    output = integrate_sphere(L, flm, glm, glm_conj=True)
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
    tests that integration two slepian functions over the
    whole sphere gives the Kronecker delta
    """
    flm = slepian_lim_lat_lon.eigenvectors[rank1]
    glm = slepian_lim_lat_lon.eigenvectors[rank2]
    output = integrate_sphere(L, flm, glm, glm_conj=True)
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
    tests that integration two slepian functions over a region on
    the sphere gives the Kronecker delta multiplied by the eigenvalue
    """
    lambda_p = slepian_polar_cap.eigenvalues[rank1]
    flm = slepian_polar_cap.eigenvectors[rank1]
    glm = slepian_polar_cap.eigenvectors[rank2]
    output = integrate_sphere(L, flm, glm, region=polar_cap_region, glm_conj=True)
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
    tests that integration two slepian functions over a region on
    the sphere gives the Kronecker delta multiplied by the eigenvalue
    """
    lambda_p = slepian_lim_lat_lon.eigenvalues[rank1]
    flm = slepian_lim_lat_lon.eigenvectors[rank1]
    glm = slepian_lim_lat_lon.eigenvectors[rank2]
    output = integrate_sphere(L, flm, glm, region=lim_lat_lon_region, glm_conj=True)
    if rank1 == rank2:
        assert_allclose(output, lambda_p, atol=0.4)
    else:
        assert_allclose(output, 0, atol=0.7)


@pytest.mark.slow
def test_integrate_two_slepian_polar_functions_whole_sphere_matrix(
    slepian_polar_cap,
) -> None:
    """
    tests that integration two slepian functions over the
    whole sphere gives the identity matrix
    """
    evecs = slepian_polar_cap.eigenvectors
    N = len(evecs)
    result = np.zeros((N, N), dtype=complex)
    for i, flm in enumerate(evecs):
        for j, glm in enumerate(evecs):
            if i <= j:
                result[i][j] = integrate_sphere(L, flm, glm, glm_conj=True)
    i_upper = np.triu_indices(result.shape[0])
    result[i_upper] = result.T[i_upper]
    desired = np.identity(N)
    output = np.abs(result - desired).mean()
    assert_allclose(output, 0, atol=1e-4)


@pytest.mark.slow
def test_integrate_two_slepian_lim_lat_lon_functions_whole_sphere_matrix(
    slepian_lim_lat_lon,
) -> None:
    """
    tests that integration two slepian functions over the
    whole sphere gives the identity matrix
    """
    evecs = slepian_lim_lat_lon.eigenvectors
    N = len(evecs)
    result = np.zeros((N, N), dtype=complex)
    for i, flm in enumerate(evecs):
        for j, glm in enumerate(evecs):
            if i <= j:
                result[i][j] = integrate_sphere(L, flm, glm, glm_conj=True)
    i_upper = np.triu_indices(result.shape[0])
    result[i_upper] = result.T[i_upper]
    desired = np.identity(N)
    output = np.abs(result - desired).mean()
    assert_allclose(output, 0, atol=1e-6)


@pytest.mark.slow
def test_integrate_two_slepian_polar_functions_region_sphere_matrix(
    slepian_polar_cap, polar_cap_region
) -> None:
    """
    tests that integration two slepian functions over the
    whole sphere gives the identity matrix
    """
    evecs = slepian_polar_cap.eigenvectors
    evals = slepian_polar_cap.eigenvalues
    N = len(evecs)
    result = np.zeros((N, N), dtype=complex)
    for i, flm in enumerate(evecs):
        for j, glm in enumerate(evecs):
            if i <= j:
                result[i][j] = integrate_sphere(
                    L, flm, glm, region=polar_cap_region, glm_conj=True
                )
    i_upper = np.triu_indices(result.shape[0])
    result[i_upper] = result.T[i_upper]
    desired = evals * np.identity(N)
    output = np.abs(result - desired).mean()
    assert_allclose(output, 0, atol=0.02)


@pytest.mark.slow
def test_integrate_two_slepian_lim_lat_lon_functions_region_sphere_matrix(
    slepian_lim_lat_lon, lim_lat_lon_region
) -> None:
    """
    tests that integration two slepian functions over the
    whole sphere gives the identity matrix
    """
    evecs = slepian_lim_lat_lon.eigenvectors
    evals = slepian_lim_lat_lon.eigenvalues
    N = len(evecs)
    result = np.zeros((N, N), dtype=complex)
    for i, flm in enumerate(evecs):
        for j, glm in enumerate(evecs):
            if i <= j:
                result[i][j] = integrate_sphere(
                    L, flm, glm, region=lim_lat_lon_region, glm_conj=True
                )
    i_upper = np.triu_indices(result.shape[0])
    result[i_upper] = result.T[i_upper]
    desired = evals * np.identity(N)
    output = np.abs(result - desired).mean()
    assert_allclose(output, 0, atol=1e-5)
