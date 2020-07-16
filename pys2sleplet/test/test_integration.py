import numpy as np
import pytest
from hypothesis import given, seed, settings
from hypothesis.strategies import SearchStrategy, integers
from numpy.testing import assert_allclose, assert_raises

from pys2sleplet.test.constants import L_SMALL as L
from pys2sleplet.test.constants import ORDER, RANDOM_SEED
from pys2sleplet.utils.mask_methods import create_mask_region
from pys2sleplet.utils.slepian_methods import (
    integrate_two_slepian_functions_per_rank,
    integrate_whole_matrix_slepian_functions,
)


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
def test_integrate_two_slepian_polar_functions_whole_sphere_per_rank(
    slepian_polar_cap, rank1, rank2
) -> None:
    """
    tests that integration of two slepian poalr functions over the
    whole sphere gives the Kronecker delta
    """
    output = integrate_two_slepian_functions_per_rank(
        slepian_polar_cap.eigenvectors, L, slepian_polar_cap.resolution, rank1, rank2
    )
    if rank1 == rank2:
        assert_allclose(output, 1, atol=0.04)
    else:
        assert_allclose(output, 0, atol=0.04)


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
    output = integrate_two_slepian_functions_per_rank(
        slepian_lim_lat_lon.eigenvectors,
        L,
        slepian_lim_lat_lon.resolution,
        rank1,
        rank2,
    )
    if rank1 == rank2:
        assert_allclose(output, 1, atol=1e-4)
    else:
        assert_allclose(output, 0, atol=1e-3)


@seed(RANDOM_SEED)
@settings(max_examples=8, deadline=None)
@given(rank1=valid_polar_ranks(), rank2=valid_polar_ranks())
def test_integrate_two_slepian_polar_functions_region_sphere_per_rank(
    slepian_polar_cap, rank1, rank2
) -> None:
    """
    tests that integration of two slepian polar functions over a region on
    the sphere gives the Kronecker delta multiplied by the eigenvalue
    """
    output = integrate_two_slepian_functions_per_rank(
        slepian_polar_cap.eigenvectors,
        L,
        slepian_polar_cap.resolution,
        rank1,
        rank2,
        mask=slepian_polar_cap.mask,
    )
    lambda_p = slepian_polar_cap.eigenvalues[rank1]
    if rank1 == rank2:
        assert_allclose(output, lambda_p, rtol=0.04)
    else:
        assert_allclose(output, 0, atol=0.04)


@seed(RANDOM_SEED)
@settings(max_examples=8, deadline=None)
@given(rank1=valid_lim_lat_lon_ranks(), rank2=valid_lim_lat_lon_ranks())
def test_integrate_two_slepian_lim_lat_lon_functions_region_sphere_per_rank(
    slepian_lim_lat_lon, rank1, rank2
) -> None:
    """
    tests that integration of two slepian lim lat lon functions over a region on
    the sphere gives the Kronecker delta multiplied by the eigenvalue
    """
    output = integrate_two_slepian_functions_per_rank(
        slepian_lim_lat_lon.eigenvectors,
        L,
        slepian_lim_lat_lon.resolution,
        rank1,
        rank2,
        mask=slepian_lim_lat_lon.mask,
    )
    lambda_p = slepian_lim_lat_lon.eigenvalues[rank1]
    if rank1 == rank2:
        assert_allclose(output, lambda_p, rtol=1)
    else:
        assert_allclose(output, 0, atol=1e-2)


def test_integrate_two_slepian_polar_cap_functions_whole_sphere_matrix(
    slepian_polar_cap,
) -> None:
    """
    tests that integration of two slepian polar cap functions over the
    whole sphere gives the identity matrix
    """
    output = integrate_whole_matrix_slepian_functions(
        slepian_polar_cap.eigenvectors, L, slepian_polar_cap.resolution
    )
    desired = np.identity(output.shape[0])
    assert_allclose(np.abs(output - desired).mean(), 0, atol=1e-2)


@pytest.mark.slow
def test_integrate_two_slepian_lim_lat_lon_functions_whole_sphere_matrix(
    slepian_lim_lat_lon,
) -> None:
    """
    tests that integration of two slepian lim lat lon functions over the
    whole sphere gives the identity matrix
    """
    output = integrate_whole_matrix_slepian_functions(
        slepian_lim_lat_lon.eigenvectors, L, slepian_lim_lat_lon.resolution
    )
    desired = np.identity(output.shape[0])
    assert_allclose(np.abs(output - desired).mean(), 0, atol=1e-3)


def test_integrate_two_slepian_polar_cap_functions_region_sphere_matrix(
    slepian_polar_cap,
) -> None:
    """
    tests that integration of two slepian lim lat lon functions over a region on
    the sphere gives the identity matrix multiplied by the eigenvalue
    """
    output = integrate_whole_matrix_slepian_functions(
        slepian_polar_cap.eigenvectors,
        L,
        slepian_polar_cap.resolution,
        mask=slepian_polar_cap.mask,
    )
    desired = slepian_polar_cap.eigenvalues * np.identity(output.shape[0])
    assert_allclose(np.abs(output - desired).mean(), 0, atol=1e-2)


@pytest.mark.slow
def test_integrate_two_slepian_lim_lat_lon_functions_region_sphere_matrix(
    slepian_lim_lat_lon,
) -> None:
    """
    tests that integration of two slepian lim lat lon functions over a region on
    the sphere gives the identity matrix multiplied by the eigenvalue
    """
    output = integrate_whole_matrix_slepian_functions(
        slepian_lim_lat_lon.eigenvectors,
        L,
        slepian_lim_lat_lon.resolution,
        mask=slepian_lim_lat_lon.mask,
    )
    desired = slepian_lim_lat_lon.eigenvalues * np.identity(output.shape[0])
    assert_allclose(np.abs(output - desired).mean(), 0, atol=1e-4)


def test_pass_incorrect_mask_size_to_integrate_region(
    slepian_polar_cap, polar_cap_region
) -> None:
    """
    tests an exception is thrown if the mask passed to the function is the wrong shape
    """
    mask = create_mask_region(L, polar_cap_region)
    assert_raises(
        AttributeError,
        integrate_whole_matrix_slepian_functions,
        slepian_polar_cap.eigenvectors,
        L,
        slepian_polar_cap.resolution,
        mask=mask,
    )
