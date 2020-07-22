import numpy as np
from numpy.testing import assert_allclose, assert_raises

from pys2sleplet.test.constants import L_SMALL as L
from pys2sleplet.utils.mask_methods import create_mask_region
from pys2sleplet.utils.slepian_methods import integrate_whole_matrix_slepian_functions


def test_integrate_two_slepian_polar_cap_functions_whole_sphere_matrix(
    slepian_polar_cap,
) -> None:
    """
    tests that integration of two slepian polar cap functions over the
    whole sphere gives the identity matrix
    """
    output = integrate_whole_matrix_slepian_functions(
        slepian_polar_cap.eigenvectors[: slepian_polar_cap.shannon],
        L,
        slepian_polar_cap.resolution,
    )
    desired = np.identity(output.shape[0])
    assert_allclose(np.abs(output - desired).mean(), 0, atol=1e-3)


def test_integrate_two_slepian_lim_lat_lon_functions_whole_sphere_matrix(
    slepian_lim_lat_lon,
) -> None:
    """
    tests that integration of two slepian lim lat lon functions over the
    whole sphere gives the identity matrix
    """
    output = integrate_whole_matrix_slepian_functions(
        slepian_lim_lat_lon.eigenvectors[: slepian_lim_lat_lon.shannon],
        L,
        slepian_lim_lat_lon.resolution,
    )
    desired = np.identity(output.shape[0])
    assert_allclose(np.abs(output - desired).mean(), 0, atol=1e-4)


def test_integrate_two_slepian_polar_cap_functions_region_sphere_matrix(
    slepian_polar_cap,
) -> None:
    """
    tests that integration of two slepian lim lat lon functions over a region on
    the sphere gives the identity matrix multiplied by the eigenvalue
    """
    output = integrate_whole_matrix_slepian_functions(
        slepian_polar_cap.eigenvectors[: slepian_polar_cap.shannon],
        L,
        slepian_polar_cap.resolution,
        mask=slepian_polar_cap.mask,
    )
    desired = slepian_polar_cap.eigenvalues[: slepian_polar_cap.shannon] * np.identity(
        output.shape[0]
    )
    assert_allclose(np.abs(output - desired).mean(), 0, atol=1e-2)


def test_integrate_two_slepian_lim_lat_lon_functions_region_sphere_matrix(
    slepian_lim_lat_lon,
) -> None:
    """
    tests that integration of two slepian lim lat lon functions over a region on
    the sphere gives the identity matrix multiplied by the eigenvalue
    """
    output = integrate_whole_matrix_slepian_functions(
        slepian_lim_lat_lon.eigenvectors[: slepian_lim_lat_lon.shannon],
        L,
        slepian_lim_lat_lon.resolution,
        mask=slepian_lim_lat_lon.mask,
    )
    desired = slepian_lim_lat_lon.eigenvalues[
        : slepian_lim_lat_lon.shannon
    ] * np.identity(output.shape[0])
    assert_allclose(np.abs(output - desired).mean(), 0, atol=0.02)


def test_pass_incorrect_mask_size_to_integrate_region(slepian_polar_cap) -> None:
    """
    tests an exception is thrown if the mask passed to the function is the wrong shape
    """
    mask = create_mask_region(L, slepian_polar_cap.region)
    assert_raises(
        AttributeError,
        integrate_whole_matrix_slepian_functions,
        slepian_polar_cap.eigenvectors,
        L,
        slepian_polar_cap.resolution,
        mask=mask,
    )
