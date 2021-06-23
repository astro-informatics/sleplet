import numpy as np
from igl import gaussian_curvature
from numpy.testing import assert_allclose, assert_equal

from pys2sleplet.utils.mesh_methods import (
    compute_shannon,
    create_mesh_region,
    mesh_forward,
    mesh_inverse,
)


def test_forward_inverse_transform_recovery(mesh) -> None:
    """
    tests that a given function is recovered after an
    forward and inverse transform on the mesh
    """
    kernel = gaussian_curvature(mesh.vertices, mesh.faces)
    u_i = mesh_forward(mesh.vertices, mesh.faces, mesh.basis_functions, kernel)
    kernel_recov = mesh_inverse(mesh.basis_functions, u_i)
    assert_allclose(np.abs(kernel - kernel_recov).mean(), 0, atol=0.2)
    assert_equal(mesh.vertices.shape[0], kernel_recov.shape[0])


def test_shannon_less_than_basis_functions(mesh) -> None:
    """
    Shannon number should be less than the total number of basis functions
    """
    shannon = compute_shannon(mesh)
    assert shannon < mesh.basis_functions.shape[0]


def test_mesh_region_is_some_fraction_of_totel(mesh) -> None:
    """
    the region should be some fraction of the total nodes
    """
    region = create_mesh_region(mesh.name, mesh.vertices)
    assert region.sum() < region.shape[0]
