import numpy as np
from igl import doublearea, gaussian_curvature
from numpy.testing import assert_allclose, assert_equal

from pys2sleplet.utils.mesh_methods import (
    compute_shannon,
    integrate_whole_mesh,
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


def test_integrate_whole_mesh_equals_area(mesh) -> None:
    """
    ensures that integrating a whole mesh equals area of the mesh
    when the function defined over is just one at the vertices
    """
    integral = integrate_whole_mesh(mesh.vertices, mesh.faces, 1)
    area = (doublearea(mesh.vertices, mesh.faces) / 2).sum()
    assert_equal(integral, area)


def test_shannon_less_than_basis_functions(mesh) -> None:
    """
    Shannon number should be less than the total number of basis functions
    """
    shannon = compute_shannon(mesh.vertices, mesh.faces, mesh.region)
    assert shannon < mesh.basis_functions.shape[0]
