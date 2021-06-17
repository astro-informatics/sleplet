import numpy as np
from igl import doublearea, gaussian_curvature
from numpy.testing import assert_allclose, assert_equal

from pys2sleplet.utils.mesh_methods import (
    integrate_whole_mesh,
    mesh_forward,
    mesh_inverse,
)


def test_forward_inverse_transform_recovery(
    bird_mesh, bird_mesh_eigendecomposition
) -> None:
    """
    tests that a given function is recovered after an
    forward and inverse transform on the mesh
    """
    vertices, faces = bird_mesh
    _, basis_functions = bird_mesh_eigendecomposition
    kernel = gaussian_curvature(vertices, faces)
    u_i = mesh_forward(vertices, faces, basis_functions, kernel)
    kernel_recov = mesh_inverse(basis_functions, u_i)
    assert_allclose(np.abs(kernel - kernel_recov).mean(), 0, atol=0.2)


def test_integrate_whole_mesh_equals_area(bird_mesh) -> None:
    """
    ensures that integrating a whole mesh equals area of the mesh
    when the function defined over is just one at the vertices
    """
    vertices, faces = bird_mesh
    integral = integrate_whole_mesh(vertices, faces, 1)
    area = (doublearea(vertices, faces) / 2).sum()
    assert_equal(integral, area)
