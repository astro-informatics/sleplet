import numpy as np
from igl import gaussian_curvature
from numpy.testing import assert_allclose

from pys2sleplet.utils.mesh_methods import mesh_forward, mesh_inverse


def test_forward_inverse_transform_recovery(
    bunny_mesh, bunny_mesh_eigendecomposition
) -> None:
    """
    tests that a given function is recovered after an
    forward and inverse transform on the mesh
    """
    vertices, faces = bunny_mesh
    _, basis_functions = bunny_mesh_eigendecomposition
    kernel = gaussian_curvature(vertices, faces)
    u_i = mesh_forward(vertices, faces, basis_functions, kernel)
    kernel_recov = mesh_inverse(basis_functions, u_i)
    assert_allclose(np.abs(kernel - kernel_recov).mean(), 0, atol=0.8)
