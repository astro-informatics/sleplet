import numpy as np
import pytest
from igl import gaussian_curvature
from numpy.testing import assert_allclose, assert_equal

from pys2sleplet.utils.mesh_methods import (
    create_mesh_region,
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


def test_mesh_region_is_some_fraction_of_total(mesh) -> None:
    """
    the region should be some fraction of the total nodes
    """
    region = create_mesh_region(mesh.name, mesh.vertices)
    assert region.sum() < region.shape[0]


@pytest.mark.slow
def test_orthonormality_over_mesh_full(mesh) -> None:
    """
    for the computation of the Slepian D matrix the basis
    functions must be orthornomal over the whole mesh
    """
    orthonormality = np.zeros(
        (mesh.basis_functions.shape[0], mesh.basis_functions.shape[0])
    )
    for i, phi_i in enumerate(mesh.basis_functions):
        for j, phi_j in enumerate(mesh.basis_functions):
            orthonormality[i, j] = integrate_whole_mesh(
                mesh.vertices, mesh.faces, phi_i, phi_j
            )
    identity = np.identity(mesh.basis_functions.shape[0])
    np.testing.assert_allclose(np.abs(orthonormality - identity).mean(), 0, atol=0.04)
