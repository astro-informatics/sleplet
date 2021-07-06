import numpy as np
from hypothesis import given, seed
from hypothesis.strategies import SearchStrategy, integers
from igl import gaussian_curvature
from numpy.testing import assert_allclose, assert_equal

from pys2sleplet.utils.mesh_methods import (
    create_mesh_region,
    integrate_whole_mesh,
    mesh_forward,
    mesh_inverse,
)
from pys2sleplet.utils.vars import RANDOM_SEED


def valid_indices() -> SearchStrategy[int]:
    """
    index can be in the range 0 to num_basis_function - 1
    """
    return integers(min_value=0, max_value=10)


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


@seed(RANDOM_SEED)
@given(i=valid_indices(), j=valid_indices())
def test_orthonormality_over_mesh(mesh, i, j) -> None:
    """
    for the computation of the Slepian D matrix the basis
    functions must be orthornomal over the whole mesh
    """
    integral = integrate_whole_mesh(
        mesh.vertices, mesh.faces, mesh.basis_functions[i], mesh.basis_functions[j]
    )
    assert_allclose(integral, 1) if i == j else assert_allclose(integral, 0, atol=0.3)
