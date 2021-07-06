import numpy as np
import pytest
from numpy.testing import assert_allclose

from pys2sleplet.utils.mesh_methods import mesh_forward
from pys2sleplet.utils.slepian_mesh_methods import (
    compute_shannon,
    slepian_mesh_forward,
    slepian_mesh_inverse,
)


def test_shannon_less_than_basis_functions(mesh) -> None:
    """
    Shannon number should be less than the total number of basis functions
    """
    shannon = compute_shannon(mesh)
    assert shannon < mesh.basis_functions.shape[0]


@pytest.mark.slow
def test_decompose_all_mesh(slepian_mesh, mesh_field_masked) -> None:
    """
    tests that all three methods produce the same coefficients for the mesh
    """
    harmonic_coefficients = mesh_forward(
        mesh_field_masked.mesh.vertices,
        mesh_field_masked.mesh.faces,
        mesh_field_masked.mesh.basis_functions,
        mesh_field_masked.field_values,
    )
    harmonic_sum_p = slepian_mesh_forward(
        slepian_mesh.mesh,
        slepian_mesh.slepian_eigenvalues,
        slepian_mesh.slepian_functions,
        slepian_mesh.N,
        u_i=harmonic_coefficients,
    )
    integrate_sphere_p = slepian_mesh_forward(
        slepian_mesh.mesh,
        slepian_mesh.slepian_eigenvalues,
        slepian_mesh.slepian_functions,
        slepian_mesh.N,
        u=mesh_field_masked.field_values,
    )
    integrate_region_p = slepian_mesh_forward(
        slepian_mesh.mesh,
        slepian_mesh.slepian_eigenvalues,
        slepian_mesh.slepian_functions,
        slepian_mesh.N,
        u=slepian_mesh.mesh.region,
    )
    assert_allclose(
        np.abs(integrate_sphere_p - harmonic_sum_p)[: slepian_mesh.N].mean(),
        0,
        atol=1e-16,
    )
    assert_allclose(
        np.abs(integrate_region_p - harmonic_sum_p)[: slepian_mesh.N].mean(),
        0,
        atol=0.2,
    )


def test_forward_inverse_transform_slepian(slepian_mesh, mesh_field_masked) -> None:
    """
    tests that the Slepian forward and inverse transforms recover the field
    """
    f_p = slepian_mesh_forward(
        slepian_mesh.mesh,
        slepian_mesh.slepian_eigenvalues,
        slepian_mesh.slepian_functions,
        slepian_mesh.N,
        u=mesh_field_masked.field_values,
    )
    f_slepian = slepian_mesh_inverse(
        f_p, slepian_mesh.mesh, slepian_mesh.slepian_functions, slepian_mesh.N
    )
    assert_allclose(
        np.abs(f_slepian - mesh_field_masked.field_values)[
            slepian_mesh.mesh.region
        ].mean(),
        0,
        atol=0.5,
    )
