# import numpy as np
# import pytest
# from numpy.testing import assert_allclose

# from pys2sleplet.utils.harmonic_methods import mesh_forward, mesh_inverse
# from pys2sleplet.utils.integration_methods import integrate_whole_mesh
# from pys2sleplet.utils.mask_methods import create_mesh_region


# def test_forward_inverse_transform_recovery(mesh_field_region) -> None:
#     """
#     tests that a given function is recovered after an
#     forward and inverse transform on the mesh
#     """
#     u_i = mesh_forward(
#         mesh_field_region.mesh_field.mesh.basis_functions,
#         mesh_field_region.field_values,
#     )
#     kernel_recov = mesh_inverse(mesh_field_region.mesh_field.mesh.basis_functions, u_i)
#     assert_allclose(
#         np.abs(mesh_field_region.field_values - kernel_recov).mean(), 0, atol=1e-14
#     )


# def test_mesh_region_is_some_fraction_of_total(mesh) -> None:
#     """
#     the region should be some fraction of the total nodes
#     """
#     region = create_mesh_region(mesh.name, mesh.vertices)
#     assert region.sum() < region.shape[0]


# @pytest.mark.slow
# def test_orthonormality_over_mesh_full(mesh) -> None:
#     """
#     for the computation of the Slepian D matrix the basis
#     functions must be orthornomal over the whole mesh
#     """
#     orthonormality = np.zeros(
#         (mesh.basis_functions.shape[0], mesh.basis_functions.shape[0])
#     )
#     for i, phi_i in enumerate(mesh.basis_functions):
#         for j, phi_j in enumerate(mesh.basis_functions):
#             orthonormality[i, j] = integrate_whole_mesh(phi_i, phi_j)
#     identity = np.identity(mesh.basis_functions.shape[0])
#     np.testing.assert_allclose(np.abs(orthonormality - identity).mean(), 0, atol=1e-16)
