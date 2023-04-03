import numpy as np
from numpy.testing import assert_allclose

import sleplet


def test_decompose_all_mesh(mesh_slepian, mesh_field_region) -> None:
    """Tests that all three methods produce the same coefficients for the mesh."""
    field = sleplet.harmonic_methods.mesh_inverse(
        mesh_slepian.mesh,
        mesh_field_region.coefficients,
    )
    harmonic_sum_p = sleplet.slepian_methods.slepian_mesh_forward(
        mesh_slepian,
        u_i=mesh_field_region.coefficients,
    )
    integrate_sphere_p = sleplet.slepian_methods.slepian_mesh_forward(
        mesh_slepian,
        u=field,
    )
    integrate_region_p = sleplet.slepian_methods.slepian_mesh_forward(
        mesh_slepian,
        u=field,
        mask=True,
    )
    assert_allclose(
        np.abs(integrate_sphere_p - harmonic_sum_p)[: mesh_slepian.N].mean(),
        0,
        atol=1e-14,
    )
    assert_allclose(
        np.abs(integrate_region_p - harmonic_sum_p)[: mesh_slepian.N].mean(),
        0,
        atol=1e-14,
    )


def test_forward_inverse_transform_slepian(mesh_slepian, mesh_field_region) -> None:
    """Tests that the Slepian forward and inverse transforms recover the field."""
    f_p = sleplet.slepian_methods.slepian_mesh_forward(
        mesh_slepian,
        u_i=mesh_field_region.coefficients,
    )
    f_slepian = sleplet.slepian_methods.slepian_mesh_inverse(mesh_slepian, f_p)
    f_harmonic = sleplet.harmonic_methods.mesh_inverse(
        mesh_slepian.mesh,
        mesh_field_region.coefficients,
    )
    assert_allclose(
        np.abs(f_slepian - f_harmonic)[mesh_slepian.mesh.region].mean(),
        0,
        atol=7e-3,
    )


def test_synthesis_mesh(mesh_slepian_wavelets, mesh_field_region) -> None:
    """Tests that Slepian polar wavelet synthesis matches the coefficients."""
    coefficients = sleplet.slepian_methods.slepian_mesh_forward(
        mesh_slepian_wavelets.mesh_slepian,
        u_i=mesh_field_region.coefficients,
    )
    wav_coeffs = sleplet.wavelet_methods.slepian_wavelet_forward(
        coefficients,
        mesh_slepian_wavelets.wavelets,
        mesh_slepian_wavelets.mesh_slepian.N,
    )
    f_p = sleplet.wavelet_methods.slepian_wavelet_inverse(
        wav_coeffs,
        mesh_slepian_wavelets.wavelets,
        mesh_slepian_wavelets.mesh_slepian.N,
    )
    assert_allclose(
        np.abs(f_p - coefficients)[: mesh_slepian_wavelets.mesh_slepian.N].mean(),
        0,
        atol=1e-16,
    )
