import numpy as np
import pyssht as ssht
from numpy.testing import assert_allclose, assert_raises

from pys2sleplet.flm.maps.earth import Earth
from pys2sleplet.slepian.slepian_decomposition import SlepianDecomposition
from pys2sleplet.test.constants import L_SMALL as L
from pys2sleplet.utils.mask_methods import create_mask_region
from pys2sleplet.utils.slepian_methods import slepian_inverse
from pys2sleplet.utils.vars import SAMPLING_SCHEME


def test_decompose_all_polar(polar_cap_decomposition) -> None:
    """
    tests that all three methods produce the same coefficients for polar cap
    """
    f_p = polar_cap_decomposition.decompose_all(method="integrate_region")
    g_p = polar_cap_decomposition.decompose_all(method="integrate_sphere")
    h_p = polar_cap_decomposition.decompose_all(method="harmonic_sum")
    assert_allclose(np.abs(f_p - h_p)[: polar_cap_decomposition.N].mean(), 0, atol=8)
    assert_allclose(np.abs(g_p - h_p)[: polar_cap_decomposition.N].mean(), 0, atol=2)


def test_decompose_all_lim_lat_lon(lim_lat_lon_decomposition) -> None:
    """
    tests that all three methods produce the same coefficients for
    limited latitude longitude region
    """
    f_p = lim_lat_lon_decomposition.decompose_all(method="integrate_region")
    g_p = lim_lat_lon_decomposition.decompose_all(method="integrate_sphere")
    h_p = lim_lat_lon_decomposition.decompose_all(method="harmonic_sum")
    assert_allclose(np.abs(f_p - h_p)[: lim_lat_lon_decomposition.N].mean(), 0, atol=18)
    assert_allclose(
        np.abs(g_p - h_p)[: lim_lat_lon_decomposition.N].mean(), 0, atol=1e-2
    )


def test_equality_to_harmonic_transform_polar(polar_cap_decomposition) -> None:
    """
    tests that fp*Sp up to N is roughly equal to flm*Ylm
    """
    f_p = polar_cap_decomposition.decompose_all()
    f_slepian = slepian_inverse(
        L, f_p, polar_cap_decomposition.s_p_lms, coefficients=polar_cap_decomposition.N
    )
    f_harmonic = ssht.inverse(polar_cap_decomposition.flm, L, Method=SAMPLING_SCHEME)
    mask = create_mask_region(L, polar_cap_decomposition.function.region)
    assert_allclose(np.abs(f_slepian - f_harmonic)[mask].mean(), 0, atol=14)


def test_equality_to_harmonic_transform_lim_lat_lon(lim_lat_lon_decomposition) -> None:
    """
    tests that fp*Sp up to N is roughly equal to flm*Ylm
    """
    f_p = lim_lat_lon_decomposition.decompose_all()
    f_slepian = slepian_inverse(
        L,
        f_p,
        lim_lat_lon_decomposition.s_p_lms,
        coefficients=lim_lat_lon_decomposition.N,
    )
    f_harmonic = ssht.inverse(lim_lat_lon_decomposition.flm, L, Method=SAMPLING_SCHEME)
    mask = create_mask_region(L, lim_lat_lon_decomposition.function.region)
    assert_allclose(np.abs(f_slepian - f_harmonic)[mask].mean(), 0, atol=123)


def test_pass_function_without_region() -> None:
    """
    tests that the class throws an exception if no Region is passed to the function
    """
    earth = Earth(L)
    assert_raises(AttributeError, SlepianDecomposition, earth)


def test_pass_rank_higher_than_available(polar_cap_decomposition) -> None:
    """
    tests that asking for a Slepian coefficient above the limit fails
    """
    assert_raises(ValueError, polar_cap_decomposition.decompose, L ** 2)
