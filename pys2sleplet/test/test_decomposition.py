import numpy as np
import pyssht as ssht
from numpy.testing import assert_allclose, assert_raises

from pys2sleplet.slepian.slepian_decomposition import SlepianDecomposition
from pys2sleplet.utils.mask_methods import create_mask_region
from pys2sleplet.utils.slepian_methods import slepian_forward, slepian_inverse
from pys2sleplet.utils.vars import SAMPLING_SCHEME


def test_decompose_all_polar(slepian_polar_cap, earth_polar_cap) -> None:
    """
    tests that all three methods produce the same coefficients for polar cap
    """
    field = ssht.inverse(
        earth_polar_cap.coefficients, slepian_polar_cap.L, Method=SAMPLING_SCHEME
    )
    harmonic_sum_p = slepian_forward(
        slepian_polar_cap.L, slepian_polar_cap, flm=earth_polar_cap.coefficients
    )
    integrate_sphere_p = slepian_forward(
        slepian_polar_cap.L, slepian_polar_cap, f=field
    )
    integrate_region_p = slepian_forward(
        slepian_polar_cap.L, slepian_polar_cap, f=field, mask=slepian_polar_cap.mask
    )
    assert_allclose(
        np.abs(integrate_sphere_p - harmonic_sum_p)[: slepian_polar_cap.N].mean(),
        0,
        atol=12,
    )
    assert_allclose(
        np.abs(integrate_region_p - harmonic_sum_p)[: slepian_polar_cap.N].mean(),
        0,
        atol=17,
    )


def test_decompose_all_lim_lat_lon(slepian_lim_lat_lon, earth_lim_lat_lon) -> None:
    """
    tests that all three methods produce the same coefficients for
    limited latitude longitude region
    """
    field = ssht.inverse(
        earth_lim_lat_lon.coefficients, slepian_lim_lat_lon.L, Method=SAMPLING_SCHEME
    )
    harmonic_sum_p = slepian_forward(
        slepian_lim_lat_lon.L, slepian_lim_lat_lon, flm=earth_lim_lat_lon.coefficients
    )
    integrate_sphere_p = slepian_forward(
        slepian_lim_lat_lon.L, slepian_lim_lat_lon, f=field
    )
    integrate_region_p = slepian_forward(
        slepian_lim_lat_lon.L,
        slepian_lim_lat_lon,
        f=field,
        mask=slepian_lim_lat_lon.mask,
    )
    assert_allclose(
        np.abs(integrate_sphere_p - harmonic_sum_p)[: slepian_lim_lat_lon.N].mean(),
        0,
        atol=0.8,
    )
    assert_allclose(
        np.abs(integrate_region_p - harmonic_sum_p)[: slepian_lim_lat_lon.N].mean(),
        0,
        atol=72,
    )


def test_equality_to_harmonic_transform_polar(
    slepian_polar_cap, earth_polar_cap
) -> None:
    """
    tests that fp*Sp up to N is roughly equal to flm*Ylm
    """
    f_p = slepian_forward(
        slepian_polar_cap.L, slepian_polar_cap, flm=earth_polar_cap.coefficients
    )
    f_slepian = slepian_inverse(f_p, slepian_polar_cap.L, slepian_polar_cap)
    f_harmonic = ssht.inverse(
        earth_polar_cap.coefficients, slepian_polar_cap.L, Method=SAMPLING_SCHEME
    )
    mask = create_mask_region(slepian_polar_cap.L, slepian_polar_cap.region)
    assert_allclose(np.abs(f_slepian - f_harmonic)[mask].mean(), 0, atol=14)


def test_equality_to_harmonic_transform_lim_lat_lon(
    slepian_lim_lat_lon, earth_lim_lat_lon
) -> None:
    """
    tests that fp*Sp up to N is roughly equal to flm*Ylm
    """
    f_p = slepian_forward(
        slepian_lim_lat_lon.L, slepian_lim_lat_lon, flm=earth_lim_lat_lon.coefficients
    )
    f_slepian = slepian_inverse(f_p, slepian_lim_lat_lon.L, slepian_lim_lat_lon)
    f_harmonic = ssht.inverse(
        earth_lim_lat_lon.coefficients, slepian_lim_lat_lon.L, Method=SAMPLING_SCHEME
    )
    mask = create_mask_region(slepian_lim_lat_lon.L, slepian_lim_lat_lon.region)
    assert_allclose(np.abs(f_slepian - f_harmonic)[mask].mean(), 0, atol=123)


def test_pass_rank_higher_than_available(slepian_polar_cap, earth_polar_cap) -> None:
    """
    tests that asking for a Slepian coefficients above the limit fails
    """
    sd = SlepianDecomposition(
        slepian_polar_cap.L, slepian_polar_cap, flm=earth_polar_cap.coefficients
    )
    assert_raises(ValueError, sd.decompose, slepian_polar_cap.L**2)


def test_no_method_found_for_decomposition(slepian_polar_cap) -> None:
    """
    checks that no method has been found when inputs haven't been set
    """
    assert_raises(
        RuntimeError, SlepianDecomposition, slepian_polar_cap.L, slepian_polar_cap
    )


def test_proportion_wavelet_energy_leakage_polar(
    slepian_polar_cap, earth_polar_cap
) -> None:
    """
    tests that the proportion of function energy leakage is as expected
    """
    field = ssht.inverse(
        earth_polar_cap.coefficients, slepian_polar_cap.L, Method=SAMPLING_SCHEME
    )
    integrate_region_p = slepian_forward(
        slepian_polar_cap.L,
        slepian_polar_cap,
        f=field,
        mask=slepian_polar_cap.mask,
        n_coeffs=slepian_polar_cap.L**2,
    )
    slepian_energy = np.abs(integrate_region_p) ** 2
    proportion_inside = (
        np.sort(slepian_energy)[::-1][: slepian_polar_cap.N].sum()
        / slepian_energy.sum()
    )
    assert_allclose(1 - proportion_inside, 0, atol=7e-5)


def test_proportion_wavelet_energy_leakage_lim_lat_lon(
    slepian_lim_lat_lon, earth_lim_lat_lon
) -> None:
    """
    tests that the proportion of function energy leakage is as expected
    """
    field = ssht.inverse(
        earth_lim_lat_lon.coefficients, slepian_lim_lat_lon.L, Method=SAMPLING_SCHEME
    )
    integrate_region_p = slepian_forward(
        slepian_lim_lat_lon.L,
        slepian_lim_lat_lon,
        f=field,
        mask=slepian_lim_lat_lon.mask,
        n_coeffs=slepian_lim_lat_lon.L**2,
    )
    slepian_energy = np.abs(integrate_region_p) ** 2
    proportion_inside = (
        np.sort(slepian_energy)[::-1][: slepian_lim_lat_lon.N].sum()
        / slepian_energy.sum()
    )
    assert_allclose(1 - proportion_inside, 0, atol=0.2)
