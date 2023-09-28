import numpy as np
import s2fft

import sleplet


def test_decompose_all_polar(slepian_polar_cap, earth_polar_cap) -> None:
    """Tests that all three methods produce the same coefficients for polar cap."""
    field = s2fft.inverse(
        s2fft.sampling.s2_samples.flm_1d_to_2d(
            earth_polar_cap.coefficients,
            slepian_polar_cap.L,
        ),
        slepian_polar_cap.L,
        method=sleplet._vars.EXECUTION_MODE,
        sampling=sleplet._vars.SAMPLING_SCHEME,
    )
    harmonic_sum_p = sleplet.slepian_methods.slepian_forward(
        slepian_polar_cap.L,
        slepian_polar_cap,
        flm=earth_polar_cap.coefficients,
    )
    integrate_sphere_p = sleplet.slepian_methods.slepian_forward(
        slepian_polar_cap.L,
        slepian_polar_cap,
        f=field,
    )
    integrate_region_p = sleplet.slepian_methods.slepian_forward(
        slepian_polar_cap.L,
        slepian_polar_cap,
        f=field,
        mask=slepian_polar_cap.mask,
    )
    np.testing.assert_allclose(
        np.abs(integrate_sphere_p - harmonic_sum_p)[: slepian_polar_cap.N].mean(),
        0,
        atol=12,
    )
    np.testing.assert_allclose(
        np.abs(integrate_region_p - harmonic_sum_p)[: slepian_polar_cap.N].mean(),
        0,
        atol=17,
    )


def test_decompose_all_lim_lat_lon(slepian_lim_lat_lon, earth_lim_lat_lon) -> None:
    """
    Tests that all three methods produce the same coefficients for
    limited latitude longitude region.
    """
    field = s2fft.inverse(
        s2fft.sampling.s2_samples.flm_1d_to_2d(
            earth_lim_lat_lon.coefficients,
            slepian_lim_lat_lon.L,
        ),
        slepian_lim_lat_lon.L,
        method=sleplet._vars.EXECUTION_MODE,
        sampling=sleplet._vars.SAMPLING_SCHEME,
    )
    harmonic_sum_p = sleplet.slepian_methods.slepian_forward(
        slepian_lim_lat_lon.L,
        slepian_lim_lat_lon,
        flm=earth_lim_lat_lon.coefficients,
    )
    integrate_sphere_p = sleplet.slepian_methods.slepian_forward(
        slepian_lim_lat_lon.L,
        slepian_lim_lat_lon,
        f=field,
    )
    integrate_region_p = sleplet.slepian_methods.slepian_forward(
        slepian_lim_lat_lon.L,
        slepian_lim_lat_lon,
        f=field,
        mask=slepian_lim_lat_lon.mask,
    )
    np.testing.assert_allclose(
        np.abs(integrate_sphere_p - harmonic_sum_p)[: slepian_lim_lat_lon.N].mean(),
        0,
        atol=3.5,
    )
    np.testing.assert_allclose(
        np.abs(integrate_region_p - harmonic_sum_p)[: slepian_lim_lat_lon.N].mean(),
        0,
        atol=90,
    )


def test_equality_to_harmonic_transform_polar(
    slepian_polar_cap,
    earth_polar_cap,
) -> None:
    """Tests that fp*Sp up to N is roughly equal to flm*Ylm."""
    f_p = sleplet.slepian_methods.slepian_forward(
        slepian_polar_cap.L,
        slepian_polar_cap,
        flm=earth_polar_cap.coefficients,
    )
    f_slepian = sleplet.slepian_methods.slepian_inverse(
        f_p,
        slepian_polar_cap.L,
        slepian_polar_cap,
    )
    f_harmonic = s2fft.inverse(
        s2fft.sampling.s2_samples.flm_1d_to_2d(
            earth_polar_cap.coefficients,
            slepian_polar_cap.L,
        ),
        slepian_polar_cap.L,
        method=sleplet._vars.EXECUTION_MODE,
        sampling=sleplet._vars.SAMPLING_SCHEME,
    )
    mask = sleplet._mask_methods.create_mask_region(
        slepian_polar_cap.L,
        slepian_polar_cap.region,
    )
    np.testing.assert_allclose(np.abs(f_slepian - f_harmonic)[mask].mean(), 0, atol=89)


def test_equality_to_harmonic_transform_lim_lat_lon(
    slepian_lim_lat_lon,
    earth_lim_lat_lon,
) -> None:
    """Tests that fp*Sp up to N is roughly equal to flm*Ylm."""
    f_p = sleplet.slepian_methods.slepian_forward(
        slepian_lim_lat_lon.L,
        slepian_lim_lat_lon,
        flm=earth_lim_lat_lon.coefficients,
    )
    f_slepian = sleplet.slepian_methods.slepian_inverse(
        f_p,
        slepian_lim_lat_lon.L,
        slepian_lim_lat_lon,
    )
    f_harmonic = s2fft.inverse(
        s2fft.sampling.s2_samples.flm_1d_to_2d(
            earth_lim_lat_lon.coefficients,
            slepian_lim_lat_lon.L,
        ),
        slepian_lim_lat_lon.L,
        method=sleplet._vars.EXECUTION_MODE,
        sampling=sleplet._vars.SAMPLING_SCHEME,
    )
    mask = sleplet._mask_methods.create_mask_region(
        slepian_lim_lat_lon.L,
        slepian_lim_lat_lon.region,
    )
    np.testing.assert_allclose(np.abs(f_slepian - f_harmonic)[mask].mean(), 0, atol=248)


def test_pass_rank_higher_than_available(slepian_polar_cap, earth_polar_cap) -> None:
    """Tests that asking for a Slepian coefficients above the limit fails."""
    sd = sleplet.slepian._slepian_decomposition.SlepianDecomposition(
        slepian_polar_cap.L,
        slepian_polar_cap,
        flm=earth_polar_cap.coefficients,
    )
    np.testing.assert_raises(ValueError, sd.decompose, slepian_polar_cap.L**2)


def test_no_method_found_for_decomposition(slepian_polar_cap) -> None:
    """Checks that no method has been found when inputs haven't been set."""
    np.testing.assert_raises(
        RuntimeError,
        sleplet.slepian._slepian_decomposition.SlepianDecomposition,
        slepian_polar_cap.L,
        slepian_polar_cap,
    )
