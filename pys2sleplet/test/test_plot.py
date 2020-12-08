import cmocean
import numpy as np
import pyssht as ssht
from numpy.testing import assert_equal, assert_raises

from pys2sleplet.test.constants import L_LARGE, L_SMALL, PHI_0, PL_ENTRIES, THETA_1
from pys2sleplet.utils.plot_methods import (
    _create_max_amplitues_dict,
    calc_nearest_grid_point,
    calc_plot_resolution,
    convert_colourscale,
    find_max_amplitude,
)
from pys2sleplet.utils.vars import SAMPLING_SCHEME


def test_resolution_values() -> None:
    """
    verifies the correct resolution is chosen for a given bandlimit
    """
    arguments = [1, 10, 100, 1000]
    output = [64, 80, 800, 2000]
    for c, arg in enumerate(arguments):
        assert_equal(calc_plot_resolution(arg), output[c])


def test_create_colourscale() -> None:
    """
    test creates a plotly compatible colourscale
    """
    colourscale = convert_colourscale(cmocean.cm.ice, pl_entries=PL_ENTRIES)
    assert_equal(len(colourscale), PL_ENTRIES)


def test_find_nearest_grid_point() -> None:
    """
    test to find nearest grid point to provided angles
    """
    alpha, beta = calc_nearest_grid_point(L_LARGE, PHI_0 / np.pi, THETA_1 / np.pi)
    assert_equal(alpha, 0.5154175447295755)
    assert_equal(beta, 1.055378782065321)


def test_amplitudes_dict(random_flm) -> None:
    """
    ensures the amplitudes dict is of the expected form
    """
    field = ssht.inverse(random_flm, L_SMALL, Method=SAMPLING_SCHEME)
    amplitudes = _create_max_amplitues_dict(field)
    assert_equal(amplitudes.keys(), {"abs", "imag", "real", "sum"})
    assert all(v >= 0 for v in amplitudes.values())


def test_find_max_amplitude_1d(random_flm) -> None:
    """
    amplitude max method is not designed for 1D inputs
    """
    assert_raises(np.AxisError, find_max_amplitude, L_SMALL, random_flm)


def test_find_max_amplitude_harmonic(random_nd_flm) -> None:
    """
    ampltiude max method for a harmonic example
    """
    amplitudes = find_max_amplitude(L_SMALL, random_nd_flm)
    assert_equal(amplitudes.keys(), {"abs", "imag", "real", "sum"})
    assert all(v >= 0 for v in amplitudes.values())


def test_find_max_amplitude_slepian(random_nd_flm, slepian_polar_cap) -> None:
    """
    ampltiude max method for a slepian example
    """
    amplitudes = find_max_amplitude(L_SMALL, random_nd_flm, slepian=slepian_polar_cap)
    assert_equal(amplitudes.keys(), {"abs", "imag", "real", "sum"})
    assert all(v >= 0 for v in amplitudes.values())
