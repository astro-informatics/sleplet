from pathlib import Path

import numpy as np
import pyssht as ssht
from numpy import typing as npt

from sleplet.data.other.earth.create_earth_flm import create_flm
from sleplet.utils.plot_methods import (
    rotate_earth_to_africa,
    rotate_earth_to_south_america,
)
from sleplet.utils.vars import AFRICA_RANGE, SAMPLING_SCHEME, SOUTH_AMERICA_RANGE

_data_path = Path(__file__).resolve().parents[3] / "data"


def _create_africa_mask(
    L: int, earth_flm: npt.NDArray[np.complex_]
) -> npt.NDArray[np.float_]:
    """
    creates the Africa region mask
    """
    rot_flm = rotate_earth_to_africa(earth_flm, L)
    earth_f = ssht.inverse(rot_flm, L, Reality=True, Method=SAMPLING_SCHEME)
    thetas, _ = ssht.sample_positions(L, Grid=True, Method=SAMPLING_SCHEME)
    return (thetas <= AFRICA_RANGE) & (earth_f >= 0)


def _create_south_america_mask(
    L: int, earth_flm: npt.NDArray[np.complex_]
) -> npt.NDArray[np.float_]:
    """
    creates the Africa region mask
    """
    rot_flm = rotate_earth_to_south_america(earth_flm, L)
    earth_f = ssht.inverse(rot_flm, L, Reality=True, Method=SAMPLING_SCHEME)
    thetas, _ = ssht.sample_positions(L, Grid=True, Method=SAMPLING_SCHEME)
    return (thetas <= SOUTH_AMERICA_RANGE) & (earth_f >= 0)


def create_mask(L: int, mask_name: str) -> None:
    """
    creates the South America region mask
    """
    earth_flm = create_flm(L)
    if mask_name == "africa":
        mask = _create_africa_mask(L, earth_flm)
    elif mask_name == "south_america":
        mask = _create_south_america_mask(L, earth_flm)
    else:
        raise ValueError(f"Mask name {mask_name} not recognised")
    np.save(_data_path / f"slepian_masks_{mask_name}_L{L}.npy", mask)
